"""Train joint-only or IMU-based history policies with action DAgger.

Following the deployable-student protocol used by OmniH2O, the student controls
every rollout and the frozen privileged teacher labels the states visited by
that student. Both baselines use the same model, normalization, replay,
optimizer, and budget; their only experimental difference is the six
deployable IMU channels in the per-frame observation.
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="JOSE GMT-style history-policy distillation")
parser.add_argument("--method", choices=("joint_only", "imu"), required=True)
parser.add_argument("--teacher-checkpoint", required=True)
parser.add_argument("--task", default="Isaac-G1-AMP-Walk-JOSE-Direct-v0")
parser.add_argument("--agent", default="skrl_amp_cfg_entry_point")
parser.add_argument("--adapter", choices=("amp", "ppo_walk"), default="amp")
parser.add_argument("--window", type=int, default=21)
parser.add_argument("--num-envs", type=int, default=1024)
parser.add_argument("--num-iterations", type=int, default=300)
parser.add_argument("--rollout-steps", type=int, default=250)
parser.add_argument("--train-steps", type=int, default=0)
parser.add_argument("--batch-size", type=int, default=1024)
parser.add_argument("--lr", type=float, default=1.0e-4)
parser.add_argument("--weight-decay", type=float, default=1.0e-4)
parser.add_argument("--buffer-capacity", type=int, default=200_000)
parser.add_argument("--eval-interval", type=int, default=20)
parser.add_argument(
    "--eval-steps", type=int, default=None,
    help="Rollout steps per evaluation call (default: the task's own max episode length, "
    "so evaluation reaches the same natural episode boundary as the teacher)",
)
parser.add_argument("--save-interval", type=int, default=50)
parser.add_argument("--gyro-noise-std", type=float, default=0.015)
parser.add_argument("--gyro-bias-std", type=float, default=0.01)
parser.add_argument("--gravity-tilt-std-rad", type=float, default=0.015)
parser.add_argument("--max-latency-steps", type=int, default=2)
parser.add_argument("--disable-sensor-corruption", action="store_true")
parser.add_argument("--log-dir", default=None)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

from jose.teacher_setup import resolve_agent_entry_point  # noqa: E402

# `--adapter ppo_walk` implies the rsl-rl runner config unless overridden.
args_cli.agent = resolve_agent_entry_point(args_cli.adapter, args_cli.agent)

sys.argv = [sys.argv[0], *hydra_args]
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from datetime import datetime
import json
import math
from pathlib import Path
import time

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from isaaclab_tasks.utils.hydra import hydra_task_config
import isaaclab_tasks  # noqa: F401

from jose.distillation.history import (
    IMU_FRAME_DIM,
    JOINT_FRAME_DIM,
    HistoryMLPStudent,
    ObservationHistory,
    build_imu_frame,
    build_joint_frame,
)
from jose.distillation.imu import IMUObservationSpec, SensorCorruptionCfg, SensorCorruptor
from jose.estimator.adapters import make_policy_adapter
from jose.estimator.models import ReplayBuffer, RunningNormalizer
from jose.schema import SCHEMA_VERSION
from jose.teacher_setup import build_env_and_teacher
from jose.skrl_compat import amp_reward_components, force_skrl_isaaclab_reset, require_skrl_2


def _sensor_state(core) -> dict[str, torch.Tensor]:
    if not hasattr(core, "get_distillation_sensor_state"):
        raise RuntimeError("Task does not implement the JOSE distillation sensor contract")
    state = core.get_distillation_sensor_state()
    forbidden = {"base_linear_velocity", "linear_acceleration"}.intersection(state)
    if forbidden:
        raise RuntimeError(f"Deploy student was exposed to forbidden features: {sorted(forbidden)}")
    return state


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    if args_cli.window <= 0:
        raise ValueError("--window must be positive")
    torch.manual_seed(args_cli.seed)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed
    if args_cli.eval_steps is None:
        # Match DirectRLEnv.max_episode_length so evaluation runs long enough to
        # reach a natural termination/truncation, same as the teacher's own
        # rollout -- otherwise a policy that never falls within a shorter fixed
        # window silently reports that window length as its episode length.
        args_cli.eval_steps = math.ceil(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))
    # `teacher_setup` hides the SKRL/rsl-rl difference, so `--adapter ppo_walk`
    # loads the manager-based walk teacher through the same call.
    env, teacher_agent = build_env_and_teacher(
        args_cli.task,
        args_cli.adapter,
        env_cfg,
        agent_cfg,
        args_cli.teacher_checkpoint,
        args_cli.device,
        seed=args_cli.seed,
    )
    teacher_agent.enable_training_mode(False, apply_to_models=True)
    adapter = make_policy_adapter(args_cli.adapter, env, "all")
    core = adapter.core_env
    device = torch.device(core.device)

    corruption_cfg = SensorCorruptionCfg(
        gyro_noise_std=args_cli.gyro_noise_std,
        gyro_bias_std=args_cli.gyro_bias_std,
        gravity_tilt_std_rad=args_cli.gravity_tilt_std_rad,
        max_latency_steps=args_cli.max_latency_steps,
        enabled=not args_cli.disable_sensor_corruption,
    )
    corruptor = SensorCorruptor(args_cli.num_envs, device, corruption_cfg)
    imu_spec = IMUObservationSpec()
    frame_dim = JOINT_FRAME_DIM if args_cli.method == "joint_only" else IMU_FRAME_DIM
    student = HistoryMLPStudent(frame_dim, window=args_cli.window).to(device)
    history = ObservationHistory(args_cli.num_envs, args_cli.window, frame_dim, device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args_cli.lr, weight_decay=args_cli.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args_cli.num_iterations, eta_min=args_cli.lr * 0.1
    )
    observation_normalizer = RunningNormalizer(student.input_dim, device)
    action_normalizer = RunningNormalizer(29, device, clip=10.0)
    replay = ReplayBuffer(args_cli.buffer_capacity, student.input_dim, 29, device)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path(args_cli.log_dir or f"logs/jose_g1/distillation/{args_cli.method}/{timestamp}").resolve()
    checkpoints = log_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir / "tensorboard"))
    evaluations: list[dict] = []
    best_metrics: dict = {}
    best_length = float("-inf")
    best_iteration = 0

    def frame(use_corruption: bool = True) -> torch.Tensor:
        state = _sensor_state(core)
        if args_cli.method == "joint_only":
            return build_joint_frame(state["joint_position"], state["joint_velocity"], state["previous_action"])
        quaternion = state["quaternion_wxyz"]
        # q and -q represent the same attitude. Random sign changes make that
        # invariance explicit in the simulator-to-LowState training contract.
        if use_corruption:
            signs = torch.where(
                torch.rand(quaternion.shape[0], 1, device=quaternion.device) < 0.5,
                -torch.ones(1, device=quaternion.device),
                torch.ones(1, device=quaternion.device),
            )
            quaternion = quaternion * signs
        observation = imu_spec.observe(quaternion, state["angular_velocity"], timestamp_s=0.0)
        if not observation.valid:
            raise RuntimeError(f"Simulation IMU adapter fault: {observation.fault.value}")
        gyro, gravity = observation.angular_velocity, observation.projected_gravity
        if use_corruption:
            gyro, gravity = corruptor(gyro, gravity)
        return build_imu_frame(
            state["joint_position"], state["joint_velocity"], state["previous_action"], gyro, gravity
        )

    def payload(iteration: int) -> dict:
        return {
            "jose_schema_version": SCHEMA_VERSION,
            "kind": f"{args_cli.method}_history_student",
            "method": args_cli.method,
            "task": args_cli.task,
            "adapter": args_cli.adapter,
            "skrl_version": require_skrl_2(),
            "iteration": iteration,
            "distillation_protocol": "student_rollout_teacher_action_labels",
            "rollout_policy": "student",
            "model_config": student.config(),
            "model_state_dict": student.state_dict(),
            "observation_normalizer": observation_normalizer.state_dict(),
            "action_normalizer": action_normalizer.state_dict(),
            "sensor_corruption": corruption_cfg.__dict__,
            "observation_features": (
                ["joint_position", "joint_velocity", "previous_action"]
                if args_cli.method == "joint_only"
                else ["joint_position", "joint_velocity", "previous_action", "body_gyro", "quaternion_projected_gravity"]
            ),
            "explicit_linear_velocity": False,
            "raw_accelerometer": False,
        }

    def save(iteration: int, name: str) -> None:
        torch.save(payload(iteration), checkpoints / name)

    @torch.no_grad()
    def evaluate(use_corruption: bool) -> dict:
        nonlocal history, corruptor
        student.eval()
        force_skrl_isaaclab_reset(env)
        observations, _ = env.reset()
        history.reset()
        corruptor.reset()
        lengths = torch.zeros(args_cli.num_envs, device=device)
        returns = torch.zeros_like(lengths)
        previous = torch.zeros(args_cli.num_envs, 29, device=device)
        completed_lengths: list[float] = []
        completed_returns: list[float] = []
        deaths = timeouts = 0
        mse = smoothness = saturation = torque = energy = 0.0
        torque_saturation = raw_task_reward = base_linear_speed = base_angular_speed = 0.0
        amp_totals = {name: 0.0 for name in ("raw_style", "scaled_task", "scaled_style", "effective_reward")}
        amp_steps = 0
        inference_elapsed = 0.0
        for _ in range(args_cli.eval_steps):
            flattened = history.push(frame(use_corruption))
            teacher = adapter.action(teacher_agent, observations)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_started = time.perf_counter()
            action = action_normalizer.denormalize(student(observation_normalizer.normalize(flattened)))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_elapsed += time.perf_counter() - inference_started
            mse += float(nn.functional.mse_loss(action, teacher))
            smoothness += float((action - previous).square().mean())
            saturation += float((action.abs() >= 0.99).float().mean())
            previous = action
            observations, rewards, terminated, truncated, _ = env.step(action)
            if hasattr(core.robot.data, "computed_torque"):
                applied = core.robot.data.computed_torque
                torque += float(applied.square().mean().sqrt())
                energy += float((applied * core.robot.data.joint_vel).abs().mean() * core.step_dt)
                limits = core.robot.data.joint_effort_limits.clamp_min(1.0e-6)
                torque_saturation += float((applied.abs() >= limits).float().mean())
            base_linear_speed += float(core.robot.data.body_lin_vel_w[:, core.ref_body_index].norm(dim=-1).mean())
            base_angular_speed += float(core.robot.data.body_ang_vel_w[:, core.ref_body_index].norm(dim=-1).mean())
            raw_task_reward += float(rewards.mean())
            extras = getattr(core, "extras", {})
            components = (
                amp_reward_components(teacher_agent, extras.get("amp_obs"), rewards)
                if extras.get("amp_obs") is not None else None
            )
            if components is not None:
                for name in amp_totals:
                    amp_totals[name] += float(components[name].mean())
                amp_steps += 1
            lengths += 1
            returns += rewards.flatten()
            done = (terminated | truncated).flatten()
            if done.any():
                completed_lengths.extend(lengths[done].cpu().tolist())
                completed_returns.extend(returns[done].cpu().tolist())
                deaths += int(terminated.sum())
                timeouts += int((truncated & ~terminated).sum())
                ids = done.nonzero(as_tuple=False).squeeze(-1)
                history.reset(ids)
                corruptor.reset(ids)
                lengths[done] = 0
                returns[done] = 0
        completed = deaths + timeouts
        count = float(args_cli.eval_steps)
        return {
            "episode_length_mean": sum(completed_lengths) / len(completed_lengths) if completed_lengths else float(lengths.mean()),
            "return_mean": sum(completed_returns) / len(completed_returns) if completed_returns else float(returns.mean()),
            "deaths": deaths,
            "timeouts": timeouts,
            "death_rate": 100.0 * deaths / completed if completed else 0.0,
            "success_rate": 100.0 * timeouts / completed if completed else 0.0,
            "teacher_action_mse": mse / count,
            "action_smoothness": smoothness / count,
            "action_saturation": saturation / count,
            "torque_rms": torque / count,
            "energy": energy / count,
            "torque_saturation": torque_saturation / count,
            "base_linear_speed": base_linear_speed / count,
            "base_angular_speed": base_angular_speed / count,
            "raw_task_reward": raw_task_reward / count,
            **{f"amp_{name}": value / max(amp_steps, 1) for name, value in amp_totals.items()},
            "parameters": sum(parameter.numel() for parameter in student.parameters()),
            "inference_ms_per_sample": inference_elapsed * 1000.0 / (args_cli.eval_steps * args_cli.num_envs),
            "sensor_corruption_enabled": use_corruption and corruption_cfg.enabled,
        }

    force_skrl_isaaclab_reset(env)
    observations, _ = env.reset()
    started = time.monotonic()
    try:
        for iteration in range(1, args_cli.num_iterations + 1):
            student.eval()
            collected_x, collected_y = [], []
            for _ in range(args_cli.rollout_steps):
                flattened = history.push(frame(True))
                observation_normalizer.update(flattened)
                with torch.no_grad():
                    teacher = adapter.action(teacher_agent, observations)
                    predicted = action_normalizer.denormalize(student(observation_normalizer.normalize(flattened)))
                action_normalizer.update(teacher)
                # OmniH2O-style on-policy DAgger: the deployable student owns
                # the trajectory; the privileged teacher is only an oracle for
                # labels at states that the student actually visits.
                action = predicted
                # ``flattened`` is a view of the in-place history ring. Snapshot
                # both sides before the next environment step mutates any
                # rollout-owned storage.
                collected_x.append(flattened.detach().clone())
                collected_y.append(teacher.detach().clone())
                observations, _, terminated, truncated, _ = env.step(action)
                done = (terminated | truncated).flatten()
                if done.any():
                    ids = done.nonzero(as_tuple=False).squeeze(-1)
                    history.reset(ids)
                    corruptor.reset(ids)
            replay.add(torch.cat(collected_x), torch.cat(collected_y))
            student.train()
            train_steps = args_cli.train_steps or max(100, 2 * args_cli.num_envs * args_cli.rollout_steps // args_cli.batch_size)
            loss_total = 0.0
            for _ in range(train_steps):
                inputs, labels = replay.sample(args_cli.batch_size)
                loss = nn.functional.mse_loss(
                    student(observation_normalizer.normalize(inputs)), action_normalizer.normalize(labels)
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                loss_total += float(loss)
            scheduler.step()
            writer.add_scalar("Loss/mse", loss_total / train_steps, iteration)
            save(iteration, "student_latest.pt")
            if iteration % args_cli.save_interval == 0:
                save(iteration, f"student_iter_{iteration:05d}.pt")
            print(
                f"[{args_cli.method} {iteration:04d}/{args_cli.num_iterations}] "
                f"loss={loss_total / train_steps:.6f} rollout=student buffer={replay.size:,} "
                f"elapsed={time.monotonic() - started:.0f}s",
                flush=True,
            )
            if iteration % args_cli.eval_interval == 0 or iteration == args_cli.num_iterations:
                corrupted = evaluate(True)
                clean = evaluate(False) if args_cli.method == "imu" else dict(corrupted)
                row = {"iteration": iteration, **corrupted, "clean_sensor_metrics": clean}
                evaluations.append(row)
                if corrupted["episode_length_mean"] > best_length:
                    best_length = corrupted["episode_length_mean"]
                    best_iteration = iteration
                    best_metrics = row
                    save(iteration, "student_best_eval.pt")
                force_skrl_isaaclab_reset(env)
                observations, _ = env.reset()
                history.reset()
                corruptor.reset()
    finally:
        writer.close()
        env.close()
    result = {
        "metrics": {**best_metrics, "best_iteration": best_iteration},
        "evaluations": evaluations,
        "method": args_cli.method,
        "window": args_cli.window,
        "frame_dim": frame_dim,
        "input_dim": student.input_dim,
        "distillation_protocol": "student_rollout_teacher_action_labels",
        "rollout_policy": "student",
        "explicit_linear_velocity": False,
        "sensor_corruption": corruption_cfg.__dict__,
    }
    (log_dir / "training.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
    simulation_app.close()
