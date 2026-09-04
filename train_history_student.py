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
parser.add_argument(
    "--buffer-capacity", type=int, default=250_000,
    help="DAgger dataset cap. Matches train_state_estimator.py --max-dataset-size so the "
    "estimator and the distillation students train on equally sized datasets.",
)
parser.add_argument("--eval-interval", type=int, default=20)
parser.add_argument(
    "--mpjpe-horizon", type=int, default=100,
    help="Steps of teacher-paired rollout used for the MPJPE motion-fidelity metrics. "
    "Bounded because the two trajectories diverge chaotically once the policies differ.",
)
parser.add_argument(
    "--eval-steps", type=int, default=None,
    help="Rollout steps per evaluation call (default: the task's own max episode length, "
    "so evaluation reaches the same natural episode boundary as the teacher)",
)
parser.add_argument(
    "--grid-settle-s", type=float, default=1.0,
    help="Locomotion command grid: settling seconds before measurement (ppo_walk only). "
    "Matches train_state_estimator.py so every method's tracking numbers are comparable.",
)
parser.add_argument(
    "--grid-measure-s", type=float, default=4.0,
    help="Locomotion command grid: measured seconds per command (ppo_walk only)",
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
)
from jose.distillation.command_eval import (
    build_frame_fn,
    build_student_policy,
    evaluate_student_command_grid,
)
from jose.distillation.imu import IMUObservationSpec, SensorCorruptionCfg, SensorCorruptor
from jose.estimator.adapters import make_policy_adapter
from jose.estimator.metrics import MetricAccumulator, step_metrics
from jose.estimator.models import ReplayBuffer, RunningNormalizer
from jose.estimator.pipeline import evaluate_paired_motion_fidelity, uses_locomotion_eval
from jose.schema import SCHEMA_VERSION
from jose.teacher_setup import build_env_and_teacher
from jose.skrl_compat import force_skrl_isaaclab_reset, require_skrl_2


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
    best_state: dict | None = None
    fidelity: dict = {}
    command_metrics: dict = {}

    # Built by distillation/command_eval.py so eval_distillation_grid.py feeds a
    # restored checkpoint exactly what training fed it.
    frame = build_frame_fn(core, args_cli.method, imu_spec, corruptor)

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
        metrics = MetricAccumulator()
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
            observations, rewards, terminated, truncated, _ = env.step(action)
            metrics.add(
                step_metrics(
                    core, adapter, teacher_agent,
                    action=action, previous_action=previous, rewards=rewards,
                    policy_action=action, teacher_action=teacher,
                )
            )
            previous = action
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
                previous[done] = 0.0
        completed = deaths + timeouts
        return {
            "episode_length_mean": sum(completed_lengths) / len(completed_lengths) if completed_lengths else float(lengths.mean()),
            "return_mean": sum(completed_returns) / len(completed_returns) if completed_returns else float(returns.mean()),
            "deaths": deaths,
            "timeouts": timeouts,
            "death_rate": 100.0 * deaths / completed if completed else 0.0,
            "success_rate": 100.0 * timeouts / completed if completed else 0.0,
            **metrics.mean(),
            "parameters": sum(parameter.numel() for parameter in student.parameters()),
            "inference_ms_per_sample": inference_elapsed * 1000.0 / (args_cli.eval_steps * args_cli.num_envs),
            "sensor_corruption_enabled": use_corruption and corruption_cfg.enabled,
        }

    force_skrl_isaaclab_reset(env)
    observations, _ = env.reset()
    started = time.monotonic()
    cumulative_gradient_steps = 0
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
            cumulative_gradient_steps += train_steps
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
                row = {"iteration": iteration, "step": cumulative_gradient_steps, **corrupted, "clean_sensor_metrics": clean}
                evaluations.append(row)
                if corrupted["episode_length_mean"] > best_length:
                    best_length = corrupted["episode_length_mean"]
                    best_iteration = iteration
                    best_metrics = row
                    best_state = {name: value.detach().cpu().clone() for name, value in student.state_dict().items()}
                    save(iteration, "student_best_eval.pt")
                force_skrl_isaaclab_reset(env)
                observations, _ = env.reset()
                history.reset()
                corruptor.reset()
        # Teacher-relative motion fidelity, measured once on the reported
        # checkpoint. Deliberately after the loop: it rolls the env twice more
        # and would otherwise perturb training's random stream.
        if best_state is not None:
            student.load_state_dict(best_state)
        student.eval()

        # Shared with eval_distillation_grid.py so a back-filled run and a fresh
        # one measure the identical policy: normalize, forward, denormalize, with
        # the history (and the IMU corruptor) reset in step with the environment.
        student_policy, on_reset_ids = build_student_policy(
            student, history, observation_normalizer, action_normalizer,
            lambda: frame(True), extra_resets=(corruptor.reset,),
        )

        def reset_student_state() -> None:
            history.reset()
            corruptor.reset()

        fidelity = evaluate_paired_motion_fidelity(
            env, adapter, teacher_agent, student_policy,
            seed=args_cli.seed, horizon=args_cli.mpjpe_horizon, on_reset=reset_student_state,
        )
        # Command tracking. Episode length and death rate pin to their ceilings
        # on this task for any competent policy, so they separate nothing; this
        # is the column that does, and it is what puts this run in the same
        # report table as the teacher and JOSE (reporting.py dispatches on
        # `track_error_norm`). AMP tasks have no velocity command and skip it.
        if uses_locomotion_eval(adapter):
            reset_student_state()
            command_metrics = evaluate_student_command_grid(
                env, adapter, student_policy, on_reset_ids,
                settle_s=args_cli.grid_settle_s, measure_s=args_cli.grid_measure_s,
                seed=args_cli.seed,
            )
    finally:
        writer.close()
        env.close()
    # One entry per evaluated iteration on a step-indexed x-axis, so reporting.py can plot
    # it against train_state_estimator.py's own learning_curve on a shared "compute spent" axis.
    learning_curve = [
        {key: value for key, value in row.items() if key != "clean_sensor_metrics"} for row in evaluations
    ]
    result = {
        "metrics": {
            **best_metrics, "best_iteration": best_iteration,
            "total_gradient_steps": cumulative_gradient_steps, "learning_curve": learning_curve,
            **fidelity, **command_metrics,
        },
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
