"""Roll out one AMP teacher episode per environment and explain early terminations."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from types import MethodType

import numpy as np

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--teacher-checkpoint", required=True)
parser.add_argument("--task", default="Isaac-G1-AMP-Jump-JOSE-Direct-v0")
parser.add_argument("--agent", default="skrl_amp_cfg_entry_point")
parser.add_argument("--adapter", default="amp", choices=("amp",))
parser.add_argument("--num-envs", type=int, default=4096)
parser.add_argument(
    "--episodes",
    type=int,
    default=None,
    help="Completed episodes to collect (defaults to one per environment).",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--fixed-reset-time",
    type=float,
    default=None,
    help="Use one motion time for every reset instead of uniform random sampling.",
)
parser.add_argument("--output", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from isaaclab.utils.math import quat_apply_inverse  # noqa: E402

from jose.estimator.adapters import make_policy_adapter  # noqa: E402
from jose.skrl_compat import (  # noqa: E402
    disable_velocity_termination_for_evaluation,
    force_skrl_isaaclab_reset,
)
from jose.teacher_setup import build_env_and_teacher  # noqa: E402


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    values = values.float().cpu()
    if values.numel() == 0:
        return {}
    probs = torch.tensor([0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    result = torch.quantile(values, probs)
    return {
        name: float(value)
        for name, value in zip(("min", "p01", "p05", "median", "p95", "p99", "max"), result)
    }


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed
    training_velocity_threshold = disable_velocity_termination_for_evaluation(env_cfg)

    env, teacher_agent = build_env_and_teacher(
        args_cli.task,
        args_cli.adapter,
        env_cfg,
        agent_cfg,
        args_cli.teacher_checkpoint,
        args_cli.device,
        seed=args_cli.seed,
    )
    adapter = make_policy_adapter(args_cli.adapter, env, "all")
    core = adapter.core_env
    count = core.num_envs
    device = core.device
    target_episodes = args_cli.episodes or count
    if target_episodes <= 0:
        raise ValueError("--episodes must be positive")
    pelvis = core.ref_body_index

    # Per-environment state for the episode currently in flight. Replacing the
    # reset sampler lets us retain the exact random motion phase after every
    # asynchronous auto-reset, not just the first reset.
    current_initial_time = torch.full((count,), float("nan"), device=device)
    current_initial_reference_height = torch.full_like(current_initial_time, float("nan"))
    current_initial_sim_height = torch.full_like(current_initial_time, float("nan"))
    previous_height = torch.full_like(current_initial_time, float("nan"))
    minimum_height = torch.full_like(current_initial_time, float("inf"))
    minimum_height_step = torch.zeros(count, dtype=torch.long, device=device)
    episode_index = torch.zeros(count, dtype=torch.long, device=device)

    original_sample_motion_reset = core._sample_motion_reset

    def diagnostic_sample_motion_reset(self, env_ids, start: bool = False):
        reset_count = env_ids.shape[0]
        if args_cli.fixed_reset_time is not None:
            times = np.full(reset_count, args_cli.fixed_reset_time)
        else:
            times = np.zeros(reset_count) if start else self._motion_loader.sample_times(reset_count)
        dof_pos, dof_vel, body_pos, body_rot, body_lin, body_ang = self._motion_loader.sample(
            reset_count, times
        )
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] = body_pos[:, self.motion_ref_body_index] + self.scene.env_origins[env_ids]
        root_state[:, 2] += 0.05
        root_state[:, 3:7] = body_rot[:, self.motion_ref_body_index]
        root_state[:, 7:10] = body_lin[:, self.motion_ref_body_index]
        root_state[:, 10:13] = body_ang[:, self.motion_ref_body_index]
        amp = self.collect_reference_motions(reset_count, times)
        self.amp_observation_buffer[env_ids] = amp.view(
            reset_count, self.cfg.num_amp_observations, -1
        )
        times_tensor = torch.as_tensor(times, dtype=torch.float32, device=device)
        reference_height = body_pos[:, self.motion_ref_body_index, 2]
        current_initial_time[env_ids] = times_tensor
        current_initial_reference_height[env_ids] = reference_height
        current_initial_sim_height[env_ids] = reference_height + 0.05
        previous_height[env_ids] = reference_height + 0.05
        minimum_height[env_ids] = reference_height + 0.05
        minimum_height_step[env_ids] = 0
        return (
            root_state,
            dof_pos[:, self.motion_dof_indexes],
            dof_vel[:, self.motion_dof_indexes],
        )

    core._sample_motion_reset = MethodType(diagnostic_sample_motion_reset, core)
    force_skrl_isaaclab_reset(env)
    observations, _ = env.reset()
    if bool(torch.isnan(current_initial_time).any()):
        raise RuntimeError("Could not capture the random motion phases used by reset")

    # Fixed-size device buffers avoid synchronizing every completed episode.
    record_env_id = torch.full((target_episodes,), -1, dtype=torch.long, device=device)
    record_episode_index = torch.full_like(record_env_id, -1)
    record_initial_time = torch.full((target_episodes,), float("nan"), device=device)
    record_initial_reference_height = torch.full_like(record_initial_time, float("nan"))
    record_initial_sim_height = torch.full_like(record_initial_time, float("nan"))
    record_terminated = torch.zeros(target_episodes, dtype=torch.bool, device=device)
    record_timed_out = torch.zeros(target_episodes, dtype=torch.bool, device=device)
    finish_step = torch.full((target_episodes,), -1, dtype=torch.long, device=device)
    finish_height = torch.full((target_episodes,), float("nan"), device=device)
    finish_previous_height = torch.full_like(finish_height, float("nan"))
    record_minimum_height = torch.full_like(finish_height, float("nan"))
    record_minimum_height_step = torch.full_like(finish_step, -1)
    finish_quaternion = torch.full((target_episodes, 4), float("nan"), device=device)
    finish_linear_velocity = torch.full((target_episodes, 3), float("nan"), device=device)
    finish_angular_velocity = torch.full((target_episodes, 3), float("nan"), device=device)
    finish_projected_gravity = torch.full((target_episodes, 3), float("nan"), device=device)
    finish_action = torch.full(
        (target_episodes, core.actions.shape[1]), float("nan"), device=device
    )

    original_get_dones = core._get_dones
    completed = 0

    def diagnostic_get_dones(self):
        nonlocal completed, previous_height
        terminated, timed_out = original_get_dones()
        height = self.robot.data.body_pos_w[:, pelvis, 2]
        lower = height < minimum_height
        minimum_height[lower] = height[lower]
        minimum_height_step[lower] = self.episode_length_buf[lower]
        done_ids = (terminated | timed_out).nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() and completed < target_episodes:
            take = min(done_ids.numel(), target_episodes - completed)
            source = done_ids[:take]
            destination = torch.arange(completed, completed + take, device=device)
            quaternion = self.robot.data.body_quat_w[:, pelvis]
            angular_velocity = self.robot.data.body_ang_vel_w[:, pelvis]
            gravity_world = torch.zeros_like(angular_velocity)
            gravity_world[:, 2] = -1.0
            record_env_id[destination] = source
            record_episode_index[destination] = episode_index[source]
            record_initial_time[destination] = current_initial_time[source]
            record_initial_reference_height[destination] = current_initial_reference_height[source]
            record_initial_sim_height[destination] = current_initial_sim_height[source]
            record_terminated[destination] = terminated[source]
            record_timed_out[destination] = timed_out[source] & ~terminated[source]
            finish_step[destination] = self.episode_length_buf[source]
            finish_height[destination] = height[source]
            finish_previous_height[destination] = previous_height[source]
            record_minimum_height[destination] = minimum_height[source]
            record_minimum_height_step[destination] = minimum_height_step[source]
            finish_quaternion[destination] = quaternion[source]
            finish_linear_velocity[destination] = self.robot.data.body_lin_vel_w[source, pelvis]
            finish_angular_velocity[destination] = angular_velocity[source]
            finish_projected_gravity[destination] = quat_apply_inverse(
                quaternion[source], gravity_world[source]
            )
            finish_action[destination] = self.actions[source]
            completed += take
        if done_ids.numel():
            episode_index[done_ids] += 1
        previous_height = height.clone()
        return terminated, timed_out

    core._get_dones = MethodType(diagnostic_get_dones, core)
    teacher_agent.enable_training_mode(False, apply_to_models=True)
    episode_waves = (target_episodes + count - 1) // count
    max_steps = (episode_waves + 2) * core.max_episode_length
    with torch.inference_mode():
        for rollout_step in range(max_steps):
            actions = adapter.action(teacher_agent, observations)
            observations, _, _, _, _ = env.step(actions)
            if (rollout_step + 1) % 250 == 0 or completed >= target_episodes:
                print(
                    f"rollout step {rollout_step + 1}/{max_steps}: "
                    f"episodes={completed}/{target_episodes}"
                )
            if completed >= target_episodes:
                break
    core._get_dones = original_get_dones
    core._sample_motion_reset = original_sample_motion_reset

    if completed < target_episodes:
        raise RuntimeError(f"Only {completed}/{target_episodes} episodes finished")

    env_id_cpu = record_env_id.cpu()
    episode_index_cpu = record_episode_index.cpu()
    initial_times = record_initial_time.cpu().numpy()
    initial_reference_height = record_initial_reference_height.cpu()
    initial_height = record_initial_sim_height.cpu()
    terminated_cpu = record_terminated.cpu()
    timed_out_cpu = record_timed_out.cpu()
    finish_step_cpu = finish_step.cpu()
    finish_height_cpu = finish_height.cpu()
    finish_previous_height_cpu = finish_previous_height.cpu()
    minimum_height_cpu = record_minimum_height.cpu()
    minimum_height_step_cpu = record_minimum_height_step.cpu()
    finish_quaternion_cpu = finish_quaternion.cpu()
    finish_linear_velocity_cpu = finish_linear_velocity.cpu()
    finish_angular_velocity_cpu = finish_angular_velocity.cpu()
    finish_projected_gravity_cpu = finish_projected_gravity.cpu()
    finish_action_cpu = finish_action.cpu()

    death_ids = terminated_cpu.nonzero(as_tuple=False).squeeze(-1).tolist()
    timeout_ids = timed_out_cpu.nonzero(as_tuple=False).squeeze(-1)
    phase_edges = np.linspace(0.0, core._motion_loader.duration, 19)
    all_phase_histogram, _ = np.histogram(initial_times, bins=phase_edges)
    death_phase_histogram, _ = np.histogram(initial_times[death_ids], bins=phase_edges)
    deaths = []
    for record_id in death_ids:
        deaths.append(
            {
                "record_id": record_id,
                "env_id": int(env_id_cpu[record_id]),
                "env_episode_index": int(episode_index_cpu[record_id]),
                "initial_motion_time_s": float(initial_times[record_id]),
                "initial_reference_pelvis_height_m": float(initial_reference_height[record_id]),
                "initial_sim_pelvis_height_m": float(initial_height[record_id]),
                "termination_step": int(finish_step_cpu[record_id]),
                "termination_time_s": float(finish_step_cpu[record_id] * core.step_dt),
                "previous_pelvis_height_m": float(finish_previous_height_cpu[record_id]),
                "termination_pelvis_height_m": float(finish_height_cpu[record_id]),
                "minimum_pelvis_height_m": float(minimum_height_cpu[record_id]),
                "minimum_height_step": int(minimum_height_step_cpu[record_id]),
                "pelvis_quaternion_wxyz": finish_quaternion_cpu[record_id].tolist(),
                "pelvis_linear_velocity_m_s": finish_linear_velocity_cpu[record_id].tolist(),
                "pelvis_angular_velocity_rad_s": finish_angular_velocity_cpu[record_id].tolist(),
                "projected_gravity_body": finish_projected_gravity_cpu[record_id].tolist(),
                "policy_action": finish_action_cpu[record_id].tolist(),
            }
        )

    report = {
        "created_at": datetime.now().isoformat(),
        "checkpoint": str(Path(args_cli.teacher_checkpoint).resolve()),
        "task": args_cli.task,
        "seed": args_cli.seed,
        "num_envs": count,
        "episodes": target_episodes,
        "fixed_reset_time_s": args_cli.fixed_reset_time,
        "max_episode_length": core.max_episode_length,
        "step_dt_s": core.step_dt,
        "termination_config": {
            "early_termination": bool(core.cfg.early_termination),
            "height_threshold_m": float(core.cfg.termination_height),
            "training_velocity_threshold_m_s": training_velocity_threshold,
            "evaluation_velocity_threshold_m_s": float(core.cfg.vel_window_min_vx),
        },
        "summary": {
            "deaths": len(death_ids),
            "timeouts": int(timed_out_cpu.sum()),
            "death_rate_percent": 100.0 * len(death_ids) / target_episodes,
            "episode_length_mean": float(finish_step_cpu.float().mean()),
            "episode_length_std": float(finish_step_cpu.float().std(unbiased=False)),
            "termination_step_quantiles": _quantiles(finish_step_cpu),
            "minimum_pelvis_height_all_m": _quantiles(minimum_height_cpu),
            "minimum_pelvis_height_timeouts_m": _quantiles(minimum_height_cpu[timeout_ids]),
            "minimum_pelvis_height_deaths_m": _quantiles(minimum_height_cpu[terminated_cpu]),
        },
        "initial_phase_histogram": {
            "bin_edges_s": phase_edges.tolist(),
            "all": all_phase_histogram.tolist(),
            "deaths": death_phase_histogram.tolist(),
        },
        "deaths": deaths,
    }
    output = Path(args_cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"diagnosis deaths={len(death_ids)}/{target_episodes} "
        f"({report['summary']['death_rate_percent']:.4f}%), output={output.resolve()}"
    )
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
