"""Evaluate the ground-truth privileged teacher baseline."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--teacher-checkpoint", required=True)
parser.add_argument("--task", required=True)
parser.add_argument("--agent", required=True)
parser.add_argument("--adapter", choices=("amp", "ppo", "ppo_walk"), required=True)
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--collect-steps", type=int, default=2000)
parser.add_argument("--epochs", type=int, default=0, help="Accepted for ablation command compatibility")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output-dir", default="logs/jose_g1/teacher")
parser.add_argument("--run-name", default=None)
parser.add_argument(
    "--keep-velocity-termination",
    action="store_true",
    help="Keep the training-time low-forward-velocity termination during evaluation.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

from jose.teacher_setup import resolve_agent_entry_point  # noqa: E402

# `--adapter ppo_walk` implies the rsl-rl runner config unless overridden.
args_cli.agent = resolve_agent_entry_point(args_cli.adapter, args_cli.agent)

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np
import time
from isaaclab_tasks.utils.hydra import hydra_task_config
import isaaclab_tasks  # noqa: F401

from jose.estimator.adapters import make_policy_adapter
from jose.estimator.pipeline import collect_rollout
from jose.skrl_compat import disable_velocity_termination_for_evaluation
from jose.teacher_setup import build_env_and_teacher, teacher_policy_module


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed
    velocity_threshold = (
        float(env_cfg.vel_window_min_vx) if hasattr(env_cfg, "vel_window_min_vx") else None
    )
    if not args_cli.keep_velocity_termination:
        threshold = disable_velocity_termination_for_evaluation(env_cfg)
        if threshold is not None and threshold > 0.0:
            print(
                f"Disabled low-velocity termination for evaluation "
                f"(training threshold: {threshold:g} m/s); height termination remains enabled."
            )
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
    collection_started = time.monotonic()
    _, metrics = collect_rollout(env, adapter, teacher_agent, args_cli.collect_steps, window=1)
    metrics["collection_duration_s"] = time.monotonic() - collection_started
    metrics["collection_samples_per_s"] = metrics["samples"] / max(metrics["collection_duration_s"], 1.0e-9)
    observations, _ = env.reset()
    policy = teacher_policy_module(teacher_agent)
    metrics["parameters"] = sum(parameter.numel() for parameter in policy.parameters()) if policy is not None else 0
    benchmark_steps = 100
    with torch.inference_mode():
        for _ in range(10):
            adapter.action(teacher_agent, observations)
        device = adapter.core_env.device
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(benchmark_steps):
            adapter.action(teacher_agent, observations)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
    metrics["inference_ms_per_sample"] = (
        (time.perf_counter() - started) * 1000.0 / (benchmark_steps * observations.shape[0])
    )
    default_name = (
        f"{args_cli.task}_TeacherGT_seed{args_cli.seed}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    )
    output = Path(args_cli.output_dir) / (args_cli.run_name or default_name)
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "teacher_checkpoint": str(Path(args_cli.teacher_checkpoint).resolve()),
        "task": args_cli.task,
        "agent": args_cli.agent,
        "adapter": args_cli.adapter,
        "num_envs": args_cli.num_envs,
        "collect_steps": args_cli.collect_steps,
        "seed": args_cli.seed,
        "device": args_cli.device,
        "evaluation_domain_randomization": False,
        "evaluation_action_noise": 0.0,
        "evaluation_velocity_termination": bool(
            args_cli.keep_velocity_termination
            and velocity_threshold is not None
            and velocity_threshold > 0.0
        ),
        "training_velocity_threshold_m_s": velocity_threshold,
    }
    (output / "training.json").write_text(
        json.dumps({"config": config, "metrics": metrics}, indent=2), encoding="utf-8"
    )
    print(
        f"TeacherGT eplen={metrics['episode_length_mean']:.2f} "
        f"return={metrics['return_mean']:.4f} success={metrics['success_rate']:.2f}%"
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
