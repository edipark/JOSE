"""Evaluate the paper's five locomotion methods at one fixed friction value.

This is the simulator worker for :mod:`jose.run_friction_sweep`.  One process
owns one ``(friction, evaluation seed)`` pair and evaluates every requested
checkpoint in the same built environment.  Friction is fixed, not sampled: the
sweep is a zero-shot dynamics shift, not domain-randomized training.
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher


PAPER_METHODS = ("teacher", "jose", "joint_only", "imu_clean", "set")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--friction", type=float, required=True)
parser.add_argument("--checkpoint-seeds", type=int, nargs="+", required=True)
parser.add_argument("--eval-seed", type=int, required=True)
parser.add_argument("--methods", nargs="+", default=list(PAPER_METHODS))
parser.add_argument("--study", required=True)
parser.add_argument("--set-study", required=True)
parser.add_argument("--teacher-checkpoint", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--task", default="Isaac-G1-PPO-Walk-Estimator-JOSE-v0")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--adapter", choices=("ppo_walk",), default="ppo_walk")
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--grid-settle-s", type=float, default=1.0)
parser.add_argument("--grid-measure-s", type=float, default=4.0)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0], *hydra_args]
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from datetime import datetime  # noqa: E402
import importlib.metadata as metadata  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import random  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402

from jose.distillation.command_eval import evaluate_student_command_grid  # noqa: E402
from jose.estimator.adapters import make_policy_adapter  # noqa: E402
from jose.robustness.methods import (  # noqa: E402
    load_jose,
    load_set,
    load_student,
    load_teacher,
)
from jose.robustness.registry import resolve  # noqa: E402
from jose.set_baseline.adapter import SETPolicyAdapter  # noqa: E402
from jose.teacher_setup import build_env_and_teacher  # noqa: E402


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _validate_args() -> None:
    if not np.isfinite(args_cli.friction) or args_cli.friction <= 0.0:
        raise ValueError("--friction must be finite and positive")
    if args_cli.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    if not args_cli.checkpoint_seeds or len(set(args_cli.checkpoint_seeds)) != len(
        args_cli.checkpoint_seeds
    ):
        raise ValueError("--checkpoint-seeds must contain unique integers")
    if not args_cli.methods or len(set(args_cli.methods)) != len(args_cli.methods):
        raise ValueError("--methods must contain unique entries")
    unknown = set(args_cli.methods) - set(PAPER_METHODS)
    if unknown:
        raise ValueError(f"Unsupported friction methods: {sorted(unknown)}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg) -> None:
    _validate_args()
    study = Path(args_cli.study).resolve()
    set_study = Path(args_cli.set_study).resolve()
    teacher_checkpoint = Path(args_cli.teacher_checkpoint).resolve()
    out = Path(args_cli.out).resolve()

    # The ground material is 1.0 and the task uses multiply combination, so a
    # fixed robot material of mu produces an effective ground-contact mu.  Keep
    # the existing all-body target: before a fall only the feet touch the plane,
    # and after a fall the survival outcome has already been decided.
    material = env_cfg.events.physics_material
    material.params["static_friction_range"] = (args_cli.friction, args_cli.friction)
    material.params["dynamic_friction_range"] = (args_cli.friction, args_cli.friction)
    material.params["restitution_range"] = (0.0, 0.0)
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.eval_seed
    _seed_everything(args_cli.eval_seed)

    env, teacher_agent = build_env_and_teacher(
        args_cli.task,
        args_cli.adapter,
        env_cfg,
        agent_cfg,
        teacher_checkpoint,
        args_cli.device,
        seed=args_cli.eval_seed,
    )
    teacher_agent.enable_training_mode(False, apply_to_models=True)
    base_adapter = make_policy_adapter(args_cli.adapter, env, "all")
    core = base_adapter.core_env
    device = torch.device(core.device)
    command_seed_base = args_cli.eval_seed * 100
    rows: list[dict] = []

    jobs: list[tuple[str, int | None]] = []
    for method in args_cli.methods:
        if method == "teacher":
            jobs.append((method, None))
        else:
            jobs.extend((method, seed) for seed in args_cli.checkpoint_seeds)

    print(
        f"[friction] mu={args_cli.friction:g} eval_seed={args_cli.eval_seed} "
        f"jobs={len(jobs)} envs={args_cli.num_envs}",
        flush=True,
    )
    try:
        for index, (method, checkpoint_seed) in enumerate(jobs, 1):
            adapter = base_adapter
            if method == "teacher":
                kind, checkpoint = "teacher", None
                act, on_step, info = load_teacher(adapter, teacher_agent)
                info["checkpoint"] = str(teacher_checkpoint)
            else:
                assert checkpoint_seed is not None
                kind, checkpoint = resolve(method, study, set_study, checkpoint_seed)
                if checkpoint is None or not checkpoint.is_file():
                    raise FileNotFoundError(f"{method} seed {checkpoint_seed}: {checkpoint}")
                if method == "jose":
                    act, on_step, info = load_jose(
                        adapter, teacher_agent, checkpoint, device, args_cli.num_envs
                    )
                elif method == "set":
                    adapter = SETPolicyAdapter(base_adapter)
                    act, on_step, info = load_set(
                        adapter,
                        teacher_agent,
                        checkpoint,
                        device,
                        args_cli.num_envs,
                        imu_scale=0.0,
                    )
                else:
                    act, on_step, info = load_student(
                        core, checkpoint, device, args_cli.num_envs, imu_scale=0.0
                    )

            metrics = evaluate_student_command_grid(
                env,
                adapter,
                act,
                on_step,
                settle_s=args_cli.grid_settle_s,
                measure_s=args_cli.grid_measure_s,
                seed=args_cli.eval_seed,
                command_seed_base=command_seed_base,
            )
            row = {
                "format_version": 1,
                "experiment": "zero_shot_friction_sweep",
                "task": args_cli.task,
                "method": method,
                "method_kind": kind,
                "checkpoint_seed": checkpoint_seed,
                "eval_seed": args_cli.eval_seed,
                "env_build_seed": args_cli.eval_seed,
                "command_seed_base": command_seed_base,
                "friction": {
                    "static": args_cli.friction,
                    "dynamic": args_cli.friction,
                    "ground": 1.0,
                    "combine_mode": "multiply",
                    "effective": args_cli.friction,
                    "body_pattern": material.params["asset_cfg"].body_names,
                    "restitution": 0.0,
                },
                "sensor_corruption": False,
                "num_envs": args_cli.num_envs,
                "grid_settle_s": args_cli.grid_settle_s,
                "grid_measure_s": args_cli.grid_measure_s,
                "method_info": info,
                "metrics": metrics,
                "runtime": {
                    "measured_at": datetime.now().isoformat(),
                    "torch": torch.__version__,
                    "isaaclab": _version("isaaclab"),
                    "isaacsim": _version("isaacsim"),
                    "rsl_rl": _version("rsl-rl-lib"),
                    "device": str(device),
                },
            }
            rows.append(row)
            print(
                f"  [{index:02d}/{len(jobs):02d}] {method:12s} "
                f"ckpt={checkpoint_seed!s:>4s} "
                f"track={metrics['track_error_norm']:.5f} "
                f"survival={metrics['grid_survival_rate']:.4f}",
                flush=True,
            )
    finally:
        env.close()

    out.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    out.write_text(payload, encoding="utf-8")
    print(f"[friction] wrote {len(rows)} rows to {out}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
    simulation_app.close()
