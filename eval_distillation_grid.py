"""Back-fill command-tracking metrics onto a finished distillation run.

``train_history_student.py`` now reports the locomotion command grid natively,
but the runs already in the catalog predate that. Retraining them costs about
3.5 hours; they do not need it. ``student_best_eval.pt`` carries the model
config, the weights and both normalizers, so the reported policy can be restored
exactly and driven over the grid in a couple of minutes.

Without this the two distillation arms emit no ``track_*`` keys, ``reporting.py``
sends them to a different table from the teacher and JOSE, and the locomotion
block of the paper compares four methods on survival alone -- a column where
every competent policy sits at the ceiling. The students look best there while
being 350-372 mm off the teacher's trajectory against JOSE's 10 mm.

Scope note: this back-fills tracking only. The episode-based survival numbers
keep the 1000-step window they were measured with, half the 2000-step window the
teacher and JOSE use. That window is unbiased -- it just halves the episode count
-- so the fix for future runs is the existing ``--eval-steps`` flag, not a
re-measurement here.

Usage:

    python -m JOSE.eval_distillation_grid \
        --run-dir <study>/methods/imu_based_distillation/window_25/joints_all/seed_42 \
        --teacher-checkpoint <walk teacher .pt> --seed 42 --headless
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Back-fill command tracking onto a distillation run")
parser.add_argument(
    "--run-dir", required=True,
    help="Seed directory holding checkpoints/student_best_eval.pt and training.json",
)
parser.add_argument("--teacher-checkpoint", required=True)
parser.add_argument("--task", default="Isaac-G1-PPO-Walk-Estimator-JOSE-v0")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--adapter", choices=("amp", "ppo_walk"), default="ppo_walk")
parser.add_argument(
    "--num-envs", type=int, default=256,
    help="Must match the run being back-filled: ObservationHistory and SensorCorruptor "
    "are sized at construction and their push/reset validate the batch shape.",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--grid-settle-s", type=float, default=1.0)
parser.add_argument("--grid-measure-s", type=float, default=4.0)
parser.add_argument(
    "--checkpoint-name", default="student_best_eval.pt",
    help="Which checkpoint under run-dir/checkpoints to evaluate",
)
parser.add_argument(
    "--sensor-corruption", action=argparse.BooleanOptionalAction, default=True,
    help="IMU corruption during evaluation. ON reproduces the run's own reported condition; "
    "--no-sensor-corruption gives the student an ideal IMU, which is the matched condition for "
    "the main table where every other method reads ideal sensors. Metrics are written under "
    "distinct keys so both can coexist.",
)
parser.add_argument(
    "--dry-run", action="store_true",
    help="Measure and print, but leave training.json untouched",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

from jose.teacher_setup import resolve_agent_entry_point  # noqa: E402

args_cli.agent = resolve_agent_entry_point(args_cli.adapter, args_cli.agent)

sys.argv = [sys.argv[0], *hydra_args]
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402

from jose.distillation.command_eval import (  # noqa: E402
    build_frame_fn,
    build_student_policy,
    evaluate_student_command_grid,
)
from jose.distillation.history import HistoryMLPStudent, ObservationHistory  # noqa: E402
from jose.distillation.imu import IMUObservationSpec, SensorCorruptionCfg, SensorCorruptor  # noqa: E402
from jose.estimator.adapters import make_policy_adapter  # noqa: E402
from jose.estimator.models import RunningNormalizer  # noqa: E402
from jose.estimator.pipeline import uses_locomotion_eval  # noqa: E402
from jose.schema import SCHEMA_VERSION  # noqa: E402
from jose.teacher_setup import build_env_and_teacher  # noqa: E402


def _load_student(path: Path, device, num_envs: int):
    """Restore the reported policy. Mirrors play_history_student.py:49-79.

    Every shape comes from ``model_config``, never from a module constant: the
    standalone default window is 21 while ``run_method_comparison.py`` trains at
    25, so a hardcoded window silently builds the wrong network.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    method = checkpoint.get("method")
    if checkpoint.get("jose_schema_version") != SCHEMA_VERSION or method not in ("joint_only", "imu"):
        raise ValueError(f"{path} is not a JOSE joint-only or IMU history student")
    if checkpoint.get("explicit_linear_velocity") is not False:
        raise ValueError("History checkpoint does not satisfy the deploy observation contract")

    config = checkpoint["model_config"]
    student = HistoryMLPStudent(
        config["frame_dim"], config["action_dim"], config["window"], tuple(config["hidden_dims"])
    ).to(device)
    student.load_state_dict(checkpoint["model_state_dict"])
    student.eval()

    observation_normalizer = RunningNormalizer(config["input_dim"], device)
    observation_normalizer.load_state_dict(checkpoint["observation_normalizer"])
    # clip=10.0 matches train_history_student.py; load_state_dict restores it, but
    # constructing with the wrong value would clip a fresh normalizer differently.
    action_normalizer = RunningNormalizer(config["action_dim"], device, clip=10.0)
    action_normalizer.load_state_dict(checkpoint["action_normalizer"])

    history = ObservationHistory(num_envs, config["window"], config["frame_dim"], device)
    corruptor = SensorCorruptor(
        num_envs, device, SensorCorruptionCfg(**checkpoint["sensor_corruption"])
    )
    return method, student, history, observation_normalizer, action_normalizer, corruptor, checkpoint


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    run_dir = Path(args_cli.run_dir).resolve()
    checkpoint_path = run_dir / "checkpoints" / args_cli.checkpoint_name
    training_json = run_dir / "training.json"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not training_json.is_file():
        raise FileNotFoundError(training_json)

    torch.manual_seed(args_cli.seed)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed

    env, teacher_agent = build_env_and_teacher(
        args_cli.task, args_cli.adapter, env_cfg, agent_cfg,
        args_cli.teacher_checkpoint, args_cli.device, seed=args_cli.seed,
    )
    print(f"[set] env and teacher ready ({args_cli.num_envs} envs)", flush=True)
    teacher_agent.enable_training_mode(False, apply_to_models=True)
    adapter = make_policy_adapter(args_cli.adapter, env, "all")
    if not uses_locomotion_eval(adapter):
        raise RuntimeError("Command tracking is only defined for the locomotion task")
    core = adapter.core_env
    device = torch.device(core.device)

    method, student, history, obs_norm, act_norm, corruptor, checkpoint = _load_student(
        checkpoint_path, device, args_cli.num_envs
    )
    print(f"[set] restored {method} student from {checkpoint_path.name}", flush=True)
    if checkpoint["task"] != args_cli.task:
        raise ValueError(f"Checkpoint task {checkpoint['task']!r} != --task {args_cli.task!r}")

    imu_spec = IMUObservationSpec()
    frame = build_frame_fn(core, method, imu_spec, corruptor)
    corruption = bool(args_cli.sensor_corruption)
    # joint_only has no IMU channels, so the flag cannot change anything for it.
    if method == "joint_only" and not corruption:
        print("[set] joint_only has no IMU input; --no-sensor-corruption is a no-op", flush=True)
    act, on_step = build_student_policy(
        student, history, obs_norm, act_norm,
        lambda: frame(corruption), extra_resets=(corruptor.reset,),
    )

    print("[set] running the command grid", flush=True)
    try:
        metrics = evaluate_student_command_grid(
            env, adapter, act, on_step,
            settle_s=args_cli.grid_settle_s, measure_s=args_cli.grid_measure_s,
            seed=args_cli.seed,
        )
    finally:
        env.close()

    print(
        f"[{method}] seed {args_cli.seed} imu={'noisy' if corruption else 'ideal'}  "
        f"vx={metrics['track_vx_rmse']:.4f} vy={metrics['track_vy_rmse']:.4f} "
        f"yaw={metrics['track_yaw_rmse']:.4f} norm={metrics['track_error_norm']:.4f} "
        f"grid_survival={metrics['grid_survival_rate']:.4f}",
        flush=True,
    )
    if args_cli.dry_run:
        print("dry run: training.json untouched")
        return

    payload = json.loads(training_json.read_text(encoding="utf-8"))
    training_json.with_suffix(".json.bak").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Ideal-IMU numbers go under a prefix so the deployment-condition ones stay
    # addressable: the main table wants matched ideal sensors, the robustness
    # section wants the deployment condition, and both come from this checkpoint.
    prefix = "" if corruption else "clean_sensor_"
    payload.setdefault("metrics", {}).update({f"{prefix}{k}": v for k, v in metrics.items()})
    payload.setdefault("command_grid_backfill", {})[
        "corrupted" if corruption else "clean"
    ] = {
        "checkpoint": str(checkpoint_path),
        "settle_s": args_cli.grid_settle_s,
        "measure_s": args_cli.grid_measure_s,
        "seed": args_cli.seed,
        "num_envs": args_cli.num_envs,
        "sensor_corruption": corruption,
    }
    training_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"merged into {training_json} (previous version at {training_json.with_suffix('.json.bak')})")


if __name__ == "__main__":
    # Hydra's decorator can swallow a traceback and still exit 0, which silently
    # looks like "ran and produced nothing". Surface it and fail loudly instead.
    import traceback

    try:
        main()
    except Exception:
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
    simulation_app.close()
