"""Re-measure teacher-relative motion fidelity from a distillation checkpoint.

``train_history_student.py`` reports MPJPE from memory at the end of training,
and that report is wrong whenever the best checkpoint is not the last one. It
restores ``best_state`` -- which clones ``student.state_dict()`` and nothing else
-- while the two ``RunningNormalizer``s keep updating to the end of training. The
reported policy is therefore the best weights paired with the final normalizers,
a combination that was never saved and never runs anywhere.

``student_best_eval.pt`` holds the weights *and* both normalizers as of the
iteration it was saved at, so loading it restores the policy that actually
deploys. On the locomotion task the gap between the two is monotone in how early
the best checkpoint was found -- zero when it is the last iteration, 0.021 in
command RMSE when it is iteration 60 of 300 -- and the same script trains the
AMP students, which have no independent measurement of their own.

This is the fidelity counterpart to ``eval_distillation_grid.py``, which does the
same job for the locomotion command grid. Nothing is retrained: the checkpoints
were always right, only the numbers taken beside them were not.

Usage:

    python eval_distillation_fidelity.py --task Isaac-G1-AMP-Dance-JOSE-Direct-v0 \
        --adapter amp --run-dir <study>/methods/joint_only_distillation/.../seed_42 \
        --teacher-checkpoint <dance teacher .pt> --seed 42 --headless
"""


from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Re-measure motion fidelity from a distillation checkpoint")
parser.add_argument(
    "--run-dir", required=True,
    help="Seed directory holding checkpoints/student_best_eval.pt and training.json",
)
parser.add_argument("--teacher-checkpoint", required=True)
parser.add_argument("--task", default="Isaac-G1-PPO-Walk-Estimator-JOSE-v0")
parser.add_argument(
    "--agent", default=None,
    help="Config entry point. Left unset so resolve_agent_entry_point picks the one "
         "the adapter needs: AMP tasks register no rsl_rl entry point, and naming it "
         "anyway makes hydra report the environment itself as missing.",
)
parser.add_argument("--adapter", choices=("amp", "ppo_walk"), default="ppo_walk")
parser.add_argument(
    "--num-envs", type=int, default=256,
    help="Must match the run being back-filled: ObservationHistory and SensorCorruptor "
    "are sized at construction and their push/reset validate the batch shape.",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--mpjpe-horizon", type=int, default=100)
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
    "--self-test", action="store_true",
    help="Roll the teacher against itself instead of the student. Both rollouts run the "
    "identical policy, so any non-zero MPJPE is the harness failing to reproduce its own "
    "seeded reset rather than a difference between policies.",
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
    checkpoint_command_conditioning,
    require_command_conditioning,
    build_student_policy,
)
from jose.distillation.history import HistoryMLPStudent, ObservationHistory  # noqa: E402
from jose.distillation.imu import IMUObservationSpec, SensorCorruptionCfg, SensorCorruptor  # noqa: E402
from jose.estimator.adapters import make_policy_adapter  # noqa: E402
from jose.estimator.models import RunningNormalizer  # noqa: E402
from jose.estimator.pipeline import evaluate_paired_motion_fidelity  # noqa: E402
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
    core = adapter.core_env
    device = torch.device(core.device)

    method, student, history, obs_norm, act_norm, corruptor, checkpoint = _load_student(
        checkpoint_path, device, args_cli.num_envs
    )
    print(f"[set] restored {method} student from {checkpoint_path.name}", flush=True)
    if checkpoint["task"] != args_cli.task:
        raise ValueError(f"Checkpoint task {checkpoint['task']!r} != --task {args_cli.task!r}")

    imu_spec = IMUObservationSpec()
    # ppo_walk is the command-conditioned adapter; AMP tasks have no command.
    command_conditioned = checkpoint.get("adapter") == "ppo_walk"
    require_command_conditioning(checkpoint, command_conditioned, str(checkpoint_path.name))
    frame = build_frame_fn(
        core, method, imu_spec, corruptor, command_conditioned=command_conditioned
    )
    corruption = bool(args_cli.sensor_corruption)
    # joint_only has no IMU channels, so the flag cannot change anything for it.
    if method == "joint_only" and not corruption:
        print("[set] joint_only has no IMU input; --no-sensor-corruption is a no-op", flush=True)
    act, on_step = build_student_policy(
        student, history, obs_norm, act_norm,
        lambda: frame(corruption), extra_resets=(corruptor.reset,),
    )

    # `on_reset` takes no arguments and clears the whole batch, unlike the
    # per-env `on_step` the command grid uses. This mirrors train_history_student.py's
    # `reset_student_state`, so the two measurements differ only in the normalizers.
    def reset_student_state() -> None:
        history.reset()
        corruptor.reset()

    print("[set] running paired motion fidelity", flush=True)
    try:
        # Same call the training script makes, with the same horizon, so the two
        # numbers differ only in which normalizers the policy carries.
        measured = (lambda obs: adapter.action(teacher_agent, obs)) if args_cli.self_test else act
        if args_cli.self_test:
            print("[set] SELF-TEST: teacher against teacher, expect ~0", flush=True)
        metrics = evaluate_paired_motion_fidelity(
            env, adapter, teacher_agent, measured,
            seed=args_cli.seed, horizon=args_cli.mpjpe_horizon, on_reset=reset_student_state,
        )
    finally:
        env.close()

    print(
        f"[{method}] seed {args_cli.seed} imu={'noisy' if corruption else 'ideal'}  "
        f"mpjpe_g={metrics['mpjpe_g']:.2f} mpjpe_l={metrics['mpjpe_l']:.2f} "
        f"horizon={metrics.get('mpjpe_horizon')}",
        flush=True,
    )
    # Where the two rollouts part company. For identical policies this should be
    # nowhere; a step index says the divergence is an event in the environment
    # rather than a difference between the policies.
    steps = metrics.get("mpjpe_by_step") or []
    first = next((s["step"] for s in steps if s["mpjpe_g"] > 1e-6), None)
    print(
        f"[set] first divergent step: {first}  "
        + "  ".join(f"s{s['step']}={s['mpjpe_g']:.1f}" for s in steps[:8]),
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
    payload.setdefault("fidelity_backfill", {})[
        "corrupted" if corruption else "clean"
    ] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_iteration": checkpoint.get("iteration"),
        "mpjpe_horizon": args_cli.mpjpe_horizon,
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
