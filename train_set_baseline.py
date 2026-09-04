"""Train and evaluate the SET baseline on one task and seed.

SET (Yu et al., IROS 2024) is the joint+IMU counterpart to JOSE's joint-only
estimator: both bolt onto an already-frozen expert, which is what makes them
comparable and what rules out the concurrent-training estimators (Ji et al. 2022,
DreamWaQ's CENet) that need the policy retrained around them.

Implemented as published -- offline expert rollouts, no DAgger. Modifying a
baseline to use our data protocol would not be a comparison against SET, and
JOSE's own DAgger ablation already measures what the on-policy protocol is worth.

Every metric name here matches what ``run_method_comparison.py`` produces, so a
SET row drops straight into that table: the runner reads only the top-level
``metrics`` object of ``training.json``.

The paper fixes six blocks, H=20 and an MSE loss; model width, heads, dropout,
optimizer and batch size are unspecified there, so they are chosen here and
recorded under ``set_config`` for the write-up.
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="SET baseline: joint+IMU state estimation transformer")
parser.add_argument("--teacher-checkpoint", required=True)
parser.add_argument("--task", default="Isaac-G1-AMP-Walk-JOSE-Direct-v0")
parser.add_argument("--agent", default="skrl_amp_cfg_entry_point")
parser.add_argument("--adapter", choices=("amp", "ppo_walk"), default="amp")
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--context", type=int, default=20,
    help="SET's H. The paper's value; JOSE uses a 25-step window chosen by its own sweep.",
)
parser.add_argument("--blocks", type=int, default=6, help="Paper value")
parser.add_argument("--width", type=int, default=128, help="Not specified in the paper")
parser.add_argument("--heads", type=int, default=4, help="Not specified in the paper")
parser.add_argument("--dropout", type=float, default=0.1, help="Not specified in the paper")
parser.add_argument(
    "--collect-steps", type=int, default=2000,
    help="Offline expert rollout steps. With --max-dataset-size this matches JOSE's budget.",
)
parser.add_argument("--max-dataset-size", type=int, default=250_000)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch-size", type=int, default=1024)
parser.add_argument("--lr", type=float, default=1.0e-3)
parser.add_argument("--eval-episodes", type=int, default=200)
parser.add_argument("--max-episode-steps", type=int, default=1000)
parser.add_argument("--eval-seed-offset", type=int, default=1000)
parser.add_argument("--mpjpe-horizon", type=int, default=100)
parser.add_argument("--grid-settle-s", type=float, default=1.0)
parser.add_argument("--grid-measure-s", type=float, default=4.0)
parser.add_argument(
    "--imu-noise-scale", type=float, default=0.0,
    help="Multiplier on the nominal IMU model, applied to SET's own sensor input "
         "during collection and evaluation. 0 reproduces the published method; "
         "anything else is the randomized arm and is ours, not SET's.",
)
parser.add_argument("--output-dir", default="logs/jose_g1/set_baseline")
parser.add_argument("--run-name", default="artifact")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

from jose.teacher_setup import resolve_agent_entry_point  # noqa: E402

args_cli.agent = resolve_agent_entry_point(args_cli.adapter, args_cli.agent)

sys.argv = [sys.argv[0], *hydra_args]
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json  # noqa: E402
from pathlib import Path  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402

from jose.distillation.command_eval import evaluate_student_command_grid  # noqa: E402
from jose.estimator.adapters import make_policy_adapter  # noqa: E402
from jose.estimator.pipeline import (  # noqa: E402
    HistoryBuffer,
    evaluate_paired_motion_fidelity,
    train_estimator,
    uses_locomotion_eval,
)
from jose.schema import SCHEMA_VERSION  # noqa: E402
from jose.set_baseline import targets as set_targets  # noqa: E402
from jose.set_baseline.adapter import SETPolicyAdapter  # noqa: E402
from jose.set_baseline.collect import collect_expert_rollout  # noqa: E402
from jose.set_baseline.evaluate import evaluate_set_closed_loop  # noqa: E402
from jose.set_baseline.open_loop import evaluate_predictions_chunked  # noqa: E402
from jose.set_baseline.model import SETEstimator  # noqa: E402
from jose.teacher_setup import build_env_and_teacher  # noqa: E402


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    started = time.monotonic()
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed

    env, teacher_agent = build_env_and_teacher(
        args_cli.task, args_cli.adapter, env_cfg, agent_cfg,
        args_cli.teacher_checkpoint, args_cli.device, seed=args_cli.seed,
    )
    teacher_agent.enable_training_mode(False, apply_to_models=True)
    imu_corruptor = None
    if args_cli.imu_noise_scale > 0.0:
        from jose.distillation.imu import SensorCorruptionCfg, SensorCorruptor
        from jose.robustness.noise import scaled_imu_cfg

        # The same nominal model the distillation students recorded, so the two
        # hardened arms on this axis were hardened against the same sensor.
        imu_corruptor = SensorCorruptor(
            args_cli.num_envs, args_cli.device,
            scaled_imu_cfg(SensorCorruptionCfg(), args_cli.imu_noise_scale),
        )
        print(f"[set] IMU randomization at {args_cli.imu_noise_scale}x nominal", flush=True)
    adapter = SETPolicyAdapter(
        make_policy_adapter(args_cli.adapter, env, "all"), imu_corruptor=imu_corruptor
    )
    locomotion = uses_locomotion_eval(adapter)

    estimated, pass_through = set_targets.split(adapter.name())
    target_dim = adapter.schema.estimator_target_dim
    observation_dim = adapter.input_dim
    estimator = SETEstimator(
        observation_dim, target_dim,
        context=args_cli.context, width=args_cli.width, heads=args_cli.heads,
        blocks=args_cli.blocks, dropout=args_cli.dropout,
        estimated_indices=estimated, pass_through_indices=pass_through,
    ).to(args_cli.device)
    split_description = set_targets.describe(adapter.name())
    print(
        f"[set] o={observation_dim} target={target_dim} "
        f"estimates={len(estimated)} passes={len(pass_through)} "
        f"params={sum(p.numel() for p in estimator.parameters()):,}",
        flush=True,
    )

    output = Path(args_cli.output_dir).resolve() / args_cli.run_name
    output.mkdir(parents=True, exist_ok=True)

    # -- offline data, as published ----------------------------------------
    collection_started = time.monotonic()
    dataset, collection = collect_expert_rollout(
        env, adapter, teacher_agent, args_cli.collect_steps, args_cli.context,
        estimated, max_samples=args_cli.max_dataset_size,
    )
    collection["duration_s"] = time.monotonic() - collection_started
    print(f"[set] collected {collection['samples']} samples in {collection['duration_s']:.0f}s", flush=True)

    def epoch_logger(row):
        print(
            f"  epoch {row['epoch']:03d}: train={row['train_mse']:.6f} val={row['validation_mse']:.6f}",
            flush=True,
        )

    training = train_estimator(
        estimator, dataset, "SET", args_cli.epochs, args_cli.batch_size,
        args_cli.lr, args_cli.device, epoch_logger, seed=args_cli.seed,
    )

    # -- closed loop, same protocol as every other row ---------------------
    pass_through_fn = adapter.pass_through_values if pass_through else None
    metrics = evaluate_set_closed_loop(
        env, adapter, teacher_agent, estimator, args_cli.context,
        pass_through_fn=pass_through_fn,
        episodes=args_cli.eval_episodes, max_episode_steps=args_cli.max_episode_steps,
        seed=args_cli.seed + args_cli.eval_seed_offset,
    )

    # Open-loop error on the teacher-forced dataset. Reported separately and
    # never as `rmse`: teacher forcing hides the exposure bias an autoregressive
    # estimator actually pays, so the closed-loop figure above is the honest one.
    open_loop = evaluate_predictions_chunked(estimator, dataset, args_cli.device)
    metrics["open_loop_rmse"] = open_loop.get("rmse")
    metrics["open_loop_r2"] = open_loop.get("r2")
    metrics["r2"] = open_loop.get("r2")

    history = HistoryBuffer(args_cli.num_envs, args_cli.context, observation_dim, torch.device(args_cli.device))

    def act(observations: torch.Tensor) -> torch.Tensor:
        sequence = history.push(adapter.estimator_input())
        estimate = estimator.predict_step(sequence, None if pass_through_fn is None else pass_through_fn())
        return adapter.action(teacher_agent, adapter.inject_estimate(observations, estimate))

    def on_step(dones: torch.Tensor) -> None:
        history.reset(dones)
        estimator.reset(dones)

    if locomotion:
        estimator.reset()
        history.values.zero_()
        metrics.update(
            evaluate_student_command_grid(
                env, adapter, act, on_step,
                settle_s=args_cli.grid_settle_s, measure_s=args_cli.grid_measure_s,
                seed=args_cli.seed + args_cli.eval_seed_offset,
            )
        )
    else:
        estimator.reset()
        history.values.zero_()
        metrics.update(
            evaluate_paired_motion_fidelity(
                env, adapter, teacher_agent, act,
                seed=args_cli.seed + args_cli.eval_seed_offset,
                horizon=args_cli.mpjpe_horizon,
                on_reset=lambda: (history.values.zero_(), estimator.reset()),
            )
        )

    # A pass-through dimension is copied from the sensor, not regressed, so its
    # closed-loop error is identically zero -- unless the value SET copied is not
    # the value the sensor actually held at that step. A frozen or stale sensor
    # read is exactly that, and it is otherwise nearly invisible: the validation
    # loss barely moves, because the fit is made and scored on the same degenerate
    # inputs, while deployment collapses because the policy is handed a constant
    # attitude. Deliberate IMU randomization breaks the invariant on purpose, so
    # the check is skipped there.
    if pass_through and args_cli.imu_noise_scale == 0.0:
        leaked = {
            index: round(metrics["target_rmse"][index], 6)
            for index in pass_through
            if metrics["target_rmse"][index] != 0.0
        }
        if leaked:
            raise RuntimeError(
                "Pass-through dimensions must have exactly zero error; got "
                f"{leaked}. SET copied something other than the live sensor read."
            )

    metrics["parameters"] = sum(parameter.numel() for parameter in estimator.parameters())
    metrics["best_validation_mse"] = training["best_validation_mse"]
    metrics["total_gradient_steps"] = training["gradient_steps"]
    metrics["learning_curve"] = [{"step": row["epoch"], **row} for row in training["epochs"]]
    metrics["collection"] = collection
    metrics["wall_time_s"] = time.monotonic() - started

    torch.save(
        {
            "jose_schema_version": SCHEMA_VERSION,
            "kind": "set_baseline",
            "task": args_cli.task,
            "adapter": args_cli.adapter,
            "model_config": estimator.config(),
            "model_state_dict": estimator.state_dict(),
            "target_split": split_description,
        },
        output / "set_estimator.pt",
    )
    (output / "training.json").write_text(
        json.dumps(
            {
                "config": {
                    "method": "SET",
                    "task": args_cli.task,
                    "adapter": args_cli.adapter,
                    "seed": args_cli.seed,
                    "num_envs": args_cli.num_envs,
                    "collect_steps": args_cli.collect_steps,
                    "imu_noise_scale": args_cli.imu_noise_scale,
                    "max_dataset_size": args_cli.max_dataset_size,
                    "epochs": args_cli.epochs,
                },
                "set_config": {
                    **estimator.config(),
                    "unspecified_in_paper": ["width", "heads", "dropout", "optimizer", "lr", "batch_size"],
                    "data_protocol": "offline_expert_rollout",
                },
                "target_split": split_description,
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    env.close()
    print(
        f"[set] done in {metrics['wall_time_s'] / 60:.1f} min  "
        f"eplen={metrics['episode_length_mean']:.1f} death={metrics['death_rate']:.2f}% "
        f"rmse={metrics['rmse']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    import traceback

    try:
        main()
    except Exception:
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
    simulation_app.close()
