"""Train a G1 state estimator with initial teacher data and optional DAgger rounds."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

from isaaclab.app import AppLauncher


def _checkpoint_fingerprint(path: str | Path) -> dict:
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "size": path.stat().st_size, "sha256": digest.hexdigest()}


parser = argparse.ArgumentParser(description="JOSE G1 state-estimator training")
parser.add_argument("--teacher_checkpoint", "--teacher-checkpoint", dest="teacher_checkpoint", required=True)
parser.add_argument("--task", default="Isaac-G1-AMP-Walk-JOSE-Direct-v0")
parser.add_argument("--agent_cfg_entry_point", "--agent", dest="agent", default="skrl_amp_cfg_entry_point")
parser.add_argument("--adapter", choices=("amp", "ppo_walk"), default="amp")
parser.add_argument("--joint_preset", "--joint-preset", dest="joint_preset", choices=("all", "legs", "upper"), default="all")
parser.add_argument("--est_type", "--estimator", dest="estimator", choices=("LSTM", "TCN", "MLP", "HISTORY_MLP"), default="LSTM")
parser.add_argument(
    "--window", type=int, default=None,
    help="History window (default: 25 for AMP Walk, 50 for other tasks)",
)
parser.add_argument("--hidden_size", "--hidden-size", dest="hidden_size", type=int, default=256)
parser.add_argument("--num_layers", "--num-layers", dest="num_layers", type=int, default=2)
parser.add_argument("--tcn_channels", "--tcn-channels", dest="tcn_channels", type=int, nargs="+", default=[64, 128, 128])
parser.add_argument("--collect_steps", "--collect-steps", dest="collect_steps", type=int, default=2000)
parser.add_argument("--noise_levels", "--noise-levels", dest="noise_levels", type=float, nargs="+", default=[0.0, 0.01, 0.02])
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--dagger_epochs", "--dagger-epochs", dest="dagger_epochs", type=int, default=10, help="Epochs for each DAgger refit")
parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=1024)
parser.add_argument("--lr", type=float, default=1.0e-3)
parser.add_argument(
    "--mpjpe-horizon", type=int, default=100,
    help="Steps of teacher-paired rollout used for the MPJPE motion-fidelity metrics. "
    "Bounded because the two trajectories diverge chaotically once the policies differ.",
)
parser.add_argument("--dagger_rounds", "--dagger-rounds", dest="dagger_rounds", type=int, default=10)
parser.add_argument("--dagger_est_ratio", "--dagger-est-ratio", dest="dagger_est_ratio", type=float, default=0.8)
parser.add_argument("--dagger_est_ratio_final", "--dagger-est-ratio-final", dest="dagger_est_ratio_final", type=float, default=1.0)
parser.add_argument("--dagger_est_ratio_schedule", "--dagger-est-ratio-schedule", dest="dagger_est_ratio_schedule", choices=("linear", "constant"), default="linear")
parser.add_argument("--dagger_extra_rounds", "--dagger-extra-rounds", dest="dagger_extra_rounds", type=int, default=0)
parser.add_argument("--max_dataset_size", "--max-dataset-size", dest="max_dataset_size", type=int, default=250000)
parser.add_argument("--dataset-cache", default=None, help="Reusable initial teacher dataset cache (.pt)")
parser.add_argument(
    "--dataset-cache-window", type=int, default=None,
    help="History window stored in the all-joint cache; may be larger than --window",
)
parser.add_argument("--eval_episodes", "--eval-episodes", dest="eval_episodes", type=int, default=200)
parser.add_argument("--max_episode_steps", "--max-episode-steps", dest="max_episode_steps", type=int, default=1000)
parser.add_argument("--eval-seed-offset", type=int, default=10000)
parser.add_argument("--num_envs", "--num-envs", dest="num_envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_dir", "--output-dir", dest="output_dir", default="logs/jose_g1/estimators")
parser.add_argument("--run-name", default=None, help="Explicit output subdirectory name")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.window is None:
    args_cli.window = 25 if args_cli.task == "Isaac-G1-AMP-Walk-JOSE-Direct-v0" else 50

from jose.teacher_setup import resolve_agent_entry_point  # noqa: E402

# `--adapter ppo_walk` implies the rsl-rl runner config unless overridden.
args_cli.agent = resolve_agent_entry_point(args_cli.adapter, args_cli.agent)

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from isaaclab_tasks.utils.hydra import hydra_task_config
import isaaclab_tasks  # noqa: F401

from jose.estimator.adapters import make_policy_adapter
from jose.estimator.models import build_estimator
from jose.estimator.pipeline import (
    HistoryBuffer,
    RolloutDataset,
    collect_rollout,
    evaluate_estimator_closed_loop,
    evaluate_paired_motion_fidelity,
    evaluate_predictions,
    save_jose_checkpoint,
    train_estimator,
)
from jose.schema import JOINT_PRESETS, SCHEMA_VERSION
from jose.teacher_setup import build_env_and_teacher, teacher_policy_module


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed
    teacher_fingerprint = _checkpoint_fingerprint(args_cli.teacher_checkpoint)
    env, teacher_agent = build_env_and_teacher(
        args_cli.task,
        args_cli.adapter,
        env_cfg,
        agent_cfg,
        teacher_fingerprint["path"],
        args_cli.device,
        seed=args_cli.seed,
    )
    adapter = make_policy_adapter(args_cli.adapter, env, args_cli.joint_preset)
    window = 1 if args_cli.estimator == "MLP" else args_cli.window
    cache_window = args_cli.dataset_cache_window or window
    if cache_window < window:
        raise ValueError("--dataset-cache-window cannot be smaller than the estimator window")
    cache_adapter = make_policy_adapter(args_cli.adapter, env, "all") if args_cli.dataset_cache else adapter
    estimator = build_estimator(
        args_cli.estimator, adapter.input_dim, adapter.schema.estimator_target_dim,
        args_cli.hidden_size, args_cli.num_layers, tuple(args_cli.tcn_channels), window,
    )
    default_run_name = (
        f"{args_cli.task}_{args_cli.estimator}_w{window}_{args_cli.joint_preset}_seed{args_cli.seed}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    )
    run_name = args_cli.run_name or default_run_name
    output = Path(args_cli.output_dir) / run_name
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.jsonl"
    writer = SummaryWriter(str(output / "tensorboard"))
    started = time.monotonic()

    def log_event(phase: str, **values):
        row = {"elapsed_s": time.monotonic() - started, "phase": phase, **values}
        with progress_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        details = " ".join(f"{key}={value}" for key, value in values.items())
        print(f"[{row['elapsed_s']:8.1f}s] {phase} {details}", flush=True)

    config = {
        "teacher_checkpoint": teacher_fingerprint,
        "task": args_cli.task,
        "agent": args_cli.agent,
        "adapter": args_cli.adapter,
        "estimator": args_cli.estimator,
        "window": window,
        "joint_preset": args_cli.joint_preset,
        "hidden_size": args_cli.hidden_size,
        "num_layers": args_cli.num_layers,
        "tcn_channels": list(args_cli.tcn_channels),
        "collect_steps": args_cli.collect_steps,
        "noise_levels": list(args_cli.noise_levels),
        "epochs": args_cli.epochs,
        "dagger_epochs": args_cli.dagger_epochs,
        "batch_size": args_cli.batch_size,
        "learning_rate": args_cli.lr,
        "dagger_rounds": args_cli.dagger_rounds,
        "dagger_extra_rounds": args_cli.dagger_extra_rounds,
        "dagger_estimator_ratio_initial": args_cli.dagger_est_ratio,
        "dagger_estimator_ratio_final": args_cli.dagger_est_ratio_final,
        "dagger_estimator_ratio_schedule": args_cli.dagger_est_ratio_schedule,
        "max_dataset_size": args_cli.max_dataset_size,
        "dagger_max_samples_per_round": args_cli.max_dataset_size,
        "dataset_aggregation": "uniform_random_subsample_after_each_round",
        "eval_episodes": args_cli.eval_episodes,
        "max_episode_steps": args_cli.max_episode_steps,
        "eval_seed_offset": args_cli.eval_seed_offset,
        "num_envs": args_cli.num_envs,
        "seed": args_cli.seed,
        "device": args_cli.device,
        "dataset_cache": args_cli.dataset_cache,
        "dataset_cache_window": cache_window,
        "evaluation_domain_randomization": False,
        "evaluation_action_noise": 0.0,
    }
    config_path = output / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("\nJOSE G1 estimator")
    print(f"  task={args_cli.task} adapter={args_cli.adapter} preset={args_cli.joint_preset}")
    print(
        f"  model={args_cli.estimator} window={window} input={adapter.input_dim} "
        f"target={adapter.schema.estimator_target_dim}"
    )
    print("  velocity_source=sim_joint_velocity")
    cache_metadata = {
        "cache_format": 2,
        "jose_schema_version": SCHEMA_VERSION,
        "teacher_checkpoint": teacher_fingerprint,
        "task": args_cli.task,
        "adapter": args_cli.adapter,
        "joint_preset": "all" if args_cli.dataset_cache else args_cli.joint_preset,
        "window": cache_window,
        "seed": args_cli.seed,
        "collect_steps": args_cli.collect_steps,
        "noise_levels": list(args_cli.noise_levels),
        "max_dataset_size": args_cli.max_dataset_size,
    }
    dataset = None
    initial_collections = []
    cache_path = Path(args_cli.dataset_cache).resolve() if args_cli.dataset_cache else None
    if cache_path is not None and cache_path.exists():
        cached, metadata = RolloutDataset.load_with_metadata(cache_path)
        if metadata != cache_metadata:
            raise ValueError(f"Dataset cache metadata mismatch: {cache_path}")
        dataset = cached
        initial_collections = [{"cache": "hit", "samples": len(dataset.targets)}]
        log_event("phase1/cache_hit", path=str(cache_path), samples=len(dataset.targets))
    else:
        initial_sample_cap = max(1, args_cli.max_dataset_size // max(len(args_cli.noise_levels), 1))
        log_event("phase1/start", noise_levels=args_cli.noise_levels)
        for noise in args_cli.noise_levels:
            collection_started = time.monotonic()
            batch, collection = collect_rollout(
                env,
                cache_adapter,
                teacher_agent,
                args_cli.collect_steps,
                cache_window,
                action_noise=noise,
                max_samples=initial_sample_cap,
            )
            dataset = batch if dataset is None else dataset.append(batch, args_cli.max_dataset_size)
            collection_duration = time.monotonic() - collection_started
            initial_collections.append({
                "action_noise": noise,
                "duration_s": collection_duration,
                **collection,
            })
            log_event(
                "phase1/collection", noise=noise, samples=len(batch.targets),
                duration_s=round(collection_duration, 3),
            )
        if cache_path is not None:
            dataset.save(cache_path, cache_metadata)
            log_event("phase1/cache_saved", path=str(cache_path), samples=len(dataset.targets))
    assert dataset is not None
    if args_cli.dataset_cache:
        source_dataset = dataset
        cache_joint_names = JOINT_PRESETS["all"]
        projection_ids = tuple(cache_joint_names.index(name) for name in JOINT_PRESETS[args_cli.joint_preset])
        dataset = source_dataset.project_joint_history(
            projection_ids,
            source_window=cache_window,
            target_window=window,
            full_joint_count=cache_adapter.input_dim // 2,
        )
        if dataset is not source_dataset:
            del source_dataset
            if "cached" in locals():
                del cached
        log_event(
            "phase1/cache_projection", source_window=cache_window, target_window=window,
            joint_preset=args_cli.joint_preset, input_dim=dataset.frames.shape[-1],
        )
    config["dataset"] = {
        "initial_samples": len(dataset.targets),
        "history_shape": list(dataset.histories.shape[1:]),
        "target_dim": dataset.targets.shape[-1],
        "frame_dim": dataset.frames.shape[-1],
        "teacher_action_dim": dataset.teacher_actions.shape[-1],
        "dtype": str(dataset.histories.dtype),
        "bytes": sum(value.numel() * value.element_size() for value in dataset.__dict__.values()),
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    log_event("phase2/start", samples=len(dataset.targets), epochs=args_cli.epochs)
    epoch_offset = 0

    def epoch_logger(row):
        step = epoch_offset + row["epoch"]
        writer.add_scalar("Loss/train_mse", row["train_mse"], step)
        writer.add_scalar("Loss/validation_mse", row["validation_mse"], step)
        log_event("phase2/epoch", global_epoch=step, **row)
        print(
            f"  epoch {row['epoch']:03d}: train={row['train_mse']:.6f} "
            f"val={row['validation_mse']:.6f}", flush=True,
        )

    training = train_estimator(
        estimator, dataset, args_cli.estimator, args_cli.epochs, args_cli.batch_size,
        args_cli.lr, args_cli.device, epoch_logger,
    )
    epoch_offset += args_cli.epochs
    # Counts optimizer steps, not epochs, so it lines up with train_history_student.py's
    # cumulative_gradient_steps on a shared "compute spent" x-axis for learning-curve plots.
    cumulative_gradient_steps = training["gradient_steps"]
    log_event("phase2/complete", best_validation_mse=training["best_validation_mse"])
    closed_loop = evaluate_estimator_closed_loop(
        env,
        adapter,
        teacher_agent,
        estimator,
        args_cli.estimator,
        window,
        args_cli.eval_episodes,
        args_cli.max_episode_steps,
        args_cli.seed + args_cli.eval_seed_offset,
    )
    rounds = [{
        "round": 0,
        "dataset_size": len(dataset.targets),
        "collection": initial_collections,
        "training": training,
        "evaluation": closed_loop,
        "cumulative_gradient_steps": cumulative_gradient_steps,
    }]
    for name, value in closed_loop.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f"Evaluation/{name}", value, 0)
    log_event(
        "evaluation", round=0,
        **{key: value for key, value in closed_loop.items() if isinstance(value, (int, float))},
    )
    best_score = (
        closed_loop["episode_length_mean"],
        -closed_loop["death_rate"],
        -closed_loop["rmse"],
    )
    best_round = 0
    best_state = {name: value.detach().cpu().clone() for name, value in estimator.state_dict().items()}
    save_jose_checkpoint(output / "estimator_round_0.pt", estimator, adapter, args_cli.task, window, closed_loop)

    total_rounds = args_cli.dagger_rounds + args_cli.dagger_extra_rounds
    for round_index in range(1, total_rounds + 1):
        if round_index > args_cli.dagger_rounds:
            ratio = 1.0
        elif args_cli.dagger_est_ratio_schedule == "constant" or args_cli.dagger_rounds <= 1:
            ratio = args_cli.dagger_est_ratio
        else:
            progress = (round_index - 1) / (args_cli.dagger_rounds - 1)
            ratio = args_cli.dagger_est_ratio + progress * (
                args_cli.dagger_est_ratio_final - args_cli.dagger_est_ratio
            )
        # Each round continues from wherever the previous one landed, and the
        # eplen-first best_score below still decides which round is reported and
        # saved. Resuming from best_state instead was tried and reverted: rounds
        # 4-10 then all restarted from the same peak and each one destroyed it,
        # which showed the regression is in the per-round fit (best epoch is
        # picked by held-out regression loss, which doesn't track closed-loop
        # performance), not in the round-to-round handoff. Keeping the original
        # handoff also keeps this comparable to the architecture/window/
        # joint_scope studies, which were all run this way.
        log_event("dagger/start", round=round_index, total=total_rounds, estimator_ratio=ratio)
        collection_started = time.monotonic()
        new_data, collection = collect_rollout(
            env,
            adapter,
            teacher_agent,
            args_cli.collect_steps,
            window,
            estimator,
            ratio,
            action_noise=0.01,
            max_samples=args_cli.max_dataset_size,
        )
        collection_duration = time.monotonic() - collection_started
        collection["duration_s"] = collection_duration
        log_event(
            "dagger/collection", round=round_index, samples=len(new_data.targets),
            duration_s=round(collection_duration, 3),
        )
        dataset = dataset.append(new_data, args_cli.max_dataset_size)
        log_event("dagger/dataset", round=round_index, samples=len(dataset.targets))
        round_epoch_offset = epoch_offset

        def dagger_epoch_logger(row, round_index=round_index, round_epoch_offset=round_epoch_offset):
            step = round_epoch_offset + row["epoch"]
            writer.add_scalar("Loss/train_mse", row["train_mse"], step)
            writer.add_scalar("Loss/validation_mse", row["validation_mse"], step)
            log_event("dagger/epoch", round=round_index, global_epoch=step, **row)
            print(
                f"  round {round_index:02d} epoch {row['epoch']:03d}: "
                f"train={row['train_mse']:.6f} val={row['validation_mse']:.6f}", flush=True,
            )
        training = train_estimator(
            estimator,
            dataset,
            args_cli.estimator,
            args_cli.dagger_epochs,
            args_cli.batch_size,
            args_cli.lr * 0.5,
            args_cli.device,
            dagger_epoch_logger,
        )
        epoch_offset += args_cli.dagger_epochs
        cumulative_gradient_steps += training["gradient_steps"]
        closed_loop = evaluate_estimator_closed_loop(
            env,
            adapter,
            teacher_agent,
            estimator,
            args_cli.estimator,
            window,
            args_cli.eval_episodes,
            args_cli.max_episode_steps,
            args_cli.seed + args_cli.eval_seed_offset,
        )
        rounds.append(
            {
                "round": round_index,
                "estimator_ratio": ratio,
                "dataset_size": len(dataset.targets),
                "collection": collection,
                "training": training,
                "evaluation": closed_loop,
                "cumulative_gradient_steps": cumulative_gradient_steps,
            }
        )
        writer.add_scalar("DAgger/estimator_ratio", ratio, round_index)
        writer.add_scalar("DAgger/dataset_size", len(dataset.targets), round_index)
        for name, value in closed_loop.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(f"Evaluation/{name}", value, round_index)
        log_event(
            "evaluation", round=round_index,
            **{key: value for key, value in closed_loop.items() if isinstance(value, (int, float))},
        )
        score = (closed_loop["episode_length_mean"], -closed_loop["death_rate"], -closed_loop["rmse"])
        if score > best_score:
            best_score = score
            best_round = round_index
            best_state = {name: value.detach().cpu().clone() for name, value in estimator.state_dict().items()}
        save_jose_checkpoint(
            output / f"estimator_round_{round_index}.pt", estimator, adapter, args_cli.task, window, closed_loop
        )
        print(
            f"  round={round_index}/{total_rounds} ratio={ratio:.2f} samples={len(dataset.targets):,} "
            f"episode={closed_loop['episode_length_mean']:.1f} death={closed_loop['death_rate']:.1f}%"
        )

    estimator.load_state_dict(best_state)
    evaluation_data, _ = collect_rollout(
        env,
        adapter,
        teacher_agent,
        min(args_cli.collect_steps, 200),
        window,
        estimator,
        estimator_ratio=1.0,
        max_samples=10000,
    )
    metrics = evaluate_predictions(
        estimator,
        evaluation_data,
        args_cli.estimator,
        args_cli.device,
        adapter.schema.estimator_target_names,
    )
    metrics.update(rounds[best_round]["evaluation"])
    metrics["estimator_parameters"] = metrics["parameters"]
    policy = teacher_policy_module(teacher_agent)
    metrics["teacher_policy_parameters"] = (
        sum(parameter.numel() for parameter in policy.parameters()) if policy is not None else 0
    )
    metrics["parameters"] = metrics["estimator_parameters"] + metrics["teacher_policy_parameters"]
    metrics["estimator_inference_ms_per_sample"] = metrics["inference_ms_per_sample"]
    benchmark_observations, _ = env.reset()
    benchmark_history = HistoryBuffer(
        benchmark_observations.shape[0], window, adapter.input_dim, benchmark_observations.device
    )
    benchmark_steps = 100
    with torch.inference_mode():
        for _ in range(10):
            frame = adapter.estimator_input()
            sequence = benchmark_history.push(frame)
            model_input = frame if args_cli.estimator == "MLP" else sequence
            estimate = estimator.predict(model_input)
            adapter.action(teacher_agent, adapter.inject_estimate(benchmark_observations, estimate))
        if str(args_cli.device).startswith("cuda"):
            torch.cuda.synchronize(args_cli.device)
        latency_started = time.perf_counter()
        for _ in range(benchmark_steps):
            frame = adapter.estimator_input()
            sequence = benchmark_history.push(frame)
            model_input = frame if args_cli.estimator == "MLP" else sequence
            estimate = estimator.predict(model_input)
            adapter.action(teacher_agent, adapter.inject_estimate(benchmark_observations, estimate))
        if str(args_cli.device).startswith("cuda"):
            torch.cuda.synchronize(args_cli.device)
    metrics["inference_ms_per_sample"] = (
        (time.perf_counter() - latency_started) * 1000.0 /
        (benchmark_steps * benchmark_observations.shape[0])
    )
    metrics["best_round"] = best_round
    metrics["rounds"] = rounds
    metrics["total_gradient_steps"] = cumulative_gradient_steps
    # One entry per evaluated round on a step-indexed x-axis, so reporting.py can plot it
    # against train_history_student.py's own learning_curve on a shared "compute spent" axis.
    metrics["learning_curve"] = [
        {"step": row["cumulative_gradient_steps"], **row["evaluation"]} for row in rounds
    ]
    # Teacher-relative motion fidelity, on the reported checkpoint. Runs last of
    # everything because it rolls the env twice more; it restores the RNG state
    # it borrows, but keeping it after every other measurement means it cannot
    # perturb any of them even if that guarantee ever weakens.
    fidelity_history = HistoryBuffer(
        benchmark_observations.shape[0], window, adapter.input_dim, benchmark_observations.device
    )

    def estimator_policy(observations: torch.Tensor) -> torch.Tensor:
        frame = adapter.estimator_input()
        sequence = fidelity_history.push(frame)
        model_input = frame if args_cli.estimator.upper() == "MLP" else sequence
        return adapter.action(teacher_agent, adapter.inject_estimate(observations, estimator.predict(model_input)))

    metrics.update(
        evaluate_paired_motion_fidelity(
            env, adapter, teacher_agent, estimator_policy,
            seed=args_cli.seed + args_cli.eval_seed_offset,
            horizon=args_cli.mpjpe_horizon,
            on_reset=lambda: fidelity_history.values.zero_(),
        )
    )
    metrics["wall_time_s"] = time.monotonic() - started
    save_jose_checkpoint(output / "best_estimator.pt", estimator, adapter, args_cli.task, window, metrics)
    (output / "training.json").write_text(
        json.dumps({"config": config, "metrics": metrics}, indent=2), encoding="utf-8"
    )
    print(
        f"  saved={output / 'best_estimator.pt'} best_round={best_round} "
        f"rmse={metrics['rmse']:.5f} r2={metrics['r2']:.4f}"
    )
    log_event("complete", best_round=best_round, rmse=metrics["rmse"], episode_length=metrics["episode_length_mean"])
    writer.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
