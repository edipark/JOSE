"""Shared JOSE estimator-ablation execution, cache, resume, and report helper.

One invocation evaluates one task and one teacher checkpoint. Walk is the
default. Isaac Sim jobs remain sequential on one GPU; compatible estimator
experiments reuse an on-disk initial rollout dataset.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

try:
    from .reporting import generate_report
except ImportError:
    from reporting import generate_report


TASKS = {
    "amp_walk": ("Isaac-G1-AMP-Walk-JOSE-Direct-v0", "amp", "skrl_amp_cfg_entry_point"),
    "amp_dance": ("Isaac-G1-AMP-Dance-JOSE-Direct-v0", "amp", "skrl_amp_cfg_entry_point"),
    "amp_jump": ("Isaac-G1-AMP-Jump-JOSE-Direct-v0", "amp", "skrl_amp_cfg_entry_point"),
    "ppo_walk": ("Isaac-G1-PPO-Walk-JOSE-Direct-v0", "ppo", "skrl_cfg_entry_point"),
}

ABLATION_FORMAT_VERSION = 2


def _file_fingerprint(path: str | Path) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        return {"path": str(path), "exists": False}
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "size": path.stat().st_size, "sha256": digest.hexdigest()}


def _implementation_fingerprint() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for relative in (
        "ablation_runner.py", "train_state_estimator.py", "evaluate_teacher.py", "train_dagger.py",
        "estimator/pipeline.py", "estimator/adapters.py", "estimator/models.py", "schema.py",
        "g1_amp_env.py", "g1_amp_env_cfg.py", "task_math.py",
    ):
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _acquire_run_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        owner = handle.read().strip() or "unknown process"
        handle.close()
        raise RuntimeError(f"Another ablation is already using this output directory ({owner})") from None
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} started={datetime.now().isoformat()}\n")
    handle.flush()
    return handle


def _extract_metrics(training_json: Path) -> dict:
    if not training_json.exists():
        return {}
    data = json.loads(training_json.read_text(encoding="utf-8"))
    source = data.get("metrics", data)
    metrics = {key: value for key, value in source.items() if isinstance(value, (int, float))}
    rounds = source.get("rounds", data.get("rounds", []))
    if isinstance(rounds, list):
        metrics["round_count"] = len(rounds)
    return metrics


def _successful_keys(path: Path, signature: str) -> set[tuple[str, int, str]]:
    keys = set()
    if not path.exists():
        return keys
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("status") == "ok" and record.get("run_signature") == signature:
            keys.add((record.get("task"), record.get("seed"), record.get("experiment")))
    return keys


def _latest_records(path: Path, signature: str) -> list[dict]:
    latest: dict[tuple[str, int, str], dict] = {}
    if not path.exists():
        return []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("run_signature") == signature:
            latest[(record.get("task"), record.get("seed"), record.get("experiment"))] = record
    return list(latest.values())


def _run_live(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as stream:
        process = subprocess.Popen(
            command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1
        )
        assert process.stdout is not None
        for line in process.stdout:
            print("  | " + line, end="", flush=True)
            stream.write(line)
        return process.wait()


def _parser(experiments: tuple[tuple[str, str, int, str, int], ...], study_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"JOSE G1 {study_name} ablation")
    parser.add_argument(
        "--teacher_checkpoint", "--teacher-checkpoint", dest="teacher_checkpoint", required=True,
        help="Path to one SKRL teacher checkpoint",
    )
    parser.add_argument("--task", choices=tuple(TASKS), default="amp_walk", help="Task preset (default: walk)")
    parser.add_argument("--task-id", default=None, help="Override Gym task id")
    parser.add_argument("--agent_cfg_entry_point", "--agent", dest="agent", default=None)
    parser.add_argument("--adapter", choices=("amp", "ppo"), default=None)
    parser.add_argument("--seeds", type=int, default=3, help="Number of consecutive seeds")
    parser.add_argument("--seed-start", "--seed_start", dest="seed_start", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--collect-steps", type=int, default=2000, help="Estimator collection steps")
    parser.add_argument("--epochs", type=int, default=50, help="Initial estimator epochs")
    parser.add_argument("--dagger-epochs", type=int, default=10, help="Epochs per estimator DAgger round")
    parser.add_argument("--max-dataset-size", type=int, default=500000)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument(
        "--experiments", nargs="+", choices=tuple(item[0] for item in experiments), default=None,
        help="Run only selected estimator experiments (TeacherGT is always included)",
    )
    parser.add_argument("--skip-student", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", default=f"logs/jose_g1/ablation/{study_name}")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    headless = parser.add_mutually_exclusive_group()
    headless.add_argument("--headless", dest="headless", action="store_true")
    headless.add_argument("--no-headless", dest="headless", action="store_false")
    parser.set_defaults(headless=True)
    parser.add_argument("--rerun", action="store_true", help="Rerun successful entries")
    return parser


def main(experiments: tuple[tuple[str, str, int, str, int], ...], study_name: str):
    if not experiments:
        raise ValueError("At least one ablation experiment is required")
    args = _parser(experiments, study_name).parse_args()
    if args.seeds <= 0:
        raise ValueError("--seeds must be positive")
    teacher = str(Path(args.teacher_checkpoint).resolve())
    if not args.dry_run and not Path(teacher).is_file():
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher}")

    default_task_id, default_adapter, default_agent = TASKS[args.task]
    task_id = args.task_id or default_task_id
    adapter = args.adapter or default_adapter
    agent = args.agent or default_agent
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    if args.fast:
        args.collect_steps, args.epochs, args.dagger_epochs = 100, 2, 2
        args.max_dataset_size = min(args.max_dataset_size, 20000)
        args.eval_episodes = min(args.eval_episodes, 20)

    teacher_fingerprint = _file_fingerprint(teacher)
    implementation_fingerprint = _implementation_fingerprint()
    signature_payload = {
        "format_version": ABLATION_FORMAT_VERSION,
        "teacher": teacher_fingerprint, "implementation": implementation_fingerprint,
        "task": args.task, "task_id": task_id, "adapter": adapter, "agent": agent,
        "num_envs": args.num_envs, "collect_steps": args.collect_steps, "epochs": args.epochs,
        "dagger_epochs": args.dagger_epochs, "max_dataset_size": args.max_dataset_size,
        "eval_episodes": args.eval_episodes, "experiments": args.experiments,
    }
    run_signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_lock = None if args.dry_run else _acquire_run_lock(output / ".active.lock")
    session_output = output / "sessions" / run_signature
    session_output.mkdir(parents=True, exist_ok=True)
    raw_path = session_output / "raw_results.jsonl"
    latest_path = session_output / "results_latest.jsonl"
    attempt_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_pid{os.getpid()}"
    attempt_output = session_output / "attempts" / attempt_id
    run_output = attempt_output / "runs"
    matrix = []
    selected_experiments = (
        tuple(item for item in experiments if item[0] in args.experiments)
        if args.experiments is not None else experiments
    )
    for seed in seeds:
        matrix.append((seed, "TeacherGT", "TEACHER", 1, "all", 0))
        matrix.extend((seed, *experiment) for experiment in selected_experiments)
    if args.fast:
        matrix = matrix[: min(len(matrix), 6)]
    seed_short_cache_windows = {
        seed: max(
            (row_window for row_seed, _, row_model, row_window, _, _ in matrix
             if row_seed == seed and row_model != "TEACHER" and row_window <= 50),
            default=1,
        )
        for seed in seeds
    }

    print("\n" + "=" * 72)
    print(f"JOSE G1 {study_name} Ablation")
    print("=" * 72)
    print(f"task={args.task} ({task_id})")
    print(f"teacher={teacher}")
    print(f"session={run_signature} output={session_output}")
    print(f"seeds={seeds} num_envs={args.num_envs}")
    print(f"estimator: collect={args.collect_steps}, epochs={args.epochs}/{args.dagger_epochs}, max_dataset={args.max_dataset_size:,}")
    print("policy-distillation baselines are isolated in run_method_comparison.py")
    print("execution: sequential on one GPU; compatible initial datasets are cached")
    print("=" * 72, flush=True)

    completed = set() if args.rerun else _successful_keys(raw_path, run_signature)
    records = []
    start_all = time.monotonic()
    for index, (seed, name, model, window, preset, dagger_rounds) in enumerate(matrix, 1):
        if (args.task, seed, name) in completed:
            print(f"[{index:03d}/{len(matrix):03d}] {args.task}/{name}/seed{seed} SKIP", flush=True)
            continue

        if model == "TEACHER":
            script = Path(__file__).with_name("evaluate_teacher.py")
            command = [
                sys.executable, str(script), "--teacher-checkpoint", teacher,
                "--task", task_id, "--agent", agent, "--adapter", adapter,
                "--seed", str(seed), "--num-envs", str(args.num_envs),
                "--collect-steps", str(args.collect_steps), "--output-dir", str(run_output),
                "--run-name", f"{task_id}_TeacherGT_seed{seed}",
            ]
            artifact = run_output / f"{task_id}_TeacherGT_seed{seed}" / "training.json"
        else:
            script = Path(__file__).with_name("train_state_estimator.py")
            run_name = f"{task_id}_{name}_seed{seed}"
            # Share the maximum requested all-joint teacher cache. Each model
            # still collects its own closed-loop DAgger rounds.
            cache_window = window if window > 50 else seed_short_cache_windows[seed]
            cache_spec = {
                "format_version": ABLATION_FORMAT_VERSION,
                "teacher": teacher_fingerprint, "implementation": implementation_fingerprint,
                "task_id": task_id, "adapter": adapter, "seed": seed,
                "window": cache_window, "preset": "all", "collect_steps": args.collect_steps,
                "max_dataset_size": args.max_dataset_size, "noise_levels": [0.0, 0.01, 0.02],
            }
            cache_id = hashlib.sha256(
                json.dumps(cache_spec, sort_keys=True).encode("utf-8")
            ).hexdigest()[:12]
            cache = output / "dataset_cache" / f"{args.task}_seed{seed}_w{cache_window}_all_{cache_id}.pt"
            command = [
                sys.executable, str(script), "--teacher_checkpoint", teacher,
                "--task", task_id, "--agent_cfg_entry_point", agent, "--adapter", adapter,
                "--est_type", model, "--window", str(window), "--joint_preset", preset,
                "--dagger_rounds", str(dagger_rounds), "--dagger_epochs", str(args.dagger_epochs),
                "--collect_steps", str(args.collect_steps), "--epochs", str(args.epochs),
                "--max_dataset_size", str(args.max_dataset_size),
                "--eval_episodes", str(args.eval_episodes),
                "--num_envs", str(args.num_envs), "--seed", str(seed),
                "--dataset-cache", str(cache), "--run-name", run_name,
                "--dataset-cache-window", str(cache_window),
                "--output-dir", str(run_output),
            ]
            artifact = run_output / run_name / "training.json"
        if args.headless:
            command.append("--headless")

        elapsed = time.monotonic() - start_all
        eta = elapsed / max(index - 1, 1) * (len(matrix) - index + 1) if index > 1 else 0.0
        print(f"[{index:03d}/{len(matrix):03d}] {args.task}/{name}/seed{seed} ETA={eta / 60:.1f}m", flush=True)
        print("  $ " + " ".join(command), flush=True)
        if args.dry_run:
            continue

        started = time.monotonic()
        process_log = attempt_output / "process_logs" / f"{args.task}_{name}_seed{seed}.log"
        returncode = _run_live(command, process_log)
        record = {
            "task": args.task, "task_id": task_id, "teacher_checkpoint": teacher,
            "teacher_fingerprint": teacher_fingerprint,
            "run_signature": run_signature,
            "experiment": name, "seed": seed, "duration_s": time.monotonic() - started,
            "returncode": returncode,
            "status": "ok" if returncode == 0 else "failed",
            "process_log": str(process_log), "artifact": str(artifact) if artifact.exists() else None,
        }
        if returncode:
            record["error"] = process_log.read_text(encoding="utf-8", errors="replace")[-4000:]
            print(f"  FAILED ({returncode}); continuing", flush=True)
        else:
            record["metrics"] = _extract_metrics(artifact)
            if name == "TeacherGT" and record["metrics"].get("episode_length_mean", 0.0) < 100.0:
                print(
                    "  WARNING: TeacherGT mean episode length < 100; checkpoint/environment mismatch likely",
                    flush=True,
                )
            summary = " ".join(
                f"{key}={value:.4g}" for key, value in record["metrics"].items()
                if isinstance(value, (int, float))
            )
            print("  " + summary, flush=True)
        records.append(record)
        with raw_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")

    if not args.dry_run:
        latest_records = _latest_records(raw_path, run_signature)
        latest_path.write_text(
            "".join(json.dumps(record) + "\n" for record in latest_records), encoding="utf-8"
        )
        result = generate_report(latest_path, session_output)
        print(f"Completed new work in {(time.monotonic() - start_all) / 60:.1f}m; report={result['output']}/report.md")
    if run_lock is not None:
        run_lock.close()
