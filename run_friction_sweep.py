"""Reproducible queue for the zero-shot locomotion friction sweep."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

try:
    from .ablation_common import acquire_run_lock, file_fingerprint
    from .robustness.registry import resolve
except ImportError:
    from ablation_common import acquire_run_lock, file_fingerprint
    from robustness.registry import resolve


FORMAT_VERSION = 1
PAPER_METHODS = ("teacher", "jose", "joint_only", "imu_clean", "set")
DEFAULT_FRICTIONS = (0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2)
DEFAULT_CHECKPOINT_SEEDS = (42, 43, 44)
DEFAULT_EVAL_SEEDS = (10042, 10043, 10044)
COMMAND_COUNT = 15

# Every source file whose bytes can change environment construction, checkpoint
# loading, policy execution, reset seeding, or metric calculation.
SOURCE_FILES = (
    "__init__.py",
    "check_friction_reproducibility.py",
    "eval_friction_robustness.py",
    "run_friction_sweep.py",
    "report_friction_robustness.py",
    "teacher_setup.py",
    "schema.py",
    "task_math.py",
    "distillation/command_eval.py",
    "distillation/history.py",
    "distillation/imu.py",
    "estimator/adapters.py",
    "estimator/locomotion.py",
    "estimator/models.py",
    "ppo_walk/g1_asset.py",
    "ppo_walk/rsl_rl_teacher.py",
    "ppo_walk/walk_env_cfg.py",
    "ppo_walk/walk_estimator_env.py",
    "ppo_walk/walk_estimator_env_cfg.py",
    "ppo_walk/agents/rsl_rl_ppo_cfg.py",
    "ppo_walk/mdp/rewards.py",
    "robustness/methods.py",
    "robustness/registry.py",
    "set_baseline/adapter.py",
    "set_baseline/model.py",
    "set_baseline/targets.py",
)


def canonical_digest(payload: object, length: int | None = None) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    value = hashlib.sha256(encoded).hexdigest()
    return value[:length] if length else value


def validate_unique(values, name: str):
    values = tuple(values)
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{name} must contain unique values")
    return values


def validate_frictions(values) -> tuple[float, ...]:
    values = tuple(float(value) for value in values)
    validate_unique(values, "frictions")
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("frictions must be finite and positive")
    if tuple(sorted(values)) != values:
        raise ValueError("frictions must be strictly increasing")
    return values


def command_seed_base(eval_seed: int) -> int:
    return int(eval_seed) * 100


def cell_key(row: dict) -> tuple[float, str, int | None, int]:
    return (
        float(row["friction"]["effective"]),
        str(row["method"]),
        row.get("checkpoint_seed"),
        int(row["eval_seed"]),
    )


def expected_keys(frictions, methods, checkpoint_seeds, eval_seeds):
    keys = set()
    for friction in frictions:
        for eval_seed in eval_seeds:
            for method in methods:
                seeds = (None,) if method == "teacher" else checkpoint_seeds
                for checkpoint_seed in seeds:
                    keys.add((float(friction), method, checkpoint_seed, int(eval_seed)))
    return keys


def source_fingerprints(root: Path) -> dict[str, dict]:
    fingerprints = {}
    for relative in SOURCE_FILES:
        got = file_fingerprint(root / relative)
        if not got.get("sha256"):
            raise FileNotFoundError(f"Required sweep source is missing: {root / relative}")
        fingerprints[relative] = {"size": got["size"], "sha256": got["sha256"]}
    return fingerprints


def git_state(root: Path) -> dict:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, text=True, capture_output=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"], cwd=root, check=True, text=True, capture_output=True
    ).stdout
    return {
        "head": head,
        "dirty": bool(status.strip()),
        "status": status.splitlines(),
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
    }


def checkpoint_fingerprints(
    teacher: Path, study: Path, set_study: Path, methods, checkpoint_seeds
) -> dict[str, dict]:
    paths: dict[str, Path] = {"teacher": teacher}
    for method in methods:
        if method == "teacher":
            continue
        for seed in checkpoint_seeds:
            _kind, path = resolve(method, study, set_study, seed)
            if path is None:
                raise FileNotFoundError(f"No checkpoint path for {method} seed {seed}")
            paths[f"{method}/seed_{seed}"] = path
    fingerprints = {}
    for name, path in paths.items():
        got = file_fingerprint(path)
        if not got.get("sha256"):
            raise FileNotFoundError(f"Missing checkpoint for {name}: {path}")
        fingerprints[name] = got
    learned_hashes = [value["sha256"] for key, value in fingerprints.items() if key != "teacher"]
    if len(learned_hashes) != len(set(learned_hashes)):
        raise RuntimeError("Two learned-method checkpoint slots resolve to identical files")
    return fingerprints


def _tracked_and_untracked(root: Path) -> list[str]:
    output = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    return sorted(item.decode() for item in output.split(b"\0") if item)


def make_source_snapshot(root: Path, destination: Path) -> Path:
    """Copy the exact working source once; workers never import the live tree."""
    package = destination / "jose"
    if package.exists():
        return package
    package.mkdir(parents=True)
    for relative in _tracked_and_untracked(root):
        source = root / relative
        target = package / relative
        if source.is_dir():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return package


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _mark_manifest_complete(
    manifest: dict, *, completed_cells: int, expected_cells: int, results_path: Path, run_dir: Path
) -> None:
    manifest.update(
        {
            "status": "complete",
            "finished_at": datetime.now().isoformat(),
            "rows": completed_cells,
            "expected_cells": expected_cells,
            "completed_cells": completed_cells,
            "results": str(results_path),
            "report": str(run_dir / "report.md"),
            "figure": str(run_dir / "friction_sweep.png"),
        }
    )
    # A resumed run may carry failure metadata from an earlier attempt.
    manifest.pop("error", None)
    manifest.pop("updated_at", None)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Invalid JSON in {path}:{number}: {error}") from error
    return rows


def _append_rows_atomic(path: Path, rows: list[dict]) -> None:
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    addition = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        stream.write(existing)
        stream.write(addition)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _validate_worker_rows(rows, friction, eval_seed, methods, checkpoint_seeds) -> None:
    expected = expected_keys((friction,), methods, checkpoint_seeds, (eval_seed,))
    got = {cell_key(row) for row in rows}
    if got != expected or len(rows) != len(expected):
        raise RuntimeError(
            f"Worker returned wrong cells: missing={sorted(expected - got)} extra={sorted(got - expected)}"
        )
    for row in rows:
        metrics = row.get("metrics", {})
        for name in ("track_error_norm", "grid_survival_rate", "feet_slide_penalty"):
            value = metrics.get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise RuntimeError(f"{cell_key(row)} has invalid {name}: {value}")
        if len(metrics.get("command_tracking", [])) != COMMAND_COUNT:
            raise RuntimeError(f"{cell_key(row)} does not contain all {COMMAND_COUNT} commands")


def _run_worker(command, log_path: Path, env: dict[str, str], cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as stream:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print("  | " + line, end="", flush=True)
            stream.write(line)
        return process.wait()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--set-study", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", default="logs/jose_g1/friction_robustness")
    parser.add_argument("--frictions", type=float, nargs="+", default=list(DEFAULT_FRICTIONS))
    parser.add_argument(
        "--checkpoint-seeds", type=int, nargs="+", default=list(DEFAULT_CHECKPOINT_SEEDS)
    )
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=list(DEFAULT_EVAL_SEEDS))
    parser.add_argument("--methods", nargs="+", default=list(PAPER_METHODS))
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-dirty", action="store_true", help="Smoke/debug only")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fast", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    root = Path(__file__).resolve().parent
    study = Path(args.study).resolve()
    set_study = Path(args.set_study).resolve()
    teacher = Path(args.teacher_checkpoint).resolve()
    frictions = validate_frictions(args.frictions)
    checkpoint_seeds = validate_unique(args.checkpoint_seeds, "checkpoint seeds")
    eval_seeds = validate_unique(args.eval_seeds, "evaluation seeds")
    methods = validate_unique(args.methods, "methods")
    if set(methods) - set(PAPER_METHODS):
        raise ValueError(f"Methods must be chosen from {PAPER_METHODS}")
    if set(checkpoint_seeds).intersection(eval_seeds):
        raise ValueError("checkpoint and evaluation seed namespaces must be disjoint")
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    if args.fast:
        frictions = (1.0,)
        checkpoint_seeds = checkpoint_seeds[:1]
        eval_seeds = eval_seeds[:1]
        args.num_envs = min(args.num_envs, 16)

    git = git_state(root)
    if git["dirty"] and not args.allow_dirty:
        detail = "\n".join(git["status"][:20])
        raise RuntimeError(
            "Production friction sweeps require a clean worktree. Commit or otherwise preserve "
            f"the current changes first, or use --allow-dirty only for smoke runs:\n{detail}"
        )
    sources = source_fingerprints(root)
    checkpoints = checkpoint_fingerprints(teacher, study, set_study, methods, checkpoint_seeds)
    config = {
        "format_version": FORMAT_VERSION,
        "experiment": "zero_shot_friction_sweep",
        "git": git,
        "source_fingerprints": sources,
        "study": str(study),
        "set_study": str(set_study),
        "teacher_checkpoint": str(teacher),
        "checkpoint_fingerprints": checkpoints,
        "frictions": frictions,
        "checkpoint_seeds": checkpoint_seeds,
        "eval_seeds": eval_seeds,
        "methods": methods,
        "num_envs": args.num_envs,
        "device": args.device,
        "commands": 15,
        "grid_settle_s": 1.0,
        "grid_measure_s": 4.0,
        "sensor_corruption": False,
    }
    signature = canonical_digest(config, 16)
    run_dir = Path(args.output_dir).resolve() / args.run_name
    print(f"friction sweep signature={signature} output={run_dir}")
    print(
        f"workers={len(frictions) * len(eval_seeds)} cells="
        f"{len(expected_keys(frictions, methods, checkpoint_seeds, eval_seeds))}"
    )
    if args.dry_run:
        for friction in frictions:
            for eval_seed in eval_seeds:
                print(f"  mu={friction:g} eval_seed={eval_seed}")
        return

    run_dir.mkdir(parents=True, exist_ok=args.resume)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("signature") != signature:
            raise RuntimeError("Existing run has a different signature; choose another --run-name")
    else:
        manifest = {
            "signature": signature,
            "status": "running",
            "created_at": datetime.now().isoformat(),
            "config": config,
        }
        _write_json(manifest_path, manifest)

    lock = acquire_run_lock(run_dir / ".active.lock")
    snapshot_repo = make_source_snapshot(root, run_dir / "source_snapshot")
    snapshot_sources = source_fingerprints(snapshot_repo)
    if snapshot_sources != sources:
        raise RuntimeError("Immutable source snapshot does not match the recorded source files")

    results_path = run_dir / "results.jsonl"
    completed_rows = _read_jsonl(results_path)
    completed = {cell_key(row) for row in completed_rows if row.get("run_signature") == signature}
    worker_env = os.environ.copy()
    snapshot_parent = str(snapshot_repo.parent)
    worker_env["PYTHONPATH"] = snapshot_parent + os.pathsep + worker_env.get("PYTHONPATH", "")
    worker_env["PYTHONDONTWRITEBYTECODE"] = "1"

    total_workers = len(frictions) * len(eval_seeds)
    worker_index = 0
    try:
        for friction in frictions:
            for eval_seed in eval_seeds:
                worker_index += 1
                batch_expected = expected_keys(
                    (friction,), methods, checkpoint_seeds, (eval_seed,)
                )
                if args.resume and batch_expected.issubset(completed):
                    print(
                        f"[{worker_index}/{total_workers}] mu={friction:g} "
                        f"eval_seed={eval_seed} SKIP",
                        flush=True,
                    )
                    continue
                stem = f"mu_{friction:g}_eval_{eval_seed}"
                temporary = run_dir / "cells" / f"{stem}.jsonl"
                log = run_dir / "process_logs" / f"{stem}.log"
                command = [
                    sys.executable,
                    str(snapshot_repo / "eval_friction_robustness.py"),
                    "--friction",
                    str(friction),
                    "--checkpoint-seeds",
                    *map(str, checkpoint_seeds),
                    "--eval-seed",
                    str(eval_seed),
                    "--methods",
                    *methods,
                    "--study",
                    str(study),
                    "--set-study",
                    str(set_study),
                    "--teacher-checkpoint",
                    str(teacher),
                    "--num-envs",
                    str(args.num_envs),
                    "--device",
                    args.device,
                    "--out",
                    str(temporary),
                ]
                if args.headless:
                    command.append("--headless")
                print(
                    f"[{worker_index}/{total_workers}] mu={friction:g} eval_seed={eval_seed}",
                    flush=True,
                )
                before = checkpoint_fingerprints(
                    teacher, study, set_study, methods, checkpoint_seeds
                )
                started = time.monotonic()
                returncode = _run_worker(command, log, worker_env, snapshot_repo.parent)
                after = checkpoint_fingerprints(
                    teacher, study, set_study, methods, checkpoint_seeds
                )
                if before != checkpoints or after != checkpoints:
                    raise RuntimeError("A checkpoint changed while the sweep was running")
                if returncode != 0:
                    raise RuntimeError(f"Worker failed ({returncode}); see {log}")
                rows = _read_jsonl(temporary)
                _validate_worker_rows(rows, friction, eval_seed, methods, checkpoint_seeds)
                new_rows = []
                for row in rows:
                    key = cell_key(row)
                    if key in completed:
                        continue
                    checkpoint_key = (
                        "teacher"
                        if row["method"] == "teacher"
                        else f"{row['method']}/seed_{row['checkpoint_seed']}"
                    )
                    row.update(
                        {
                            "run_signature": signature,
                            "source_signature": canonical_digest(sources),
                            "git_head": git["head"],
                            "checkpoint_sha256": checkpoints[checkpoint_key]["sha256"],
                            "worker_duration_s": time.monotonic() - started,
                        }
                    )
                    new_rows.append(row)
                    completed.add(key)
                _append_rows_atomic(results_path, new_rows)

        wanted = expected_keys(frictions, methods, checkpoint_seeds, eval_seeds)
        final_rows = _read_jsonl(results_path)
        final_keys = {cell_key(row) for row in final_rows if row.get("run_signature") == signature}
        if final_keys != wanted or len(final_keys) != len(wanted):
            raise RuntimeError(
                f"Incomplete sweep: missing={len(wanted - final_keys)} extra={len(final_keys - wanted)}"
            )
        report_command = [
            sys.executable,
            str(snapshot_repo / "report_friction_robustness.py"),
            "--results",
            str(results_path),
            "--out-dir",
            str(run_dir),
        ]
        report_log = run_dir / "process_logs" / "report.log"
        if _run_worker(report_command, report_log, worker_env, snapshot_repo.parent) != 0:
            raise RuntimeError(f"Report generation failed; see {report_log}")
        _mark_manifest_complete(
            manifest,
            completed_cells=len(final_keys),
            expected_cells=len(wanted),
            results_path=results_path,
            run_dir=run_dir,
        )
        _write_json(manifest_path, manifest)
        print(f"friction sweep complete: {run_dir / 'report.md'}")
    except BaseException as error:
        manifest.update(
            {
                "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "updated_at": datetime.now().isoformat(),
                "error": str(error),
            }
        )
        _write_json(manifest_path, manifest)
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
