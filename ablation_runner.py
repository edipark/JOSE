"""Teacher-scoped JOSE estimator-ablation catalog, execution, and reports.

Each invocation creates a human-readable dated study. Jobs and datasets live
in a catalog below the teacher checkpoint identity, so equivalent jobs can be
reused across window, architecture, and joint-scope studies.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
    from .ablation_catalog import (
        CATALOG_FORMAT_VERSION,
        AblationExperiment,
        TeacherCatalog,
        canonical_digest,
        normalize_experiments,
        write_json,
    )
    from .reporting import aggregate, generate_report
except ImportError:
    from ablation_catalog import (
        CATALOG_FORMAT_VERSION,
        AblationExperiment,
        TeacherCatalog,
        canonical_digest,
        normalize_experiments,
        write_json,
    )
    from reporting import aggregate, generate_report


TASKS = {
    "amp_walk": ("Isaac-G1-AMP-Walk-JOSE-Direct-v0", "amp", "skrl_amp_cfg_entry_point"),
    "amp_dance": ("Isaac-G1-AMP-Dance-JOSE-Direct-v0", "amp", "skrl_amp_cfg_entry_point"),
    "amp_jump": ("Isaac-G1-AMP-Jump-JOSE-Direct-v0", "amp", "skrl_amp_cfg_entry_point"),
    "ppo_walk": ("Isaac-G1-PPO-Walk-JOSE-Direct-v0", "ppo", "skrl_cfg_entry_point"),
}

TRAINING_IMPLEMENTATION = (
    "train_state_estimator.py",
    "estimator/pipeline.py",
    "estimator/adapters.py",
    "estimator/models.py",
    "schema.py",
    "skrl_compat.py",
    "g1_amp_env.py",
    "g1_amp_env_cfg.py",
    "task_math.py",
)
TEACHER_IMPLEMENTATION = (
    "evaluate_teacher.py",
    "estimator/adapters.py",
    "schema.py",
    "skrl_compat.py",
    "g1_amp_env.py",
    "g1_amp_env_cfg.py",
    "task_math.py",
)
TASK_IMPLEMENTATION = {
    "amp_walk": ("__init__.py", "g1_cfg.py", "motions/motion_loader.py", "motions/G1_walk.npz", "agents/skrl_g1_walk_amp_cfg.yaml"),
    "amp_dance": ("__init__.py", "g1_cfg.py", "motions/motion_loader.py", "motions/G1_dance.npz", "agents/skrl_g1_dance_amp_cfg.yaml"),
    "amp_jump": ("__init__.py", "g1_cfg.py", "motions/motion_loader.py", "motions/G1_jump.npz", "agents/skrl_g1_jump_amp_cfg.yaml"),
    "ppo_walk": ("__init__.py", "g1_cfg.py", "g1_ppo_env.py", "motions/motion_loader.py", "motions/G1_walk.npz", "agents/skrl_g1_walk_ppo_cfg.yaml"),
}


def _file_fingerprint(path: str | Path) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        return {"path": str(path), "exists": False}
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "size": path.stat().st_size, "sha256": digest.hexdigest()}


def _content_identity(fingerprint: dict) -> dict:
    if fingerprint.get("sha256"):
        return {"size": fingerprint.get("size"), "sha256": fingerprint["sha256"]}
    return {"exists": False, "path": fingerprint.get("path")}


def _implementation_fingerprint(paths: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for relative in paths:
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
        raise RuntimeError(f"Another ablation is already using this teacher/task catalog ({owner})") from None
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
    if not isinstance(source, dict):
        raise ValueError(f"Expected a JSON metrics object in {training_json}")
    # Keep nested DAgger rounds, per-target errors, and representative traces.
    # Reporting consumes these fields later to produce learning curves and
    # target/trace plots; retaining only scalar metrics silently discarded them.
    metrics = dict(source)
    rounds = metrics.get("rounds", data.get("rounds", []))
    if isinstance(rounds, list):
        metrics["rounds"] = rounds
        metrics["round_count"] = len(rounds)
    return metrics


def _extract_run_config(training_json: Path) -> dict:
    if not training_json.exists():
        return {}
    data = json.loads(training_json.read_text(encoding="utf-8"))
    config = data.get("config", {}) if isinstance(data, dict) else {}
    if isinstance(config, dict) and config:
        return config
    config_path = training_json.parent / "config.json"
    if config_path.is_file():
        fallback = json.loads(config_path.read_text(encoding="utf-8"))
        return fallback if isinstance(fallback, dict) else {}
    return {}


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


def _parser(experiments: tuple[AblationExperiment, ...], study_name: str) -> argparse.ArgumentParser:
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
    parser.add_argument("--max-dataset-size", type=int, default=250000)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument(
        "--experiments", nargs="+", choices=tuple(item.slug for item in experiments), default=None,
        help="Run only selected canonical experiments (teacher_gt is always included)",
    )
    parser.add_argument("--skip-student", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", default="logs/jose_g1/ablation", help="Teacher catalog root")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    headless = parser.add_mutually_exclusive_group()
    headless.add_argument("--headless", dest="headless", action="store_true")
    headless.add_argument("--no-headless", dest="headless", action="store_false")
    parser.set_defaults(headless=True)
    parser.add_argument("--rerun", action="store_true", help="Run jobs even when the catalog is complete")
    parser.add_argument(
        "--import-legacy", action=argparse.BooleanOptionalAction, default=True,
        help="Register compatible successful results from the former study/session layout",
    )
    return parser


def _job_spec(
    experiment: AblationExperiment | None,
    *,
    teacher: dict,
    task: str,
    task_id: str,
    adapter: str,
    agent: str,
    seed: int,
    args,
    implementation: str,
) -> dict:
    common = {
        "catalog_format_version": CATALOG_FORMAT_VERSION,
        "teacher": _content_identity(teacher),
        "task": task,
        "task_id": task_id,
        "adapter": adapter,
        "agent": agent,
        "seed": seed,
        "num_envs": args.num_envs,
        "collect_steps": args.collect_steps,
        "implementation": implementation,
    }
    if experiment is None:
        return {"kind": "teacher_evaluation", **common}
    return {
        "kind": "estimator",
        **common,
        "experiment": asdict(experiment),
        "epochs": args.epochs,
        "dagger_epochs": args.dagger_epochs,
        "max_dataset_size": args.max_dataset_size,
        "dagger_max_samples_per_round": args.max_dataset_size,
        "dataset_aggregation": "uniform_random_subsample_after_each_round",
        "eval_episodes": args.eval_episodes,
        "dataset_cache_window": max(50, experiment.window),
    }


def _legacy_config_matches(config: dict, spec: dict) -> bool:
    experiment = spec.get("experiment")
    if not experiment:
        return True
    expected = {
        "task": spec["task_id"],
        "adapter": spec["adapter"],
        "estimator": experiment["estimator"],
        "window": experiment["window"],
        "joint_preset": experiment["joint_preset"],
        "collect_steps": spec["collect_steps"],
        "epochs": spec["epochs"],
        "dagger_epochs": spec["dagger_epochs"],
        "dagger_rounds": experiment["dagger_rounds"],
        "max_dataset_size": spec["max_dataset_size"],
        "eval_episodes": spec["eval_episodes"],
        "num_envs": spec["num_envs"],
        "seed": spec["seed"],
        "dataset_cache_window": spec["dataset_cache_window"],
    }
    return all(config.get(key) == value for key, value in expected.items())


def _import_legacy_result(
    output_root: Path,
    entry: Path,
    spec: dict,
    experiment_name: str,
    teacher_fingerprint: dict,
) -> dict | None:
    """Register, without moving it, an exact successful legacy artifact."""
    for raw_path in sorted(output_root.glob("*/sessions/*/raw_results.jsonl"), reverse=True):
        try:
            lines = raw_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in reversed(lines):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status") != "ok" or record.get("seed") != spec["seed"]:
                continue
            if record.get("task_id") != spec["task_id"]:
                continue
            old_sha = record.get("teacher_fingerprint", {}).get("sha256")
            if old_sha != teacher_fingerprint.get("sha256"):
                continue
            artifact = Path(record.get("artifact") or "")
            if not artifact.is_file():
                continue
            if spec["kind"] == "teacher_evaluation":
                if record.get("experiment") != "TeacherGT":
                    continue
                expected_samples = spec["collect_steps"] * spec["num_envs"]
                if record.get("metrics", {}).get("samples") != expected_samples:
                    continue
            else:
                config_path = artifact.parent / "config.json"
                checkpoint_path = artifact.parent / "best_estimator.pt"
                if not config_path.is_file() or not checkpoint_path.is_file():
                    continue
                try:
                    config = json.loads(config_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if not _legacy_config_matches(config, spec):
                    continue
            imported = {
                **record,
                "experiment": experiment_name,
                "catalog_action": "imported_legacy",
                "catalog_spec": spec,
                "catalog_spec_digest": canonical_digest(spec),
                "source_record": str(raw_path),
            }
            legacy_id = "legacy_" + canonical_digest({"artifact": str(artifact)}, 12)
            TeacherCatalog.write_attempt(entry, legacy_id, imported, make_current=True)
            return imported
    return None


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _result_eplen(record: dict, *, seed: int, experiment: str) -> float:
    value = record.get("metrics", {}).get("episode_length_mean")
    if not isinstance(value, (int, float)):
        raise ValueError(
            f"Complete catalog result is missing numeric episode_length_mean: "
            f"{experiment}/seed{seed}"
        )
    return float(value)


def _write_intermediate_results(
    path: Path,
    *,
    study_manifest: dict,
    study_config: dict,
    jobs: list[dict],
    actions: dict[tuple[int, str], str],
    failed: dict[tuple[int, str], dict],
    status: str = "running",
) -> tuple[list[dict], list[dict], list[dict]]:
    """Atomically publish all currently available study results as one JSON object."""
    rows: list[dict] = []
    job_states: list[dict] = []
    missing: list[dict] = []
    for job in jobs:
        key = (job["seed"], job["name"])
        base = {
            "seed": key[0],
            "experiment": key[1],
            "spec_digest": job["digest"],
            "action": actions.get(key, "PENDING"),
        }
        if key in failed:
            rows.append(failed[key])
            job_states.append({**base, "status": "failed"})
            missing.append({"seed": key[0], "experiment": key[1], "reason": "attempt_failed"})
            continue
        record = TeacherCatalog.read_complete(
            job["entry"], require_checkpoint=job["experiment"] is not None
        )
        if record is None:
            job_states.append({**base, "status": "missing"})
            missing.append({"seed": key[0], "experiment": key[1], "reason": "not_complete"})
            continue
        study_record = dict(record)
        study_record["catalog_action"] = actions.get(key, "REUSE_RESULT")
        study_record["source_artifact"] = record.get("artifact")
        rows.append(study_record)
        eplen = _result_eplen(record, seed=key[0], experiment=key[1])
        job_states.append(
            {
                **base,
                "status": "complete",
                "source_artifact": record.get("artifact"),
                "eplen": eplen,
            }
        )

    complete_count = sum(item["status"] == "complete" for item in job_states)
    failed_count = sum(item["status"] == "failed" for item in job_states)
    payload = {
        "catalog_format_version": CATALOG_FORMAT_VERSION,
        "kind": "ablation_intermediate_results",
        "study": study_manifest["study"],
        "run_id": study_manifest["run_id"],
        "status": status,
        "updated_at": datetime.now().isoformat(),
        "teacher_id": study_manifest["teacher_id"],
        "teacher": study_manifest["teacher"],
        "task": study_manifest["task"],
        "task_id": study_manifest["task_id"],
        "config": study_config,
        "progress": {
            "total": len(jobs),
            "complete": complete_count,
            "failed": failed_count,
            "missing": len(jobs) - complete_count - failed_count,
        },
        "jobs": job_states,
        "results": rows,
        "summary": aggregate(rows),
    }
    write_json(path, payload)
    # Keep the compact job view in the manifest identical to the intermediate
    # result while the study is running as well as after finalization.
    study_manifest["jobs"] = job_states
    write_json(path.parent / "manifest.json", study_manifest)
    print(
        "intermediate_result=" + json.dumps(
            {"path": str(path), "status": status, **payload["progress"]},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return rows, job_states, missing


def main(
    experiments: tuple[AblationExperiment | tuple[str, str, int, str, int], ...],
    study_name: str,
):
    experiments = normalize_experiments(experiments)
    if not experiments:
        raise ValueError("At least one ablation experiment is required")
    args = _parser(experiments, study_name).parse_args()
    if args.seeds <= 0:
        raise ValueError("--seeds must be positive")
    teacher_path = str(Path(args.teacher_checkpoint).resolve())
    if not args.dry_run and not Path(teacher_path).is_file():
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_path}")

    default_task_id, default_adapter, default_agent = TASKS[args.task]
    task_id = args.task_id or default_task_id
    adapter = args.adapter or default_adapter
    agent = args.agent or default_agent
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    if args.fast:
        args.collect_steps, args.epochs, args.dagger_epochs = 100, 2, 2
        args.max_dataset_size = min(args.max_dataset_size, 20000)
        args.eval_episodes = min(args.eval_episodes, 20)

    teacher_fingerprint = _file_fingerprint(teacher_path)
    output_root = Path(args.output_dir).resolve()
    catalog = TeacherCatalog.open(
        output_root, teacher_path, teacher_fingerprint, create=not args.dry_run
    )
    run_id = catalog.next_run_id(args.task, study_name)
    study_output = catalog.study_path(args.task, study_name, run_id)
    run_lock = None if args.dry_run else _acquire_run_lock(catalog.task_root(args.task) / ".active.lock")
    if not args.dry_run:
        study_output.mkdir(parents=True, exist_ok=False)

    selected = (
        tuple(item for item in experiments if item.slug in args.experiments)
        if args.experiments is not None else experiments
    )
    matrix: list[tuple[int, str, AblationExperiment | None]] = []
    for seed in seeds:
        matrix.append((seed, "teacher_gt", None))
        matrix.extend((seed, experiment.slug, experiment) for experiment in selected)

    task_implementation = TASK_IMPLEMENTATION[args.task]
    training_implementation = _implementation_fingerprint(TRAINING_IMPLEMENTATION + task_implementation)
    teacher_implementation = _implementation_fingerprint(TEACHER_IMPLEMENTATION + task_implementation)
    jobs = []
    for seed, name, experiment in matrix:
        spec = _job_spec(
            experiment,
            teacher=teacher_fingerprint,
            task=args.task,
            task_id=task_id,
            adapter=adapter,
            agent=agent,
            seed=seed,
            args=args,
            implementation=teacher_implementation if experiment is None else training_implementation,
        )
        digest = canonical_digest(spec)
        jobs.append(
            {
                "seed": seed,
                "name": name,
                "experiment": experiment,
                "spec": spec,
                "digest": digest,
                "entry": catalog.entry_path(
                    args.task,
                    seed,
                    name,
                    digest,
                    estimator=None if experiment is None else experiment.estimator,
                    window=None if experiment is None else experiment.window,
                    joint_preset=None if experiment is None else experiment.joint_preset,
                ),
            }
        )

    study_config = {
        "seeds": seeds,
        "experiments": [item.slug for item in selected],
        "num_envs": args.num_envs,
        "collect_steps": args.collect_steps,
        "initial_epochs": args.epochs,
        "dagger_epochs": args.dagger_epochs,
        "max_dataset_size": args.max_dataset_size,
        "eval_episodes": args.eval_episodes,
        "headless": args.headless,
        "aggregation": "uniform_random_subsample_after_each_round",
    }
    study_manifest = {
        "catalog_format_version": CATALOG_FORMAT_VERSION,
        "run_id": run_id,
        "study": study_name,
        "status": "planned" if args.dry_run else "running",
        "teacher_id": catalog.teacher_root.name,
        "teacher": teacher_fingerprint,
        "task": args.task,
        "task_id": task_id,
        "created_at": datetime.now().isoformat(),
        "config": study_config,
        "jobs": [
            {"seed": job["seed"], "experiment": job["name"], "spec_digest": job["digest"]}
            for job in jobs
        ],
    }
    if not args.dry_run:
        for job in jobs:
            spec_path = job["entry"] / "spec.json"
            if spec_path.exists():
                existing_spec = json.loads(spec_path.read_text(encoding="utf-8"))
                if existing_spec != job["spec"]:
                    raise RuntimeError(f"Catalog digest collision or corrupt spec: {spec_path}")
            else:
                write_json(spec_path, job["spec"])
        write_json(study_output / "manifest.json", study_manifest)

    print("\n" + "=" * 76)
    print(f"JOSE G1 {study_name} Ablation Catalog")
    print("=" * 76)
    print(f"teacher_catalog={catalog.teacher_root}")
    print(f"task={args.task} ({task_id})")
    print(f"study={run_id} output={study_output}")
    print(f"seeds={seeds} jobs={len(jobs)}")
    print("execution: fill missing catalog entries, then finalize the study report")
    print("=" * 76, flush=True)

    actions: dict[tuple[int, str], str] = {}
    failed: dict[tuple[int, str], dict] = {}
    intermediate_path = study_output / "intermediate_results.json"
    if not args.dry_run:
        _write_intermediate_results(
            intermediate_path,
            study_manifest=study_manifest,
            study_config=study_config,
            jobs=jobs,
            actions=actions,
            failed=failed,
        )
    start_all = time.monotonic()
    try:
        for index, job in enumerate(jobs, 1):
            seed = job["seed"]
            name = job["name"]
            experiment = job["experiment"]
            entry = job["entry"]
            current = TeacherCatalog.read_complete(entry, require_checkpoint=experiment is not None)
            if current is None and args.import_legacy and not args.dry_run:
                current = _import_legacy_result(
                    output_root, entry, job["spec"], name, teacher_fingerprint
                )
                if current is not None:
                    actions[(seed, name)] = "IMPORT_LEGACY"
            if current is not None and not args.rerun:
                action = actions.get((seed, name), "REUSE_RESULT")
                actions[(seed, name)] = action
                result_eplen = _result_eplen(current, seed=seed, experiment=name)
                print(
                    f"[{index:03d}/{len(jobs):03d}] {args.task}/{name}/seed{seed} "
                    f"{action} eplen={result_eplen:.2f} source={current.get('artifact')}", flush=True,
                )
                _write_intermediate_results(
                    intermediate_path,
                    study_manifest=study_manifest,
                    study_config=study_config,
                    jobs=jobs,
                    actions=actions,
                    failed=failed,
                )
                continue

            actions[(seed, name)] = "RUN"
            attempt_id = f"{study_name}__{run_id}"
            attempt = entry / "attempts" / attempt_id
            artifact_dir = attempt / "artifact"
            artifact = artifact_dir / "training.json"
            if experiment is None:
                script = Path(__file__).with_name("evaluate_teacher.py")
                command = [
                    sys.executable, str(script), "--teacher-checkpoint", teacher_path,
                    "--task", task_id, "--agent", agent, "--adapter", adapter,
                    "--seed", str(seed), "--num-envs", str(args.num_envs),
                    "--collect-steps", str(args.collect_steps), "--output-dir", str(attempt),
                    "--run-name", "artifact",
                ]
            else:
                script = Path(__file__).with_name("train_state_estimator.py")
                cache_window = job["spec"]["dataset_cache_window"]
                cache_spec = {
                    "catalog_format_version": CATALOG_FORMAT_VERSION,
                    "teacher": _content_identity(teacher_fingerprint),
                    "implementation": training_implementation,
                    "task_id": task_id,
                    "adapter": adapter,
                    "agent": agent,
                    "seed": seed,
                    "num_envs": args.num_envs,
                    "window": cache_window,
                    "joint_preset": "all",
                    "collect_steps": args.collect_steps,
                    "max_dataset_size": args.max_dataset_size,
                    "noise_levels": [0.0, 0.01, 0.02],
                }
                cache_digest = canonical_digest(cache_spec, 12)
                cache = catalog.dataset_path(args.task, seed, cache_window, "all", cache_digest)
                if not args.dry_run:
                    write_json(cache.with_suffix(".json"), cache_spec)
                command = [
                    sys.executable, str(script), "--teacher_checkpoint", teacher_path,
                    "--task", task_id, "--agent_cfg_entry_point", agent, "--adapter", adapter,
                    "--est_type", experiment.estimator, "--window", str(experiment.window),
                    "--joint_preset", experiment.joint_preset,
                    "--dagger_rounds", str(experiment.dagger_rounds),
                    "--dagger_epochs", str(args.dagger_epochs),
                    "--collect_steps", str(args.collect_steps), "--epochs", str(args.epochs),
                    "--max_dataset_size", str(args.max_dataset_size),
                    "--eval_episodes", str(args.eval_episodes),
                    "--num_envs", str(args.num_envs), "--seed", str(seed),
                    "--dataset-cache", str(cache), "--run-name", "artifact",
                    "--dataset-cache-window", str(cache_window), "--output-dir", str(attempt),
                ]
            if args.headless:
                command.append("--headless")

            elapsed = time.monotonic() - start_all
            eta = elapsed / max(index - 1, 1) * (len(jobs) - index + 1) if index > 1 else 0.0
            print(
                f"[{index:03d}/{len(jobs):03d}] {args.task}/{name}/seed{seed} RUN "
                f"ETA={eta / 60:.1f}m", flush=True,
            )
            print("  $ " + " ".join(command), flush=True)
            if args.dry_run:
                continue

            started = time.monotonic()
            started_at = datetime.now().isoformat()
            process_log = attempt / "process.log"
            returncode = _run_live(command, process_log)
            record = {
                "task": args.task,
                "task_id": task_id,
                "teacher_checkpoint": teacher_path,
                "teacher_fingerprint": teacher_fingerprint,
                "experiment": name,
                "seed": seed,
                "started_at": started_at,
                "finished_at": datetime.now().isoformat(),
                "duration_s": time.monotonic() - started,
                "returncode": returncode,
                "status": "ok" if returncode == 0 and artifact.is_file() else "failed",
                "process_log": str(process_log),
                "command": command,
                "artifact": str(artifact) if artifact.is_file() else None,
                "catalog_action": "run",
                "catalog_spec": job["spec"],
                "catalog_spec_digest": job["digest"],
                "producer_study": str(study_output),
            }
            if record["status"] == "failed":
                record["error"] = (
                    process_log.read_text(encoding="utf-8", errors="replace")[-4000:]
                    if process_log.exists() else "Process produced no complete artifact"
                )
                failed[(seed, name)] = record
                print(f"  FAILED ({returncode}); catalog entry remains incomplete", flush=True)
            else:
                record["metrics"] = _extract_metrics(artifact)
                record["run_config"] = _extract_run_config(artifact)
                if name == "teacher_gt" and record["metrics"].get("episode_length_mean", 0.0) < 100.0:
                    print("  WARNING: teacher mean episode length < 100", flush=True)
                if name == "teacher_gt":
                    print(
                        f"  teacher_eval eplen={record['metrics'].get('episode_length_mean', float('nan')):.2f}",
                        flush=True,
                    )
                print(
                    "  " + " ".join(
                        f"{key}={value:.4g}" for key, value in record["metrics"].items()
                        if isinstance(value, (int, float))
                    ), flush=True,
                )
            TeacherCatalog.write_attempt(
                entry, attempt_id, record, make_current=record["status"] == "ok"
            )
            _write_intermediate_results(
                intermediate_path,
                study_manifest=study_manifest,
                study_config=study_config,
                jobs=jobs,
                actions=actions,
                failed=failed,
            )

        if args.dry_run:
            return

        final_rows, final_job_states, missing = _write_intermediate_results(
            intermediate_path,
            study_manifest=study_manifest,
            study_config=study_config,
            jobs=jobs,
            actions=actions,
            failed=failed,
            status="finalizing",
        )

        results_path = study_output / "results.jsonl"
        _write_jsonl(results_path, final_rows)
        study_manifest["jobs"] = final_job_states
        if missing:
            study_manifest.update(
                {
                    "status": "incomplete",
                    "catalog_complete": False,
                    "missing": missing,
                    "finished_at": datetime.now().isoformat(),
                }
            )
            write_json(study_output / "manifest.json", study_manifest)
            _write_intermediate_results(
                intermediate_path,
                study_manifest=study_manifest,
                study_config=study_config,
                jobs=jobs,
                actions=actions,
                failed=failed,
                status="incomplete",
            )
            print(
                f"Catalog incomplete ({len(missing)} missing/failed); final report was not generated. "
                f"study={study_output}", flush=True,
            )
            raise RuntimeError("Ablation catalog is incomplete; rerun to fill the missing entries")

        report_output = study_output / "report"
        result = generate_report(results_path, report_output, require_plots=True)
        study_manifest.update(
            {
                "status": "complete",
                "catalog_complete": True,
                "finished_at": datetime.now().isoformat(),
                "results": str(results_path),
                "report": str(report_output / "report.md"),
            }
        )
        write_json(study_output / "manifest.json", study_manifest)
        _write_intermediate_results(
            intermediate_path,
            study_manifest=study_manifest,
            study_config=study_config,
            jobs=jobs,
            actions=actions,
            failed=failed,
            status="complete",
        )
        print(
            f"Catalog complete; finalized report={result['output']}/report.md "
            f"elapsed={(time.monotonic() - start_all) / 60:.1f}m", flush=True,
        )
    finally:
        if run_lock is not None:
            run_lock.close()
