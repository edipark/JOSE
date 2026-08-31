"""Teacher-scoped Teacher/IMU/Joint-only/JOSE comparison over 1--3 tasks."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

try:
    from .ablation_catalog import (
        TASKS as CATALOG_TASK_REGISTRY,
        TASK_IMPLEMENTATION,
        TRAINING_IMPLEMENTATION,
        TeacherCatalog,
        write_json,
    )
    from .ablation_common import (
        acquire_run_lock, content_identity, default_estimator_window, file_fingerprint,
        implementation_fingerprint, run_live_subprocess,
    )
    from .reporting import generate_report
except ImportError:
    from ablation_catalog import (
        TASKS as CATALOG_TASK_REGISTRY,
        TASK_IMPLEMENTATION,
        TRAINING_IMPLEMENTATION,
        TeacherCatalog,
        write_json,
    )
    from ablation_common import (
        acquire_run_lock, content_identity, default_estimator_window, file_fingerprint,
        implementation_fingerprint, run_live_subprocess,
    )
    from reporting import generate_report


TASKS = {
    "walk": CATALOG_TASK_REGISTRY["amp_walk"][0],
    "dance": CATALOG_TASK_REGISTRY["amp_dance"][0],
    "jump": CATALOG_TASK_REGISTRY["amp_jump"][0],
    "locomotion": CATALOG_TASK_REGISTRY["locomotion"][0],
}
# Gym task id -> (estimator adapter kind, agent config entry point), so the
# teacher/estimator/student commands below stay correct for a non-AMP task
# instead of the "amp" adapter being silently assumed everywhere.
TASK_ADAPTERS = {
    task_id: (adapter, agent)
    for task_id, adapter, agent in CATALOG_TASK_REGISTRY.values()
    if task_id in TASKS.values()
}
METHODS = ("PrivilegedTeacher", "IMU-BasedDistillation", "Joint-OnlyDistillation", "JOSE")
# Target number of student evaluation snapshots per run.
LEARNING_CURVE_POINTS = 15

FORMAT_VERSION = 4
# Gym task id -> the ablation_catalog.TASKS key it corresponds to (used to
# place this comparison's studies in the same per-task catalog layout).
CATALOG_TASKS = {
    task_id: key for key, (task_id, _, _) in CATALOG_TASK_REGISTRY.items() if task_id in TASKS.values()
}
METHOD_SLUGS = {
    "PrivilegedTeacher": "privileged_teacher",
    "IMU-BasedDistillation": "imu_based_distillation",
    "Joint-OnlyDistillation": "joint_only_distillation",
    "JOSE": "jose",
}


def parse_cases(values: list[list[str]]) -> tuple[tuple[str, str], ...]:
    if not 1 <= len(values) <= 3:
        raise ValueError("--case must be supplied between one and three times")
    cases = []
    seen = set()
    for task, checkpoint in values:
        key = task.lower()
        task_id = TASKS.get(key, task)
        if task_id not in TASKS.values():
            raise ValueError(
                f"Unsupported task {task!r}; use walk, dance, jump, locomotion, or a matching JOSE task id"
            )
        if task_id in seen:
            raise ValueError(f"Duplicate comparison task: {task_id}")
        seen.add(task_id)
        cases.append((task_id, str(Path(checkpoint).resolve())))
    return tuple(cases)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="JOSE four-way method comparison")
    parser.add_argument(
        "--case", nargs=2, action="append", required=True, metavar=("TASK", "TEACHER_CHECKPOINT"),
        help="Repeat 1--3 times for any subset of walk/dance/jump/locomotion",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--collect-steps", type=int, default=2000)
    parser.add_argument("--estimator-epochs", type=int, default=50)
    parser.add_argument("--estimator-dagger-rounds", type=int, default=10)
    parser.add_argument("--estimator-dagger-epochs", type=int, default=10)
    parser.add_argument("--estimator-max-dataset-size", type=int, default=250000)
    parser.add_argument("--student-iterations", type=int, default=300)
    parser.add_argument("--student-rollout-steps", type=int, default=250)
    parser.add_argument(
        "--student-train-steps", type=int, default=None,
        help="Gradient steps per student iteration (default: sized so total student gradient "
        "steps across --student-iterations matches the estimator's own training budget -- "
        "see _estimator_gradient_steps)",
    )
    parser.add_argument(
        "--eval-steps", type=int, default=None,
        help="Rollout steps per student evaluation call (default: the task's own max episode "
        "length, matching PrivilegedTeacher's evaluation window; --fast overrides to 50)",
    )
    parser.add_argument("--output-dir", default="logs/jose_g1/ablation", help="Teacher catalog root")
    parser.add_argument(
        "--run-name", default=None,
        help="Human-readable study id; supply the same value to resume a previous comparison",
    )
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fast", action="store_true")
    return parser


def _estimator_gradient_steps(args: argparse.Namespace) -> int:
    """Expected optimizer steps across the JOSE estimator's DAgger rounds.

    Mirrors estimator/pipeline.py:_fit_model's batching: each epoch iterates the
    dataset once, holding out min(10%, 10000) samples for validation, at a fixed
    batch_size=1024 (that script's own default; not exposed as a flag here).

    Deliberately excludes the initial --estimator-epochs supervised warm start
    on round 0: that phase fits a fixed dataset with no on-policy component and
    has no equivalent in the student's training (which is on-policy DAgger from
    iteration 1), so it isn't a comparable unit of "training" to size the
    student's budget against -- only the --estimator-dagger-rounds rounds of
    on-policy refinement are.
    """
    batch_size = 1024
    dataset_size = args.estimator_max_dataset_size
    validation_size = max(1, min(dataset_size // 10, 10000))
    batches_per_epoch = math.ceil((dataset_size - validation_size) / batch_size)
    return args.estimator_dagger_rounds * args.estimator_dagger_epochs * batches_per_epoch


def _fingerprint(path: str, dry_run: bool) -> dict:
    fingerprint = file_fingerprint(path)
    if not dry_run and fingerprint.get("exists") is False:
        raise FileNotFoundError(f"Teacher checkpoint not found: {path}")
    return fingerprint


def validate_run_name(value: str) -> str:
    """Slugify an explicit --run-name, rejecting one with no usable characters.

    Deliberately not ``ablation_catalog._slug`` (that one falls back to the
    literal ``"checkpoint"`` for a checkpoint-derived display id, which is the
    wrong behavior for a run name the user typed themselves).
    """
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    if not value:
        raise ValueError("--run-name must contain at least one letter or number")
    return value


def _read_success(path: Path, signature: str) -> set[tuple[str, int, str]]:
    completed = set()
    if not path.exists():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("signature") == signature and row.get("status") == "ok":
            completed.add((row["task_id"], row["seed"], row["experiment"]))
    return completed


def _metrics(path: Path) -> dict:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", payload)
    return {key: value for key, value in metrics.items() if isinstance(value, (int, float, list, dict, bool))}


def _method_output(method: str, task: str, seed: int, run_root: Path) -> Path:
    output = run_root / "methods" / METHOD_SLUGS[method]
    if method in ("IMU-BasedDistillation", "Joint-OnlyDistillation", "JOSE"):
        output = output / f"window_{default_estimator_window(task)}" / "joints_all"
    return output / f"seed_{seed}"


def _command(method: str, task: str, teacher: str, seed: int, args, run_root: Path) -> tuple[list[str], Path]:
    root = Path(__file__).parent
    # Every trained method sees the same history length, so window size isn't
    # a confound in the comparison.
    window = default_estimator_window(task)
    common_student = [
        "--teacher-checkpoint", teacher, "--task", task, "--seed", str(seed),
        "--num-envs", str(args.num_envs), "--num-iterations", str(args.student_iterations),
        "--rollout-steps", str(args.student_rollout_steps), "--train-steps", str(args.student_train_steps),
        "--window", str(window),
        # Keep the learning curve at a readable resolution however many
        # iterations the env-sample budget works out to, instead of letting the
        # fixed default thin it to a handful of points on short runs.
        "--eval-interval", str(max(1, args.student_iterations // LEARNING_CURVE_POINTS)),
    ]
    if args.eval_steps is not None:
        # Otherwise train_history_student.py falls back to its own per-task default.
        common_student.extend(["--eval-steps", str(args.eval_steps)])
    output = _method_output(method, task, seed, run_root)
    adapter, agent = TASK_ADAPTERS[task]
    if method == "PrivilegedTeacher":
        command = [
            sys.executable, str(root / "evaluate_teacher.py"), "--teacher-checkpoint", teacher,
            "--task", task, "--agent", agent, "--adapter", adapter,
            "--seed", str(seed), "--num-envs", str(args.num_envs),
            "--collect-steps", str(args.collect_steps), "--output-dir", str(output.parent),
            "--run-name", output.name,
        ]
    elif method in ("IMU-BasedDistillation", "Joint-OnlyDistillation"):
        script = "train_imu_distillation.py" if method.startswith("IMU") else "train_joint_only_distillation.py"
        command = [
            sys.executable, str(root / script), *common_student, "--adapter", adapter,
            # Same dataset cap as the estimator, and the same reservoir eviction
            # behind it (estimator/models.py:ReplayBuffer), so "how much data the
            # method gets to learn from" isn't a confound either.
            "--buffer-capacity", str(args.estimator_max_dataset_size),
            "--log-dir", str(output),
        ]
    else:
        command = [
            sys.executable, str(root / "train_state_estimator.py"), "--teacher-checkpoint", teacher,
            "--task", task, "--adapter", adapter, "--seed", str(seed), "--num-envs", str(args.num_envs),
            "--collect-steps", str(args.collect_steps), "--epochs", str(args.estimator_epochs),
            "--dagger-rounds", str(args.estimator_dagger_rounds),
            "--dagger-epochs", str(args.estimator_dagger_epochs), "--estimator", "LSTM",
            "--max-dataset-size", str(args.estimator_max_dataset_size),
            "--window", str(window), "--joint-preset", "all", "--output-dir", str(output.parent),
            "--run-name", output.name,
        ]
    if args.headless:
        command.append("--headless")
    return command, output / "training.json"


def main() -> None:
    args = _parser().parse_args()
    cases = parse_cases(args.case)
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must contain unique integers")
    if args.fast:
        args.num_envs = min(args.num_envs, 32)
        args.collect_steps = 50
        args.estimator_epochs = 2
        args.estimator_dagger_rounds = 1
        args.estimator_dagger_epochs = 2
        args.estimator_max_dataset_size = min(args.estimator_max_dataset_size, 20000)
        args.student_iterations = 2
        args.student_rollout_steps = 20
        args.student_train_steps = 5
        args.eval_steps = 50
    elif args.student_train_steps is None:
        # Size the student's per-iteration gradient steps so its total across
        # --student-iterations matches the estimator's own training budget --
        # see _estimator_gradient_steps for why this isn't a per-round multiply.
        args.student_train_steps = max(1, _estimator_gradient_steps(args) // args.student_iterations)
    fingerprints = {task: _fingerprint(checkpoint, args.dry_run) for task, checkpoint in cases}
    # Every task in the comparison contributes its own env/config/motion files.
    task_implementation = tuple(
        dict.fromkeys(path for task_id, _ in cases for path in TASK_IMPLEMENTATION[CATALOG_TASKS[task_id]])
    )
    signature_payload = {
        "version": FORMAT_VERSION,
        # Without this the cache key is hyperparameters only, so editing the
        # training code leaves every row marked complete and --resume silently
        # mixes results from two different implementations into one study.
        "implementation": implementation_fingerprint(TRAINING_IMPLEMENTATION + task_implementation),
        "cases": {task: content_identity(value) for task, value in fingerprints.items()},
        "seeds": args.seeds,
        "num_envs": args.num_envs, "collect_steps": args.collect_steps,
        "estimator_epochs": args.estimator_epochs, "estimator_dagger_rounds": args.estimator_dagger_rounds,
        "estimator_dagger_epochs": args.estimator_dagger_epochs,
        "estimator_max_dataset_size": args.estimator_max_dataset_size,
        "student_buffer_capacity": args.estimator_max_dataset_size,
        "student_iterations": args.student_iterations, "student_rollout_steps": args.student_rollout_steps,
        "student_train_steps": args.student_train_steps, "eval_steps": args.eval_steps,
    }
    signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode()).hexdigest()[:16]
    output_root = Path(args.output_dir).resolve()
    run_id = validate_run_name(args.run_name) if args.run_name else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    studies: dict[str, Path] = {}
    catalogs: dict[str, TeacherCatalog] = {}
    for task, checkpoint in cases:
        catalog = TeacherCatalog.open(
            output_root, checkpoint, fingerprints[task], create=not args.dry_run
        )
        catalogs[task] = catalog
        studies[task] = catalog.study_path(CATALOG_TASKS[task], "method_comparison", run_id)
    if not args.dry_run:
        for study in studies.values():
            study.mkdir(parents=True, exist_ok=args.resume)
            manifest_path = study / "manifest.json"
            if manifest_path.exists():
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                if existing.get("config_signature") != signature:
                    raise RuntimeError(
                        f"Study {study} already exists with a different configuration; "
                        "choose another --run-name"
                    )
            else:
                task = next(task for task, candidate in studies.items() if candidate == study)
                write_json(
                    manifest_path,
                    {
                        "catalog_format_version": 1,
                        "study": "method_comparison",
                        "run_id": run_id,
                        "status": "running",
                        "teacher_id": catalogs[task].teacher_root.name,
                        "teacher": fingerprints[task],
                        "task": CATALOG_TASKS[task],
                        "task_id": task,
                        "created_at": datetime.now().isoformat(),
                        "config_signature": signature,
                        "config": signature_payload,
                    },
                )
    locks = [] if args.dry_run else [acquire_run_lock(study / ".active.lock") for study in studies.values()]
    raws = {task: study / "results.jsonl" for task, study in studies.items()}
    completed = set()
    if args.resume:
        for raw in raws.values():
            completed.update(_read_success(raw, signature))
    matrix = [(task, checkpoint, seed, method) for task, checkpoint in cases for seed in args.seeds for method in METHODS]
    print(f"JOSE 4-way comparison: {len(cases)} task(s), {len(args.seeds)} seed(s), {len(matrix)} jobs")
    for task, study in studies.items():
        print(f"teacher_catalog={catalogs[task].teacher_root}")
        print(f"study={CATALOG_TASKS[task]}/method_comparison/{run_id} output={study}")
    try:
        for index, (task, teacher, seed, method) in enumerate(matrix, 1):
            key = (task, seed, method)
            if key in completed:
                print(f"[{index}/{len(matrix)}] {task}/{method}/seed{seed} SKIP")
                continue
            study = studies[task]
            command, artifact = _command(method, task, teacher, seed, args, study)
            print(
                f"[{index}/{len(matrix)}] {task}/{method}/seed{seed} output={artifact.parent}"
                f"\n  $ {' '.join(command)}"
            )
            if args.dry_run:
                continue
            started = time.monotonic()
            started_at = datetime.now().isoformat()
            log = _method_output(method, task, seed, study) / "process.log"
            returncode = run_live_subprocess(command, log)
            row = {
                "signature": signature, "task": task, "task_id": task, "experiment": method,
                "seed": seed, "teacher_fingerprint": fingerprints[task], "returncode": returncode,
                "status": "ok" if returncode == 0 else "failed", "duration_s": time.monotonic() - started,
                "started_at": started_at, "finished_at": datetime.now().isoformat(),
                "artifact": str(artifact) if artifact.exists() else None, "process_log": str(log),
                "command": command,
            }
            if returncode == 0:
                row["metrics"] = _metrics(artifact)
            else:
                row["error"] = log.read_text(encoding="utf-8", errors="replace")[-4000:]
            with raws[task].open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
        if not args.dry_run:
            for task, study in studies.items():
                report_output = study / "report"
                generate_report(raws[task], report_output, require_plots=True)
                manifest_path = study / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest.update({
                    "status": "complete",
                    "finished_at": datetime.now().isoformat(),
                    "results": str(raws[task]),
                    "report": str(report_output / "report.md"),
                })
                write_json(manifest_path, manifest)
                print(f"Comparison complete: {report_output / 'report.md'}")
    finally:
        for lock in locks:
            lock.close()


if __name__ == "__main__":
    main()
