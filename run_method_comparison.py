"""Reproducible Teacher/IMU/Joint-only/JOSE comparison over 1--3 tasks."""

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
    "walk": "Isaac-G1-AMP-Walk-JOSE-Direct-v0",
    "dance": "Isaac-G1-AMP-Dance-JOSE-Direct-v0",
    "jump": "Isaac-G1-AMP-Jump-JOSE-Direct-v0",
}
METHODS = ("PrivilegedTeacher", "IMU-BasedDistillation", "Joint-OnlyDistillation", "JOSE")
FORMAT_VERSION = 2


def parse_cases(values: list[list[str]]) -> tuple[tuple[str, str], ...]:
    if not 1 <= len(values) <= 3:
        raise ValueError("--case must be supplied between one and three times")
    cases = []
    seen = set()
    for task, checkpoint in values:
        key = task.lower()
        task_id = TASKS.get(key, task)
        if task_id not in TASKS.values():
            raise ValueError(f"Unsupported task {task!r}; use walk, dance, jump, or a matching JOSE task id")
        if task_id in seen:
            raise ValueError(f"Duplicate comparison task: {task_id}")
        seen.add(task_id)
        cases.append((task_id, str(Path(checkpoint).resolve())))
    return tuple(cases)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="JOSE four-way method comparison")
    parser.add_argument(
        "--case", nargs=2, action="append", required=True, metavar=("TASK", "TEACHER_CHECKPOINT"),
        help="Repeat 1--3 times for any subset of walk/dance/jump",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--collect-steps", type=int, default=2000)
    parser.add_argument("--estimator-epochs", type=int, default=50)
    parser.add_argument("--estimator-dagger-rounds", type=int, default=10)
    parser.add_argument("--estimator-max-dataset-size", type=int, default=250000)
    parser.add_argument("--student-iterations", type=int, default=300)
    parser.add_argument("--student-rollout-steps", type=int, default=250)
    parser.add_argument("--student-train-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=600)
    parser.add_argument("--output-dir", default="logs/jose_g1/method_comparison")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fast", action="store_true")
    return parser


def _fingerprint(path: str, dry_run: bool) -> dict:
    target = Path(path)
    if not target.is_file():
        if dry_run:
            return {"path": path, "exists": False}
        raise FileNotFoundError(f"Teacher checkpoint not found: {path}")
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": path, "size": target.stat().st_size, "sha256": digest.hexdigest()}


def _lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f"Comparison output is locked by another process: {path}") from None
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} started={datetime.now().isoformat()}\n")
    handle.flush()
    return handle


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


def _run(command: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8", buffering=1) as stream:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert process.stdout is not None
        for line in process.stdout:
            print("  | " + line, end="", flush=True)
            stream.write(line)
        return process.wait()


def _metrics(path: Path) -> dict:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", payload)
    return {key: value for key, value in metrics.items() if isinstance(value, (int, float, list, dict, bool))}


def _command(method: str, task: str, teacher: str, seed: int, args, run_root: Path) -> tuple[list[str], Path]:
    root = Path(__file__).parent
    common_student = [
        "--teacher-checkpoint", teacher, "--task", task, "--seed", str(seed),
        "--num-envs", str(args.num_envs), "--num-iterations", str(args.student_iterations),
        "--rollout-steps", str(args.student_rollout_steps), "--train-steps", str(args.student_train_steps),
        "--eval-steps", str(args.eval_steps),
    ]
    name = f"{task}_{method}_seed{seed}"
    output = run_root / name
    if method == "PrivilegedTeacher":
        command = [
            sys.executable, str(root / "evaluate_teacher.py"), "--teacher-checkpoint", teacher,
            "--task", task, "--seed", str(seed), "--num-envs", str(args.num_envs),
            "--collect-steps", str(args.collect_steps), "--output-dir", str(run_root), "--run-name", name,
        ]
    elif method in ("IMU-BasedDistillation", "Joint-OnlyDistillation"):
        script = "train_imu_distillation.py" if method.startswith("IMU") else "train_joint_only_distillation.py"
        command = [sys.executable, str(root / script), *common_student, "--log-dir", str(output)]
    else:
        command = [
            sys.executable, str(root / "train_state_estimator.py"), "--teacher-checkpoint", teacher,
            "--task", task, "--seed", str(seed), "--num-envs", str(args.num_envs),
            "--collect-steps", str(args.collect_steps), "--epochs", str(args.estimator_epochs),
            "--dagger-rounds", str(args.estimator_dagger_rounds), "--estimator", "LSTM",
            "--max-dataset-size", str(args.estimator_max_dataset_size),
            "--window", "50", "--joint-preset", "all", "--output-dir", str(run_root), "--run-name", name,
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
        args.estimator_max_dataset_size = min(args.estimator_max_dataset_size, 20000)
        args.student_iterations = 2
        args.student_rollout_steps = 20
        args.student_train_steps = 5
        args.eval_steps = 50
    fingerprints = {task: _fingerprint(checkpoint, args.dry_run) for task, checkpoint in cases}
    signature_payload = {
        "version": FORMAT_VERSION, "cases": fingerprints, "seeds": args.seeds,
        "num_envs": args.num_envs, "collect_steps": args.collect_steps,
        "estimator_epochs": args.estimator_epochs, "estimator_dagger_rounds": args.estimator_dagger_rounds,
        "estimator_max_dataset_size": args.estimator_max_dataset_size,
        "student_iterations": args.student_iterations, "student_rollout_steps": args.student_rollout_steps,
        "student_train_steps": args.student_train_steps, "eval_steps": args.eval_steps,
    }
    signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode()).hexdigest()[:16]
    output = Path(args.output_dir).resolve() / signature
    output.mkdir(parents=True, exist_ok=True)
    lock = None if args.dry_run else _lock(output / ".active.lock")
    raw = output / "raw_results.jsonl"
    completed = _read_success(raw, signature) if args.resume else set()
    matrix = [(task, checkpoint, seed, method) for task, checkpoint in cases for seed in args.seeds for method in METHODS]
    print(f"JOSE 4-way comparison: {len(cases)} task(s), {len(args.seeds)} seed(s), {len(matrix)} jobs")
    try:
        for index, (task, teacher, seed, method) in enumerate(matrix, 1):
            key = (task, seed, method)
            if key in completed:
                print(f"[{index}/{len(matrix)}] {task}/{method}/seed{seed} SKIP")
                continue
            command, artifact = _command(method, task, teacher, seed, args, output / "runs")
            print(f"[{index}/{len(matrix)}] {task}/{method}/seed{seed}\n  $ {' '.join(command)}")
            if args.dry_run:
                continue
            started = time.monotonic()
            started_at = datetime.now().isoformat()
            log = output / "process_logs" / f"{task}_{method}_seed{seed}.log"
            returncode = _run(command, log)
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
            with raw.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
        if not args.dry_run:
            generate_report(raw, output)
            (output / "signature.json").write_text(json.dumps(signature_payload, indent=2), encoding="utf-8")
            print(f"Comparison complete: {output / 'report.md'}")
    finally:
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    main()
