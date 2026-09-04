"""Run the SET baseline across tasks and seeds, into its own study directory.

Deliberately separate from ``run_method_comparison.py``. SET is a baseline we
implemented from a paper with no released code, so its results should be
reproducible and re-runnable without touching the four-method comparison, and a
mistake here must not be able to invalidate anything JOSE has already logged.

The row schema and directory layout mirror ``run_method_comparison.py`` exactly,
so ``reporting.generate_report`` renders these rows the same way and a SET row
can be merged into the comparison table by concatenating the two ``results.jsonl``
files.

SET's own source files are fingerprinted separately (``SET_IMPLEMENTATION``), so
a change here is visible in the record without perturbing the JOSE fingerprints
that gate the ablation dataset caches.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

try:
    from .ablation_catalog import TASKS as CATALOG_TASK_REGISTRY, write_json
    from .ablation_common import (
        content_identity, file_fingerprint, implementation_fingerprint, run_live_subprocess,
    )
    from .reporting import generate_report
except ImportError:
    from ablation_catalog import TASKS as CATALOG_TASK_REGISTRY, write_json
    from ablation_common import (
        content_identity, file_fingerprint, implementation_fingerprint, run_live_subprocess,
    )
    from reporting import generate_report


TASKS = {name: entry[0] for name, entry in CATALOG_TASK_REGISTRY.items()}
TASK_ADAPTERS = {entry[0]: (entry[1], entry[2]) for entry in CATALOG_TASK_REGISTRY.values()}

#: Everything that defines a SET run's behaviour. Disjoint from the JOSE tuples
#: in ablation_catalog.py by construction: nothing here is hashed for JOSE, and
#: nothing hashed for JOSE is edited by this baseline.
SET_IMPLEMENTATION = (
    "train_set_baseline.py",
    "set_baseline/model.py",
    "set_baseline/adapter.py",
    "set_baseline/collect.py",
    "set_baseline/evaluate.py",
    "set_baseline/targets.py",
    "distillation/command_eval.py",
)

EXPERIMENT = "SET"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SET baseline sweep")
    parser.add_argument(
        "--case", nargs=2, action="append", required=True, metavar=("TASK", "TEACHER_CHECKPOINT"),
        help=f"Repeat per task; TASK is one of {sorted(TASKS)}",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--collect-steps", type=int, default=2000)
    parser.add_argument("--max-dataset-size", type=int, default=250_000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--context", type=int, default=20)
    parser.add_argument("--output-dir", default="logs/jose_g1/set_baseline")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _metrics(path: Path) -> dict:
    """Same contract run_method_comparison.py uses: the top-level `metrics` object."""
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", payload)
    return {key: value for key, value in metrics.items() if isinstance(value, (int, float, list, dict, bool))}


def _command(task_id: str, teacher: str, seed: int, output: Path, args) -> list[str]:
    adapter, agent = TASK_ADAPTERS[task_id]
    command = [
        sys.executable, str(Path(__file__).with_name("train_set_baseline.py")),
        "--teacher-checkpoint", teacher, "--task", task_id, "--agent", agent, "--adapter", adapter,
        "--seed", str(seed), "--num-envs", str(args.num_envs),
        "--collect-steps", str(args.collect_steps),
        "--max-dataset-size", str(args.max_dataset_size),
        "--epochs", str(args.epochs), "--context", str(args.context),
        "--output-dir", str(output.parent), "--run-name", output.name,
    ]
    if args.headless:
        command.append("--headless")
    return command


def main() -> int:
    args = _parser().parse_args()
    unknown = [name for name, _ in args.case if name not in TASKS]
    if unknown:
        raise SystemExit(f"Unknown task(s) {unknown}; choose from {sorted(TASKS)}")

    signature = implementation_fingerprint(SET_IMPLEMENTATION)
    run_name = args.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    study = Path(args.output_dir).resolve() / run_name
    raw = study / "results.jsonl"
    done: set[tuple[str, int]] = set()
    if args.resume and raw.is_file():
        for line in raw.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "ok" and row.get("signature") == signature:
                done.add((row["task_id"], row["seed"]))

    jobs = [
        (name, TASKS[name], teacher, seed)
        for name, teacher in args.case
        for seed in args.seeds
    ]
    print(f"SET baseline: {len(jobs)} job(s), signature {signature[:12]}", flush=True)

    study.mkdir(parents=True, exist_ok=True)
    rows_written: list[dict] = []
    for index, (task_name, task_id, teacher, seed) in enumerate(jobs, start=1):
        # The task key is part of the path: without it two tasks at the same seed
        # would land in one directory and silently overwrite each other.
        output = study / task_name / "methods" / "set" / f"context_{args.context}" / f"seed_{seed}"
        if (task_id, seed) in done:
            print(f"[{index}/{len(jobs)}] skip {task_id} seed {seed} (already complete)", flush=True)
            continue
        command = _command(task_id, teacher, seed, output, args)
        print(f"[{index}/{len(jobs)}] {task_id} seed {seed} -> {output}", flush=True)
        print("  $ " + " ".join(command), flush=True)
        if args.dry_run:
            continue

        output.mkdir(parents=True, exist_ok=True)
        log = output / "process.log"
        started = time.monotonic()
        started_at = datetime.now().isoformat()
        returncode = run_live_subprocess(command, log)
        artifact = output / "training.json"
        row = {
            "signature": signature,
            "task": task_id,
            "task_key": task_name,
            "task_id": task_id,
            "experiment": EXPERIMENT,
            "seed": seed,
            "teacher_fingerprint": content_identity(file_fingerprint(teacher)),
            "returncode": returncode,
            "status": "ok" if returncode == 0 else "failed",
            "duration_s": time.monotonic() - started,
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(),
            "artifact": str(artifact) if artifact.is_file() else None,
            "process_log": str(log),
            "command": command,
        }
        if returncode == 0:
            row["metrics"] = _metrics(artifact)
        else:
            row["error"] = log.read_text(encoding="utf-8", errors="replace")[-4000:]
        with raw.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        rows_written.append(row)

    if not args.dry_run and raw.is_file():
        write_json(study / "manifest.json", {"signature": signature, "created_at": datetime.now().isoformat()})
        generate_report(raw, study / "report")
        print(f"report written to {study / 'report'}", flush=True)

    # Exit non-zero when nothing usable came out. Returning 0 regardless let a
    # run where all nine jobs OOM'd be reported as a completed stage, and the
    # queue moved on to work that depended on the checkpoints it never wrote.
    failures = [job for job in rows_written if job["status"] != "ok"]
    if failures:
        print(
            f"{len(failures)}/{len(rows_written)} jobs failed: "
            + ", ".join(f"{job['task_key']} seed {job['seed']}" for job in failures),
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
