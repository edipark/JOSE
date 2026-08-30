"""Run one ablation/comparison script across every checkpoint in a training run.

`ablation_runner.py` (via `run_architecture_ablation.py`, `run_window_ablation.py`,
`run_joint_scope_ablation.py`) and `run_method_comparison.py` each operate on a
single teacher checkpoint per invocation, keyed into the on-disk catalog by
that checkpoint's content hash. Nothing in the repository repeated one of
those commands across a training run's periodic checkpoints (``agent_10000.pt``
.. ``agent_100000.pt``, ``best_agent.pt``) -- this script does that, then
merges the resulting per-checkpoint ``results.jsonl`` files into one combined
report so checkpoints can be compared side by side.

Each underlying run still writes into the normal content-addressed catalog
under ``--output-dir``, so a sweep is safe to re-run: checkpoints already
complete are reused exactly as ``ablation_runner.py``/`run_method_comparison.py`
already do on their own.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

try:
    from .ablation_catalog import TeacherCatalog, teacher_display_id, write_json
    from .ablation_common import file_fingerprint, run_live_subprocess
    from .reporting import generate_report
except ImportError:
    from ablation_catalog import TeacherCatalog, teacher_display_id, write_json
    from ablation_common import file_fingerprint, run_live_subprocess
    from reporting import generate_report


# Sweep target -> (module filename, on-disk study name).
ABLATION_SCRIPTS = {
    "architecture": ("run_architecture_ablation.py", "architecture"),
    "window": ("run_window_ablation.py", "window"),
    "joint_scope": ("run_joint_scope_ablation.py", "joint_scope"),
    "dagger": ("run_dagger_ablation.py", "dagger"),
    "method_comparison": ("run_method_comparison.py", "method_comparison"),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep an ablation/comparison script across multiple teacher checkpoints",
    )
    parser.add_argument("--ablation-script", choices=tuple(ABLATION_SCRIPTS), required=True)
    parser.add_argument("--checkpoint-dir", default=None, help="Directory to glob checkpoints from")
    parser.add_argument("--glob", default="agent_*.pt", help="Glob pattern within --checkpoint-dir")
    parser.add_argument(
        "--checkpoints", nargs="+", default=None,
        help="Explicit checkpoint paths; combined with --checkpoint-dir/--glob if both are given",
    )
    parser.add_argument(
        "--include-best", action="store_true",
        help="Also include best_agent.pt from --checkpoint-dir",
    )
    parser.add_argument(
        "--case-task", choices=("walk", "dance", "jump"), default=None,
        help="Required for --ablation-script method_comparison: the one task every checkpoint is compared on",
    )
    parser.add_argument("--output-dir", default="logs/jose_g1/ablation", help="Teacher catalog root, forwarded to each run")
    parser.add_argument(
        "--sweep-output-dir", default=None,
        help="Where the combined results/report go (default: <output-dir>/sweeps/<script>_<timestamp>)",
    )
    parser.add_argument(
        "--continue-on-error", action="store_true",
        help="Keep sweeping the remaining checkpoints after one fails instead of aborting",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print each command without running it")
    return parser


def resolve_checkpoints(args: argparse.Namespace) -> list[Path]:
    found: set[Path] = set()
    if args.checkpoint_dir:
        directory = Path(args.checkpoint_dir).resolve()
        found.update(path.resolve() for path in directory.glob(args.glob))
        if args.include_best:
            best = directory / "best_agent.pt"
            if best.is_file():
                found.add(best.resolve())
    if args.checkpoints:
        found.update(Path(item).resolve() for item in args.checkpoints)
    if not found:
        raise ValueError(
            "No checkpoints resolved; pass --checkpoint-dir (with --glob/--include-best) and/or --checkpoints"
        )
    return sorted(found)


def resolve_teacher_id(output_root: Path, checkpoint: Path) -> str:
    """Return the on-disk teacher_id the underlying run just registered for `checkpoint`."""
    fingerprint = file_fingerprint(checkpoint)
    if not fingerprint.get("sha256"):
        return teacher_display_id(checkpoint)
    catalog = TeacherCatalog.open(output_root, checkpoint, fingerprint, create=False)
    return catalog.teacher_root.name


def latest_results_jsonl(output_root: Path, teacher_id: str, study_name: str) -> Path | None:
    candidates = list((output_root / teacher_id).glob(f"*/studies/{study_name}/*/results.jsonl"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def build_command(args: argparse.Namespace, checkpoint: Path, passthrough: list[str]) -> list[str]:
    script_name, _ = ABLATION_SCRIPTS[args.ablation_script]
    script = Path(__file__).with_name(script_name)
    if args.ablation_script == "method_comparison":
        if not args.case_task:
            raise ValueError("--ablation-script method_comparison requires --case-task {walk,dance,jump}")
        command = [
            sys.executable, str(script),
            "--case", args.case_task, str(checkpoint),
            "--output-dir", args.output_dir,
        ]
    else:
        command = [
            sys.executable, str(script),
            "--teacher-checkpoint", str(checkpoint),
            "--output-dir", args.output_dir,
        ]
    command.extend(passthrough)
    return command


def main() -> None:
    args, passthrough = _parser().parse_known_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    checkpoints = resolve_checkpoints(args)
    _, study_name = ABLATION_SCRIPTS[args.ablation_script]
    output_root = Path(args.output_dir).resolve()
    sweep_output = Path(
        args.sweep_output_dir
        or output_root / "sweeps" / f"{args.ablation_script}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    ).resolve()

    print(f"JOSE checkpoint sweep: {len(checkpoints)} checkpoint(s), script={args.ablation_script}")
    for checkpoint in checkpoints:
        print(f"  {checkpoint}")

    manifest = {
        "ablation_script": args.ablation_script, "study": study_name,
        "output_dir": str(output_root), "created_at": datetime.now().isoformat(),
        "checkpoints": [],
    }
    combined_rows: list[dict] = []
    if not args.dry_run:
        sweep_output.mkdir(parents=True, exist_ok=True)

    for index, checkpoint in enumerate(checkpoints, 1):
        command = build_command(args, checkpoint, passthrough)
        print(f"[{index}/{len(checkpoints)}] {checkpoint.name}\n  $ {' '.join(command)}")
        if args.dry_run:
            continue

        log_path = sweep_output / "logs" / f"{checkpoint.stem}.log"
        returncode = run_live_subprocess(command, log_path)
        if returncode != 0:
            manifest["checkpoints"].append(
                {"checkpoint": str(checkpoint), "status": "failed", "returncode": returncode, "log": str(log_path)}
            )
            write_json(sweep_output / "sweep_manifest.json", manifest)
            if args.continue_on_error:
                print(f"  FAILED (returncode={returncode}); continuing (--continue-on-error)")
                continue
            raise RuntimeError(f"Checkpoint run failed: {checkpoint} (returncode={returncode}); see {log_path}")

        teacher_id = resolve_teacher_id(output_root, checkpoint)
        results_path = latest_results_jsonl(output_root, teacher_id, study_name)
        if results_path is None:
            manifest["checkpoints"].append(
                {"checkpoint": str(checkpoint), "status": "no_results", "teacher_id": teacher_id}
            )
            write_json(sweep_output / "sweep_manifest.json", manifest)
            message = f"No results.jsonl found for {checkpoint} under {output_root / teacher_id}"
            if args.continue_on_error:
                print(f"  {message}; continuing (--continue-on-error)")
                continue
            raise RuntimeError(message)

        rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in rows:
            row["teacher_id"] = teacher_id
            row["checkpoint_path"] = str(checkpoint)
        combined_rows.extend(rows)
        manifest["checkpoints"].append({
            "checkpoint": str(checkpoint), "status": "ok", "teacher_id": teacher_id,
            "results": str(results_path), "rows": len(rows),
        })

    if args.dry_run:
        return

    write_json(sweep_output / "sweep_manifest.json", manifest)
    combined_path = sweep_output / "combined_results.jsonl"
    combined_path.write_text("".join(json.dumps(row) + "\n" for row in combined_rows), encoding="utf-8")
    print(f"combined_results={combined_path} rows={len(combined_rows)}")

    if not combined_rows:
        print("No results were collected; skipping report generation.")
        return

    report_output = sweep_output / "report"
    generate_report(combined_path, report_output, require_plots=False)
    print(
        f"Sweep complete: {report_output / 'report.md'} "
        f"({len(combined_rows)} rows from {len(checkpoints)} checkpoint(s))"
    )


if __name__ == "__main__":
    main()
