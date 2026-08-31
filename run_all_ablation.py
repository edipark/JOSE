"""Run every ablation study against one (or a few) teacher checkpoints.

`run_architecture_ablation.py`, `run_window_ablation.py`,
`run_joint_scope_ablation.py`, `run_dagger_ablation.py`, and
`run_method_comparison.py` each answer a different question about the same
teacher checkpoint. Getting the full picture meant invoking all five by hand,
restating `--task`/`--seeds`/`--headless` identically on each one. This script
takes those settings once and applies them uniformly across all five studies
(or a chosen subset), against the checkpoint(s) given.

Each underlying run still writes into the normal content-addressed catalog
under `--output-dir`, so a sweep is safe to re-run: anything already complete
is reused exactly as the individual scripts already do on their own.

Report merging happens *within* a study, across checkpoints -- never across
studies. Two studies can legitimately share an experiment slug (architecture
and joint_scope both include `lstm_w25_all`, reused rather than retrained), so
merging their rows together would double-count that arm in the aggregate
mean/std. Keeping each study's combined report separate avoids that.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

try:
    from .ablation_catalog import TASKS as CATALOG_TASKS, TeacherCatalog, teacher_display_id, write_json
    from .ablation_common import file_fingerprint, run_live_subprocess
    from .reporting import generate_report
except ImportError:
    from ablation_catalog import TASKS as CATALOG_TASKS, TeacherCatalog, teacher_display_id, write_json
    from ablation_common import file_fingerprint, run_live_subprocess
    from reporting import generate_report


# Study name -> (entry-point script, on-disk study name), in the order they
# run by default. method_comparison last: it is the slowest and its four jobs
# per seed dwarf a single ablation-runner arm.
ABLATION_SCRIPTS = {
    "architecture": ("run_architecture_ablation.py", "architecture"),
    "window": ("run_window_ablation.py", "window"),
    "joint_scope": ("run_joint_scope_ablation.py", "joint_scope"),
    "dagger": ("run_dagger_ablation.py", "dagger"),
    "method_comparison": ("run_method_comparison.py", "method_comparison"),
}

# run_method_comparison.py uses its own short task vocabulary (walk/dance/
# jump/locomotion) rather than ablation_catalog.TASKS's keys (amp_walk/
# amp_dance/amp_jump/locomotion) -- translated here so --task takes one value
# for every study.
METHOD_COMPARISON_TASK_KEYS = {
    "amp_walk": "walk", "amp_dance": "dance", "amp_jump": "jump", "locomotion": "locomotion",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run some or all ablation studies against one or more teacher checkpoints, "
        "with one shared task/seeds/headless setting applied to every study.",
    )
    parser.add_argument(
        "--checkpoints", nargs="+", required=True, metavar="PATH",
        help="Teacher checkpoint file(s). Each study is run once per checkpoint given.",
    )
    parser.add_argument(
        "--task", choices=tuple(CATALOG_TASKS), required=True,
        help="Applied to every study; translated to run_method_comparison.py's own vocabulary for it.",
    )
    parser.add_argument(
        "--studies", nargs="+", choices=tuple(ABLATION_SCRIPTS), default=tuple(ABLATION_SCRIPTS),
        help="Which studies to run (default: all five)",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Applied to every study. run_method_comparison.py accepts any set of seeds; the "
        "ablation-runner studies (architecture/window/joint_scope/dagger) only accept a "
        "contiguous range, so non-contiguous values are rejected for those with a clear error. "
        "Omit to leave each study at its own default (3 seeds).",
    )
    parser.add_argument("--output-dir", default="logs/jose_g1/ablation", help="Teacher catalog root, forwarded to each run")
    parser.add_argument(
        "--sweep-output-dir", default=None,
        help="Where the per-study combined results/report go "
        "(default: <output-dir>/sweeps/<timestamp>)",
    )
    parser.add_argument(
        "--continue-on-error", action="store_true",
        help="Keep going after one study/checkpoint fails instead of aborting the whole sweep",
    )
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true", help="Print every command without running it")
    return parser


def resolve_checkpoints(paths: list[str]) -> list[Path]:
    resolved = sorted({Path(item).resolve() for item in paths})
    if not resolved:
        raise ValueError("--checkpoints resolved to nothing")
    return resolved


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


def _seed_args(study: str, seeds: list[int] | None) -> list[str]:
    if seeds is None:
        return []
    if study == "method_comparison":
        return ["--seeds", *[str(seed) for seed in seeds]]
    # ablation_runner.py only takes a contiguous range: --seed-start + a count.
    start, count = min(seeds), len(seeds)
    if sorted(seeds) != list(range(start, start + count)):
        raise ValueError(
            f"--seeds {seeds} is not a contiguous range, which {study} requires "
            f"(it takes --seed-start/--seeds as a starting seed and a count)"
        )
    return ["--seed-start", str(start), "--seeds", str(count)]


def build_command(study: str, args: argparse.Namespace, checkpoint: Path) -> list[str]:
    script_name, _ = ABLATION_SCRIPTS[study]
    script = Path(__file__).with_name(script_name)
    if study == "method_comparison":
        command = [
            sys.executable, str(script),
            "--case", METHOD_COMPARISON_TASK_KEYS[args.task], str(checkpoint),
            "--output-dir", args.output_dir,
        ]
    else:
        command = [
            sys.executable, str(script),
            "--teacher-checkpoint", str(checkpoint),
            "--task", args.task,
            "--output-dir", args.output_dir,
        ]
    command.extend(_seed_args(study, args.seeds))
    command.append("--headless" if args.headless else "--no-headless")
    return command


def _run_study(study: str, args: argparse.Namespace, checkpoints: list[Path], sweep_output: Path, manifest: dict) -> None:
    _, study_name = ABLATION_SCRIPTS[study]
    output_root = Path(args.output_dir).resolve()
    study_entry = {"study": study, "checkpoints": []}
    manifest["studies"].append(study_entry)
    combined_rows: list[dict] = []

    for index, checkpoint in enumerate(checkpoints, 1):
        command = build_command(study, args, checkpoint)
        print(f"[{study} {index}/{len(checkpoints)}] {checkpoint.name}\n  $ {' '.join(command)}")
        if args.dry_run:
            continue

        log_path = sweep_output / "logs" / study / f"{checkpoint.stem}.log"
        returncode = run_live_subprocess(command, log_path)
        if returncode != 0:
            study_entry["checkpoints"].append(
                {"checkpoint": str(checkpoint), "status": "failed", "returncode": returncode, "log": str(log_path)}
            )
            write_json(sweep_output / "sweep_manifest.json", manifest)
            message = f"{study} failed for {checkpoint} (returncode={returncode}); see {log_path}"
            if args.continue_on_error:
                print(f"  {message}; continuing (--continue-on-error)")
                continue
            raise RuntimeError(message)

        teacher_id = resolve_teacher_id(output_root, checkpoint)
        results_path = latest_results_jsonl(output_root, teacher_id, study_name)
        if results_path is None:
            study_entry["checkpoints"].append(
                {"checkpoint": str(checkpoint), "status": "no_results", "teacher_id": teacher_id}
            )
            write_json(sweep_output / "sweep_manifest.json", manifest)
            message = f"No results.jsonl found for {study}/{checkpoint} under {output_root / teacher_id}"
            if args.continue_on_error:
                print(f"  {message}; continuing (--continue-on-error)")
                continue
            raise RuntimeError(message)

        rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in rows:
            row["teacher_id"] = teacher_id
            row["checkpoint_path"] = str(checkpoint)
        combined_rows.extend(rows)
        study_entry["checkpoints"].append({
            "checkpoint": str(checkpoint), "status": "ok", "teacher_id": teacher_id,
            "results": str(results_path), "rows": len(rows),
        })

    if args.dry_run:
        return

    write_json(sweep_output / "sweep_manifest.json", manifest)
    study_output = sweep_output / study
    study_output.mkdir(parents=True, exist_ok=True)
    combined_path = study_output / "combined_results.jsonl"
    combined_path.write_text("".join(json.dumps(row) + "\n" for row in combined_rows), encoding="utf-8")
    study_entry["combined_results"] = str(combined_path)

    if not combined_rows:
        print(f"[{study}] no results collected; skipping report generation.")
        return
    report_output = study_output / "report"
    generate_report(combined_path, report_output, require_plots=False)
    study_entry["report"] = str(report_output / "report.md")
    print(f"[{study}] complete: {report_output / 'report.md'} ({len(combined_rows)} rows)")


def main() -> None:
    args = _parser().parse_args()
    checkpoints = resolve_checkpoints(args.checkpoints)
    sweep_output = Path(
        args.sweep_output_dir
        or Path(args.output_dir).resolve() / "sweeps" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ).resolve()

    print(
        f"JOSE checkpoint sweep: {len(checkpoints)} checkpoint(s), "
        f"{len(args.studies)} study(ies) = {', '.join(args.studies)}, task={args.task}"
    )
    for checkpoint in checkpoints:
        print(f"  {checkpoint}")

    manifest = {
        "task": args.task, "studies_requested": list(args.studies), "seeds": args.seeds,
        "output_dir": str(Path(args.output_dir).resolve()), "created_at": datetime.now().isoformat(),
        "studies": [],
    }
    if not args.dry_run:
        sweep_output.mkdir(parents=True, exist_ok=True)

    for study in args.studies:
        _run_study(study, args, checkpoints, sweep_output, manifest)


if __name__ == "__main__":
    main()
