"""Ablation aggregation, tables, plots, and human-readable report generation."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable


PRIMARY_METRICS = (
    "rmse", "mae", "r2", "return_mean", "episode_length_mean", "episode_length_std",
    "death_rate", "timeout_rate",
    "teacher_action_mse", "action_smoothness", "energy", "torque_rms", "inference_ms_per_sample", "parameters",
    "success_rate", "base_linear_speed", "base_angular_speed", "action_saturation", "torque_saturation",
    "raw_task_reward", "amp_raw_style", "amp_scaled_task", "amp_scaled_style", "amp_effective_reward",
    "wall_time_s", "collection_duration_s", "collection_samples_per_s",
    "mpjpe_g", "mpjpe_l", "root_position_error",
)

# Metrics plotted against metrics.learning_curve's "step" (cumulative gradient
# steps) when present -- see _plots()'s learning_curves figure.
LEARNING_CURVE_METRICS = ("return_mean", "episode_length_mean", "rmse", "death_rate", "teacher_action_mse")

REQUIRED_REPORT_FILES = ("summary.json", "table.md", "report.md")
# PNG only, and only the two figures worth keeping: everything else the report
# used to emit was either a bar chart of a number already in table.md or a
# per-seed diagnostic, and 26 files per study buried the two that get read.
# Per-run raw values still live in results.jsonl, outside report/.
REQUIRED_PLOT_FILES = ("dagger_learning_curve.png", "learning_curves.png")


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _numeric(values: Iterable) -> list[float]:
    return [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(value)]


def aggregate(rows: list[dict]) -> list[dict]:
    # teacher_id distinguishes rows collected from different teacher checkpoints
    # (e.g. a run_all_ablation.py comparison across checkpoints); it is
    # absent from normal single-checkpoint studies, where every row shares the
    # same default and grouping is unaffected.
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row.get("task", "unknown"), row.get("experiment", "unknown"), row.get("teacher_id", "single"))].append(row)
    output = []
    for (task, experiment, teacher_id), group in sorted(groups.items()):
        summary = {
            "task": task, "experiment": experiment, "teacher_id": teacher_id,
            "seeds": len(group), "successful": sum(row.get("status") == "ok" for row in group),
        }
        for metric in PRIMARY_METRICS:
            values = _numeric(row.get("metrics", {}).get(metric) for row in group)
            if values:
                summary[f"{metric}_mean"] = mean(values)
                summary[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
                summary[f"{metric}_ci95"] = 1.96 * summary[f"{metric}_std"] / math.sqrt(len(values))
        output.append(summary)
    teacher_lengths = {
        (row["task"], row["teacher_id"]): row["episode_length_mean_mean"]
        for row in output
        if row["experiment"].lower() in ("teachergt", "teacher_gt")
        and row.get("episode_length_mean_mean", 0.0) > 0.0
    }
    for row in output:
        baseline = teacher_lengths.get((row["task"], row["teacher_id"]))
        if baseline is not None and "episode_length_mean_mean" in row:
            row["episode_length_ratio_percent"] = 100.0 * row["episode_length_mean_mean"] / baseline
    return output


def _markdown_table(rows: list[dict]) -> str:
    columns = (
        "Task", "Experiment", "OK/Seeds", "Episode steps", "Within-run sigma",
        "Death %", "Timeout %", "Teacher ratio %", "Return", "RMSE", "R2", "Latency ms",
        "MPJPE-G mm", "MPJPE-L mm",
    )
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"

    def mean_std(row: dict, metric: str, digits: int = 1) -> str:
        value = row.get(f"{metric}_mean")
        if value is None:
            return ""
        deviation = row.get(f"{metric}_std", 0.0)
        return f"{value:.{digits}f} ± {deviation:.{digits}f}"

    body = []
    for row in rows:
        cells = (
            row["task"], row["experiment"], f"{row['successful']}/{row['seeds']}",
            mean_std(row, "episode_length_mean"), mean_std(row, "episode_length_std"),
            mean_std(row, "death_rate"), mean_std(row, "timeout_rate"),
            f"{row['episode_length_ratio_percent']:.1f}" if "episode_length_ratio_percent" in row else "",
            mean_std(row, "return_mean", 2), mean_std(row, "rmse", 4),
            mean_std(row, "r2", 4), mean_std(row, "inference_ms_per_sample", 4),
            mean_std(row, "mpjpe_g", 1), mean_std(row, "mpjpe_l", 1),
        )
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join((header, divider, *body))


def _group_label(row: dict) -> str:
    teacher_id = row.get("teacher_id")
    return f"{row.get('experiment')}/{teacher_id}" if teacher_id else str(row.get("experiment"))


def _group_colors(plt, groups) -> dict[str, str]:
    """Pin one colour per group up front, before any axis is drawn.

    A metric absent for one group means that group never calls plot() on that
    axis, so matplotlib's per-axis colour cycle ends up at a different position
    on each subplot and one group silently borrows another's colour on the
    panels where the other has no data.
    """
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    return {label: cycle[index % len(cycle)] for index, label in enumerate(sorted(groups))}


def _band(axis, by_x: dict, color: str, label: str) -> None:
    """Mean line with a shaded +/-1 std band across seeds."""
    xs = sorted(by_x)
    means = [mean(by_x[x]) for x in xs]
    spreads = [stdev(by_x[x]) if len(by_x[x]) > 1 else 0.0 for x in xs]
    axis.plot(xs, means, linewidth=2, marker="o", markersize=3, color=color, label=label)
    axis.fill_between(
        xs, [m - s for m, s in zip(means, spreads)], [m + s for m, s in zip(means, spreads)],
        color=color, alpha=0.18, linewidth=0,
    )


def _plots(output: Path, rows: list[dict], raw_rows: list[dict]) -> tuple[list[str], dict[str, str]]:
    """Render every plot for which qualifying data exists.

    Returns the artifact filenames produced and, for each plot the data ruled
    out, a human-readable reason -- callers report the reason instead of
    treating the missing plot as a failure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    artifacts = []
    skipped: dict[str, str] = {}
    dagger_rows = [
        row for row in raw_rows
        if row.get("status") == "ok" and row.get("metrics", {}).get("rounds")
    ]
    # What each panel pulls out of one entry of metrics.rounds. The middle one
    # is the point of the figure: validation loss falling does not mean the
    # policy walks better, and only the closed-loop panel shows whether the
    # DAgger rounds actually bought anything.
    dagger_panels = (
        ("best_validation_mse", lambda item: item.get("training", {}).get("best_validation_mse")),
        ("episode_length_mean", lambda item: item.get("evaluation", {}).get("episode_length_mean")),
        ("rmse", lambda item: item.get("evaluation", {}).get("rmse")),
    )
    if dagger_rows:
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in dagger_rows:
            groups[_group_label(row)].append(row)
        colors = _group_colors(plt, groups)
        figure, axes = plt.subplots(
            len(dagger_panels), 1, figsize=(9, 3.2 * len(dagger_panels)), sharex=True, squeeze=False,
        )
        axes = axes[:, 0]
        for axis, (name, extract) in zip(axes, dagger_panels):
            for label, group in sorted(groups.items()):
                by_round: dict[float, list[float]] = defaultdict(list)
                for row in group:
                    for item in row["metrics"]["rounds"]:
                        value, index = extract(item), item.get("round")
                        if isinstance(value, (int, float)) and isinstance(index, (int, float)):
                            by_round[index].append(value)
                if by_round:
                    _band(axis, by_round, colors[label], f"{label} (n={len(group)})")
            axis.set_ylabel(name)
            axis.grid(alpha=0.25)
        axes[-1].set_xlabel("DAgger round")
        axes[0].set_title("DAgger rounds (mean \u00b1 std across seeds)")
        axes[0].legend(fontsize=7, ncol=2)
        figure.tight_layout()
        path = output / "dagger_learning_curve.png"
        figure.savefig(path)
        artifacts.append(path.name)
        plt.close(figure)
    else:
        skipped["dagger_learning_curve"] = "no successful raw rows had metrics.rounds"
    curve_rows = [row for row in raw_rows if row.get("status") == "ok" and row.get("metrics", {}).get("learning_curve")]
    curve_metrics = [
        metric for metric in LEARNING_CURVE_METRICS
        if any(
            isinstance(point.get(metric), (int, float))
            for row in curve_rows for point in row["metrics"]["learning_curve"]
        )
    ] if curve_rows else []
    if curve_metrics:
        baseline_rows = [
            row for row in raw_rows
            if row.get("status") == "ok" and not row.get("metrics", {}).get("learning_curve")
        ]

        curve_groups: dict[str, list[dict]] = defaultdict(list)
        for row in curve_rows:
            curve_groups[_group_label(row)].append(row)
        baseline_groups: dict[str, list[dict]] = defaultdict(list)
        for row in baseline_rows:
            baseline_groups[_group_label(row)].append(row)
        group_colors = _group_colors(plt, curve_groups)

        figure, curve_axes = plt.subplots(
            len(curve_metrics), 1, figsize=(9, 3.2 * len(curve_metrics)), sharex=True, squeeze=False,
        )
        curve_axes = curve_axes[:, 0]
        for axis, metric in zip(curve_axes, curve_metrics):
            for label, rows in sorted(curve_groups.items()):
                # Seeds share the same eval_interval, so their snapshots land on
                # the same step values -- group by step to get a mean/std band
                # instead of one noisy line per seed.
                by_step: dict[float, list[float]] = defaultdict(list)
                for row in rows:
                    for point in row["metrics"]["learning_curve"]:
                        value, step = point.get(metric), point.get("step")
                        if isinstance(value, (int, float)) and isinstance(step, (int, float)):
                            by_step[step].append(value)
                if by_step:
                    _band(axis, by_step, group_colors[label], f"{label} (n={len(rows)})")
            for label, rows in sorted(baseline_groups.items()):
                values = [row["metrics"][metric] for row in rows if isinstance(row.get("metrics", {}).get(metric), (int, float))]
                if not values:
                    continue
                center, spread = mean(values), stdev(values) if len(values) > 1 else 0.0
                axis.axhline(center, linestyle="--", linewidth=1.5, alpha=0.7, color="0.3", label=f"{label} (fixed, n={len(values)})")
                if spread > 0.0:
                    axis.axhspan(center - spread, center + spread, color="0.3", alpha=0.1, linewidth=0)
            axis.set_ylabel(metric)
            axis.grid(alpha=0.25)
        curve_axes[-1].set_xlabel("Cumulative gradient steps")
        curve_axes[0].set_title("Training curves (mean ± std across seeds, shared gradient-step axis)")
        curve_axes[0].legend(fontsize=7, ncol=2)
        figure.tight_layout()
        path = output / "learning_curves.png"
        figure.savefig(path)
        artifacts.append(path.name)
        plt.close(figure)
    else:
        skipped["learning_curves"] = (
            "no raw rows had metrics.learning_curve" if not curve_rows
            else "learning_curve entries had none of the tracked metrics"
        )
    return artifacts, skipped


def generate_report(
    raw_jsonl: str | Path, output_dir: str | Path, *, require_plots: bool = False
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw_rows = read_jsonl(raw_jsonl)
    summary = aggregate(raw_rows)
    # summary.json carries every aggregated metric in PRIMARY_METRICS; the raw
    # per-run values stay in results.jsonl beside this directory. The CSV and
    # LaTeX mirrors of both were dropped -- nothing read them.
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    table = _markdown_table(summary)
    (output / "table.md").write_text(table + "\n", encoding="utf-8")
    try:
        plots, skipped_plots = _plots(output, summary, raw_rows)
        plot_error = None
        unavailable = output / "PLOTS_UNAVAILABLE.txt"
        if unavailable.exists():
            unavailable.unlink()
    except ImportError as exc:
        plots = []
        skipped_plots = {}
        plot_error = str(exc)
        (output / "PLOTS_UNAVAILABLE.txt").write_text(f"Install matplotlib to generate plots: {exc}\n", encoding="utf-8")
    failures = [row for row in raw_rows if row.get("status") != "ok"]
    report = [
        "# JOSE G1 Ablation Report", "", f"Runs: {len(raw_rows)}; failures: {len(failures)}", "",
        "The SOLO-aligned Walk teacher combines velocity tracking (task scale 0.5) with AMP style reward (scale 2.0).", "",
        "## Results", "", table, "", "## Artifacts", "",
        f"Plots: {', '.join(plots) if plots else 'none'}",
    ]
    if plot_error:
        report.extend(("", f"Plot generation warning: {plot_error}"))
    if skipped_plots:
        report.extend(("", "## Skipped plots", ""))
        report.extend(f"- {stem}: {reason}" for stem, reason in sorted(skipped_plots.items()))
    if failures:
        report.extend(("", "## Failed runs", "", *[f"- {row.get('task')}/{row.get('experiment')}/seed{row.get('seed')}: {row.get('error')}" for row in failures]))
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    # Tables/CSVs/summary are always producible from any non-empty raw_jsonl,
    # so a missing one indicates a real bug and is a hard failure. Plots are
    # conditional on the data actually containing what each plot needs (e.g. no
    # DAgger rounds -> no dagger_learning_curve), so a missing plot is reported
    # in report.md via `skipped_plots` above rather than raised -- only a truly
    # unavailable matplotlib install (require_plots=True) is a hard failure.
    missing = [name for name in REQUIRED_REPORT_FILES if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"Report generation omitted required artifacts: {missing}")
    if require_plots and plot_error:
        raise RuntimeError(f"Plots were required but matplotlib is unavailable: {plot_error}")
    return {
        "runs": len(raw_rows), "failures": len(failures), "plots": plots,
        "skipped_plots": skipped_plots,
        "required_files": list(REQUIRED_REPORT_FILES),
        "output": str(output),
    }
