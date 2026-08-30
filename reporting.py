"""Ablation aggregation, tables, plots, and human-readable report generation."""

from __future__ import annotations

import csv
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
)

# Metrics plotted against metrics.learning_curve's "step" (cumulative gradient
# steps) when present -- see _plots()'s learning_curves figure.
LEARNING_CURVE_METRICS = ("return_mean", "episode_length_mean", "rmse", "death_rate", "teacher_action_mse")

REQUIRED_REPORT_FILES = (
    "summary.json", "summary.csv", "results_tidy.csv", "table.md", "table.tex", "report.md",
)
REQUIRED_PLOT_FILES = tuple(
    f"{stem}.{suffix}"
    for stem in (
        "episode_length_mean", "death_rate", "timeout_rate", "return_mean", "rmse", "pareto",
        "target_rmse_heatmap", "dagger_learning_curve", "representative_trace", "learning_curves",
    )
    for suffix in ("png", "pdf")
)


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
    # (e.g. a run_checkpoint_sweep.py comparison across checkpoints); it is
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


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(rows: list[dict]) -> str:
    columns = (
        "Task", "Experiment", "OK/Seeds", "Episode steps", "Within-run sigma",
        "Death %", "Timeout %", "Teacher ratio %", "Return", "RMSE", "R2", "Latency ms",
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
        )
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join((header, divider, *body))


def _latex_table(rows: list[dict]) -> str:
    lines = [
        r"\begin{tabular}{llrrrrrrr}", r"\toprule",
        r"Task & Experiment & Episode & Death (\%) & Timeout (\%) & Ratio (\%) & Return & RMSE & $R^2$ \\",
        r"\midrule",
    ]
    for row in rows:
        experiment = row["experiment"].replace("_", r"\_")
        lines.append(
            f"{row['task']} & {experiment} & {row.get('episode_length_mean_mean', float('nan')):.1f} $\\pm$ "
            f"{row.get('episode_length_mean_std', float('nan')):.1f} & {row.get('death_rate_mean', float('nan')):.1f} & "
            f"{row.get('timeout_rate_mean', float('nan')):.1f} & {row.get('episode_length_ratio_percent', float('nan')):.1f} & "
            f"{row.get('return_mean_mean', float('nan')):.2f} & {row.get('rmse_mean', float('nan')):.4f} & "
            f"{row.get('r2_mean', float('nan')):.4f} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines)


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
    labels = [f"{row['task'].replace('Isaac-G1-', '')}\n{row['experiment']}" for row in rows]
    for metric, title in (
        ("episode_length_mean", "Mean episode length"),
        ("death_rate", "Death rate"), ("timeout_rate", "Timeout rate"),
        ("return_mean", "Closed-loop return"), ("rmse", "Estimator RMSE"),
    ):
        selected = [(index, row) for index, row in enumerate(rows) if f"{metric}_mean" in row]
        if not selected:
            skipped[metric] = f"no aggregated rows had a value for {metric}"
            continue
        x = range(len(selected))
        values = [row[f"{metric}_mean"] for _, row in selected]
        errors = [row.get(f"{metric}_ci95", 0.0) for _, row in selected]
        figure, axis = plt.subplots(figsize=(max(8, len(selected) * 0.8), 5))
        axis.errorbar(x, values, yerr=errors, fmt="o", capsize=4)
        axis.set_xticks(list(x), [labels[index] for index, _ in selected], rotation=35, ha="right")
        axis.set_title(f"{title} (mean and 95% CI)")
        axis.grid(alpha=0.25)
        figure.tight_layout()
        for suffix in ("png", "pdf"):
            path = output / f"{metric}.{suffix}"
            figure.savefig(path)
            artifacts.append(path.name)
        plt.close(figure)
    # Performance-latency Pareto plot.
    selected = [row for row in rows if "rmse_mean" in row and "inference_ms_per_sample_mean" in row]
    if selected:
        figure, axis = plt.subplots(figsize=(7, 5))
        axis.scatter([row["inference_ms_per_sample_mean"] for row in selected], [row["rmse_mean"] for row in selected])
        for row in selected:
            axis.annotate(row["experiment"], (row["inference_ms_per_sample_mean"], row["rmse_mean"]), fontsize=7)
        axis.set(xlabel="Inference latency (ms/sample)", ylabel="RMSE", title="Accuracy/latency Pareto")
        axis.grid(alpha=0.25)
        figure.tight_layout()
        for suffix in ("png", "pdf"):
            path = output / f"pareto.{suffix}"
            figure.savefig(path)
            artifacts.append(path.name)
        plt.close(figure)
    else:
        skipped["pareto"] = "no aggregated rows had both rmse and inference_ms_per_sample"
    targets_by_dim: dict[int, list[tuple[str, list, list | None]]] = defaultdict(list)
    for row in raw_rows:
        values = row.get("metrics", {}).get("target_rmse")
        if row.get("status") == "ok" and values:
            targets_by_dim[len(values)].append(
                (
                    f"{row.get('experiment')}/s{row.get('seed')}",
                    values,
                    row.get("metrics", {}).get("target_names"),
                )
            )
    for dimension, targets in targets_by_dim.items():
        figure, axis = plt.subplots(figsize=(max(10, dimension * 0.3), max(4, len(targets) * 0.35)))
        image = axis.imshow([values for _, values, _ in targets], aspect="auto")
        axis.set_yticks(range(len(targets)), [label for label, _, _ in targets], fontsize=7)
        names = next((names for _, _, names in targets if names), None) or [f"t{i}" for i in range(dimension)]
        axis.set_xticks(range(dimension), names, rotation=90, fontsize=6)
        axis.set_title(f"Estimator target RMSE heatmap ({dimension}D)")
        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        stem = "target_rmse_heatmap" if len(targets_by_dim) == 1 else f"target_rmse_heatmap_{dimension}d"
        for suffix in ("png", "pdf"):
            path = output / f"{stem}.{suffix}"
            figure.savefig(path)
            artifacts.append(path.name)
        plt.close(figure)
    if not targets_by_dim:
        skipped["target_rmse_heatmap"] = "no raw rows had metrics.target_rmse"
    dagger_rows = [row for row in raw_rows if row.get("metrics", {}).get("rounds")]
    if dagger_rows:
        figure, axis = plt.subplots(figsize=(8, 5))
        for row in dagger_rows:
            rounds = row["metrics"]["rounds"]
            x = [item["round"] for item in rounds]
            y = [item.get("training", {}).get("best_validation_mse", float("nan")) for item in rounds]
            axis.plot(x, y, alpha=0.55, label=f"{row['experiment']}/s{row['seed']}")
        axis.set(xlabel="DAgger round", ylabel="Validation MSE", title="DAgger learning curves")
        if len(dagger_rows) <= 12:
            axis.legend(fontsize=6)
        axis.grid(alpha=0.25)
        figure.tight_layout()
        for suffix in ("png", "pdf"):
            path = output / f"dagger_learning_curve.{suffix}"
            figure.savefig(path)
            artifacts.append(path.name)
        plt.close(figure)
    else:
        skipped["dagger_learning_curve"] = (
            "no raw rows had metrics.rounds (experiments with dagger_rounds=0 produce none)"
        )
    trace_row = next(
        (row for row in raw_rows if row.get("metrics", {}).get("trace_target") and row.get("metrics", {}).get("trace_prediction")),
        None,
    )
    if trace_row:
        target = trace_row["metrics"]["trace_target"]
        prediction = trace_row["metrics"]["trace_prediction"]
        trace_dimensions = min(9, len(target[0]))
        groups = math.ceil(trace_dimensions / 3)
        figure, axes = plt.subplots(groups, 1, figsize=(10, 2.6 * groups), sharex=True)
        axes = [axes] if groups == 1 else axes
        for group, axis in enumerate(axes):
            for offset in range(min(3, trace_dimensions - group * 3)):
                index = group * 3 + offset
                axis.plot([row[index] for row in target], alpha=0.65, label=f"target {index}")
                axis.plot([row[index] for row in prediction], linestyle="--", alpha=0.65, label=f"estimate {index}")
            axis.grid(alpha=0.2)
            axis.legend(ncol=3, fontsize=6)
        axes[0].set_title(f"Representative estimator trace: {trace_row['experiment']}")
        axes[-1].set_xlabel("Sample")
        figure.tight_layout()
        for suffix in ("png", "pdf"):
            path = output / f"representative_trace.{suffix}"
            figure.savefig(path)
            artifacts.append(path.name)
        plt.close(figure)
    else:
        skipped["representative_trace"] = "no raw rows had both metrics.trace_target and metrics.trace_prediction"
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

        def group_label(row: dict) -> str:
            teacher_id = row.get("teacher_id")
            return f"{row.get('experiment')}/{teacher_id}" if teacher_id else str(row.get("experiment"))

        curve_groups: dict[str, list[dict]] = defaultdict(list)
        for row in curve_rows:
            curve_groups[group_label(row)].append(row)
        baseline_groups: dict[str, list[dict]] = defaultdict(list)
        for row in baseline_rows:
            baseline_groups[group_label(row)].append(row)
        # Fix each group's color up front. A given metric (e.g. rmse) isn't
        # present for every group, so per-axis auto-cycling would otherwise
        # assign colors in whatever order each subplot happens to draw them in
        # -- e.g. JOSE silently taking IMU-BasedDistillation's blue on the one
        # subplot where IMU has no data to plot.
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
        group_colors = {
            label: color_cycle[index % len(color_cycle)]
            for index, label in enumerate(sorted(curve_groups))
        }

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
                if not by_step:
                    continue
                steps = sorted(by_step)
                means = [mean(by_step[step]) for step in steps]
                spreads = [stdev(by_step[step]) if len(by_step[step]) > 1 else 0.0 for step in steps]
                color = group_colors[label]
                axis.plot(steps, means, linewidth=2, marker="o", markersize=3, color=color, label=f"{label} (n={len(rows)})")
                axis.fill_between(
                    steps, [m - s for m, s in zip(means, spreads)], [m + s for m, s in zip(means, spreads)],
                    color=color, alpha=0.18, linewidth=0,
                )
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
        for suffix in ("png", "pdf"):
            path = output / f"learning_curves.{suffix}"
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
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(output / "results_tidy.csv", [
        {**{key: value for key, value in row.items() if key != "metrics"}, **row.get("metrics", {})} for row in raw_rows
    ])
    _write_csv(output / "summary.csv", summary)
    table = _markdown_table(summary)
    (output / "table.md").write_text(table + "\n", encoding="utf-8")
    (output / "table.tex").write_text(_latex_table(summary) + "\n", encoding="utf-8")
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
