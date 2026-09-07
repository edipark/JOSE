"""Aggregate and plot a completed zero-shot friction sweep."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics as st


METHOD_ORDER = ("teacher", "jose", "joint_only", "imu_clean", "set")
METHOD_LABELS = {
    "teacher": "Privileged Teacher",
    "jose": "JOSE",
    "joint_only": "Joint-only Distill.",
    "imu_clean": "IMU Distill.",
    "set": "SET",
}
METRICS = {
    "survival": ("grid_survival_rate", 100.0),
    "track": ("track_error_norm", 1.0),
    "slide": ("feet_slide_penalty", 1.0),
}


def read_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    keys = [
        (
            float(row["friction"]["effective"]),
            row["method"],
            row.get("checkpoint_seed"),
            int(row["eval_seed"]),
        )
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate friction result cells")
    return rows


def aggregate(rows: list[dict]) -> dict:
    """Average eval seeds within checkpoints, then summarize checkpoint seeds."""
    teacher_track = {
        (float(row["friction"]["effective"]), int(row["eval_seed"])): row["metrics"][
            "track_error_norm"
        ]
        for row in rows
        if row["method"] == "teacher"
    }
    values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    regrets = defaultdict(lambda: defaultdict(list))
    for row in rows:
        friction = float(row["friction"]["effective"])
        method = row["method"]
        checkpoint = row.get("checkpoint_seed")
        # Teacher's evaluation seed is its replication unit. Learned methods
        # use the trained checkpoint as the outer replication unit.
        replicate = int(row["eval_seed"]) if method == "teacher" else int(checkpoint)
        for metric, (source, scale) in METRICS.items():
            values[(method, friction)][replicate][metric].append(scale * row["metrics"][source])
        if method != "teacher":
            reference = teacher_track[(friction, int(row["eval_seed"]))]
            regrets[(method, friction)][replicate].append(
                row["metrics"]["track_error_norm"] - reference
            )

    summary = {}
    for key, replicates in values.items():
        method, friction = key
        record = {}
        for metric in METRICS:
            outer = [st.mean(bucket[metric]) for bucket in replicates.values()]
            record[metric] = {
                "mean": st.mean(outer),
                "min": min(outer),
                "max": max(outer),
                "outer_replicates": len(outer),
            }
        if method != "teacher":
            outer = [st.mean(bucket) for bucket in regrets[key].values()]
            record["track_regret"] = {
                "mean": st.mean(outer),
                "min": min(outer),
                "max": max(outer),
                "outer_replicates": len(outer),
            }
        summary[f"{method}|{friction:g}"] = record
    return summary


def _cell(record: dict | None, digits: int) -> str:
    if not record:
        return "--"
    return f"{record['mean']:.{digits}f} [{record['min']:.{digits}f}, {record['max']:.{digits}f}]"


def render_report(rows: list[dict], summary: dict) -> str:
    frictions = sorted({float(row["friction"]["effective"]) for row in rows})
    lines = [
        "# Zero-shot friction sweep",
        "",
        f"Raw cells: {len(rows)}. Static and dynamic friction are equal at every point.",
        "Learned methods average evaluation seeds within each checkpoint first; brackets are the",
        "observed range over checkpoint seeds. Teacher brackets are over evaluation seeds.",
        "",
    ]
    for metric, title, digits in (
        ("survival", "Grid survival (%)", 1),
        ("track", "Normalized command RMSE", 4),
        ("track_regret", "Teacher-relative tracking regret", 4),
        ("slide", "Feet-slide penalty", 4),
    ):
        lines += [f"## {title}", "", "| method | " + " | ".join(f"μ={mu:g}" for mu in frictions) + " |", "|---|" + "---|" * len(frictions)]
        for method in METHOD_ORDER:
            cells = []
            for friction in frictions:
                record = summary.get(f"{method}|{friction:g}", {})
                cells.append(_cell(record.get(metric), digits))
            lines.append(f"| {METHOD_LABELS[method]} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def plot(summary: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    frictions = sorted(
        {float(key.split("|", 1)[1]) for key in summary}
    )
    colours = {
        "teacher": "#202020",
        "jose": "#0072B2",
        "joint_only": "#D55E00",
        "imu_clean": "#E69F00",
        "set": "#009E73",
    }
    markers = {"teacher": "o", "jose": "s", "joint_only": "^", "imu_clean": "v", "set": "D"}
    figure, axes = plt.subplots(2, 1, figsize=(3.4, 4.0), sharex=True)
    for method in METHOD_ORDER:
        for axis, metric in zip(axes, ("survival", "track")):
            records = [summary.get(f"{method}|{mu:g}", {}).get(metric) for mu in frictions]
            if any(record is None for record in records):
                continue
            mean = np.asarray([record["mean"] for record in records])
            low = np.asarray([record["min"] for record in records])
            high = np.asarray([record["max"] for record in records])
            axis.plot(
                frictions,
                mean,
                label=METHOD_LABELS[method],
                color=colours[method],
                marker=markers[method],
                markersize=3.2,
                linewidth=1.2,
            )
            axis.fill_between(frictions, low, high, color=colours[method], alpha=0.12, linewidth=0)
    axes[0].set_ylabel("Survival (%)")
    axes[0].set_ylim(0, 105)
    axes[1].set_ylabel("Command RMSE")
    axes[1].set_xlabel("Static and dynamic friction coefficient")
    for axis in axes:
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.5)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(labelsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=6.5)
    figure.subplots_adjust(top=0.86, left=0.18, right=0.98, bottom=0.11, hspace=0.18)
    figure.savefig(out, dpi=300)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    rows = read_rows(Path(args.results))
    if not rows:
        raise RuntimeError("No friction rows to report")
    summary = aggregate(rows)
    for value in summary.values():
        for metric in value.values():
            if not math.isfinite(metric["mean"]):
                raise RuntimeError("Non-finite aggregate")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "report.md").write_text(render_report(rows, summary), encoding="utf-8")
    plot(summary, out / "friction_sweep.png")
    print(f"wrote {out / 'report.md'} and {out / 'friction_sweep.png'}")


if __name__ == "__main__":
    main()
