"""Window-length sweep figures for the JOSE paper.

Writes one single-column figure, fig_window.png, with two stacked panels that
share the window axis:

  (a) survival, already a 0--100 scale on every task, so the tasks share one
      axis without rescaling;
  (b) estimation RMSE, with relative inference cost on a second axis. The RMSE
      is the raw value: it is an average over a target vector whose components
      carry mixed units, so it is readable within a task but not across tasks,
      and the caption has to say so. Cost is a ratio because it is measured
      batched across environments and its absolute value is not a deployment
      latency; it also moves in the opposite direction over a much wider range,
      which is why it gets the right-hand axis.

Usage:
    python figures/plot_window.py [--outdir figures] [--spread minmax|std]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ABLATION_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs", "jose_g1", "ablation",
)

# Display name and colour per task, in the order the legend should read. The
# Okabe-Ito blue / vermillion / green stay separable under the common forms of
# colour blindness.
TASKS = [
    ("amp_walk", "Walk", "#0072B2"),
    # Okabe-Ito's reddish purple darkened from #CC79A7, which sits at 2.6:1
    # against white -- readable as a fill, thin as a 1 pt line. The hue is
    # unchanged, so the colour-blind separation the palette provides is intact
    # and only the contrast moves.
    ("amp_dance", "Dance", "#B03A75"),
    ("amp_jump", "Jump", "#D55E00"),
    ("locomotion", "Locomotion", "#009E73"),
]

# Drawn at the final \columnwidth so nothing is rescaled on inclusion, which
# would otherwise change the effective font size. The legend is placed inside
# the canvas rather than cropped in, so savefig does not use bbox_inches.
FIGURE_SIZE = {"stacked": (3.4, 3.45), "side": (3.4, 1.72)}
COST_COLOUR = "#8A8A8A"
RULE_COLOUR = "#B8B8B8"
CHOSEN_WINDOW = 25
WINDOW_TICKS = [1, 5, 10, 25, 50]


def latest_window_study(task: str) -> str | None:
    """Newest window study for one task, or None if it has not been run yet.

    Studies are timestamped directories, so the lexicographic maximum is the
    most recent run. Returning None rather than raising lets a task be listed
    in TASKS before its sweep exists -- dance is declared there and will start
    appearing in both figures the moment its window study lands.
    """
    pattern = os.path.join(ABLATION_ROOT, "*", task, "studies", "window", "*", "results.jsonl")
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def read_sweep(task: str) -> dict[int, dict[str, list[float]]]:
    """Per-seed metrics keyed by window length, for one task."""
    per_window: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    results = latest_window_study(task)
    if results is None:
        return per_window
    for line in open(results):
        record = json.loads(line)
        match = re.fullmatch(r"lstm_w(\d+)_all", record.get("experiment", ""))
        if match is None or record.get("status") != "ok":
            continue  # teacher_gt, non-sweep rows and failed jobs
        window = int(match.group(1))
        metrics = record.get("metrics", {})
        if "death_rate" in metrics:
            per_window[window]["survival"].append(100.0 - metrics["death_rate"])
        if "rmse" in metrics:
            per_window[window]["rmse"].append(metrics["rmse"])
        if "inference_ms_per_sample" in metrics:
            per_window[window]["cost"].append(metrics["inference_ms_per_sample"])
    return per_window


def series(sweep, key, spread: str, scale: float | None = None):
    """Windows, centres and asymmetric error-bar half-widths for one metric.

    ``spread='minmax'`` draws the full range over the seeds. With three seeds a
    standard deviation implies a symmetric distribution the data do not have --
    on walk the short windows are bimodal, with seeds at 0 and at 75 -- so the
    observed range is the honest depiction. ``spread='std'`` is offered for
    consistency with tables that quote one.
    """
    windows = np.array(sorted(w for w in sweep if sweep[w].get(key)), dtype=float)
    values = [np.asarray(sweep[int(w)][key], dtype=float) for w in windows]
    if scale is None:
        scale = 1.0
    elif scale == "first":
        scale = float(np.mean(values[0]))
    centre = np.array([v.mean() for v in values]) / scale
    if spread == "std":
        half = np.array([v.std() for v in values]) / scale
        error = np.vstack([half, half])
    else:
        error = np.vstack([
            centre - np.array([v.min() for v in values]) / scale,
            np.array([v.max() for v in values]) / scale - centre,
        ])
    return windows, centre, error


def style() -> None:
    """Match the figures to the IEEEtran body text.

    ``Nimbus Roman`` is the URW clone of Times that pdflatex actually loads for
    IEEEtran's ``ptm``, so the axis labels come out in the same face as the
    surrounding prose; the STIX math set is Times-metric and keeps ``$W$``
    consistent with the body. Sizes are set in points against the 10 pt body,
    and the figures are drawn at their final width so nothing is rescaled on
    inclusion.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.grid": True,
        "grid.color": "#D9D9D9",
        "grid.alpha": 1.0,
        "grid.linewidth": 0.4,
        "axes.spines.top": False,
        "axes.edgecolor": "#4D4D4D",
        "axes.linewidth": 0.6,
        "axes.labelcolor": "#1A1A1A",
        "text.color": "#1A1A1A",
        "xtick.color": "#4D4D4D",
        "ytick.color": "#4D4D4D",
        "xtick.labelcolor": "#1A1A1A",
        "ytick.labelcolor": "#1A1A1A",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 0.9,
        "lines.markersize": 1.9,
        "legend.handletextpad": 0.5,
        "legend.borderpad": 0.2,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
    })


def available(sweeps, key):
    """The declared tasks that actually have data for one metric."""
    return [(task, label, colour) for task, label, colour in TASKS if sweeps[task] and
            any(sweeps[task][w].get(key) for w in sweeps[task])]


def save(figure, out: str, dpi: int, legend_rows: float = 0, w_pad: float | None = None) -> None:
    """Lay out inside the fixed canvas and write the file at exactly that size.

    ``bbox_inches="tight"`` would crop each figure to its own content and hand
    back two different image sizes, so the rectangle is reserved here instead
    and the canvas is written as declared.
    """
    top = 1.0 - 0.055 * legend_rows
    figure.tight_layout(pad=0.3, rect=(0, 0, 1, top), **({} if w_pad is None else {"w_pad": w_pad}))
    figure.savefig(out, dpi=dpi)
    print(f"wrote {out}")


def window_axis(axis, tight: bool = False) -> None:
    axis.set_xscale("log")
    axis.set_xticks(WINDOW_TICKS)
    axis.set_xticklabels([str(t) for t in WINDOW_TICKS])
    axis.minorticks_off()
    axis.set_xlabel("$W$ (frames)" if tight else "Window length $W$ (frames)")
    axis.set_xlim(0.85, 60)


def plot_window(sweeps, spread: str, out: str, dpi: int, layout: str) -> None:
    stacked = layout == "stacked"
    figure, (left, right) = plt.subplots(
        2 if stacked else 1, 1 if stacked else 2,
        figsize=FIGURE_SIZE[layout], sharex=stacked,
    )
    # Side by side the panels are about 1.5 in wide, so the chrome has to give:
    # shorter labels, smaller ticks and only the windows that carry the story.
    tight = not stacked

    left.spines["right"].set_visible(False)
    for task, label, colour in available(sweeps, "survival"):
        windows, centre, error = series(sweeps[task], "survival", spread)
        left.errorbar(
            windows, centre, yerr=error, color=colour, label=label, marker="o",
            capsize=1.6, elinewidth=0.5, capthick=0.5, zorder=3,
        )
    left.set_ylabel("Survival (%)", labelpad=1.5)
    left.set_ylim(-8, 108)
    left.set_yticks([0, 25, 50, 75, 100])
    left.text(0.0, 1.02, "(a)", transform=left.transAxes, va="bottom")

    for task, label, colour in available(sweeps, "rmse"):
        windows, centre, error = series(sweeps[task], "rmse", spread)
        right.errorbar(
            windows, centre, yerr=error, color=colour, marker="o",
            capsize=1.6, elinewidth=0.5, capthick=0.5, zorder=3,
        )
    right.set_yscale("log")
    right.set_ylabel("Estimation RMSE", labelpad=1.5)
    ticks = [0.01, 0.02, 0.05, 0.1] if tight else [0.005, 0.01, 0.02, 0.05, 0.1]
    right.set_yticks(ticks)
    right.set_yticklabels([("%g" % t) for t in ticks])
    right.set_ylim(0.005, 0.22)
    right.text(0.0, 1.02, "(b)", transform=right.transAxes, va="bottom")

    cost_axis = right.twinx()
    cost_axis.grid(False)
    windows, cost, _ = series(sweeps["amp_walk"], "cost", spread, scale="first")
    cost_axis.plot(
        windows, cost, color=COST_COLOUR, linestyle=(0, (3.5, 2)), linewidth=0.6,
        marker="s", markersize=1.4, label="Inference cost", zorder=3,
    )
    cost_axis.set_ylabel(
        "Cost (rel.)" if tight else "Inference cost (rel. $W\\!=\\!1$)",
        color=COST_COLOUR, labelpad=1.5,
    )
    cost_axis.tick_params(axis="y", colors=COST_COLOUR, pad=1.5)
    cost_axis.spines["right"].set_color(COST_COLOUR)
    cost_axis.spines["top"].set_visible(False)
    cost_axis.set_ylim(0, 4.4)
    cost_axis.set_yticks([1, 2, 3, 4])

    for axis in (left, right):
        axis.axvline(CHOSEN_WINDOW, color=RULE_COLOUR, linewidth=0.5,
                     linestyle=(0, (1, 2.5)), zorder=1)
        window_axis(axis, tight)
    if stacked:
        left.set_xlabel("")

    # One legend above both panels: the task colours are shared, and putting it
    # outside keeps it off the curves as tasks are added.
    handles, labels = left.get_legend_handles_labels()
    extra = cost_axis.get_legend_handles_labels()
    entries = len(handles) + len(extra[0])
    columns = min(entries, 4) if tight else (2 if entries <= 4 else 3)
    figure.legend(
        handles + extra[0], labels + extra[1], loc="upper center",
        bbox_to_anchor=(0.5, 1.0), frameon=False, handlelength=1.6,
        ncol=columns, columnspacing=0.9,
    )
    if stacked:
        figure.align_ylabels((left, right))
    rows = 1 + (entries - 1) // columns
    # The panel letters sit above the axes, so the reserved strip has to hold
    # the legend and them, not just the legend.
    save(figure, out, dpi, legend_rows=rows * (2.7 if tight else 1.6),
         w_pad=1.8 if tight else None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--spread", choices=("minmax", "std"), default="minmax")
    parser.add_argument("--layout", choices=("stacked", "side"), default="side")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    style()
    sweeps = {task: read_sweep(task) for task, _, _ in TASKS}
    plot_window(sweeps, args.spread, os.path.join(args.outdir, "fig_window.png"), args.dpi, args.layout)

    for task, label, _ in TASKS:
        windows, survival, _ = series(sweeps[task], "survival", args.spread)
        _, rmse, _ = series(sweeps[task], "rmse", args.spread)
        print(f"{label:11s} " + "  ".join(
            f"W{int(w):<2d} surv {s:5.1f} rmse {r:.4f}" for w, s, r in zip(windows, survival, rmse)
        ))


if __name__ == "__main__":
    main()
