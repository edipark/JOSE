"""Hyperparameter-sweep figure for the JOSE paper.

Writes fig_sweeps.png, a 2x2 that replaces fig_window.png and fig_dagger.png.
The top row sweeps the observed history length, the bottom row the number of
aggregation rounds; the left column is closed-loop survival and the right the
open-loop estimation error. Both sweeps make the same point from different
directions -- what the estimator needs is not architecture but data, in one case
more of the window and in the other more of its own distribution -- so reading
them side by side is what the merged layout buys, on top of the column of space
the two separate floats were costing.

Data reading, typography, colours and the seed-spread convention come from
``plot_window.py`` and ``plot_dagger.py`` so no figure can drift from the study
it reports.

Usage:
    python figures/plot_sweeps.py [--outdir figures] [--spread minmax|std]
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_window as win
import plot_dagger as dag

#: Two stacked rows at the final \columnwidth. The two source figures were
#: 1.72 in each, so the merge only pays for itself below their sum plus the
#: caption it drops; 2.85 in is where the axes still read at 7.5 pt ticks.
FIGURE_SIZE = (3.4, 2.85)

#: Panel headings, shared with plot_degradation.py so the two figures in the
#: results section carry the same lettering.
TITLE_SIZE = 7.5
TITLE_PAD = 2.5

#: The cost curve reads against its own right-hand axis. It is drawn in the body
#: text colour rather than grey: the dashed stroke and square marker already
#: separate it from the four task curves, and grey made the axis it owns look
#: like disabled chrome.
COST_COLOUR = "#1A1A1A"


def _curves(axis, sweeps, key, spread, label_them):
    for task, label, colour in win.available(sweeps, key):
        x, centre, error = win.series(sweeps[task], key, spread)
        axis.errorbar(
            x, centre, yerr=error, color=colour, marker="o",
            label=label if label_them else None,
            capsize=1.4, elinewidth=0.5, capthick=0.5, zorder=3,
        )


def _rmse_axis(axis, ticks, limits, tag):
    axis.set_yscale("log")
    axis.set_ylabel("Estimation RMSE", labelpad=1.0)
    axis.set_yticks(ticks)
    axis.set_yticklabels([("%g" % t) for t in ticks])
    axis.set_ylim(*limits)
    axis.set_title(tag, fontsize=TITLE_SIZE, pad=TITLE_PAD, loc="left")


def _survival_axis(axis, tag):
    axis.set_ylabel("Survival (%)", labelpad=1.0)
    axis.set_ylim(-8, 108)
    axis.set_yticks([0, 25, 50, 75, 100])
    axis.set_title(tag, fontsize=TITLE_SIZE, pad=TITLE_PAD, loc="left")


def plot(window_sweeps, dagger_sweeps, spread: str, out: str, dpi: int) -> None:
    figure, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)
    (w_surv, w_rmse), (d_surv, d_rmse) = axes
    for axis in axes.ravel():
        axis.spines["right"].set_visible(False)

    _curves(w_surv, window_sweeps, "survival", spread, label_them=True)
    _survival_axis(w_surv, "(a) Window length")
    _curves(w_rmse, window_sweeps, "rmse", spread, label_them=False)
    _rmse_axis(w_rmse, [0.01, 0.02, 0.05, 0.1], (0.005, 0.22), "(b) Window length")

    # The cost curve rides the window row only: it is a property of the window,
    # and aggregation rounds do not change the network that runs.
    cost_axis = w_rmse.twinx()
    cost_axis.grid(False)
    windows, cost, _ = win.series(window_sweeps["amp_walk"], "cost", spread, scale="first")
    cost_axis.plot(
        windows, cost, color=COST_COLOUR, linestyle=(0, (3.5, 2)), linewidth=0.6,
        marker="s", markersize=1.4, label="Inference cost", zorder=3,
    )
    cost_axis.set_ylabel("Cost (rel.)", color=COST_COLOUR, labelpad=1.0)
    cost_axis.tick_params(axis="y", colors=COST_COLOUR, pad=1.0)
    cost_axis.spines["right"].set_color(COST_COLOUR)
    cost_axis.spines["top"].set_visible(False)
    cost_axis.set_ylim(0, 4.4)
    cost_axis.set_yticks([1, 2, 3, 4])

    _curves(d_surv, dagger_sweeps, "survival", spread, label_them=False)
    _survival_axis(d_surv, "(c) Aggregation rounds")
    _curves(d_rmse, dagger_sweeps, "rmse", spread, label_them=False)
    _rmse_axis(d_rmse, [0.01, 0.02, 0.05, 0.1, 0.2], (0.008, 0.3), "(d) Aggregation rounds")

    for axis in (w_surv, w_rmse):
        axis.axvline(win.CHOSEN_WINDOW, color=win.RULE_COLOUR, linewidth=0.5,
                     linestyle=(0, (1, 2.5)), zorder=1)
        win.window_axis(axis, tight=True)
    for axis in (d_surv, d_rmse):
        axis.axvline(0, color=win.RULE_COLOUR, linewidth=0.5,
                     linestyle=(0, (1, 2.5)), zorder=1)
        dag.round_axis(axis)
        axis.set_xlabel("Aggregation round $K$")

    # Two legends rather than one five-column row: the four tasks share an
    # encoding and belong on one line, while the cost curve reads against a
    # different axis and is set apart underneath. A single ncol=5 legend fills
    # column-major and would break the four tasks across two lines instead.
    handles, labels = w_surv.get_legend_handles_labels()
    cost_handles, cost_labels = cost_axis.get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.005),
        frameon=False, handlelength=1.6, ncol=4, columnspacing=1.1,
        handletextpad=0.4,
    )
    figure.legend(
        cost_handles, cost_labels, loc="upper center", bbox_to_anchor=(0.5, 0.947),
        frameon=False, handlelength=1.6, handletextpad=0.4,
    )

    # Explicit margins rather than tight_layout: the canvas is fixed at the
    # column width, so every hundredth of an inch not spent on padding goes to
    # the axes themselves.
    figure.subplots_adjust(left=0.108, right=0.883, top=0.850, bottom=0.132,
                           wspace=0.60, hspace=0.62)
    figure.savefig(out, dpi=dpi)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--spread", choices=("minmax", "std"), default="minmax")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    win.style()
    window_sweeps = {task: win.read_sweep(task) for task, _, _ in win.TASKS}
    dagger_sweeps = {task: dag.read_sweep(task) for task, _, _ in win.TASKS}
    plot(window_sweeps, dagger_sweeps, args.spread,
         os.path.join(args.outdir, "fig_sweeps.png"), args.dpi)


if __name__ == "__main__":
    main()
