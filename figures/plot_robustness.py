"""Encoder-degradation figure for the JOSE paper.

Writes fig_robustness.png: survival and command-tracking error for every method,
with and without encoder randomization at training time, at two noise levels.

Why bars and not a sweep. The clean condition is already Table~I, so plotting it
again adds a point that carries no new information and anchors every curve at
the same place. Once it is dropped only two noise levels remain, and a line
through two points is a single segment -- it looks like a trend without being
measured as one.

Why the noise level is the grouped variable and the randomization is a position
on the axis, rather than the other way round. Both comparisons matter, but only
one of them is a quantity: nominal and twice nominal are ordered, so they get the
ordered channel, a lightness ramp, and a group that darkens upward is an arm that
degrades. The randomization pair then sits as two adjacent entries, which keeps
it legible without spending a second visual channel on it. It also makes this
figure and fig_imu_robustness read the same way, with a given level drawn in the
same shade in both.

The teacher is not drawn. It is measured, and at nominal it does not move -- that
level is inside its own training distribution -- but it has no pair, so as a line
it ranked nobody while cutting across the bars and the n/a markers that do. The
fact is carried in the text, where it costs nothing to state.

Usage:
    python figures/plot_robustness.py [--outdir figures]
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

import plot_window as shared


ROBUSTNESS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs", "jose_g1", "robustness",
)

#: (method key, label), as two blocks: every method as trained, then every
#: method with encoder randomization, in Table~I's order within each block.
#:
#: Blocked rather than interleaved in pairs. The pair comparison is one method
#: against itself, but the finding is not about any one method -- it is that the
#: whole left block collapses and the whole right block holds. Blocking puts that
#: in the shape of the panel, where interleaving made the reader assemble it from
#: four separate comparisons.
ARMS = [
    ("jose", "JOSE"),
    ("joint_only", "Joint-\nonly"),
    ("imu_clean", "Distill."),
    ("set", "SET"),
    ("jose_enc", "JOSE\n$+$ DR"),
    ("joint_only_enc", "Joint-only\n$+$ DR"),
    ("imu_clean_enc", "Distill.\n$+$ DR"),
    ("set_enc", "SET\n$+$ DR"),
]
#: Where the randomized block starts; the axis opens a gap there so the two
#: blocks read as two, without a rule that would add ink for no new information.
BLOCK_SPLIT = 4
BLOCK_GAP = 0.55

SCALES = (1.0, 2.0)
#: A green lightness ramp, the same hue family plot_window.py gives the
#: locomotion task. Both
#: degradation figures measure only that task, so hue names the task here
#: exactly as it does in the sweeps, and lightness is left free to carry the
#: one ordered quantity these figures vary. A saturated second hue would have
#: implied a second category that does not exist.
SCALE_COLOURS = {1.0: "#CDE8C8", 2.0: "#2E7D32"}
SCALE_EDGES = {1.0: "#2E7D32", 2.0: "#2E7D32"}
SCALE_LABELS = {1.0: "nominal", 2.0: "$2\\times$ nominal"}

FIGURE_SIZE = (3.4, 3.7)
BAR_WIDTH = 0.36
#: (metric key, axis label, limits, whether to print values on the bars)
METRICS = (
    ("survival", "Survival (%)", (0, 118), True),
    ("track", "Command RMSE", (0, 1.32), False),
)


def read_axis(axis: str):
    per_method = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    path = os.path.join(ROBUSTNESS_DIR, f"{axis}_axis.jsonl")
    if not os.path.isfile(path):
        return per_method
    for line in open(path):
        if not line.strip():
            continue
        record = json.loads(line)
        metrics = record.get("metrics", {})
        bucket = per_method[record["method"]][float(record["scale"])]
        if "track_error_norm" in metrics:
            bucket["track"].append(metrics["track_error_norm"])
        if "grid_survival_rate" in metrics:
            bucket["survival"].append(100.0 * metrics["grid_survival_rate"])
    return per_method


def stat(sweeps, method, scale, key):
    """Mean and observed range over seeds, or None when the arm is missing."""
    values = sweeps.get(method, {}).get(scale, {}).get(key)
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    centre = float(array.mean())
    return centre, np.array([[centre - array.min()], [array.max() - centre]])


def arm_positions() -> np.ndarray:
    positions = np.arange(len(ARMS), dtype=float)
    positions[BLOCK_SPLIT:] += BLOCK_GAP
    return positions


def panel(axis, sweeps, key, ylabel, ylim, annotate) -> None:
    positions = arm_positions()
    for offset, scale in enumerate(SCALES):
        for slot, (method, _label) in enumerate(ARMS):
            measured = stat(sweeps, method, scale, key)
            x = positions[slot] + (offset - 0.5) * (BAR_WIDTH + 0.04)
            if measured is None:
                # Nothing drawn and nothing implied: an absent arm must not read
                # as a zero-height bar.
                axis.text(x, ylim[0] + 0.04 * (ylim[1] - ylim[0]), "n/a", ha="center",
                          fontsize=5.0, color="#9A9A9A", rotation=90)
                continue
            centre, error = measured
            axis.bar(x, centre, BAR_WIDTH, color=SCALE_COLOURS[scale],
                     edgecolor=SCALE_EDGES[scale], linewidth=0.5, zorder=3)
            axis.errorbar(x, centre, yerr=error, color="#1A1A1A", elinewidth=0.5,
                          capsize=1.2, capthick=0.5, zorder=4)
            if annotate:
                # Above the whisker, not above the bar: on the arms with real
                # seed spread the two collide otherwise.
                top = centre + float(error[1][0])
                axis.text(x, top + 0.035 * (ylim[1] - ylim[0]), f"{centre:.0f}",
                          ha="center", va="bottom", fontsize=5,
                          fontweight="bold", color="#1A1A1A")

    axis.set_xticks(positions)
    axis.set_xticklabels([label for _method, label in ARMS], fontsize=5.5)
    axis.tick_params(axis="x", pad=1.0)
    axis.tick_params(axis="y", pad=1.2)
    axis.set_ylabel(ylabel, labelpad=1.5)
    axis.set_ylim(*ylim)
    axis.set_xlim(positions[0] - 0.6, positions[-1] + 0.6)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="x", visible=False)


def plot(sweeps, out: str, dpi: int) -> None:
    """Two stacked panels, one per metric.

    Both are drawn because they answer different questions and one does not
    substitute for the other. Survival says whether the robot is still walking;
    command RMSE says how well it follows the command while it is. A method can
    hold the first and lose the second -- that is the whole reason this paper
    reports tracking at all.

    RMSE stays on a linear axis. A bar encodes its value as length from zero, and
    a log axis has no zero, so log-scaled bars misstate every ratio a reader
    takes off them.
    """
    figure, axes = plt.subplots(len(METRICS), 1, figsize=FIGURE_SIZE, squeeze=False)
    for row, (key, ylabel, ylim, annotate) in enumerate(METRICS):
        axis = axes[row][0]
        # Labelled on both panels: the reader should not have to carry eight
        # positions down from one panel to the other.
        panel(axis, sweeps, key, ylabel, ylim, annotate)
        axis.text(0.0, 1.02, f"({'ab'[row]}) {ylabel.split(' (')[0]}",
                  transform=axis.transAxes, va="bottom")

    handles = [Patch(facecolor=SCALE_COLOURS[s], edgecolor=SCALE_EDGES[s], linewidth=0.5)
               for s in SCALES]
    figure.legend(handles, [SCALE_LABELS[s] for s in SCALES], loc="upper center",
                  bbox_to_anchor=(0.5, 1.0), frameon=False, handlelength=1.4,
                  ncol=2, columnspacing=1.2)
    shared.save(figure, out, dpi, legend_rows=1.1, w_pad=1.2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    shared.style()
    sweeps = read_axis("encoder")
    plot(sweeps, os.path.join(args.outdir, "fig_robustness.png"), args.dpi)

    for method, label in ARMS:
        row = []
        for scale in SCALES:
            s = stat(sweeps, method, scale, "survival")
            t = stat(sweeps, method, scale, "track")
            row.append(f"{scale:g}x " + ("--" if s is None else f"{t[0]:.3f}/{s[0]:5.1f}%"))
        print(f"{label.replace(chr(10), ' '):18s} " + "  ".join(row))


if __name__ == "__main__":
    main()
