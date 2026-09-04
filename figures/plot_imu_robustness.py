"""IMU-degradation figure for the JOSE paper.

Writes fig_imu_robustness.png: survival and command-tracking error for every
method that could in principle read an IMU, at three noise levels.

Why the encoding differs from fig_robustness. That figure pairs each method's
clean arm against its randomized one, because on the encoder axis every method
has both. On this axis only the IMU distillation baseline does -- JOSE and the
joint-only student read no IMU and so cannot be hardened against one, and SET is
reproduced as published. Copying the paired encoding here would leave one full
group and three half-empty ones. So the pair becomes two entries on the axis and
the bars within a group carry the noise level instead, which also puts the thing
the section is about -- how steeply a method falls as its IMU gets worse -- into
the shape of each group rather than into a comparison across groups.

Why JOSE and the joint-only student are drawn at all when they cannot move.
Their flatness is the claim, and a claim the reader can see measured is worth
more than the same claim asserted in a sentence. They are run through the same
sweep as everything else, and their bars are also the check that the IMU
corruption stays inside the methods that read an IMU.

Why there is no teacher here. On the encoder axis it is the ceiling, because the
noise reaches it and it still holds. On this axis nothing reaches it, so its line
would be a constant that ranks nobody, and the flatness it used to certify is now
certified by JOSE's own bars. The figure is a comparison among the methods that
are being compared.

Usage:
    python figures/plot_imu_robustness.py [--outdir figures]
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

#: (method key, label). JOSE first, then the baselines in Table~I's order, with
#: the two IMU distillation arms adjacent so the randomization pair reads as one
#: comparison even though it occupies two slots.
METHODS = [
    ("jose", "JOSE"),
    ("joint_only", "Joint-\nonly"),
    ("imu_clean", "IMU\ndist."),
    ("set", "SET"),
    ("imu_dr", "IMU dist.\n$+$ DR"),
    ("set_imu_dr", "SET\n$+$ DR"),
]
#: Where the randomized block starts, matching fig_robustness. Both methods that
#: read an IMU appear in it: hardening one and not the other would show which arm
#: got the treatment rather than which method tolerates a bad sensor.
BLOCK_SPLIT = 4
BLOCK_GAP = 0.55

#: Noise level as a lightness ramp on the one hue fig_robustness already uses.
#: Level is an ordered quantity, so it gets an ordered channel; hue would imply
#: three unrelated categories. Light to dark is the direction a reader assumes
#: for less to more, so a group that darkens upward is a method that degrades.
SCALES = (1.0, 2.0, 4.0)
SCALE_COLOURS = {1.0: "#DFD0EE", 2.0: "#A67FC8", 4.0: "#653D8F"}
SCALE_EDGES = {1.0: "#7E52AD", 2.0: "#A67FC8", 4.0: "#653D8F"}
SCALE_LABELS = {1.0: "nominal", 2.0: "$2\\times$ nominal", 4.0: "$4\\times$ nominal"}

FIGURE_SIZE = (3.4, 3.7)
BAR_WIDTH = 0.26
METRICS = (
    ("survival", "Survival (%)", (0, 112)),
    ("track", "Command RMSE", (0, 0.78)),
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
    positions = np.arange(len(METHODS), dtype=float)
    positions[BLOCK_SPLIT:] += BLOCK_GAP
    return positions


def panel(axis, sweeps, key, ylabel, ylim) -> None:
    positions = arm_positions()
    for offset, scale in enumerate(SCALES):
        for slot, (method, _label) in enumerate(METHODS):
            measured = stat(sweeps, method, scale, key)
            x = positions[slot] + (offset - 1) * (BAR_WIDTH + 0.03)
            if measured is None:
                # An absent arm must not read as a zero-height bar.
                axis.text(x, ylim[0] + 0.04 * (ylim[1] - ylim[0]), "n/a", ha="center",
                          fontsize=5.5, color="#9A9A9A", rotation=90)
                continue
            centre, error = measured
            axis.bar(
                x, centre, BAR_WIDTH, color=SCALE_COLOURS[scale],
                edgecolor=SCALE_EDGES[scale], linewidth=0.5, zorder=3,
            )
            axis.errorbar(x, centre, yerr=error, color="#1A1A1A", elinewidth=0.5,
                          capsize=1.4, capthick=0.5, zorder=4)

    axis.set_xticks(positions)
    axis.set_xticklabels([label for _method, label in METHODS], fontsize=5.5)
    axis.tick_params(axis="x", pad=1.0)
    axis.tick_params(axis="y", pad=1.2)
    axis.set_ylabel(ylabel, labelpad=1.5)
    axis.set_ylim(*ylim)
    axis.set_xlim(positions[0] - 0.6, positions[-1] + 0.6)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="x", visible=False)


def plot(sweeps, out: str, dpi: int) -> None:
    figure, axes = plt.subplots(len(METRICS), 1, figsize=FIGURE_SIZE, squeeze=False)
    for row, (key, ylabel, ylim) in enumerate(METRICS):
        axis = axes[row][0]
        # Labelled on both panels: the reader should not have to carry the
        # positions down from one panel to the other.
        panel(axis, sweeps, key, ylabel, ylim)
        axis.text(0.0, 1.02, f"({'ab'[row]}) {ylabel.split(' (')[0]}",
                  transform=axis.transAxes, va="bottom")

    # Proxy handles rather than the drawn artists: the first group on the axis
    # can be an arm that is missing, and then no bar would carry the label.
    handles = [Patch(facecolor=SCALE_COLOURS[s], edgecolor=SCALE_EDGES[s], linewidth=0.5)
               for s in SCALES]
    labels = [SCALE_LABELS[s] for s in SCALES]
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
                  frameon=False, handlelength=1.4, ncol=3, columnspacing=1.0)
    shared.save(figure, out, dpi, legend_rows=1.1, w_pad=1.2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    shared.style()
    sweeps = read_axis("imu")
    plot(sweeps, os.path.join(args.outdir, "fig_imu_robustness.png"), args.dpi)

    for method, label in METHODS:
        row = []
        for scale in SCALES:
            s = stat(sweeps, method, scale, "survival")
            t = stat(sweeps, method, scale, "track")
            row.append(f"{scale:g}x " + ("--" if s is None else f"{t[0]:.3f}/{s[0]:5.1f}%"))
        print(f"{label.replace(chr(10), ' '):16s} " + "  ".join(row))


if __name__ == "__main__":
    main()
