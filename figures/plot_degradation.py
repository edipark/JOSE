"""Both sensor-degradation axes in one single-column figure.

Writes fig_degradation.png: survival and command-tracking error against encoder
noise (top row) and IMU noise (bottom row), replacing fig_robustness.png and
fig_imu_robustness.png. The two were separate column-width floats; merged they
cost one, and a float is worth a page at the length this paper runs.

The merge is not free and the cost lands on the horizontal axis. Four panels in
one column give each about 1.5 in, so the arm labels no longer fit lying down and
are set vertically, and the value annotations that sat above the survival bars
are dropped -- at this width they collided with the whiskers, and every one of
them is already stated in the text. What the panels keep is the comparison they
were drawn for: the block of arms as trained against the block with
randomization, in Table~I's order inside each block.

Rows are the sensor being degraded and columns are the metric, so a reader moves
down to change what is broken and across to change what is measured. Each row
keeps its own arm set: the encoder axis has a randomized partner for all four
methods, the IMU axis only for the two that read an inertial unit, and forcing
them onto shared categories would have implied arms that were never run.

Levels stay a lightness ramp on one hue, matching both source figures, so a
given level is the same shade in all four panels and a group that darkens upward
is an arm that degrades. The two rows ramp over different level sets (2 and 3),
which is why each row carries its own legend rather than one shared key that
would list levels absent from half the figure.

Usage:
    python figures/plot_degradation.py [--outdir figures]
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
    "logs", "jose_g1", "robustness_cmdfix",
)

#: One entry per row: which axis file, its arms, its levels, and how the level
#: ramp is coloured. Kept as data so the two rows cannot drift apart in style
#: the way two separate scripts could.
ROWS = (
    {
        "axis": "encoder",
        "title": "Encoder noise",
        "arms": [
            ("jose", "JOSE (Ours)"), ("joint_only", "Joint-only"), ("imu_clean", "Distill."),
            ("set", "SET"), ("jose_enc", "JOSE (Ours)"), ("joint_only_enc", "Joint-only"),
            ("imu_clean_enc", "Distill."), ("set_enc", "SET"),
        ],
        "split": 4,
        "scales": (1.0, 2.0),
        "track_lim": (0, 1.32),
        "label_drop": -34,
    },
    {
        "axis": "imu",
        "title": "IMU noise",
        "arms": [
            ("jose", "JOSE (Ours)"), ("joint_only", "Joint-only"), ("imu_clean", "Distill."),
            ("set", "SET"), ("imu_dr", "Distill."), ("set_imu_dr", "SET"),
        ],
        "split": 4,
        "scales": (1.0, 2.0, 4.0),
        "track_lim": (0, 0.78),
        "label_drop": -27,
    },
)

#: One ramp shared by both rows, indexed by the level itself rather than by the
#: level's position within a row. The encoder axis has two levels and the IMU
#: axis three; keyed by position, "2x" would have been the ramp's dark end in one
#: row and its middle in the other, which is what the two source figures did.
#: Keyed by value, a shade means the same multiple everywhere in the figure.
LEVEL_COLOURS = {1.0: "#CDE8C8", 2.0: "#5FA463", 4.0: "#1B5E20"}
LEVEL_EDGE = "#1B5E20"
LEVEL_LABELS = {1.0: "nominal", 2.0: "$2\\times$", 4.0: "$4\\times$"}
#: A wash behind the randomized block. The block needs to be visible at a glance
#: and the figure has no vertical room to spend, so it is marked by ground rather
#: than by a suffix on all eight ticks or a caption-sized note above them.
BLOCK_TINT = "#EFEFE9"

BLOCK_GAP = 0.55
FIGURE_SIZE = (3.4, 3.98)


SURVIVAL_LIM = (0, 112)


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
        if metrics.get("track_error_norm") is not None:
            bucket["track"].append(metrics["track_error_norm"])
        if metrics.get("grid_survival_rate") is not None:
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


def panel(axis, row, sweeps, key, ylabel, ylim, show_labels, show_header) -> None:
    arms, scales = row["arms"], row["scales"]
    positions = np.arange(len(arms), dtype=float)
    positions[row["split"]:] += BLOCK_GAP
    # Bars fill the slot with a hair of air between groups; with three levels in
    # 1.5 in there is no room for the wider gap the single-panel figures used.
    width = 0.86 / len(scales)

    for offset, scale in enumerate(scales):
        for slot, (method, _label) in enumerate(arms):
            measured = stat(sweeps, method, scale, key)
            x = positions[slot] + (offset - (len(scales) - 1) / 2.0) * width
            if measured is None:
                # Nothing drawn and nothing implied: an absent arm must not read
                # as a zero-height bar.
                axis.text(x, ylim[0] + 0.05 * (ylim[1] - ylim[0]), "n/a", ha="center",
                          fontsize=4.0, color="#9A9A9A", rotation=90)
                continue
            centre, error = measured
            axis.bar(x, centre, width, color=LEVEL_COLOURS[scale],
                     edgecolor=LEVEL_EDGE, linewidth=0.4, zorder=3)
            axis.errorbar(x, centre, yerr=error, color="#1A1A1A", elinewidth=0.4,
                          capsize=1.0, capthick=0.4, zorder=4)

    axis.set_ylim(*ylim)
    axis.set_ylabel(ylabel, fontsize=6.3, labelpad=1.5)
    axis.tick_params(axis="y", labelsize=5.5, length=1.8, pad=0.8)
    axis.set_xlim(positions[0] - 0.7, positions[-1] + 0.7)
    axis.set_xticks(positions)
    if show_labels:
        axis.set_xticklabels([label for _key, label in arms], rotation=90, fontsize=5.2)
    else:
        axis.set_xticklabels([])
    axis.tick_params(axis="x", length=0, pad=1.5)
    axis.grid(axis="x", visible=False)

    # Mark the two blocks under the axis, and shade the randomized one. Naming
    # both is what makes the split legible: marking only the right block left
    # the reader to infer what the left one was. A legend cannot do this job --
    # a legend maps a visual channel to a meaning, and the block is a position
    # on the axis, not a property of any bar.
    split = row["split"]
    axis.axvspan(positions[split] - 0.5, positions[-1] + 0.7,
                 facecolor=BLOCK_TINT, edgecolor="none", zorder=0)
    if show_labels:
        axis.annotate(
            "$+$ randomization",
            xy=((positions[split] + positions[-1]) / 2.0, 0),
            xytext=(0, row["label_drop"]), textcoords="offset points",
            xycoords=("data", "axes fraction"),
            ha="center", va="top", fontsize=5.2, color="#4D4D4D",
            annotation_clip=False,
        )

    # JOSE is this paper's method; bolding its tick makes it findable in a grid
    # of four panels without adding a colour that would collide with the ramp.
    for label in axis.get_xticklabels():
        if label.get_text().startswith("JOSE"):
            label.set_fontweight("bold")


def plot(out: str, dpi: int) -> None:
    figure, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)

    for r, row in enumerate(ROWS):
        sweeps = read_axis(row["axis"])
        for c, (key, ylabel, ylim) in enumerate((
            ("survival", "Survival (%)", SURVIVAL_LIM),
            ("track", "Command RMSE", row["track_lim"]),
        )):
            axis = axes[r][c]
            panel(axis, row, sweeps, key, ylabel, ylim, show_labels=True, show_header=(c == 0))
            # The title names the degraded sensor and the y-label names the
            # metric. Carrying both in the title needs 6.2 pt to fit the panel,
            # which reads as a second typographic system beside the other two
            # results figures; the size comes from plot_window.py so it cannot
            # drift from them.
            axis.set_title(f"({'abcd'[r * 2 + c]}) {row['title']}",
                           fontsize=shared.TITLE_SIZE, pad=shared.TITLE_PAD,
                           loc="left")

        # One key per row: the rows ramp over different level sets, and a shared
        # key would list a level that half the figure never measured.
        axes[r][1].legend(
            handles=[Patch(facecolor=LEVEL_COLOURS[s], edgecolor=LEVEL_EDGE,
                           linewidth=0.4, label=LEVEL_LABELS[s]) for s in row["scales"]],
            loc="upper right", fontsize=5.0, frameon=False, handlelength=0.9,
            handleheight=0.7, borderpad=0.1, labelspacing=0.25, borderaxespad=0.2,
        )

    figure.subplots_adjust(left=0.092, right=0.999, top=0.958, bottom=0.163,
                           wspace=0.26, hspace=0.70)
    figure.savefig(out, dpi=dpi)
    plt.close(figure)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()
    shared.style()
    plot(os.path.join(args.outdir, "fig_degradation.png"), args.dpi)


if __name__ == "__main__":
    main()
