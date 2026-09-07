"""Figure 1: the JOSE framework, in three phases.

Writes fig_framework.png at two-column width.

What the panels have to carry, and why each is there. Phase 1 is the setup the
paper does not change -- a teacher trained on the full observation, by whatever
algorithm that teacher's task calls for -- and it is drawn so the reader can see
that o^p enters the teacher as an input, not as a target. Phase 2 is the
contribution: the teacher is frozen, and a regression onto o^p is fitted from a
window of encoder history, with aggregation closing the distribution gap. Phase 3
is the claim: the same frozen teacher runs, unmodified, on a robot whose inertial
unit is never read.

The three phases share one visual grammar so the eye can follow one object
across them. The teacher is the same box in all three, shaded once it freezes;
o^p is the same colour wherever it appears, hollow once it is an estimate rather
than a measurement; and the arrow into the teacher enters at the same point in
every panel, so what changes between phases is visibly only where that arrow
comes from.

Usage:
    python figures/plot_framework.py [--outdir figures]
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import plot_window as shared

# Two columns of IEEEtran with the standard gutter. The axes use hundredths of
# an inch as their unit in BOTH directions, so a box drawn 60x36 really is
# 0.60in x 0.36in on the page: with x and y on different scales a "square" box
# comes out as a sliver, which is what the first draft did.
W_IN, H_IN = 7.16, 2.45
FIGURE_SIZE = (W_IN, H_IN)
XMAX, YMAX = W_IN * 100, H_IN * 100

INK = "#1A1A1A"
MUTED = "#6E6E6E"
RULE = "#B8B8B8"
#: o^d, the encoder half of the observation: measured in every phase.
DEPLOY = "#0072B2"
#: o^p, the privileged half: measured in phase 1, estimated from phase 2 on.
PRIV = "#7E52AD"
#: The estimator, the one part of the pipeline this paper fits.
EST = "#E8AE00"
#: The teacher, shaded once frozen.
LIVE = "#FFFFFF"
FROZEN = "#EDEDED"


def box(ax, x, y, w, h, label, *, face=LIVE, edge=INK, lw=0.8, fs=7.0,
        tc=None, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, linewidth=lw, edgecolor=edge,
                                facecolor=face, zorder=3,
                                boxstyle="round,pad=1.2,rounding_size=3"))
    if dashed:
        ax.add_patch(FancyBboxPatch((x, y), w, h, linewidth=lw, edgecolor=edge,
                                    facecolor="none", zorder=5,
                                    linestyle=(0, (2.6, 1.6)),
                                    boxstyle="round,pad=1.2,rounding_size=3"))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs,
            color=tc or INK, zorder=6, linespacing=1.35)


def arrow(ax, start, end, *, color=INK, lw=0.8, rad=0.0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=7,
                                 linewidth=lw, color=color, zorder=2,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=0.5, shrinkB=0.5))


def phase_title(ax, x, number, title):
    ax.text(x, YMAX - 18, f"{number}  {title}", fontsize=7.8, color=INK,
            fontweight="bold", ha="left", va="center")


def panel_one(ax, x0):
    """Teacher training: the observation is complete, so nothing is estimated."""
    phase_title(ax, x0, "1", "Train the teacher")
    box(ax, x0, 118, 62, 34, "$o^d_t$  encoders", edge=DEPLOY, fs=6.8)
    box(ax, x0, 62, 62, 34, "$o^p_t$  privileged", edge=PRIV, fs=6.8)
    box(ax, x0 + 96, 84, 44, 62, "$\\pi_T$", fs=9)
    arrow(ax, (x0 + 62, 135), (x0 + 96, 124), color=DEPLOY)
    arrow(ax, (x0 + 62, 79), (x0 + 96, 106), color=PRIV)
    arrow(ax, (x0 + 140, 115), (x0 + 176, 115))
    ax.text(x0 + 158, 126, "$a_t$", fontsize=7.2, ha="center", color=INK)
    ax.text(x0, 34, "AMP (walk, dance, jump)\nor PPO (locomotion)", fontsize=6.4,
            color=MUTED, ha="left", va="center", linespacing=1.4)


def panel_two(ax, x0):
    """Estimator fitting: the teacher is frozen and only E_theta is trained."""
    phase_title(ax, x0, "2", "Fit the estimator")
    box(ax, x0, 118, 74, 34, "$h_t$: last $W$\nframes of $o^d$", edge=DEPLOY, fs=6.4)
    box(ax, x0 + 96, 118, 38, 34, "$E_\\theta$", face=EST, fs=9)
    arrow(ax, (x0 + 74, 135), (x0 + 96, 135), color=DEPLOY)
    arrow(ax, (x0 + 134, 135), (x0 + 158, 135), color=PRIV)
    ax.text(x0 + 162, 135, "$\\hat{o}^p_t$", fontsize=7.6, color=PRIV,
            ha="left", va="center")
    # The target is simulator state, not a teacher output: say so under the
    # arrow rather than beside it, where panel 3 begins.
    ax.text(x0 + 146, 111, "MSE vs.\n$o^p_t$ (simulator)", fontsize=6.2,
            color=MUTED, ha="center", va="top", linespacing=1.35)
    box(ax, x0 + 96, 44, 38, 38, "$\\pi_T$\nfrozen", face=FROZEN, fs=6.6)
    # Aggregation: the estimator drives the rollout that becomes its next data.
    arrow(ax, (x0 + 115, 118), (x0 + 115, 86), color=MUTED, lw=0.7)
    arrow(ax, (x0 + 96, 63), (x0 + 37, 63), color=MUTED, lw=0.7)
    arrow(ax, (x0 + 37, 63), (x0 + 37, 116), color=MUTED, lw=0.7)
    ax.text(x0 + 66, 30, "aggregate, $K$ rounds", fontsize=6.4, color=MUTED,
            ha="center", va="center")


def panel_three(ax, x0):
    """Deployment: the same frozen teacher, with no inertial unit in the loop."""
    phase_title(ax, x0, "3", "Deploy")
    box(ax, x0, 118, 52, 34, "$o^d_t$\nencoders", edge=DEPLOY, fs=6.6)
    box(ax, x0 + 68, 118, 36, 34, "$E_\\theta$", face=EST, fs=9)
    arrow(ax, (x0 + 52, 135), (x0 + 68, 135), color=DEPLOY)
    # Hollow and dashed: a prediction where phase 1 had a measurement.
    box(ax, x0 + 120, 118, 38, 34, "$\\hat{o}^p_t$", edge=PRIV, fs=8, lw=0.9,
        dashed=True)
    arrow(ax, (x0 + 104, 135), (x0 + 120, 135), color=PRIV)
    box(ax, x0 + 176, 84, 42, 62, "$\\pi_T$\nfrozen", face=FROZEN, fs=7.0)
    arrow(ax, (x0 + 158, 132), (x0 + 176, 124), color=PRIV)
    arrow(ax, (x0 + 26, 118), (x0 + 176, 102), color=DEPLOY, rad=-0.24)
    arrow(ax, (x0 + 218, 115), (x0 + 248, 115))
    ax.text(x0 + 233, 126, "$a_t$", fontsize=7.2, ha="center", color=INK)
    box(ax, x0 + 26, 22, 168, 26, "no inertial unit is read", face="#F4EFFA",
        edge=PRIV, lw=0.7, fs=7.0, tc=PRIV)


def plot(out: str, dpi: int) -> None:
    figure, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.set_xlim(0, XMAX)
    ax.set_ylim(0, YMAX)
    ax.set_aspect("equal")
    ax.axis("off")

    panel_one(ax, 12)
    panel_two(ax, 212)
    panel_three(ax, 452)

    # Dividers rather than boxes: the phases are one pipeline read left to
    # right, and three framed panels would read as three separate systems.
    for x in (196, 436):
        ax.plot([x, x], [18, YMAX - 8], color=RULE, linewidth=0.5, zorder=1)

    figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
    figure.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.01)
    plt.close(figure)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()
    shared.style()
    plot(os.path.join(args.outdir, "fig_framework.png"), args.dpi)


if __name__ == "__main__":
    main()
