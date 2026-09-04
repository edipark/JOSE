"""Aggregation-round figure for the JOSE paper.

Writes fig_dagger.png: survival and estimation error against the number of
DAgger rounds the estimator was trained through, one curve per task.

The claim this figure has to support is that the on-policy protocol is doing the
work, not the architecture. Round 0 is a warm start on teacher-driven data alone
-- a plain behaviour-cloned estimator -- so the r00 point is what JOSE would be
without aggregation, and on every task it is a policy that falls over.

Two things about the shape of the data are deliberate rather than incidental:

* **The curves are not monotone on walk.** Intermediate rounds there are
  bimodal across seeds -- at r04 the three seeds sit at 64, 86 and 82 % while at
  r05 all three sit at 0 -- so the range over seeds is drawn rather than a
  standard deviation, which would imply a symmetric spread the data do not have.
  This is also why ``train_state_estimator.py`` selects the best round on
  closed-loop behaviour instead of taking the last one.
* **Locomotion is sampled at four rounds, the AMP tasks at eleven.** The
  locomotion sweep ran {0, 2, 5, 10} because a full sweep is eleven arms times
  three seeds and the four points already resolve the shape. Markers show which
  rounds were measured.

Typography, colours and the seed-spread convention are imported from
``plot_window.py`` so the two figures cannot drift apart.

Usage:
    python figures/plot_dagger.py [--outdir figures] [--spread minmax|std]
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

import plot_window as shared


#: The unsuffixed arm is the full schedule: the ablation names rounds 0-9
#: explicitly and leaves the default-length run (``--dagger_rounds 10``) bare.
FULL_ROUNDS = 10

FIGURE_SIZE = (3.4, 1.72)
ROUND_TICKS = [0, 2, 4, 6, 8, 10]


def latest_dagger_study(task: str) -> str | None:
    """Newest aggregation study for one task, or None if it has not been run."""
    pattern = os.path.join(
        shared.ABLATION_ROOT, "*", task, "studies", "dagger", "*", "results.jsonl"
    )
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def read_sweep(task: str) -> dict[int, dict[str, list[float]]]:
    """Per-seed metrics keyed by aggregation round, for one task.

    Returns the same structure ``plot_window.series`` consumes, so the spread
    handling is shared rather than re-implemented with a different convention.
    """
    per_round: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    results = latest_dagger_study(task)
    if results is None:
        return per_round
    for line in open(results):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("status") != "ok":
            continue
        name = record.get("experiment", "")
        if name == "lstm_w25_all":
            rounds = FULL_ROUNDS
        else:
            match = re.fullmatch(r"lstm_w25_all_r(\d+)", name)
            if match is None:
                continue  # teacher_gt and any non-sweep row
            rounds = int(match.group(1))
        metrics = record.get("metrics", {})
        if "death_rate" in metrics:
            per_round[rounds]["survival"].append(100.0 - metrics["death_rate"])
        if metrics.get("rmse") is not None:
            per_round[rounds]["rmse"].append(metrics["rmse"])
    return per_round


def round_axis(axis) -> None:
    axis.set_xticks(ROUND_TICKS)
    axis.set_xlabel("Aggregation round")
    axis.set_xlim(-0.6, 10.6)


def plot_dagger(sweeps, spread: str, out: str, dpi: int) -> None:
    figure, (left, right) = plt.subplots(1, 2, figsize=FIGURE_SIZE)

    for axis in (left, right):
        axis.spines["right"].set_visible(False)

    for task, label, colour in shared.available(sweeps, "survival"):
        rounds, centre, error = shared.series(sweeps[task], "survival", spread)
        left.errorbar(
            rounds, centre, yerr=error, color=colour, label=label, marker="o",
            capsize=1.6, elinewidth=0.5, capthick=0.5, zorder=3,
        )
    left.set_ylabel("Survival (%)", labelpad=1.5)
    left.set_ylim(-8, 108)
    left.set_yticks([0, 25, 50, 75, 100])
    left.text(0.0, 1.02, "(a)", transform=left.transAxes, va="bottom")

    for task, label, colour in shared.available(sweeps, "rmse"):
        rounds, centre, error = shared.series(sweeps[task], "rmse", spread)
        right.errorbar(
            rounds, centre, yerr=error, color=colour, marker="o",
            capsize=1.6, elinewidth=0.5, capthick=0.5, zorder=3,
        )
    # Log scale: r00 is an order of magnitude above the converged rounds on
    # every task, so a linear axis would flatten everything after r01 into the
    # baseline and hide where each task actually settles.
    right.set_yscale("log")
    right.set_ylabel("Estimation RMSE", labelpad=1.5)
    ticks = [0.01, 0.02, 0.05, 0.1, 0.2]
    right.set_yticks(ticks)
    right.set_yticklabels([("%g" % t) for t in ticks])
    right.set_ylim(0.008, 0.3)
    right.text(0.0, 1.02, "(b)", transform=right.transAxes, va="bottom")

    for axis in (left, right):
        # r00 is the no-aggregation baseline, so it is worth marking as a
        # reference the rest of the curve is read against.
        axis.axvline(0, color=shared.RULE_COLOUR, linewidth=0.5,
                     linestyle=(0, (1, 2.5)), zorder=1)
        round_axis(axis)

    handles, labels = left.get_legend_handles_labels()
    columns = min(len(handles), 4)
    figure.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
        frameon=False, handlelength=1.6, ncol=columns, columnspacing=0.9,
    )
    rows = 1 + (len(handles) - 1) // max(columns, 1)
    shared.save(figure, out, dpi, legend_rows=rows * 2.7, w_pad=1.8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--spread", choices=("minmax", "std"), default="minmax")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    shared.style()
    sweeps = {task: read_sweep(task) for task, _, _ in shared.TASKS}
    plot_dagger(sweeps, args.spread, os.path.join(args.outdir, "fig_dagger.png"), args.dpi)

    for task, label, _ in shared.TASKS:
        if not sweeps[task]:
            print(f"{label:11s} (no aggregation study)")
            continue
        rounds, survival, _ = shared.series(sweeps[task], "survival", args.spread)
        _, rmse, _ = shared.series(sweeps[task], "rmse", args.spread)
        print(f"{label:11s} " + "  ".join(
            f"r{int(r):02d} surv {s:5.1f} rmse {m:.4f}" for r, s, m in zip(rounds, survival, rmse)
        ))


if __name__ == "__main__":
    main()
