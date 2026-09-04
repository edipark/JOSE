"""Markdown report for the sensor-degradation sweeps.

The ablation studies get their report from ``reporting.py``, which reads a
``results.jsonl`` whose rows are training runs. The robustness sweeps write a
different row -- one evaluation of one method at one noise scale and seed -- so
they had no report at all, and the only place their numbers appeared was inside
the figures. This writes them out so a number in the paper can be checked against
a table rather than read off a bar.

Usage:
    python -m jose.report_robustness [--dir logs/jose_g1/robustness]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

#: Column key -> (header, format). Survival is scaled to a percentage here for
#: the same reason the figures do it: a rate between 0 and 1 reads as a
#: probability, and every other survival number in the paper is a percentage.
COLUMNS = (
    ("survival", "Surv. (%)", "{:.1f}"),
    ("track", "Track RMSE", "{:.4f}"),
    ("vx", "vx", "{:.4f}"),
    ("vy", "vy", "{:.4f}"),
    ("wz", "wz", "{:.4f}"),
)

#: Reading order: ours first, then the baselines, then each hardened arm.
METHOD_ORDER = (
    "teacher",
    "jose", "joint_only", "imu_clean", "imu_dr", "set",
    "jose_enc", "joint_only_enc", "imu_clean_enc", "set_enc", "set_imu_dr",
)


def read(path: Path):
    rows = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    seeds = defaultdict(lambda: defaultdict(set))
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        metrics = record.get("metrics", {})
        bucket = rows[record["method"]][float(record["scale"])]
        seeds[record["method"]][float(record["scale"])].add(record.get("seed"))
        for key, source in (("survival", "grid_survival_rate"), ("track", "track_error_norm"),
                            ("vx", "track_vx_rmse"), ("vy", "track_vy_rmse"),
                            ("wz", "track_yaw_rmse")):
            if source in metrics:
                value = metrics[source]
                bucket[key].append(100.0 * value if key == "survival" else value)
    return rows, seeds


def cell(values, fmt: str) -> str:
    if not values:
        return "--"
    if len(values) == 1:
        return fmt.format(values[0])
    return f"{fmt.format(st.mean(values))} ± {fmt.format(st.stdev(values))}"


def render(axis: str, rows, seeds) -> list[str]:
    scales = sorted({scale for method in rows for scale in rows[method]})
    out = [f"## {axis} axis", ""]
    ordered = [m for m in METHOD_ORDER if m in rows] + sorted(set(rows) - set(METHOD_ORDER))
    missing = [m for m in METHOD_ORDER if m not in rows]
    for key, header, fmt in COLUMNS:
        if not any(rows[m][s].get(key) for m in rows for s in rows[m]):
            continue
        out += [f"### {header}", "",
                "| method | " + " | ".join(f"{s:g}x" for s in scales) + " |",
                "|---|" + "---|" * len(scales)]
        for method in ordered:
            cells = [cell(rows[method][scale].get(key, []), fmt) for scale in scales]
            out.append(f"| {method} | " + " | ".join(cells) + " |")
        out.append("")
    counts = sorted({len(seeds[m][s]) for m in rows for s in rows[m]})
    out += [f"Seeds per cell: {counts}.", ""]
    if missing:
        out += [f"Not present in this file: {', '.join(missing)}.", ""]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="logs/jose_g1/robustness")
    parser.add_argument("--out", default=None, help="Default: <dir>/report.md")
    args = parser.parse_args()

    directory = Path(args.dir).resolve()
    report = ["# Sensor degradation sweeps", "",
              "Means over seeds, ± the sample standard deviation. `Surv.` is the",
              "command-grid survival rate; `Track RMSE` is the norm over the three",
              "command components, which are also broken out. A scale multiplies that",
              "axis's nominal noise model; `0x` is the clean condition.", ""]
    for axis in ("encoder", "imu"):
        path = directory / f"{axis}_axis.jsonl"
        if not path.is_file():
            report += [f"## {axis} axis", "", "_No data file._", ""]
            continue
        rows, seeds = read(path)
        report += render(axis, rows, seeds)

    out = Path(args.out) if args.out else directory / "report.md"
    out.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
