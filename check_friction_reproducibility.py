"""Compare forward/reverse friction workers for method-order dependence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


DEFAULT_TOLERANCES = {
    "track_error_norm": 1.0e-3,
    "grid_survival_rate": 0.0,
    "feet_slide_penalty": 1.0e-3,
    "command_numeric": 5.0e-3,
}


def _read(path: Path) -> dict[tuple[str, int | None], dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    keyed = {(row["method"], row.get("checkpoint_seed")): row for row in rows}
    if len(keyed) != len(rows):
        raise RuntimeError(f"Duplicate method/checkpoint rows in {path}")
    return keyed


def _numeric_differences(left, right, path="") -> list[tuple[str, float]]:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise RuntimeError(f"Mismatched keys at {path or '<root>'}")
        differences = []
        for key in sorted(left):
            differences.extend(_numeric_differences(left[key], right[key], f"{path}.{key}"))
        return differences
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise RuntimeError(f"Mismatched list length at {path}")
        differences = []
        for index, (a, b) in enumerate(zip(left, right)):
            differences.extend(_numeric_differences(a, b, f"{path}[{index}]"))
        return differences
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        difference = abs(float(left) - float(right))
        if not math.isfinite(difference):
            raise RuntimeError(f"Non-finite value at {path}")
        return [(path, difference)]
    if left != right:
        raise RuntimeError(f"Mismatched value at {path}: {left!r} != {right!r}")
    return []


def compare(left_path: Path, right_path: Path, tolerances=None) -> dict:
    tolerances = dict(DEFAULT_TOLERANCES if tolerances is None else tolerances)
    left = _read(left_path)
    right = _read(right_path)
    if set(left) != set(right):
        raise RuntimeError("Forward and reverse workers contain different method/checkpoint cells")

    methods = {}
    passed = True
    for key in sorted(left, key=lambda item: (item[0], -1 if item[1] is None else item[1])):
        a = left[key]["metrics"]
        b = right[key]["metrics"]
        scalar = {
            name: abs(float(a[name]) - float(b[name]))
            for name in ("track_error_norm", "grid_survival_rate", "feet_slide_penalty")
        }
        command_differences = _numeric_differences(
            a["command_tracking"], b["command_tracking"], "command_tracking"
        )
        worst_path, worst_difference = max(command_differences, key=lambda item: item[1])
        checks = {
            **{name: value <= tolerances[name] for name, value in scalar.items()},
            "command_numeric": worst_difference <= tolerances["command_numeric"],
        }
        method_passed = all(checks.values())
        passed &= method_passed
        label = f"{key[0]}|{key[1] if key[1] is not None else 'teacher'}"
        methods[label] = {
            "passed": method_passed,
            "differences": {**scalar, "command_numeric": worst_difference},
            "worst_command_path": worst_path,
            "checks": checks,
        }
    return {"passed": passed, "tolerances": tolerances, "methods": methods}


METRIC_LABELS = {
    "track_error_norm": "Command RMSE",
    "grid_survival_rate": "Survival",
    "feet_slide_penalty": "Feet slide",
    "command_numeric": "Worst per-command value",
}

METRIC_ORDER = ("grid_survival_rate", "track_error_norm", "feet_slide_penalty", "command_numeric")


def render_report(result: dict, forward: Path, reverse: Path) -> str:
    verdict = "PASS" if result["passed"] else "FAIL"
    lines = [
        "# Friction sweep method-order gate",
        "",
        f"**{verdict}.** The same cells were evaluated twice in one process, once with the",
        "methods in catalog order and once reversed. Agreement means a method's numbers do",
        "not depend on which methods ran before it.",
        "",
        f"- forward: `{forward.name}`",
        f"- reverse: `{reverse.name}`",
        "",
        "## Absolute forward-reverse difference",
        "",
        "| method | " + " | ".join(METRIC_LABELS[name] for name in METRIC_ORDER) + " | verdict |",
        "|---" * (len(METRIC_ORDER) + 2) + "|",
    ]
    for label, record in result["methods"].items():
        cells = [f"{record['differences'][name]:.2e}" for name in METRIC_ORDER]
        name = label.replace("|", " / ")
        lines.append(
            f"| {name} | " + " | ".join(cells) + f" | {'pass' if record['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Tolerances",
            "",
            "| quantity | tolerance |",
            "|---|---|",
        ]
    )
    for name in METRIC_ORDER:
        lines.append(f"| {METRIC_LABELS[name]} | {result['tolerances'][name]:g} |")
    lines.extend(
        [
            "",
            "Survival is required to match exactly; the others carry a tolerance because",
            "floating-point reduction order on the GPU is not fixed across call order.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("forward")
    parser.add_argument("reverse")
    parser.add_argument("--out")
    parser.add_argument("--report")
    args = parser.parse_args()
    forward = Path(args.forward)
    reverse = Path(args.reverse)
    result = compare(forward, reverse)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
    if args.report:
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(render_report(result, forward, reverse), encoding="utf-8")
    print(payload, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
