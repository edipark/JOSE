"""LSTM history-window ablation, independently configurable with --windows."""

from __future__ import annotations

import argparse
import sys

try:
    from .ablation_runner import main
    from .ablation_catalog import window_experiments
except ImportError:
    from ablation_runner import main
    from ablation_catalog import window_experiments


DEFAULT_WINDOWS = (1, 5, 10, 25, 50)


def parse_windows(values: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    if not values or any(value <= 0 for value in values):
        raise ValueError("--windows requires one or more positive integers")
    return tuple(sorted(set(values)))


def extract_windows(argv: list[str]) -> tuple[tuple[int, ...], list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--windows", nargs="+", type=int, default=list(DEFAULT_WINDOWS))
    known, remaining = parser.parse_known_args(argv)
    return parse_windows(known.windows), remaining


if __name__ == "__main__":
    windows, remaining = extract_windows(sys.argv[1:])
    sys.argv = [sys.argv[0], *remaining]
    main(window_experiments(windows), "window")
