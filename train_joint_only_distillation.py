"""Convenience entry point for the joint-only DAgger baseline."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


if "--method" not in sys.argv:
    sys.argv[1:1] = ["--method", "joint_only"]
runpy.run_path(str(Path(__file__).with_name("train_history_student.py")), run_name="__main__")
