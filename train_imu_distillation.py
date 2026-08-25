"""Convenience entry point for the deployable IMU DAgger baseline."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


if "--method" not in sys.argv:
    sys.argv[1:1] = ["--method", "imu"]
runpy.run_path(str(Path(__file__).with_name("train_history_student.py")), run_name="__main__")
