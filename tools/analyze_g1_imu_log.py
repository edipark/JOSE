"""Replay and validate recorded or synthetic Unitree G1 IMU JSONL logs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

import torch

try:
    from ..distillation.imu import IMUObservationSpec
except ImportError:
    from jose.distillation.imu import IMUObservationSpec


def synthetic_rows(rate_hz: float = 100.0) -> list[dict]:
    rows = []
    timestamp = 0.0
    for segment, seconds, pitch_amplitude, gyro_scale in (
        ("standing", 2.0, 0.01, 0.02),
        ("walking", 4.0, 0.10, 0.8),
        ("jump_landing", 3.0, 0.28, 2.5),
    ):
        count = round(seconds * rate_hz)
        for index in range(count):
            phase = 2.0 * math.pi * index / max(count - 1, 1)
            pitch = pitch_amplitude * math.sin(phase * (2.0 if segment == "walking" else 1.0))
            rows.append(
                {
                    "segment": segment,
                    "imu_state": {
                        "quaternion": [math.cos(pitch / 2.0), 0.0, math.sin(pitch / 2.0), 0.0],
                        "gyroscope": [0.0, gyro_scale * math.cos(phase), 0.0],
                        "accelerometer": [0.0, 0.0, -9.81],
                        "timestamp_s": timestamp,
                    },
                }
            )
            timestamp += 1.0 / rate_hz
    return rows


def load_rows(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error
    if not rows:
        raise ValueError("IMU log is empty")
    return rows


def analyze(rows: list[dict], spec: IMUObservationSpec | None = None) -> dict:
    spec = spec or IMUObservationSpec(stale_after_s=1.0)
    by_segment: dict[str, list[dict]] = {}
    previous_timestamp = None
    previous_gravity = None
    faults = []
    for index, row in enumerate(rows):
        imu = row.get("imu_state", row)
        quaternion = torch.as_tensor(imu.get("quaternion", imu.get("quat")), dtype=torch.float32)
        timestamp_value = imu.get(
            "timestamp_s", imu.get("timestamp", imu.get("time", row.get("timestamp_s", row.get("timestamp"))))
        )
        if timestamp_value is None:
            raise ValueError(f"IMU log row {index} has no monotonic timestamp")
        timestamp = float(timestamp_value)
        observation = spec.from_low_state(row, timestamp_s=timestamp)
        segment = str(row.get("segment", "unlabeled"))
        gravity_delta = 0.0 if previous_gravity is None else float((observation.projected_gravity - previous_gravity).norm())
        dt = None if previous_timestamp is None else timestamp - previous_timestamp
        sample = {
            "quaternion_norm_error": abs(float(quaternion.norm()) - 1.0),
            "gyro_norm": float(observation.angular_velocity.norm()),
            "gravity_norm_error": abs(float(observation.projected_gravity.norm()) - 1.0),
            "gravity_delta": gravity_delta,
            "dt": dt,
        }
        by_segment.setdefault(segment, []).append(sample)
        if not observation.valid:
            faults.append({"index": index, "fault": observation.fault.value})
        if dt is not None and dt <= 0:
            faults.append({"index": index, "fault": "non_monotonic_timestamp"})
        previous_timestamp = timestamp
        previous_gravity = observation.projected_gravity

    summaries = {}
    all_dt = []
    for segment, samples in by_segment.items():
        dt_values = [sample["dt"] for sample in samples if sample["dt"] is not None]
        all_dt.extend(dt_values)
        summaries[segment] = {
            "samples": len(samples),
            "max_quaternion_norm_error": max(sample["quaternion_norm_error"] for sample in samples),
            "max_gyro_rad_s": max(sample["gyro_norm"] for sample in samples),
            "max_projected_gravity_norm_error": max(sample["gravity_norm_error"] for sample in samples),
            "max_projected_gravity_step": max(sample["gravity_delta"] for sample in samples),
            "median_period_s": statistics.median(dt_values) if dt_values else None,
            "max_period_s": max(dt_values) if dt_values else None,
        }
    required = {"standing", "walking", "jump_landing"}
    return {
        "format": "unitree_g1_imu_jsonl_v1",
        "samples": len(rows),
        "segments": summaries,
        "faults": faults,
        "required_segments_present": sorted(required.intersection(summaries)),
        "acceptance": {
            "parser_valid": not faults,
            "real_log": False,
            "real_world_deployment_ready": False,
            "reason": "A real G1 log is required even when the synthetic contract passes",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a Unitree G1 LowState IMU JSONL log")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input")
    source.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--mark-real", action="store_true", help="Declare that --input came from a physical G1")
    args = parser.parse_args()
    rows = synthetic_rows() if args.synthetic else load_rows(args.input)
    report = analyze(rows)
    if args.mark_real:
        required = {"standing", "walking", "jump_landing"}
        present = set(report["segments"])
        ready = report["acceptance"]["parser_valid"] and required <= present
        report["acceptance"].update(
            real_log=True,
            real_world_deployment_ready=ready,
            reason="accepted" if ready else "real log must contain standing, walking, and jump_landing labels",
        )
    text = json.dumps(report, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
