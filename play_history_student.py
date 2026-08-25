"""Play/video either 21-frame distillation baseline from one checkpoint."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="JOSE joint-only/IMU history-student play")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default=None)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--video", action="store_true")
parser.add_argument("--video-length", type=int, default=600)
parser.add_argument("--video-dir", default="logs/jose_g1/videos/history_student")
parser.add_argument("--real-time", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pathlib import Path
import time

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg
import isaaclab_tasks  # noqa: F401

from jose.distillation.history import HistoryMLPStudent, ObservationHistory, build_imu_frame, build_joint_frame
from jose.distillation.imu import IMUObservationSpec
from jose.estimator.models import RunningNormalizer
from jose.schema import SCHEMA_VERSION
from jose.tools.rollout_diagnostics import RolloutDiagnostics, unwrap_env_with_robot


def main() -> None:
    checkpoint = torch.load(Path(args_cli.checkpoint).resolve(), map_location=args_cli.device, weights_only=True)
    method = checkpoint.get("method")
    if checkpoint.get("jose_schema_version") != SCHEMA_VERSION or method not in ("joint_only", "imu"):
        raise ValueError("Checkpoint is not a JOSE joint-only or IMU history student")
    if checkpoint.get("explicit_linear_velocity") is not False:
        raise ValueError("History checkpoint does not satisfy the deploy observation contract")
    task = args_cli.task or checkpoint["task"]
    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=args_cli.num_envs)
    raw_env = gym.make(task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    core = unwrap_env_with_robot(raw_env)
    if core is None or not hasattr(core, "get_distillation_sensor_state"):
        raise RuntimeError("Task does not implement the JOSE distillation sensor contract")
    if args_cli.video:
        raw_env = gym.wrappers.RecordVideo(
            raw_env, video_folder=args_cli.video_dir, step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length, disable_logger=True,
        )
    config = checkpoint["model_config"]
    student = HistoryMLPStudent(
        config["frame_dim"], config["action_dim"], config["window"], tuple(config["hidden_dims"])
    ).to(core.device)
    student.load_state_dict(checkpoint["model_state_dict"])
    student.eval()
    observation_normalizer = RunningNormalizer(config["input_dim"], core.device)
    observation_normalizer.load_state_dict(checkpoint["observation_normalizer"])
    action_normalizer = RunningNormalizer(config["action_dim"], core.device, clip=10.0)
    action_normalizer.load_state_dict(checkpoint["action_normalizer"])
    history = ObservationHistory(args_cli.num_envs, config["window"], config["frame_dim"], core.device)
    imu_spec = IMUObservationSpec()
    diagnostics = RolloutDiagnostics(core, 0, max_steps=max(args_cli.steps, args_cli.video_length))
    raw_env.reset()
    step_limit = args_cli.video_length if args_cli.video else args_cli.steps
    try:
        for _ in range(step_limit):
            if not simulation_app.is_running():
                break
            started = time.monotonic()
            with torch.inference_mode():
                state = core.get_distillation_sensor_state()
                if method == "joint_only":
                    frame = build_joint_frame(
                        state["joint_position"], state["joint_velocity"], state["previous_action"]
                    )
                else:
                    observation = imu_spec.observe(
                        state["quaternion_wxyz"], state["angular_velocity"], timestamp_s=0.0
                    )
                    if not observation.valid:
                        raise RuntimeError(f"IMU inference fault: {observation.fault.value}")
                    frame = build_imu_frame(
                        state["joint_position"], state["joint_velocity"], state["previous_action"],
                        observation.angular_velocity, observation.projected_gravity,
                    )
                flattened = history.push(frame)
                action = action_normalizer.denormalize(student(observation_normalizer.normalize(flattened)))
                _, _, terminated, truncated, _ = raw_env.step(action)
                done = (terminated | truncated).flatten()
                if done.any():
                    history.reset(done.nonzero(as_tuple=False).squeeze(-1))
            diagnostics.record(action)
            if args_cli.real_time:
                remaining = float(core.step_dt) - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        artifacts = diagnostics.save(args_cli.video_dir, float(core.step_dt), f"{method}_diagnostics")
        if artifacts:
            print(f"Diagnostics: {artifacts}")
        raw_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
