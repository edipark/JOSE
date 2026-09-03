"""Replay, validate, and optionally record a G1 NPZ motion in Isaac Sim."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay a JOSE G1 reference motion")
parser.add_argument(
    "--file",
    "--motion",
    dest="motion",
    default=str(Path(__file__).parents[1] / "motions" / "G1_walk.npz"),
    help="Path to the G1 motion NPZ file (--motion is kept as a compatibility alias)",
)
parser.add_argument("--record-output", default=None)
parser.add_argument("--loops", type=int, default=0, help="0 repeats until the app closes")
parser.add_argument("--speed", type=float, default=1.0)
parser.add_argument("--video", action="store_true")
parser.add_argument(
    "--video-length",
    type=int,
    default=None,
    help="Length in physics steps (default: one complete motion)",
)
parser.add_argument("--video-dir", default=None)
parser.add_argument("--video-width", type=int, default=1280)
parser.add_argument("--video-height", type=int, default=720)
parser.add_argument("--video-fps", type=int, default=30)
parser.add_argument(
    "--track-camera",
    action="store_true",
    help="Keep the camera centered on the reference root while replaying",
)
parser.add_argument(
    "--camera-offset",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    default=(3.2, 3.2, 2.2),
    help="World-space camera offset from the reference root when --track-camera is set",
)
parser.add_argument(
    "--camera-lookat-height",
    type=float,
    default=0.25,
    help="Vertical look-at offset from the reference root when --track-camera is set",
)
parser.add_argument(
    "--video-codec",
    default="libx264",
    choices=("libx264", "h264_nvenc"),
    help="FFmpeg encoder used by imageio. h264_nvenc is substantially faster when available.",
)
parser.add_argument("--matplotlib", action="store_true", help="Also show the motion skeleton in matplotlib")
parser.add_argument("--print-base-velocity", action="store_true")
parser.add_argument("--print-base-velocity-interval", type=int, default=30)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import ViewerCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from jose.g1_cfg import G1_JOSE_CFG
from jose.motions.motion_loader import MotionLoader
from jose.motions.record_data import MotionRecorder
from jose.schema import G1_JOINT_NAMES


def _rgb_frame(annotator):
    raw = annotator.get_data()
    frame = raw.get("data") if isinstance(raw, dict) else raw
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[-1] < 3:
        return None
    return np.ascontiguousarray(frame[..., :3])


def main():
    if args_cli.speed <= 0.0:
        raise ValueError("--speed must be positive")
    if args_cli.video_width <= 0 or args_cli.video_height <= 0 or args_cli.video_fps <= 0:
        raise ValueError("Video width, height, and FPS must be positive")
    motion = MotionLoader(args_cli.motion, args_cli.device, expected_dof_names=G1_JOINT_NAMES)
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device, render_interval=1)
    )
    video_length = args_cli.video_length
    if video_length is None:
        video_length = max(1, math.ceil(motion.duration / (sim.get_physics_dt() * args_cli.speed)))
    elif video_length <= 0:
        raise ValueError("--video-length must be positive")
    # Use the same default viewer composition as play and recorded rollouts.
    viewer_cfg = ViewerCfg()
    sim.set_camera_view(viewer_cfg.eye, viewer_cfg.lookat)
    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.robot = G1_JOSE_CFG.replace(prim_path="/World/Robot")
    scene = InteractiveScene(scene_cfg)
    # Match the world rendered by play.py's G1 AMP environments.
    spawn_ground_plane(
        prim_path="/World/ground",
        cfg=GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            )
        ),
    )
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)
    sim.reset()
    robot = scene["robot"]
    dof_ids = motion.get_dof_index(robot.joint_names)
    root_id = motion.get_body_index(["pelvis"])[0]
    env_ids = torch.tensor([0], device=args_cli.device)
    recorder = MotionRecorder(
        robot,
        robot.joint_names,
        int(round(1.0 / sim.get_physics_dt())),
        args_cli.device,
        smoothing_window=1,
    )
    if args_cli.record_output:
        recorder.start_recording()
    print(
        "\n[INFO] G1 motion replay\n"
        f"  file: {Path(args_cli.motion).resolve()}\n"
        f"  duration: {motion.duration:.3f} s ({motion.num_frames} frames)\n"
        f"  speed: {args_cli.speed:.3f}x\n"
        f"  sim dt: {sim.get_physics_dt():.6f} s\n"
        + (f"  video length: {video_length} physics steps\n" if args_cli.video else "")
    )

    if args_cli.matplotlib:
        import threading

        import matplotlib

        matplotlib.use("TkAgg")
        from jose.motions.motion_viewer import MotionViewer

        viewer = MotionViewer(args_cli.motion, render_scene=True)
        threading.Thread(target=viewer.show, daemon=True).start()

    video_writer = annotator = render_product = None
    render_product_path = None
    video_path = None
    if args_cli.video:
        import imageio.v2 as imageio
        import omni.replicator.core as rep

        video_dir = Path(args_cli.video_dir or Path(args_cli.motion).resolve().parent / "videos")
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / f"{Path(args_cli.motion).stem}.mp4"
        writer_kwargs = {"fps": args_cli.video_fps, "codec": args_cli.video_codec}
        if args_cli.video_codec == "h264_nvenc":
            # Isaac Sim's bundled imageio-ffmpeg binary exposes h264_nvenc but
            # omits the usual rate-control flags. Use the encoder defaults and
            # optionally normalize quality with the system FFmpeg afterwards.
            writer_kwargs["quality"] = None
        else:
            writer_kwargs["quality"] = 8
        video_writer = imageio.get_writer(video_path, **writer_kwargs)
        render_product = rep.create.render_product(
            "/OmniverseKit_Persp", (args_cli.video_width, args_cli.video_height)
        )
        render_product_path = render_product if isinstance(render_product, str) else render_product.path
        annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annotator.attach([render_product_path])

    loop = step = 0
    current_time = 0.0
    capture_every = max(1, round(1.0 / (sim.get_physics_dt() * args_cli.video_fps)))
    try:
        while simulation_app.is_running() and (args_cli.loops == 0 or loop < args_cli.loops):
            if args_cli.video and step >= video_length:
                break
            sample = motion.sample(1, np.asarray([current_time]))
            dof_pos, dof_vel, body_pos, body_rot, body_lin, body_ang = sample
            root = torch.cat(
                (
                    body_pos[:, root_id],
                    body_rot[:, root_id],
                    body_lin[:, root_id] * args_cli.speed,
                    body_ang[:, root_id] * args_cli.speed,
                ),
                dim=-1,
            )
            robot.write_root_link_pose_to_sim(root[:, :7], env_ids)
            robot.write_root_com_velocity_to_sim(root[:, 7:], env_ids)
            robot.write_joint_state_to_sim(
                dof_pos[:, dof_ids], dof_vel[:, dof_ids] * args_cli.speed, None, env_ids,
            )
            if args_cli.track_camera:
                root_position = body_pos[0, root_id].detach().cpu().numpy()
                camera_eye = root_position + np.asarray(args_cli.camera_offset)
                camera_lookat = root_position + np.asarray((0.0, 0.0, args_cli.camera_lookat_height))
                sim.set_camera_view(camera_eye, camera_lookat)
            scene.write_data_to_sim()
            capture_frame = video_writer is not None and step % capture_every == 0
            # Hydra/Replicator requires continuous render updates. Skipping the
            # intermediate 120 Hz updates makes the next RGB readback slower
            # because it has to catch up the render graph in one call.
            sim.step(render=True)
            scene.update(sim.get_physics_dt())
            recorder.record_frame(step)
            if capture_frame:
                frame = _rgb_frame(annotator)
                if frame is not None:
                    video_writer.append_data(frame)
            if args_cli.print_base_velocity and step % max(1, args_cli.print_base_velocity_interval) == 0:
                linear = robot.data.root_lin_vel_w[0]
                angular = robot.data.root_ang_vel_w[0]
                print(
                    f"[replay] t={current_time:.3f}s "
                    f"linear=({float(linear[0]):+.4f}, {float(linear[1]):+.4f}, {float(linear[2]):+.4f}) "
                    f"angular=({float(angular[0]):+.4f}, {float(angular[1]):+.4f}, {float(angular[2]):+.4f})"
                )
            step += 1
            current_time += sim.get_physics_dt() * args_cli.speed
            if current_time >= motion.duration:
                current_time %= motion.duration
                loop += 1
                if args_cli.record_output:
                    break
    finally:
        if args_cli.record_output:
            recorder.stop_recording()
            recorder.save_data(args_cli.record_output)
        if annotator is not None and render_product_path is not None:
            try:
                annotator.detach([render_product_path])
            except Exception as exc:
                print(f"[WARNING] Could not detach RGB annotator: {exc}")
        if render_product is not None and not isinstance(render_product, str):
            try:
                render_product.destroy()
            except Exception as exc:
                print(f"[WARNING] Could not destroy render product: {exc}")
        if video_writer is not None:
            video_writer.close()
            print(f"[INFO] Video saved to: {video_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
