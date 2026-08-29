"""Fixed-command quantitative evaluation of a G1 PPO walk policy.

Playing a checkpoint in the viewer only tells you whether the robot looks alive.
This script answers the question the flat task actually poses: *does the policy
follow the commanded (vx, vy, yaw)?* It holds each command constant across every
environment for a fixed window and reports tracking error, survival, gait
statistics and the sanity checks that distinguish a walking policy from a
standing one.

Example
-------
    python eval_ppo_walk.py --task Isaac-G1-PPO-Walk-JOSE-v0 --headless \
        --num_envs 64 \
        --checkpoint logs/rsl_rl/isaac_g1_ppo_walk_jose_v0/<run>/model_499.pt \
        --output eval.json

A rising ``Train/mean_reward`` is not evidence of success: with ``std = 0.5``
kernels a motionless robot already collects ``exp(-|cmd|^2 / 0.25)`` every step.
Judge a run by this script and by ``Episode_Reward/feet_air_time``, which is zero
whenever the robot is not stepping.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

from jose.ppo_walk.utils import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Fixed-command evaluation of a G1 PPO walk policy.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments per command.")
parser.add_argument("--task", type=str, default="Isaac-G1-PPO-Walk-JOSE-v0", help="Name of the task.")
parser.add_argument("--settle_s", type=float, default=2.0, help="Seconds discarded after reset before measuring.")
parser.add_argument("--measure_s", type=float, default=8.0, help="Seconds of measurement per command.")
parser.add_argument("--output", type=str, default=None, help="Optional path to write the results as JSON.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import importlib.metadata as metadata
import json
import os
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path

import jose  # noqa: F401
from jose.ppo_walk import mdp as walk_mdp
from jose.ppo_walk.utils.parser_cfg import parse_env_cfg

installed_version = metadata.version("rsl-rl-lib")

FEET_BODIES = ".*ankle_roll.*"

# (vx [m/s], vy [m/s], yaw rate [rad/s]) -- the required minimum evaluation set.
EVAL_COMMANDS = [
    (0.0, 0.0, 0.0),
    (0.3, 0.0, 0.0),
    (0.6, 0.0, 0.0),
    (0.0, 0.2, 0.0),
    (0.0, -0.2, 0.0),
    (0.0, 0.0, 0.3),
    (0.0, 0.0, -0.3),
    (0.4, 0.2, 0.2),
]


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


class CommandEvaluator:
    """Runs one fixed command across all environments and accumulates statistics."""

    def __init__(self, env, feet_sensor_cfg, feet_asset_cfg, settle_steps, measure_steps):
        self.env = env
        self.unwrapped = env.unwrapped
        self.feet_sensor_cfg = feet_sensor_cfg
        self.feet_asset_cfg = feet_asset_cfg
        self.settle_steps = settle_steps
        self.measure_steps = measure_steps
        self.dt = self.unwrapped.step_dt
        self.command_term = self.unwrapped.command_manager.get_term("base_velocity")

    def _force_command(self, cmd: torch.Tensor):
        """Pin the command for every environment.

        ``ranges`` is narrowed so that any mid-episode resample reproduces the same
        value, ``is_standing_env`` is cleared so no environment is silently zeroed,
        and the buffer itself is overwritten because ``_update_command`` runs at the
        end of :meth:`step`, after the reward has already been computed.
        """
        ranges = self.command_term.cfg.ranges
        ranges.lin_vel_x = (float(cmd[0]), float(cmd[0]))
        ranges.lin_vel_y = (float(cmd[1]), float(cmd[1]))
        ranges.ang_vel_z = (float(cmd[2]), float(cmd[2]))
        self.command_term.cfg.rel_standing_envs = 0.0
        self.command_term.is_standing_env[:] = False
        self.command_term.vel_command_b[:] = cmd

    def run(self, policy, cmd_tuple: tuple[float, float, float]) -> dict:
        env, unwrapped = self.env, self.unwrapped
        device = unwrapped.device
        num_envs = unwrapped.num_envs
        cmd = torch.tensor(cmd_tuple, device=device, dtype=torch.float32)

        with torch.inference_mode():
            obs, _ = env.reset()
        self._force_command(cmd)

        robot = unwrapped.scene["robot"]
        contact_sensor = unwrapped.scene.sensors[self.feet_sensor_cfg.name]
        feet_sensor_ids = self.feet_sensor_cfg.body_ids

        # per-env accumulators over the measurement window
        alive = torch.ones(num_envs, dtype=torch.bool, device=device)
        fell = torch.zeros(num_envs, dtype=torch.bool, device=device)
        n_samples = torch.zeros(num_envs, device=device)
        sum_v = torch.zeros(num_envs, 3, device=device)  # vx, vy (yaw frame), yaw rate (base frame)
        sum_sq_err = torch.zeros(num_envs, 3, device=device)
        sum_height = torch.zeros(num_envs, device=device)
        sum_air_time_rew = torch.zeros(num_envs, device=device)
        sum_slide_pen = torch.zeros(num_envs, device=device)
        lift_count = torch.zeros(num_envs, device=device)
        double_stance_steps = torch.zeros(num_envs, device=device)
        # violation counter for the "air time must be 0 in double stance" invariant
        double_stance_reward_violations = 0
        max_air_reward_in_double_stance = 0.0
        start_xy = None
        end_xy = torch.zeros(num_envs, 2, device=device)
        prev_contact = None

        total_steps = self.settle_steps + self.measure_steps
        with torch.inference_mode():
            for step in range(total_steps):
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                self._force_command(cmd)

                measuring = step >= self.settle_steps
                # an environment that terminated (fall) inside the window stops
                # contributing samples; `dones` here includes time-outs, but the
                # play config's episode is long enough that none occur.
                terminated = unwrapped.termination_manager.terminated
                if measuring:
                    fell |= terminated & alive
                alive &= ~(dones.bool())

                if not measuring:
                    prev_contact = contact_sensor.data.current_contact_time[:, feet_sensor_ids] > 0.0
                    continue

                if start_xy is None:
                    start_xy = robot.data.root_pos_w[:, :2].clone()

                mask = alive.float()

                # -- velocity, measured the same way the reward kernels measure it
                vel_yaw = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w)
                meas = torch.stack([vel_yaw[:, 0], vel_yaw[:, 1], robot.data.root_ang_vel_b[:, 2]], dim=1)
                sum_v += meas * mask.unsqueeze(1)
                sum_sq_err += torch.square(meas - cmd.unsqueeze(0)) * mask.unsqueeze(1)
                sum_height += robot.data.root_pos_w[:, 2] * mask
                n_samples += mask

                # -- gait statistics
                in_contact = contact_sensor.data.current_contact_time[:, feet_sensor_ids] > 0.0
                lifts = (prev_contact & ~in_contact).float().sum(dim=1)
                lift_count += lifts * mask
                double_stance = in_contact.all(dim=1)
                double_stance_steps += double_stance.float() * mask
                prev_contact = in_contact

                # -- raw (unweighted) reward-term values
                air_rew = walk_mdp.feet_air_time_positive_biped(
                    unwrapped,
                    command_name="base_velocity",
                    threshold=0.4,
                    sensor_cfg=self.feet_sensor_cfg,
                )
                slide_pen = walk_mdp.feet_slide(
                    unwrapped, sensor_cfg=self.feet_sensor_cfg, asset_cfg=self.feet_asset_cfg
                )
                sum_air_time_rew += air_rew * mask
                sum_slide_pen += slide_pen * mask

                # -- invariant: no air-time reward while both feet are down
                bad = double_stance & alive & (air_rew > 0.0)
                if bad.any():
                    double_stance_reward_violations += int(bad.sum().item())
                    max_air_reward_in_double_stance = max(
                        max_air_reward_in_double_stance, float(air_rew[bad].max().item())
                    )

                end_xy = torch.where(alive.unsqueeze(1), robot.data.root_pos_w[:, :2], end_xy)

        with torch.inference_mode():
            valid = n_samples > 0
            n = n_samples.clamp(min=1.0)
            mean_v = (sum_v / n.unsqueeze(1))[valid]
            rmse = torch.sqrt(sum_sq_err / n.unsqueeze(1))[valid]
            measure_s = self.measure_steps * self.dt
            drift = (
                torch.norm(end_xy - start_xy, dim=1)[valid]
                if start_xy is not None
                else torch.zeros(1, device=device)
            )

            return {
                "command": {"vx": cmd_tuple[0], "vy": cmd_tuple[1], "yaw": cmd_tuple[2]},
                "measured": {
                    "vx": float(mean_v[:, 0].mean()),
                    "vy": float(mean_v[:, 1].mean()),
                    "yaw": float(mean_v[:, 2].mean()),
                },
                "error": {
                    "vx_mean": float((mean_v[:, 0] - cmd[0]).mean()),
                    "vy_mean": float((mean_v[:, 1] - cmd[1]).mean()),
                    "yaw_mean": float((mean_v[:, 2] - cmd[2]).mean()),
                    "vx_rmse": float(rmse[:, 0].mean()),
                    "vy_rmse": float(rmse[:, 1].mean()),
                    "yaw_rmse": float(rmse[:, 2].mean()),
                },
                "survival_rate": float(alive.float().mean()),
                "fall_count": int(fell.sum()),
                "num_envs": int(num_envs),
                "base_height_m": float((sum_height / n)[valid].mean()),
                "feet_air_time_reward": float((sum_air_time_rew / n)[valid].mean()),
                "feet_slide_penalty": float((sum_slide_pen / n)[valid].mean()),
                "foot_lifts_per_s": float((lift_count[valid] / measure_s).mean()),
                "double_stance_fraction": float((double_stance_steps / n)[valid].mean()),
                "drift_m": float(drift.mean()),
                "drift_speed_mps": float(drift.mean() / measure_s),
                "double_stance_air_reward_violations": double_stance_reward_violations,
                "max_air_reward_in_double_stance": max_air_reward_in_double_stance,
            }


def run_sanity_checks(results: dict) -> list[tuple[str, bool, str]]:
    """Return ``(name, passed, detail)`` for each required sanity check."""
    checks = []

    def r(cmd):
        return results[str(cmd)]

    zero = r((0.0, 0.0, 0.0))
    fwd_03 = r((0.3, 0.0, 0.0))
    fwd_06 = r((0.6, 0.0, 0.0))
    yaw_p = r((0.0, 0.0, 0.3))
    yaw_n = r((0.0, 0.0, -0.3))

    # 1. air-time reward must never fire while both feet are on the ground
    violations = sum(v["double_stance_air_reward_violations"] for v in results.values())
    checks.append((
        "feet_air_time is 0 in double stance",
        violations == 0,
        f"{violations} violating samples (max reward seen "
        f"{max(v['max_air_reward_in_double_stance'] for v in results.values()):.4f})",
    ))

    # 2. a nonzero command that produces no stepping must earn no air-time reward
    standing_under_command = [
        (k, v)
        for k, v in results.items()
        if (v["command"]["vx"], v["command"]["vy"]) != (0.0, 0.0) and v["double_stance_fraction"] > 0.99
    ]
    checks.append((
        "no air-time reward while standing under a nonzero command",
        all(v["feet_air_time_reward"] <= 1e-6 for _, v in standing_under_command),
        f"{len(standing_under_command)} commands stood still"
        + ("" if not standing_under_command else f": {[k for k, _ in standing_under_command]}"),
    ))

    # 3. zero command must not make the robot step in place
    checks.append((
        "zero command does not lift feet repeatedly",
        zero["foot_lifts_per_s"] < 0.5,
        f"{zero['foot_lifts_per_s']:.3f} lifts/s, air-time reward {zero['feet_air_time_reward']:.4f}",
    ))

    # 4. zero command must not drift
    checks.append((
        "zero command does not drift",
        zero["drift_speed_mps"] < 0.1,
        f"{zero['drift_speed_mps']:.3f} m/s over the window",
    ))

    # 5. measured vx must increase *meaningfully* with commanded vx. A bare
    #    monotonicity test passes on numerical noise (0.001 < 0.002), which would
    #    report a statue as responsive, so require a real margin per step.
    margin = 0.05
    checks.append((
        "measured vx increases with commanded vx",
        (fwd_03["measured"]["vx"] > zero["measured"]["vx"] + margin)
        and (fwd_06["measured"]["vx"] > fwd_03["measured"]["vx"] + margin),
        f"vx(0)={zero['measured']['vx']:.3f}, vx(0.3)={fwd_03['measured']['vx']:.3f},"
        f" vx(0.6)={fwd_06['measured']['vx']:.3f} (each step must gain > {margin})",
    ))

    # 6. yaw-only commands must produce yaw rate of the right sign
    checks.append((
        "yaw-only command produces angular velocity",
        yaw_p["measured"]["yaw"] > 0.1 and yaw_n["measured"]["yaw"] < -0.1,
        f"yaw(+0.3)={yaw_p['measured']['yaw']:.3f}, yaw(-0.3)={yaw_n['measured']['yaw']:.3f}",
    ))

    return checks


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    # deterministic measurement
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.push_robot = None
    env_cfg.events.base_external_force_torque = None

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] loading checkpoint: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    feet_sensor_cfg = SceneEntityCfg("contact_forces", body_names=FEET_BODIES)
    feet_sensor_cfg.resolve(env.unwrapped.scene)
    feet_asset_cfg = SceneEntityCfg("robot", body_names=FEET_BODIES)
    feet_asset_cfg.resolve(env.unwrapped.scene)
    if len(feet_sensor_cfg.body_ids) != 2:
        raise RuntimeError(
            f"'{FEET_BODIES}' resolved to {len(feet_sensor_cfg.body_ids)} bodies "
            f"({feet_sensor_cfg.body_names}); the biped air-time metric needs exactly 2."
        )
    print(f"[INFO] feet bodies: {feet_sensor_cfg.body_names}")

    dt = env.unwrapped.step_dt
    evaluator = CommandEvaluator(
        env,
        feet_sensor_cfg,
        feet_asset_cfg,
        settle_steps=int(round(args_cli.settle_s / dt)),
        measure_steps=int(round(args_cli.measure_s / dt)),
    )

    results = {}
    for cmd in EVAL_COMMANDS:
        print(f"[INFO] evaluating command {cmd} ...")
        results[str(cmd)] = evaluator.run(policy, cmd)

    # ---- report ----
    header = (
        f"{'command (vx,vy,yaw)':>22} | {'measured (vx,vy,yaw)':>24} | {'RMSE (vx,vy,yaw)':>22} |"
        f" {'surv':>5} {'fall':>5} {'h[m]':>6} {'air':>6} {'slide':>7} {'lift/s':>7} {'2stance':>8} {'drift':>7}"
    )
    print("\n" + "=" * len(header))
    print(f"checkpoint: {resume_path}")
    print(f"task: {args_cli.task}   num_envs: {args_cli.num_envs}   "
          f"settle: {args_cli.settle_s}s   measure: {args_cli.measure_s}s")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for cmd in EVAL_COMMANDS:
        v = results[str(cmd)]
        c, m, e = v["command"], v["measured"], v["error"]
        print(
            f"{c['vx']:>7.2f}{c['vy']:>7.2f}{c['yaw']:>8.2f} |"
            f"{m['vx']:>8.3f}{m['vy']:>8.3f}{m['yaw']:>8.3f} |"
            f"{e['vx_rmse']:>7.3f}{e['vy_rmse']:>7.3f}{e['yaw_rmse']:>8.3f} |"
            f" {v['survival_rate']:>5.2f} {v['fall_count']:>5d} {v['base_height_m']:>6.3f}"
            f" {v['feet_air_time_reward']:>6.3f} {v['feet_slide_penalty']:>7.3f}"
            f" {v['foot_lifts_per_s']:>7.2f} {v['double_stance_fraction']:>8.2f} {v['drift_m']:>7.3f}"
        )
    print("=" * len(header))

    print("\nSanity checks")
    print("-" * 13)
    checks = run_sanity_checks(results)
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}\n         {detail}")
    n_pass = sum(1 for _, p, _ in checks if p)
    print(f"\n{n_pass}/{len(checks)} sanity checks passed.")

    if args_cli.output:
        payload = {
            "checkpoint": resume_path,
            "task": args_cli.task,
            "num_envs": args_cli.num_envs,
            "settle_s": args_cli.settle_s,
            "measure_s": args_cli.measure_s,
            "results": results,
            "sanity_checks": [{"name": n, "passed": p, "detail": d} for n, p, d in checks],
        }
        with open(args_cli.output, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"[INFO] wrote {args_cli.output}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
