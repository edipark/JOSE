"""Environment configurations for G1 AMP walk, dance, and jump."""

from __future__ import annotations

import os
from dataclasses import MISSING

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from .g1_cfg import G1_SOLO_CFG
from .task_math import (
    AMP_HISTORY_STEPS,
    CONTROL_DECIMATION,
    EPISODE_LENGTH_S,
    PHYSICS_DT,
    TWIST_ACTION_SCALE,
)


MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")


@configclass
class G1AmpEnvCfg(DirectRLEnvCfg):
    episode_length_s = EPISODE_LENGTH_S
    decimation = CONTROL_DECIMATION
    observation_space = 101
    action_space = 29
    action_scale = TWIST_ACTION_SCALE
    state_space = 0
    num_amp_observations = AMP_HISTORY_STEPS
    amp_observation_space = 101

    early_termination = True
    termination_height = 0.55
    vel_window_min_vx = 0.0
    vel_window_steps = 10
    motion_file: str = MISSING
    reference_body = "pelvis"
    reset_strategy = "default"

    # World +X velocity task reward. Disabled in the base/dance/jump task and
    # enabled by G1AmpWalkEnvCfg, matching SOLO's reward equation.
    target_velocity = 0.6
    velocity_tracking_coeff = 2.0
    velocity_reward_weight = 0.0
    # mean((action[t] - action[t - 1]) ** 2) across joints.
    action_rate_penalty_weight = 0.0
    # mean((action[t] - 2 * action[t - 1] + action[t - 2]) ** 2) across joints.
    action_second_difference_penalty_weight = 0.0

    sim: SimulationCfg = SimulationCfg(
        dt=PHYSICS_DT,
        render_interval=decimation,
        physx=PhysxCfg(gpu_found_lost_pairs_capacity=2**23, gpu_total_aggregate_pairs_capacity=2**23),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)
    robot = G1_SOLO_CFG.replace(prim_path="/World/envs/env_.*/Robot")


@configclass
class G1AmpWalkEnvCfg(G1AmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "G1_walk.npz")
    velocity_reward_weight = 0.5
    action_rate_penalty_weight = 0.0
    action_second_difference_penalty_weight = 0.1


@configclass
class G1AmpDanceEnvCfg(G1AmpEnvCfg):
    robot = G1AmpEnvCfg.robot.replace(soft_joint_pos_limit_factor=1.0)
    motion_file = os.path.join(MOTIONS_DIR, "G1_dance.npz")
    reset_strategy = "default"


@configclass
class G1AmpJumpEnvCfg(G1AmpEnvCfg):
    """SOLO-aligned AMP with the jump motion and a jump-safe pelvis threshold."""

    robot = G1AmpEnvCfg.robot.replace(soft_joint_pos_limit_factor=1.0)
    motion_file = os.path.join(MOTIONS_DIR, "G1_jump.npz")
    reset_strategy = "random"
    termination_height = 0.20
