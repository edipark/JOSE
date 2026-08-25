"""Minimal SKRL PPO walking example using JOSE's 29-DOF G1 interface."""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from .g1_amp_env import G1AmpEnv
from .g1_amp_env_cfg import G1AmpWalkEnvCfg
from .task_math import WALK_TARGET_VELOCITY


@configclass
class G1PpoWalkEnvCfg(G1AmpWalkEnvCfg):
    observation_space = 99
    amp_observation_space = 101
    target_velocity = WALK_TARGET_VELOCITY
    command_resample_s = 5.0


class G1PpoWalkEnv(G1AmpEnv):
    cfg: G1PpoWalkEnvCfg

    def __init__(self, cfg: G1PpoWalkEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.commands = torch.zeros((self.num_envs, 3), device=self.device)
        self.commands[:, 0] = self.cfg.target_velocity
        self.actions = torch.zeros((self.num_envs, 29), device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.previous_actions.copy_(self.actions)
        self.actions = actions.clone()

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if hasattr(self, "actions"):
            ids = self.robot._ALL_INDICES if env_ids is None else env_ids
            self.actions[ids] = 0.0
            self.previous_actions[ids] = 0.0

    def get_estimator_target(self) -> torch.Tensor:
        return torch.cat(
            (
                self.robot.data.root_lin_vel_b,
                self.robot.data.root_ang_vel_b,
                self.robot.data.projected_gravity_b,
            ),
            dim=-1,
        )

    def _get_observations(self) -> dict:
        base_state = self.get_estimator_target()
        obs = torch.cat(
            (
                base_state,
                self.commands,
                self.robot.data.joint_pos,
                self.robot.data.joint_vel,
                self.previous_actions,
            ),
            dim=-1,
        )
        self.extras = {"log": self.extras.get("log", {}) if isinstance(self.extras, dict) else {}}
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        command_error = (self.robot.data.root_lin_vel_b[:, :2] - self.commands[:, :2]).square().sum(dim=-1)
        yaw_error = (self.robot.data.root_ang_vel_b[:, 2] - self.commands[:, 2]).square()
        tracking = torch.exp(-command_error / 0.25) + 0.5 * torch.exp(-yaw_error / 0.25)
        upright = torch.exp(-4.0 * self.robot.data.projected_gravity_b[:, :2].square().sum(dim=-1))
        action_rate = (self.actions - self.previous_actions).square().mean(dim=-1)
        torque = self.robot.data.applied_torque.square().mean(dim=-1)
        reward = tracking + 0.25 * upright - 0.01 * action_rate - 1.0e-6 * torque
        previous_log = self.extras.get("log", {}) if isinstance(self.extras.get("log"), dict) else {}
        self.extras["log"] = {
            **previous_log,
            "reward/raw_task": reward.mean().detach(),
            "reward/velocity_tracking": tracking.mean().detach(),
            "reward/upright": upright.mean().detach(),
            "metric/base_vel_x": self.robot.data.root_lin_vel_b[:, 0].mean().detach(),
        }
        return reward
