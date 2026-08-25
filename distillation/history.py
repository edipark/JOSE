"""GMT-style explicit-linear-velocity-free history students."""

from __future__ import annotations

import torch
from torch import nn


G1_DOF = 29
DISTILLATION_WINDOW = 21  # current frame plus previous 20 frames
JOINT_FRAME_DIM = 3 * G1_DOF
IMU_FRAME_DIM = JOINT_FRAME_DIM + 6


def build_joint_frame(joint_position: torch.Tensor, joint_velocity: torch.Tensor, previous_action: torch.Tensor) -> torch.Tensor:
    for name, value in (("joint_position", joint_position), ("joint_velocity", joint_velocity), ("previous_action", previous_action)):
        if value.shape[-1] != G1_DOF:
            raise ValueError(f"{name} must contain {G1_DOF} values")
    return torch.cat((joint_position, joint_velocity, previous_action), dim=-1)


def build_imu_frame(
    joint_position: torch.Tensor,
    joint_velocity: torch.Tensor,
    previous_action: torch.Tensor,
    angular_velocity: torch.Tensor,
    projected_gravity: torch.Tensor,
) -> torch.Tensor:
    if angular_velocity.shape[-1] != 3 or projected_gravity.shape[-1] != 3:
        raise ValueError("IMU features must be gyro(3) and projected gravity(3)")
    gravity_norm = projected_gravity.norm(dim=-1)
    if torch.any((gravity_norm - 1.0).abs() > 1.0e-3):
        raise ValueError("Projected gravity must be unit length")
    return torch.cat(
        (build_joint_frame(joint_position, joint_velocity, previous_action), angular_velocity, projected_gravity),
        dim=-1,
    )


class ObservationHistory:
    """Newest-first fixed history with explicit reset semantics."""

    def __init__(self, num_envs: int, window: int, frame_dim: int, device: torch.device | str):
        if window <= 0 or frame_dim <= 0:
            raise ValueError("History window and frame dimension must be positive")
        self.values = torch.zeros(num_envs, window, frame_dim, device=device)
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = torch.arange(len(self.values), device=self.values.device) if env_ids is None else env_ids
        self.values[ids] = 0.0
        self.initialized[ids] = False

    def push(self, frame: torch.Tensor) -> torch.Tensor:
        if frame.shape != self.values[:, 0].shape:
            raise ValueError(f"Expected history frame {tuple(self.values[:, 0].shape)}, got {tuple(frame.shape)}")
        fresh = ~self.initialized
        if fresh.any():
            self.values[fresh] = frame[fresh, None, :]
        existing = ~fresh
        if existing.any():
            self.values[existing, 1:] = self.values[existing, :-1].clone()
            self.values[existing, 0] = frame[existing]
        self.initialized[:] = True
        return self.flatten()

    def flatten(self) -> torch.Tensor:
        return self.values.reshape(len(self.values), -1)


class HistoryMLPStudent(nn.Module):
    """Deterministic 256-256-128 DAgger policy shared by both baselines."""

    def __init__(
        self,
        frame_dim: int,
        action_dim: int = G1_DOF,
        window: int = DISTILLATION_WINDOW,
        hidden_dims: tuple[int, ...] = (256, 256, 128),
    ):
        super().__init__()
        self.frame_dim = frame_dim
        self.window = window
        self.input_dim = frame_dim * window
        self.action_dim = action_dim
        self.hidden_dims = tuple(hidden_dims)
        layers: list[nn.Module] = []
        current = self.input_dim
        for hidden in hidden_dims:
            layers.extend((nn.Linear(current, hidden), nn.ELU()))
            current = hidden
        layers.append(nn.Linear(current, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim == 3:
            history = history.reshape(history.shape[0], -1)
        if history.shape[-1] != self.input_dim:
            raise ValueError(f"Expected flattened history dim {self.input_dim}, got {history.shape[-1]}")
        return self.network(history)

    def config(self) -> dict:
        return {
            "type": "HistoryMLPStudent",
            "frame_dim": self.frame_dim,
            "window": self.window,
            "input_dim": self.input_dim,
            "action_dim": self.action_dim,
            "hidden_dims": self.hidden_dims,
            "explicit_linear_velocity": False,
        }
