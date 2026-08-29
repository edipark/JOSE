"""Policy adapters that make AMP and PPO share the estimator pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from ..schema import (
    AMP_OBSERVATION_SCHEMA,
    PPO_WALK_OBSERVATION_SCHEMA,
    ObservationSchema,
    joint_indices,
    ppo_walk_history_target_indices,
    ppo_walk_target_scales,
    PPO_WALK_HISTORY_LENGTH,
)
from ..skrl_compat import deterministic_action
from ..task_math import inject_observation_estimate


def unwrap_direct_env(env):
    current = env
    for _ in range(32):
        candidate = getattr(current, "unwrapped", current)
        if hasattr(candidate, "get_estimator_joint_state") and hasattr(candidate, "get_estimator_target"):
            return candidate
        next_env = getattr(current, "_env", None)
        if next_env is None or next_env is current:
            break
        current = next_env
    raise RuntimeError("Environment does not implement the JOSE estimator interface")


class PolicyAdapter(ABC):
    schema: ObservationSchema

    def __init__(self, env, joint_preset: str = "all"):
        self.env = env
        self.core_env = unwrap_direct_env(env)
        _, _, names = self.core_env.get_estimator_joint_state()
        self.joint_preset = joint_preset
        self.joint_ids = joint_indices(names, joint_preset)

    @property
    def input_dim(self) -> int:
        return 2 * len(self.joint_ids)

    def estimator_input(self) -> torch.Tensor:
        # Deliberately use the simulator-provided joint velocity. There is no
        # finite-difference, encoder quantization, EMA, or hardware-noise path.
        joint_pos, joint_vel, _ = self.core_env.get_estimator_joint_state()
        ids = torch.as_tensor(self.joint_ids, device=joint_pos.device)
        return torch.cat((joint_pos.index_select(1, ids), joint_vel.index_select(1, ids)), dim=-1)

    def estimator_target(self) -> torch.Tensor:
        target = self.core_env.get_estimator_target()
        if target.shape[-1] != self.schema.estimator_target_dim:
            raise RuntimeError(
                f"Adapter expected target dim {self.schema.estimator_target_dim}, got {target.shape[-1]}"
            )
        return target

    def inject_estimate(self, observations: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
        if estimate.shape[-1] != self.schema.estimator_target_dim:
            raise ValueError("Estimator output does not match the policy schema")
        return inject_observation_estimate(observations, estimate, self.schema.estimator_target_indices)

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    def action(self, agent: Any, observations: torch.Tensor) -> torch.Tensor:
        states = self.env.state() if hasattr(self.env, "state") else None
        return deterministic_action(agent, observations, states)


class AmpPolicyAdapter(PolicyAdapter):
    schema = AMP_OBSERVATION_SCHEMA

    def name(self) -> str:
        return "amp"


class PpoWalkPolicyAdapter(PolicyAdapter):
    """Adapter for the manager-based PPO walk teacher driven by rsl-rl.

    Two things differ from the Direct-workflow adapters and both are plumbing,
    not method:

    * The teacher is an rsl-rl policy, so it is called directly instead of going
      through SKRL's ``act`` API.
    * The policy observation carries a five-step history per term, and the
      observation manager stores ``base_ang_vel`` pre-scaled by 0.2. Injection
      therefore rescales the estimate and overwrites the *entire* history block
      of the three estimated terms. Replacing only the newest frame would leave
      four frames of ground-truth privileged state visible to the policy, which
      would understate the closed-loop cost of estimation error.
    """

    schema = PPO_WALK_OBSERVATION_SCHEMA

    def __init__(self, env, joint_preset: str = "all"):
        super().__init__(env, joint_preset)
        self._history_indices = ppo_walk_history_target_indices()
        self._scales: torch.Tensor | None = None
        self._ring: torch.Tensor | None = None

    def name(self) -> str:
        return "ppo_walk"

    def action(self, agent: Any, observations: torch.Tensor) -> torch.Tensor:
        return agent(self.env.as_policy_input(observations))

    def _scaled(self, estimate: torch.Tensor) -> torch.Tensor:
        if self._scales is None:
            self._scales = torch.tensor(
                ppo_walk_target_scales(), device=estimate.device, dtype=estimate.dtype
            )
        return estimate * self._scales

    def _push(self, estimate: torch.Tensor) -> torch.Tensor:
        """Roll the estimate ring and return it flattened oldest-frame-first.

        The ring is refilled with the current estimate for environments that just
        reset, matching how the observation manager repopulates its own history
        buffer after a reset.
        """
        scaled = self._scaled(estimate)
        if self._ring is None or self._ring.shape[0] != scaled.shape[0]:
            self._ring = scaled.unsqueeze(1).repeat(1, PPO_WALK_HISTORY_LENGTH, 1)
        else:
            self._ring = torch.roll(self._ring, -1, dims=1)
            self._ring[:, -1] = scaled
            just_reset = self.core_env.episode_length_buf == 0
            if bool(just_reset.any()):
                self._ring[just_reset] = scaled[just_reset].unsqueeze(1)
        return self._ring.reshape(scaled.shape[0], -1)

    def inject_estimate(self, observations: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
        if estimate.shape[-1] != self.schema.estimator_target_dim:
            raise ValueError("Estimator output does not match the policy schema")
        return inject_observation_estimate(observations, self._push(estimate), self._history_indices)


def make_policy_adapter(kind: str, env, joint_preset: str = "all") -> PolicyAdapter:
    kind = kind.lower()
    if kind == "amp":
        return AmpPolicyAdapter(env, joint_preset)
    # "ppo" named the SKRL Direct PPO walk teacher, which was removed along with
    # its environment. It now resolves to the rsl-rl walk teacher so existing
    # commands and logged run metadata keep working.
    if kind in ("ppo_walk", "ppo"):
        return PpoWalkPolicyAdapter(env, joint_preset)
    raise ValueError(f"Unknown policy adapter {kind!r}; choose amp or ppo_walk")
