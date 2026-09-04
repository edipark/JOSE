"""SET's non-privileged observation, on top of JOSE's policy adapters.

The paper's input is ``o = (omega, phi, q, qdot, p, cmd)``: base angular
velocity, base orientation, joint positions and velocities, Cartesian foot
positions relative to the body, and the command vector. JOSE's own adapters
expose joint state only -- that asymmetry is the experiment -- so this subclass
overrides exactly one method and inherits everything else, including the
injection path that writes an estimate back into the teacher's observation.

Mapping to the G1, with the deviations from the paper made explicit:

===============  ==================================================  =====
paper            here                                                dims
===============  ==================================================  =====
``omega``        ``angular_velocity`` (body frame, from the IMU)      3
``phi``          ``projected_gravity``; the paper says "orientation"  3
``q``            ``joint_position``                                   29
``qdot``         ``joint_velocity``                                   29
``p``            two ankle_roll links in the body frame (quadruped
                 has four feet, a biped two)                          6
``cmd``          ``base_velocity``; the paper's is 5-D and carries
                 a jump height and a jump trigger this task lacks     3 / 0
===============  ==================================================  =====

Everything comes from ``get_distillation_sensor_state()``, the same contract the
IMU-distillation baseline uses, so SET is held to the identical no-privileged-
input rule: that method raises if the environment ever exposes base linear
velocity or linear acceleration.

Foot positions are a function of ``q`` -- forward kinematics through the URDF --
so reading them off the simulator is a fast way to evaluate that function, not
extra information. They are expressed in the body frame, which is what a robot
can compute without knowing its heading.
"""

from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply_inverse

from jose.estimator.adapters import PolicyAdapter

from .targets import PASS_THROUGH


FOOT_TOKEN = "ankle_roll"


class SETPolicyAdapter(PolicyAdapter):
    """A :class:`PolicyAdapter` whose estimator input is SET's ``o``."""

    def __init__(self, base: PolicyAdapter, imu_corruptor=None):
        # Wrap rather than re-derive: the base adapter already resolved the joint
        # ids, the schema and the core env, and it owns the injection logic whose
        # index bookkeeping we must not duplicate.
        self._base = base
        # Evaluation-time IMU degradation. None during training and in the clean
        # condition; the sensor-robustness sweep supplies one so that SET is
        # degraded on the IMU axis exactly as the distillation students are.
        # SET publishes no IMU randomization, which is a statement about its
        # training and not a licence to hand it a clean sensor at test time.
        self.imu_corruptor = imu_corruptor
        # One physical sensor read per step. ``estimator_input`` and
        # ``pass_through_values`` both consume the IMU, and the corruptor is
        # stateful -- it carries a per-episode bias and a latency queue -- so
        # calling it twice would advance that queue twice and draw two
        # independent samples of what is one measurement. The cache is cleared
        # once per environment step by ``invalidate_imu``.
        self._imu_cache = None
        self.env = base.env
        self.core_env = base.core_env
        self.joint_preset = base.joint_preset
        self.joint_ids = base.joint_ids
        self.schema = base.schema

        names = list(self.core_env.robot.data.body_names)
        feet = [index for index, name in enumerate(names) if FOOT_TOKEN in name]
        if len(feet) != 2:
            raise RuntimeError(f"Expected two {FOOT_TOKEN!r} bodies, found {len(feet)}")
        self._foot_ids = torch.as_tensor(feet, device=self.core_env.device)
        self._has_command = hasattr(self.core_env, "command_manager")

    def name(self) -> str:
        return self._base.name()

    def action(self, agent, observations: torch.Tensor) -> torch.Tensor:
        return self._base.action(agent, observations)

    def inject_estimate(self, observations: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
        return self._base.inject_estimate(observations, estimate)

    def _feet_in_body_frame(self) -> torch.Tensor:
        data = self.core_env.robot.data
        root_position = data.root_pos_w if hasattr(data, "root_pos_w") else data.body_pos_w[:, 0]
        root_quaternion = data.root_quat_w if hasattr(data, "root_quat_w") else data.body_quat_w[:, 0]
        offsets = data.body_pos_w.index_select(1, self._foot_ids) - root_position.unsqueeze(1)
        rotated = torch.stack(
            [quat_apply_inverse(root_quaternion, offsets[:, index]) for index in range(offsets.shape[1])],
            dim=1,
        )
        return rotated.reshape(offsets.shape[0], -1)

    def _command(self, batch: int, device, dtype) -> torch.Tensor:
        if not self._has_command:
            return torch.zeros(batch, 0, device=device, dtype=dtype)
        return self.core_env.command_manager.get_command("base_velocity").to(dtype)

    @property
    def input_dim(self) -> int:
        return self.estimator_input().shape[-1]

    def invalidate_imu(self) -> None:
        """Drop the cached sensor read. Call once per environment step."""
        self._imu_cache = None

    def _imu(self, state) -> tuple[torch.Tensor, torch.Tensor]:
        # No corruptor, no cache: the cache exists only so that the step's two
        # consumers share one *noisy* draw. Caching the clean read as well would
        # make a caller that never invalidates -- collection, SET's own
        # evaluator -- reuse the first step's IMU for the whole run.
        if self.imu_corruptor is None:
            return state["angular_velocity"], state["projected_gravity"]
        if self._imu_cache is None:
            self._imu_cache = self.imu_corruptor(
                state["angular_velocity"], state["projected_gravity"]
            )
        return self._imu_cache

    def pass_through_values(self) -> torch.Tensor | None:
        """Target dimensions SET reads instead of estimating, in target order.

        ``None`` when the adapter passes nothing through -- the AMP case, where
        the target's angular velocity is world-frame and the IMU's is
        body-frame, so no dimension matches verbatim.
        """
        measured = PASS_THROUGH[self.name()]
        if not measured:
            return None
        state = self.core_env.get_distillation_sensor_state()
        angular_velocity, projected_gravity = self._imu(state)
        reading = {"angular_velocity": angular_velocity, "projected_gravity": projected_gravity}
        return torch.stack(
            [reading.get(key, state.get(key))[:, component]
             for _, (key, component) in sorted(measured.items())], dim=-1
        )

    def estimator_input(self) -> torch.Tensor:
        state = self.core_env.get_distillation_sensor_state()
        forbidden = {"base_linear_velocity", "linear_acceleration"}.intersection(state)
        if forbidden:
            raise RuntimeError(f"SET was exposed to forbidden features: {sorted(forbidden)}")
        joint_position = state["joint_position"]
        batch = joint_position.shape[0]
        angular_velocity, projected_gravity = self._imu(state)
        return torch.cat(
            (
                angular_velocity,
                projected_gravity,
                joint_position,
                state["joint_velocity"],
                self._feet_in_body_frame(),
                self._command(batch, joint_position.device, joint_position.dtype),
            ),
            dim=-1,
        )
