"""Encoder and IMU degradation, applied at evaluation time.

Two sensor channels, two failure stories:

* **IMU** -- only the joint+IMU methods read it, so only they can lose anything.
  ``distillation/imu.py`` already models it (gyro white noise, a per-episode gyro
  bias, an attitude tilt, and staleness); this module only scales that model.
* **Encoders** -- *every* method reads them, JOSE most of all, since joint state
  is its entire input. This is JOSE's worst case and the honest place to look
  for its limit.

The unit of "1x" is not invented here. ``ppo_walk/walk_estimator_env_cfg.py``
carries the per-term noise the walk teacher was trained under::

    joint_pos_rel   AdditiveUniformNoiseCfg(-0.01, +0.01)   rad
    joint_vel_rel   AdditiveUniformNoiseCfg(-1.5,  +1.5)    rad/s

Those magnitudes are the teacher's own training assumption, which makes 1x a
statement about the system rather than a number we chose to make a plot look
right. 0x is the condition Table I reports.

Encoder noise has to be applied in two places, and applying it in only one is
the mistake this module exists to prevent:

1. **The policy observation** -- ``joint_pos_rel`` / ``joint_vel_rel`` inside the
   495-D vector the teacher acts on. Handled by :func:`apply_encoder_noise_cfg`
   before the environment is built.
2. **The estimator input** -- ``get_estimator_joint_state`` and
   ``get_distillation_sensor_state`` on the environment, which every estimator
   and student reads *directly*, bypassing the observation manager entirely.
   Handled by :func:`install_encoder_noise` after the environment is built.

Noise in (1) alone leaves the estimator reading perfect encoders while the policy
reads bad ones; noise in (2) alone does the reverse. Neither is a robot.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace

import torch

from jose.distillation.imu import SensorCorruptionCfg


#: Half-widths of the uniform noise the walk teacher was trained under, taken
#: from ppo_walk/walk_estimator_env_cfg.py. Changing these changes what "1x"
#: means, so they are named rather than inlined.
ENCODER_POSITION_HALF_WIDTH = 0.01  # rad
ENCODER_VELOCITY_HALF_WIDTH = 1.5  # rad/s

#: Observation terms whose noise is *removed* when isolating the encoder axis.
#: Leaving them on would make the sweep a general observation-noise sweep, and
#: the base-velocity terms in particular are what JOSE is estimating -- noising
#: them would corrupt the comparison rather than degrade a sensor.
NON_ENCODER_TERMS = (
    "base_lin_vel", "base_ang_vel", "projected_gravity", "velocity_commands", "last_action",
)
ENCODER_TERMS = ("joint_pos_rel", "joint_vel_rel")


@dataclass(frozen=True)
class EncoderNoiseCfg:
    """Additive encoder error, in the same shape the environment already uses.

    ``bias_fraction`` adds a per-episode constant offset drawn once per joint, on
    top of the per-step uniform draw. Real encoder error is mostly calibration
    offset and quantization, both of which persist within an episode, and a
    filter that averages white noise away cannot average a constant away. It
    defaults to 0 so that ``scale=1`` reproduces the environment's own model
    exactly, and the sweep stays a sweep of one number.
    """

    scale: float = 1.0
    position_half_width: float = ENCODER_POSITION_HALF_WIDTH
    velocity_half_width: float = ENCODER_VELOCITY_HALF_WIDTH
    bias_fraction: float = 0.0

    @property
    def enabled(self) -> bool:
        return self.scale > 0.0

    def describe(self) -> dict:
        return {
            "scale": self.scale,
            "position_half_width_rad": self.scale * self.position_half_width,
            "velocity_half_width_rad_s": self.scale * self.velocity_half_width,
            "bias_fraction": self.bias_fraction,
            "unit_source": "ppo_walk/walk_estimator_env_cfg.py joint_pos_rel/joint_vel_rel",
        }


class EncoderCorruptor:
    """Per-step uniform encoder error, plus an optional per-episode offset."""

    def __init__(self, num_envs: int, num_joints: int, device, cfg: EncoderNoiseCfg):
        self.cfg = cfg
        self.device = device
        shape = (num_envs, num_joints)
        self._position_bias = torch.zeros(shape, device=device)
        self._velocity_bias = torch.zeros(shape, device=device)
        self.reset()

    def reset(self, env_ids=None) -> None:
        """Redraw the persistent offset for environments that just reset."""
        if not self.cfg.enabled or self.cfg.bias_fraction <= 0.0:
            return
        position = self.cfg.scale * self.cfg.position_half_width * self.cfg.bias_fraction
        velocity = self.cfg.scale * self.cfg.velocity_half_width * self.cfg.bias_fraction
        if env_ids is None:
            self._position_bias.uniform_(-position, position)
            self._velocity_bias.uniform_(-velocity, velocity)
            return
        ids = env_ids.nonzero(as_tuple=True)[0] if env_ids.dtype == torch.bool else env_ids
        if not ids.numel():
            return
        self._position_bias[ids] = torch.empty_like(self._position_bias[ids]).uniform_(-position, position)
        self._velocity_bias[ids] = torch.empty_like(self._velocity_bias[ids]).uniform_(-velocity, velocity)

    def corrupt(self, joint_position: torch.Tensor, joint_velocity: torch.Tensor):
        if not self.cfg.enabled:
            return joint_position, joint_velocity
        position = self.cfg.scale * self.cfg.position_half_width
        velocity = self.cfg.scale * self.cfg.velocity_half_width
        noisy_position = joint_position + torch.empty_like(joint_position).uniform_(-position, position)
        noisy_velocity = joint_velocity + torch.empty_like(joint_velocity).uniform_(-velocity, velocity)
        if self.cfg.bias_fraction > 0.0:
            noisy_position = noisy_position + self._position_bias
            noisy_velocity = noisy_velocity + self._velocity_bias
        return noisy_position, noisy_velocity


def apply_encoder_noise_cfg(env_cfg, scale: float) -> dict:
    """Degrade the *policy observation*'s encoder terms only. Call before build.

    The teacher acts on this vector, so without this the policy would keep
    reading perfect encoders while the estimator read bad ones. Every non-encoder
    term has its noise stripped, which is what isolates the axis: the base
    velocity terms are the very quantities under estimation, and noising them
    would be corrupting the comparison, not degrading a sensor.
    """
    policy = env_cfg.observations.policy
    if scale <= 0.0:
        policy.enable_corruption = False
        return {"enable_corruption": False, "scale": 0.0}

    policy.enable_corruption = True
    for name in NON_ENCODER_TERMS:
        term = getattr(policy, name, None)
        if term is not None:
            term.noise = None
    applied = {}
    for name in ENCODER_TERMS:
        term = getattr(policy, name, None)
        if term is None or term.noise is None:
            raise RuntimeError(
                f"Observation term {name!r} carries no noise config; the 1x unit is "
                "defined by that config, so the sweep has no meaning without it."
            )
        # Multiply the config's own bounds rather than assigning constants, so
        # this keeps working if the environment's noise model is ever retuned.
        term.noise.n_min *= scale
        term.noise.n_max *= scale
        applied[name] = (term.noise.n_min, term.noise.n_max)
    return {"enable_corruption": True, "scale": scale, "terms": applied}


@contextmanager
def install_encoder_noise(core_env, cfg: EncoderNoiseCfg):
    """Degrade the *estimator's* view of the joints, for the block's duration.

    Both accessors are replaced, not just one. ``get_estimator_joint_state`` is
    what JOSE and the joint-only student read; ``get_distillation_sensor_state``
    is what the IMU student and SET read. A method reading the untouched one
    would silently be handed perfect encoders and would look robust for the wrong
    reason -- so the noise is installed on the environment, where nothing can
    route around it, rather than on each method's adapter.

    Yields the corruptor so the caller can reset its per-episode offset in step
    with the environment.
    """
    _, joint_velocity, _ = core_env.get_estimator_joint_state()
    corruptor = EncoderCorruptor(
        joint_velocity.shape[0], joint_velocity.shape[1], joint_velocity.device, cfg
    )
    if not cfg.enabled:
        yield corruptor
        return

    original_joint_state = core_env.get_estimator_joint_state
    original_sensor_state = core_env.get_distillation_sensor_state

    def noisy_joint_state():
        position, velocity, names = original_joint_state()
        position, velocity = corruptor.corrupt(position, velocity)
        return position, velocity, names

    def noisy_sensor_state():
        state = dict(original_sensor_state())
        state["joint_position"], state["joint_velocity"] = corruptor.corrupt(
            state["joint_position"], state["joint_velocity"]
        )
        return state

    core_env.get_estimator_joint_state = noisy_joint_state
    core_env.get_distillation_sensor_state = noisy_sensor_state
    try:
        yield corruptor
    finally:
        core_env.get_estimator_joint_state = original_joint_state
        core_env.get_distillation_sensor_state = original_sensor_state


def scaled_imu_cfg(base: SensorCorruptionCfg, scale: float) -> SensorCorruptionCfg:
    """The same IMU model at a different magnitude.

    Latency is left alone: it is an integer step count, not a magnitude, and
    scaling it would change the fault's character rather than its size -- mixing
    two independent axes into one number.
    """
    if scale <= 0.0:
        return replace(base, enabled=False)
    return replace(
        base,
        gyro_noise_std=base.gyro_noise_std * scale,
        gyro_bias_std=base.gyro_bias_std * scale,
        gravity_tilt_std_rad=base.gravity_tilt_std_rad * scale,
        enabled=True,
    )
