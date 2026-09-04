"""Load any of the five methods behind one calling convention.

Each loader returns ``(act, on_step, info)``:

* ``act(observations) -> action``. The teacher, JOSE and SET use the observation
  (the last two after overwriting the privileged slots with their estimate); the
  two distillation students ignore it and read the sensor dict instead, which is
  the whole point of that baseline.
* ``on_step(dones)`` clears whatever per-episode state the method keeps --
  history windows, IMU corruptor, SET's autoregressive ring.
* ``info`` is metadata for the results row.

One convention is what makes a sweep possible: the robustness driver needs to
put five differently-shaped things through the same command grid at the same
noise setting, and any per-method special-casing inside that loop is a place for
the conditions to quietly diverge.

Encoder noise is deliberately *absent* from this file. It is installed on the
environment by ``robustness/noise.py`` so that no method can route around it;
these loaders read whatever the environment hands them, exactly as they would on
a robot with a bad encoder.
"""

from __future__ import annotations

from pathlib import Path

import torch

from jose.distillation.command_eval import build_frame_fn, build_student_policy
from jose.distillation.command_eval import reset_ids
from jose.distillation.history import HistoryMLPStudent, ObservationHistory
from jose.distillation.imu import IMUObservationSpec, SensorCorruptionCfg, SensorCorruptor
from jose.estimator.models import RunningNormalizer
from jose.estimator.pipeline import HistoryBuffer, load_estimator
from jose.schema import SCHEMA_VERSION

from .noise import scaled_imu_cfg


# The set of methods and where their checkpoints live is owned by
# eval_sensor_robustness.py (METHOD_SPECS); this module only knows how to turn
# one checkpoint into an (act, on_step) pair.


def _student_checkpoint(path: Path, device, num_envs: int, imu_scale: float):
    """Restore a history student. Mirrors eval_distillation_grid.py:_load_student."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    method = checkpoint.get("method")
    if checkpoint.get("jose_schema_version") != SCHEMA_VERSION or method not in ("joint_only", "imu"):
        raise ValueError(f"{path} is not a JOSE joint-only or IMU history student")
    if checkpoint.get("explicit_linear_velocity") is not False:
        raise ValueError("History checkpoint does not satisfy the deploy observation contract")

    config = checkpoint["model_config"]
    student = HistoryMLPStudent(
        config["frame_dim"], config["action_dim"], config["window"], tuple(config["hidden_dims"])
    ).to(device)
    student.load_state_dict(checkpoint["model_state_dict"])
    student.eval()

    observation_normalizer = RunningNormalizer(config["input_dim"], device)
    observation_normalizer.load_state_dict(checkpoint["observation_normalizer"])
    action_normalizer = RunningNormalizer(config["action_dim"], device, clip=10.0)
    action_normalizer.load_state_dict(checkpoint["action_normalizer"])

    history = ObservationHistory(num_envs, config["window"], config["frame_dim"], device)
    # The checkpoint records the IMU model this student was TRAINED with. The
    # sweep asks how it behaves at other magnitudes, so the trained model is the
    # unit and the scale multiplies it -- a student trained clean and one trained
    # noisy are then measured against the same physical noise at each point.
    trained_cfg = SensorCorruptionCfg(**checkpoint["sensor_corruption"])
    base = SensorCorruptionCfg(**{**checkpoint["sensor_corruption"], "enabled": True})
    corruptor = SensorCorruptor(num_envs, device, scaled_imu_cfg(base, imu_scale))
    return method, student, history, observation_normalizer, action_normalizer, corruptor, trained_cfg


def load_student(core, path: Path, device, num_envs: int, imu_scale: float):
    """The joint-only or IMU distillation baseline."""
    method, student, history, obs_norm, act_norm, corruptor, trained = _student_checkpoint(
        path, device, num_envs, imu_scale
    )
    frame = build_frame_fn(core, method, IMUObservationSpec(), corruptor)
    # joint_only builds its frame from joint state alone, so the corruptor is
    # never consulted and imu_scale cannot affect it. That flatness is a result.
    act, on_step = build_student_policy(
        student, history, obs_norm, act_norm, lambda: frame(True), extra_resets=(corruptor.reset,)
    )
    info = {
        "checkpoint": str(path),
        "window": student.window if hasattr(student, "window") else None,
        "trained_imu_noise": trained.__dict__,
        "eval_imu_scale": imu_scale,
    }
    return act, on_step, info


def load_teacher(adapter, teacher_agent):
    """The frozen expert, reading the policy observation directly.

    It has no IMU and no estimator, so it moves only on the encoder axis -- and
    there it moves because the observation itself was degraded, which is what
    makes it the ceiling for that axis rather than a competitor on it.
    """

    def act(observations: torch.Tensor) -> torch.Tensor:
        return adapter.action(teacher_agent, observations)

    def on_step(_dones) -> None:
        return None

    return act, on_step, {"checkpoint": None}


def load_jose(adapter, teacher_agent, path: Path, device, num_envs: int):
    """JOSE: joint encoders in, privileged base state out, injected into the teacher."""
    estimator, payload = load_estimator(path, device)
    config = payload["model_config"]
    window = int(payload.get("window") or config.get("window"))
    history = HistoryBuffer(num_envs, window, adapter.input_dim, device)
    sequence_model = config["type"].upper() != "MLP"

    @torch.no_grad()
    def act(observations: torch.Tensor) -> torch.Tensor:
        frame = adapter.estimator_input()
        flattened = history.push(frame)
        estimate = estimator.predict(flattened if sequence_model else frame)
        return adapter.action(teacher_agent, adapter.inject_estimate(observations, estimate))

    def on_step(dones) -> None:
        if dones is not None and dones.any():
            history.reset(dones)

    estimator.eval()
    return act, on_step, {"checkpoint": str(path), "window": window, "estimator": config["type"]}


def load_set(adapter, teacher_agent, path: Path, device, num_envs: int, imu_scale: float = 0.0):
    """The SET baseline: joints + IMU in, privileged state out.

    ``adapter`` must already be a :class:`SETPolicyAdapter`, since SET's input is
    wider than JOSE's.

    SET publishes no IMU randomization and we add none, so on the IMU axis it is
    the un-randomized arm -- but un-randomized describes its *training*, not its
    test. It is degraded at evaluation exactly as the clean-trained distillation
    student is, and against the same physical noise, because the axis asks how a
    method behaves when its sensor goes bad and a method handed a clean sensor is
    not on the axis at all. On locomotion this matters twice over: SET does not
    merely read the IMU, it passes all six of those dimensions straight through
    into the teacher's observation, so a clean pass-through would make the sweep
    measure nothing.

    The base model is the default :class:`SensorCorruptionCfg`, which is the one
    the distillation students recorded, so a given level is the same physical
    noise for every method that reads an IMU.
    """
    from jose.set_baseline.model import SETEstimator

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = checkpoint["model_config"]
    estimator = SETEstimator(
        observation_dim=config["observation_dim"],
        target_dim=config["target_dim"],
        output_dim=config["output_dim"],
        context=config["context"],
        width=config["width"],
        blocks=config["blocks"],
        heads=config["heads"],
        estimated_indices=tuple(config["estimated_indices"]),
        pass_through_indices=tuple(config["pass_through_indices"]),
    ).to(device)
    estimator.load_state_dict(checkpoint["model_state_dict"])
    estimator.eval()

    corruptor = SensorCorruptor(num_envs, device, scaled_imu_cfg(SensorCorruptionCfg(), imu_scale))
    adapter.imu_corruptor = corruptor
    adapter.invalidate_imu()

    context = config["context"]
    history = HistoryBuffer(num_envs, context, adapter.input_dim, device)
    has_pass_through = bool(config["pass_through_indices"])

    @torch.no_grad()
    def act(observations: torch.Tensor) -> torch.Tensor:
        sequence = history.push(adapter.estimator_input())
        estimate = estimator.predict_step(
            sequence, adapter.pass_through_values() if has_pass_through else None
        )
        return adapter.action(teacher_agent, adapter.inject_estimate(observations, estimate))

    def on_step(dones) -> None:
        # Every step, not only on a reset: this is what makes the step's two IMU
        # consumers share one sensor read rather than drawing two.
        adapter.invalidate_imu()
        if dones is not None and dones.any():
            history.reset(dones)
            estimator.reset(dones)
            corruptor.reset(reset_ids(dones))

    return act, on_step, {
        "checkpoint": str(path), "context": context, "estimator": "SET",
        "eval_imu_scale": imu_scale,
    }
