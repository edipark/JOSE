"""Which target dimensions SET estimates, and which it simply reads.

SET is a joint+IMU estimator: in the original paper its input
``o = (omega, phi, q, qdot, p, cmd)`` and its output ``o' = (h, v)`` are
disjoint, so the network only ever predicts quantities it cannot measure. Our
targets were designed for a joint-encoder-only estimator and are therefore not
disjoint from an IMU input. Keeping that property is what makes this a faithful
port rather than a handicapped one: a dimension present verbatim in ``o`` is
passed through, never regressed.

"Verbatim" is meant literally, and the frame conventions decide it:

**locomotion** -- the target is
``(root_lin_vel_b, root_ang_vel_b, projected_gravity_b)``
(``ppo_walk/walk_estimator_env.py``), and ``get_distillation_sensor_state``
returns ``root_ang_vel_b`` and ``projected_gravity_b`` -- the *same expressions*
off the same ``ArticulationData``. Six dimensions pass through; SET estimates the
three base linear-velocity components, which is exactly the ``v`` of the paper.

**AMP** -- the target's ``base_ang_vel`` is ``body_ang_vel_w`` (``g1_amp_env.py``;
``compute_amp_observation`` concatenates it unrotated, so it is in the *world*
frame), while an IMU measures body-frame angular velocity, which is what
``get_distillation_sensor_state`` returns as
``quat_apply_inverse(quat, body_ang_vel_w)``. Converting between them needs the
root orientation, which is itself part of what has to be estimated. **No AMP
dimension matches verbatim**, so SET estimates all 43 -- the same output as JOSE,
which makes their RMSE directly comparable.

Tempting near-misses that are deliberately *not* passed through:

* AMP ``base_normal`` is ``R z_hat`` in world coordinates while projected gravity
  is ``R^-1 (-z_hat)`` in body coordinates. Those are different vectors unless R
  is symmetric, and treating them as one would quietly corrupt the baseline.
* AMP ``base_tangent`` needs absolute heading, which no IMU provides.
"""

from __future__ import annotations

from jose.schema import AMP_PRIVILEGED_NAMES, PPO_WALK_OBSERVATION_SCHEMA


#: Target-vector index -> the sensor entry it is read from, per adapter.
#: Empty for an adapter means every dimension is estimated.
PASS_THROUGH: dict[str, dict[int, tuple[str, int]]] = {
    # (sensor key, component) for the six body-frame dimensions the IMU measures.
    "ppo_walk": {
        3: ("angular_velocity", 0),
        4: ("angular_velocity", 1),
        5: ("angular_velocity", 2),
        6: ("projected_gravity", 0),
        7: ("projected_gravity", 1),
        8: ("projected_gravity", 2),
    },
    # World-frame target versus body-frame sensor: nothing matches verbatim.
    "amp": {},
}

TARGET_NAMES: dict[str, tuple[str, ...]] = {
    "ppo_walk": PPO_WALK_OBSERVATION_SCHEMA.estimator_target_names,
    "amp": AMP_PRIVILEGED_NAMES,
}


def split(adapter_name: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """``(estimated_indices, pass_through_indices)`` into the target vector."""
    if adapter_name not in PASS_THROUGH:
        raise ValueError(f"Unknown adapter {adapter_name!r}; choose amp or ppo_walk")
    measured = PASS_THROUGH[adapter_name]
    width = len(TARGET_NAMES[adapter_name])
    estimated = tuple(index for index in range(width) if index not in measured)
    return estimated, tuple(sorted(measured))


def describe(adapter_name: str) -> dict:
    """Human-readable record of the split, for the run's metadata."""
    estimated, measured = split(adapter_name)
    names = TARGET_NAMES[adapter_name]
    return {
        "adapter": adapter_name,
        "target_dim": len(names),
        "estimated_dim": len(estimated),
        "pass_through_dim": len(measured),
        "estimated_names": [names[index] for index in estimated],
        "pass_through_names": [names[index] for index in measured],
    }
