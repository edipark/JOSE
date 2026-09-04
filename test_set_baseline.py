"""CPU tests for the SET baseline. No simulator required.

The load-bearing one is `test_training_and_inference_pack_identically`: if the
packed tensor built during offline collection ever disagrees with the one
`SETEstimator.predict_step` assembles at inference, SET trains on one problem
and is evaluated on another. That failure is silent and would show up as "the
baseline is bad", which is the worst possible way to be wrong about a baseline.
"""

from __future__ import annotations

import pytest
import torch

from jose.distillation.command_eval import _reset_ids
from jose.set_baseline.collect import pack
from jose.set_baseline.model import SETEstimator
from jose.set_baseline.targets import PASS_THROUGH, describe, split


class _HistoryBuffer:
    """Local copy of estimator/pipeline.py's buffer so the test needs no simulator."""

    def __init__(self, num_envs, window, dim):
        self.values = torch.zeros(num_envs, window, dim)

    def push(self, frame):
        self.values = torch.roll(self.values, -1, dims=1)
        self.values[:, -1] = frame
        return self.values


def test_locomotion_passes_through_exactly_the_imu_dimensions():
    estimated, measured = split("ppo_walk")
    assert measured == (3, 4, 5, 6, 7, 8)
    assert estimated == (0, 1, 2)
    body = describe("ppo_walk")
    assert body["pass_through_names"] == [
        "base_ang_vel_x", "base_ang_vel_y", "base_ang_vel_z",
        "projected_gravity_x", "projected_gravity_y", "projected_gravity_z",
    ]
    assert body["estimated_names"] == ["base_lin_vel_x", "base_lin_vel_y", "base_lin_vel_z"]


def test_amp_passes_nothing_through():
    """The AMP target's angular velocity is world-frame; an IMU reads body-frame."""
    estimated, measured = split("amp")
    assert measured == ()
    assert len(estimated) == 43
    assert PASS_THROUGH["amp"] == {}


def test_split_partitions_the_target():
    for adapter in ("amp", "ppo_walk"):
        estimated, measured = split(adapter)
        assert sorted(estimated + measured) == list(range(len(describe(adapter)["estimated_names"]) + len(measured)))
        assert not set(estimated) & set(measured)


def test_estimated_and_passed_through_must_partition():
    with pytest.raises(ValueError):
        SETEstimator(8, 4, estimated_indices=(0, 1), pass_through_indices=(1, 2, 3))
    with pytest.raises(ValueError):
        SETEstimator(8, 4, estimated_indices=(0, 1), pass_through_indices=(2,))


def test_training_and_inference_pack_identically():
    """Offline collection and closed-loop inference must build the same tensor.

    Collection pushes the *previous* target into the privileged history; the
    model's ring is written *after* each prediction. Both must leave o'_{t-1} in
    the newest slot alongside o_t.
    """
    torch.manual_seed(0)
    envs, context, obs_dim, target_dim = 3, 5, 7, 4
    model = SETEstimator(obs_dim, target_dim, context=context, width=16, heads=2, blocks=1, dropout=0.0)
    model.eval()

    observations = [torch.randn(envs, obs_dim) for _ in range(8)]
    privileged = [torch.randn(envs, target_dim) for _ in range(8)]

    collect_obs = _HistoryBuffer(envs, context, obs_dim)
    collect_priv = _HistoryBuffer(envs, context, target_dim)
    previous = torch.zeros(envs, target_dim)
    model.privileged_ring = torch.zeros(envs, context, target_dim)
    infer_obs = _HistoryBuffer(envs, context, obs_dim)

    for step in range(8):
        collected = pack(collect_obs.push(observations[step]), collect_priv.push(previous))
        sequence = infer_obs.push(observations[step])
        inferred = torch.cat((sequence, model.privileged_ring), dim=-1)
        assert torch.equal(collected, inferred), f"packing diverged at step {step}"

        # Teacher forcing on the inference side, so both sides advance alike.
        model.privileged_ring = torch.roll(model.privileged_ring, -1, dims=1)
        model.privileged_ring[:, -1] = privileged[step]
        previous = privileged[step]


def test_predict_step_writes_the_full_vector_and_returns_it():
    estimated, measured = split("ppo_walk")
    model = SETEstimator(
        11, 9, context=4, width=16, heads=2, blocks=1, dropout=0.0,
        estimated_indices=estimated, pass_through_indices=measured,
    )
    model.eval()
    model.privileged_ring = torch.zeros(0, 4, 9)
    history = torch.randn(2, 4, 11)
    values = torch.randn(2, len(measured))
    full = model.predict_step(history, values)
    assert full.shape == (2, 9)
    assert torch.allclose(full[:, list(measured)], values), "measured dimensions were not written verbatim"
    assert torch.equal(model.privileged_ring[:, -1], full), "ring must remember the full vector"


def test_predict_step_requires_the_measured_dimensions():
    estimated, measured = split("ppo_walk")
    model = SETEstimator(
        11, 9, context=4, width=16, heads=2, blocks=1, dropout=0.0,
        estimated_indices=estimated, pass_through_indices=measured,
    )
    model.privileged_ring = torch.zeros(0, 4, 9)
    with pytest.raises(ValueError):
        model.predict_step(torch.randn(2, 4, 11))


def test_attention_is_causal():
    """The newest token may see the past; the past may not see the newest."""
    model = SETEstimator(6, 3, context=4, width=16, heads=2, blocks=2, dropout=0.0)
    model.eval()
    packed = torch.randn(2, 4, model.input_dim)
    with torch.no_grad():
        base = model(packed)
        future = packed.clone()
        future[:, -1] += 5.0
        past = packed.clone()
        past[:, 0] += 5.0
        assert not torch.allclose(base, model(future))
        assert not torch.allclose(base, model(past))
    mask = model.attention_mask
    assert bool(mask[0, 1]) and not bool(mask[1, 0]), "mask is not lower-triangular"


def test_reset_clears_only_the_selected_environments():
    model = SETEstimator(6, 3, context=4, width=16, heads=2, blocks=1, dropout=0.0)
    model.privileged_ring = torch.ones(4, 4, 3)
    model.reset(torch.tensor([True, False, True, False]))
    assert model.privileged_ring[0].abs().sum() == 0
    assert model.privileged_ring[1].abs().sum() > 0
    assert model.privileged_ring[2].abs().sum() == 0
    assert model.privileged_ring[3].abs().sum() > 0


def test_reset_ids_converts_a_boolean_mask_to_indices():
    """SensorCorruptor.reset sizes its draws with len(ids); a mask would be wrong."""
    ids = _reset_ids(torch.tensor([True, False, True]))
    assert ids.tolist() == [0, 2] and len(ids) == 2
    assert _reset_ids(None) is None
    assert _reset_ids(torch.tensor([1, 2])).tolist() == [1, 2]
