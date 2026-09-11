"""CPU tests for the SET baseline. No simulator required.

The load-bearing one is `test_training_and_inference_pack_identically`: if the
packed tensor built during offline collection ever disagrees with the one
`SETEstimator.predict_step` assembles at inference, SET trains on one problem
and is evaluated on another. That failure is silent and would show up as "the
baseline is bad", which is the worst possible way to be wrong about a baseline.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from jose.distillation.command_eval import reset_ids
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


def test_dagger_rollout_packs_and_labels_correctly():
    """Run the on-policy collector against fakes, so a real round does not spend
    an hour before discovering a shape error.

    Checks the three things that can silently go wrong: the packed width has to
    equal what ``SETEstimator`` was built to accept, the labels have to be ground
    truth at the state actually reached (and only the estimated dimensions), and
    the estimator mask has to scale with the ratio.
    """
    from jose.set_baseline.dagger import collect_dagger_rollout
    import jose.set_baseline.dagger as dagger_module

    n, obs_dim, target_dim, context = 6, 11, 9, 4
    estimated, passed = (0, 1, 2), (3, 4, 5, 6, 7, 8)

    class _Schema:
        estimator_target_dim = target_dim
        action_dim = 5

    class _Adapter:
        schema = _Schema()
        input_dim = obs_dim

        def __init__(self):
            self.t = 0

        def name(self):
            return "ppo_walk"

        def estimator_input(self):
            return torch.full((n, obs_dim), float(self.t))

        def estimator_target(self):
            return torch.arange(target_dim).float().repeat(n, 1) + self.t

        def action(self, agent, observations):
            return torch.zeros(n, 5)

        def inject_estimate(self, observations, estimate):
            assert estimate.shape == (n, target_dim)
            return observations

    class _Env:
        def reset(self):
            return torch.zeros(n, 3), {}

        def step(self, action):
            adapter.t += 1
            done = torch.zeros(n, dtype=torch.bool)
            if adapter.t == 3:
                done[:2] = True          # force a mid-rollout reset
            return torch.zeros(n, 3), None, done, torch.zeros(n, dtype=torch.bool), {}

    class _Teacher:
        def enable_training_mode(self, *args, **kwargs):
            pass

    original = dagger_module.force_skrl_isaaclab_reset
    dagger_module.force_skrl_isaaclab_reset = lambda env: None
    try:
        adapter = _Adapter()
        # Built the way train_set_baseline.py builds it: the first argument is the
        # adapter's input dim, and the model adds the target dim itself.
        estimator = SETEstimator(
            adapter.input_dim, target_dim, context=context, width=16, heads=2,
            blocks=1, dropout=0.0,
            estimated_indices=estimated, pass_through_indices=passed,
        )
        assert estimator.input_dim == obs_dim + target_dim
        pass_through_fn = lambda: torch.zeros(n, len(passed))

        seen = {}
        for ratio in (0.0, 0.5, 1.0):
            adapter.t = 0
            data, stats = collect_dagger_rollout(
                _Env(), adapter, _Teacher(), estimator, steps=5, context=context,
                estimated_indices=estimated, estimator_ratio=ratio,
                pass_through_fn=pass_through_fn,
            )
            assert data.histories.shape == (5 * n, context, estimator.input_dim)
            assert data.targets.shape == (5 * n, len(estimated))
            assert torch.isfinite(data.histories).all()
            assert stats["collection_protocol"] == "on_policy_dagger_rollout"
            assert stats["deaths"] == 2
            seen[ratio] = stats["estimator_driven_steps"]

        assert seen[0.0] == 0 and seen[1.0] == 5 * n
        assert 0 < seen[0.5] < seen[1.0], seen

        adapter.t = 0
        data, _ = collect_dagger_rollout(
            _Env(), adapter, _Teacher(), estimator, steps=2, context=context,
            estimated_indices=estimated, estimator_ratio=1.0,
            pass_through_fn=pass_through_fn,
        )
        # Ground truth at the state reached, sliced to the dimensions SET predicts.
        assert torch.allclose(data.targets[0], torch.tensor([0.0, 1.0, 2.0]))

        adapter.t = 0
        _, stats = collect_dagger_rollout(
            _Env(), adapter, _Teacher(), estimator, steps=5, context=context,
            estimated_indices=estimated, estimator_ratio=0.5,
            pass_through_fn=pass_through_fn, max_samples=10,
        )
        assert stats["samples"] == 10
    finally:
        dagger_module.force_skrl_isaaclab_reset = original


def test_set_dagger_ratio_schedule_matches_jose():
    """A SET+DAgger round must mix exactly as the JOSE round of the same index."""
    from jose.set_baseline.dagger import estimator_ratio_for

    def jose_ratio(round_index, rounds, initial, final, schedule):
        # Transcribed from train_state_estimator.py's round loop.
        if schedule == "constant" or rounds <= 1:
            return initial
        progress = (round_index - 1) / (rounds - 1)
        return initial + progress * (final - initial)

    for k in range(1, 11):
        assert estimator_ratio_for(k, 10, 0.8, 1.0, "linear") == pytest.approx(
            jose_ratio(k, 10, 0.8, 1.0, "linear")
        )
    assert estimator_ratio_for(1, 1, 0.8, 1.0, "linear") == pytest.approx(0.8)
    assert estimator_ratio_for(5, 10, 0.8, 1.0, "constant") == pytest.approx(0.8)
    # The last round is fitted entirely on the closed loop it will run at deployment.
    assert estimator_ratio_for(10, 10, 0.8, 1.0, "linear") == pytest.approx(1.0)


def test_dagger_collection_feeds_predict_step_the_observation_history_only():
    """The packing trap, from the other side.

    ``predict_step`` takes ``(batch, context, observation_dim)`` and concatenates
    the privileged half from its own ring of past outputs. The on-policy collector
    also builds a *packed* tensor -- observation history plus ground-truth lagged
    privileged history -- for the training row. Handing that packed tensor to
    ``predict_step`` instead of the observation history appends a second
    privileged block, and the estimator is driven on an input it was never fitted
    on. It is the same failure ``set_baseline/collect.py``'s module docstring
    warns about, and it is silent: the shapes only disagree at the model's first
    linear layer.
    """
    source = (Path(__file__).parent / "set_baseline" / "dagger.py").read_text(encoding="utf-8")
    body = source[source.index("if estimator_ratio > 0.0:"):source.index("estimated_obs =")]
    assert "estimator.predict_step(\n                sequence," in body, body
    assert "predict_step(\n                packed" not in body

    # And the training row keeps the packed form, which is what SET is fitted on.
    row = source[source.index("packed_rows.append"):source.index("target_rows.append")]
    assert "packed[sample_ids]" in row


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
    ids = reset_ids(torch.tensor([True, False, True]))
    assert ids.tolist() == [0, 2] and len(ids) == 2
    assert reset_ids(None) is None
    assert reset_ids(torch.tensor([1, 2])).tolist() == [1, 2]


def _nested_function(path, outer, inner):
    """Return the AST of a function defined inside another, by name.

    Parsed rather than imported: train_set_baseline.py pulls in isaaclab at
    module scope, which needs a live Omniverse app. Parsed rather than grepped
    because a substring search matches the comment that explains the call as
    readily as the call itself -- which is how a guard like this quietly stops
    guarding anything.
    """
    import ast

    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == outer:
            for child in ast.walk(node):
                if isinstance(child, ast.FunctionDef) and child.name == inner:
                    return child
    raise AssertionError(f"{inner} not found inside {outer} in {path}")


def _attribute_calls(node):
    import ast

    return {
        f"{ast.unparse(c.func.value)}.{c.func.attr}"
        for c in ast.walk(node)
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
    }


def test_command_grid_on_step_drops_the_cached_imu_read():
    """The evaluation callback must clear the adapter's one-per-step sensor cache.

    SETPolicyAdapter caches a single *corrupted* IMU draw so that
    ``estimator_input`` and ``pass_through_values`` share one measurement instead
    of advancing the latency ring twice. Nothing refills it: a caller that never
    invalidates keeps step 0's reading for the whole evaluation, so with a
    corruptor installed the policy walks believing it is not rotating and that
    gravity points where it did at t=0.

    That is not hypothetical. Every recorded ``set_imu_noise`` seed reports
    ``grid_survival_rate`` 0.0000 with ~3000 falls while its episode evaluator --
    which invalidates inside its own loop -- reports 998 of 1000 steps for the
    same weights. Two evaluators of one policy cannot both be right, and the
    disagreement is the signature of this bug.
    """
    on_step = _nested_function("train_set_baseline.py", "main", "on_step")
    calls = _attribute_calls(on_step)
    assert "adapter.invalidate_imu" in calls, (
        "on_step must call adapter.invalidate_imu(); without it the command-grid "
        "evaluation of any --imu-noise-scale run reads a frozen sensor"
    )
    assert "imu_corruptor.reset" in calls, (
        "on_step must re-draw the per-episode gyro bias for environments that "
        "reset; one fixed offset for a whole run is an easier problem than deployment"
    )
