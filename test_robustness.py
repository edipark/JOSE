"""CPU checks for the sensor-degradation experiments.

Everything here runs without a simulator, which is the point: the robustness
queue is roughly eleven GPU-hours and its arms are distinguished only by
directory names and noise magnitudes. A typo in either produces plausible
numbers for the wrong thing, and the wrong thing is not visible in a plot.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

# Plain imports: none of these modules touches isaaclab, which is what lets the
# whole file run on a machine with no simulator. test_jose.py's _load_module
# dance exists for modules that cannot be imported that way; using it here would
# have meant faking `jose.distillation` in sys.modules, which then breaks
# test_set_baseline.py's real import of it in the same pytest session.
from jose.distillation import imu
from jose.robustness import noise, registry


def _env_cfg(policy):
    """The two-level shape apply_encoder_noise_cfg reads: cfg.observations.policy."""
    observations = type("_Observations", (), {"policy": policy})()
    return type("_EnvCfg", (), {"observations": observations})()


# -- the noise models -----------------------------------------------------


def test_encoder_scale_zero_is_exactly_the_clean_condition():
    """0x must be a no-op, not "small noise": it is Table I's row."""
    position, velocity = torch.randn(4, 29), torch.randn(4, 29)
    corruptor = noise.EncoderCorruptor(4, 29, "cpu", noise.EncoderNoiseCfg(scale=0.0))
    noisy_position, noisy_velocity = corruptor.corrupt(position, velocity)
    assert torch.equal(noisy_position, position)
    assert torch.equal(noisy_velocity, velocity)


@pytest.mark.parametrize("scale", [1.0, 2.0, 4.0])
def test_encoder_noise_respects_the_environments_own_bounds(scale):
    """1x is the walk teacher's own training noise; the sweep multiplies it."""
    zeros = torch.zeros(64, 29)
    corruptor = noise.EncoderCorruptor(64, 29, "cpu", noise.EncoderNoiseCfg(scale=scale))
    position, velocity = corruptor.corrupt(zeros, zeros)
    assert position.abs().max() <= scale * noise.ENCODER_POSITION_HALF_WIDTH + 1e-9
    assert velocity.abs().max() <= scale * noise.ENCODER_VELOCITY_HALF_WIDTH + 1e-9


def test_encoder_bias_persists_within_an_episode_and_resets_per_environment():
    """A constant offset is what a filter cannot average away, so it must persist."""
    cfg = noise.EncoderNoiseCfg(scale=1.0, bias_fraction=0.5)
    corruptor = noise.EncoderCorruptor(4, 29, "cpu", cfg)
    before = corruptor._position_bias.clone()
    corruptor.reset(torch.tensor([0, 2]))
    assert not torch.equal(before[0], corruptor._position_bias[0])
    assert torch.equal(before[1], corruptor._position_bias[1])
    assert torch.equal(before[3], corruptor._position_bias[3])


def test_encoder_bias_defaults_off_so_one_x_matches_the_environment():
    cfg = noise.EncoderNoiseCfg(scale=1.0)
    corruptor = noise.EncoderCorruptor(4, 29, "cpu", cfg)
    corruptor.reset()
    assert corruptor._position_bias.abs().max() == 0.0


def test_imu_scaling_leaves_latency_alone():
    """Latency is a step count, not a magnitude: scaling it mixes two axes."""
    base = imu.SensorCorruptionCfg()
    scaled = noise.scaled_imu_cfg(base, 3.0)
    assert scaled.gyro_noise_std == pytest.approx(3.0 * base.gyro_noise_std)
    assert scaled.gyro_bias_std == pytest.approx(3.0 * base.gyro_bias_std)
    assert scaled.gravity_tilt_std_rad == pytest.approx(3.0 * base.gravity_tilt_std_rad)
    assert scaled.max_latency_steps == base.max_latency_steps
    assert noise.scaled_imu_cfg(base, 0.0).enabled is False


def test_observation_noise_isolates_the_encoder_terms():
    """Base-velocity noise would corrupt the comparison, not degrade a sensor."""

    class Term:
        def __init__(self, has_noise):
            self.noise = type("N", (), {"n_min": -1.0, "n_max": 1.0})() if has_noise else None

    class Policy:
        def __init__(self):
            self.enable_corruption = False
            for name in noise.NON_ENCODER_TERMS:
                setattr(self, name, Term(True))
            for name in noise.ENCODER_TERMS:
                setattr(self, name, Term(True))

    policy = Policy()
    applied = noise.apply_encoder_noise_cfg(_env_cfg(policy), 2.0)
    assert applied["enable_corruption"] is True
    for name in noise.NON_ENCODER_TERMS:
        assert getattr(policy, name).noise is None, f"{name} should have been stripped"
    for name in noise.ENCODER_TERMS:
        term = getattr(policy, name)
        assert (term.noise.n_min, term.noise.n_max) == (-2.0, 2.0)


def test_observation_noise_at_zero_turns_corruption_off():
    class Policy:
        enable_corruption = True

    policy = Policy()
    applied = noise.apply_encoder_noise_cfg(_env_cfg(policy), 0.0)
    assert policy.enable_corruption is False and applied["scale"] == 0.0


# -- the arm registry -----------------------------------------------------


def test_every_axis_entry_is_a_known_method():
    for axis, methods in registry.AXIS_METHODS.items():
        for method in methods:
            assert method in registry.METHOD_SPECS, f"{axis}: {method}"


def test_every_hardened_arm_has_an_unhardened_partner_on_its_axis():
    """A hardened arm with no partner is a number with nothing to compare to."""
    for axis, methods in registry.AXIS_METHODS.items():
        for method in methods:
            partner = registry.RANDOMIZATION_PAIRS.get(method)
            if partner is not None:
                assert partner in methods, f"{axis}: {method} has no partner {partner}"


def test_the_encoder_axis_hardens_every_trained_method():
    """Hardening some arms and not others measures the treatment, not the method."""
    encoder = set(registry.AXIS_METHODS["encoder"])
    trained = {
        method for method in encoder
        if registry.METHOD_SPECS[method][0] != "teacher"
        and method not in registry.RANDOMIZATION_PAIRS
    }
    hardened = {registry.RANDOMIZATION_PAIRS[m] for m in encoder if m in registry.RANDOMIZATION_PAIRS}
    assert trained == hardened, f"unhardened on the encoder axis: {sorted(trained - hardened)}"


def test_no_two_arms_share_a_directory():
    """Two arms in one directory silently overwrite each other."""
    slugs = [slug for _, slug in registry.METHOD_SPECS.values() if slug is not None]
    assert len(slugs) == len(set(slugs))


def test_resolve_separates_hardened_from_unhardened_paths():
    study, set_study = Path("/study"), Path("/set")
    seen = set()
    for method in registry.METHOD_SPECS:
        kind, path = registry.resolve(method, study, set_study, 42)
        if path is None:
            assert kind == "teacher"
            continue
        assert path not in seen, f"{method} collides with an earlier arm"
        seen.add(path)


def test_resolve_uses_the_right_filename_per_loader():
    study, set_study = Path("/study"), Path("/set")
    assert registry.resolve("jose", study, set_study, 42)[1].name == "best_estimator.pt"
    assert registry.resolve("imu_dr", study, set_study, 42)[1].name == "student_best_eval.pt"
    assert registry.resolve("set_enc", study, set_study, 42)[1].name == "set_estimator.pt"


def test_resolve_skips_set_when_no_set_study_is_given():
    kind, path = registry.resolve("set", Path("/study"), None, 42)
    assert kind == "set" and path is None


def test_resolve_rejects_an_unknown_method():
    with pytest.raises(KeyError):
        registry.resolve("nope", Path("/study"), None, 42)
