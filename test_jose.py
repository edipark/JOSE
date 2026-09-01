"""CPU unit tests for the JOSE G1 contracts and report pipeline."""

from __future__ import annotations

import importlib.util
import json
from collections import defaultdict, deque
from pathlib import Path
import subprocess
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml


JOSE_DIR = Path(__file__).parent


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


schema = _load_module("jose_g1_schema_test", JOSE_DIR / "schema.py")
compat = _load_module("jose_g1_compat_test", JOSE_DIR / "skrl_compat.py")
models = _load_module("jose_g1_models_test", JOSE_DIR / "estimator" / "models.py")
reporting = _load_module("jose_g1_reporting_test", JOSE_DIR / "reporting.py")
ablation_catalog = _load_module("jose_g1_ablation_catalog_test", JOSE_DIR / "ablation_catalog.py")
ablation_common = _load_module("jose_g1_ablation_common_test", JOSE_DIR / "ablation_common.py")
task_math = _load_module("jose_g1_task_math_test", JOSE_DIR / "task_math.py")
motion_loader = _load_module("jose_g1_motion_loader_test", JOSE_DIR / "motions" / "motion_loader.py")
imu = _load_module("jose_g1_imu_test", JOSE_DIR / "distillation" / "imu.py")
history = _load_module("jose_g1_history_test", JOSE_DIR / "distillation" / "history.py")
rollout_diagnostics = _load_module(
    "jose_g1_rollout_diagnostics_test", JOSE_DIR / "tools" / "rollout_diagnostics.py"
)


def _load_pipeline_module():
    package = "jose_pipeline_test_package"
    estimator_package = f"{package}.estimator"
    for name in (package, estimator_package):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    sys.modules[f"{package}.schema"] = schema
    sys.modules[f"{package}.skrl_compat"] = compat
    adapters_stub = types.ModuleType(f"{estimator_package}.adapters")
    adapters_stub.PolicyAdapter = object
    sys.modules[adapters_stub.__name__] = adapters_stub
    sys.modules[f"{estimator_package}.models"] = models
    # metrics.py imports ..skrl_compat, so it has to be loaded inside the stub
    # package rather than flat like the modules above.
    _load_module(f"{estimator_package}.metrics", JOSE_DIR / "estimator" / "metrics.py")
    return _load_module(f"{estimator_package}.pipeline", JOSE_DIR / "estimator" / "pipeline.py")


def test_joint_presets_and_dimensions():
    assert len(schema.G1_JOINT_NAMES) == 29
    assert len(schema.G1_LEG_JOINT_NAMES) == 12
    assert len(schema.G1_UPPER_JOINT_NAMES) == 17
    assert schema.estimator_input_dim("all") == 58
    assert schema.estimator_input_dim("legs") == 24
    assert schema.estimator_input_dim("upper") == 34
    shuffled = tuple(reversed(schema.G1_JOINT_NAMES))
    ids = schema.joint_indices(shuffled, "all")
    assert tuple(shuffled[index] for index in ids) == schema.G1_JOINT_NAMES


def test_rollout_dataset_bounded_append_and_history_projection():
    pipeline = _load_pipeline_module()

    def dataset(start, count):
        ids = torch.arange(start, start + count)
        return pipeline.RolloutDataset(
            histories=ids[:, None, None].expand(-1, 4, 6).clone(),
            targets=ids[:, None].clone(),
            frames=ids[:, None].expand(-1, 6).clone(),
            teacher_actions=ids[:, None].clone(),
        )

    torch.manual_seed(7)
    combined = dataset(0, 4).append(dataset(4, 6), max_size=5)
    assert combined.histories.shape == (5, 4, 6)
    assert torch.equal(combined.histories[:, 0, 0], combined.targets[:, 0])
    assert torch.equal(combined.frames[:, 0], combined.targets[:, 0])
    assert torch.equal(combined.teacher_actions[:, 0], combined.targets[:, 0])

    source = dataset(0, 4)
    assert source.project_joint_history((0, 1, 2), 4, 4, 3) is source
    projected = source.project_joint_history((0, 2), 4, 2, 3)
    assert projected.histories.shape == (4, 2, 4)
    assert projected.frames.shape == (4, 4)


def test_joint_validation_fails_early():
    with pytest.raises(ValueError, match="missing"):
        schema.joint_indices(schema.G1_JOINT_NAMES[:-1], "all")
    with pytest.raises(ValueError, match="Duplicate"):
        schema.joint_indices((*schema.G1_JOINT_NAMES, schema.G1_JOINT_NAMES[0]), "all")


def test_reference_motions_have_the_same_joint_set():
    import numpy as np

    for name in ("G1_walk.npz", "G1_dance.npz", "G1_jump.npz"):
        motion = np.load(JOSE_DIR / "motions" / name)
        names = motion["dof_names"].tolist()
        assert len(names) == len(set(names)) == 29
        assert set(names) == set(schema.G1_JOINT_NAMES)


def test_motion_loader_preserves_recorded_timing_and_velocities():
    import numpy as np

    motion_path = JOSE_DIR / "motions" / "G1_walk.npz"
    data = np.load(motion_path)
    motion = motion_loader.MotionLoader(str(motion_path), torch.device("cpu"))
    assert motion.dt == pytest.approx(1.0 / float(data["fps"]))
    assert torch.equal(
        motion.dof_velocities, torch.as_tensor(data["dof_velocities"], dtype=torch.float32)
    )
    assert torch.equal(
        motion.body_linear_velocities,
        torch.as_tensor(data["body_linear_velocities"], dtype=torch.float32),
    )
    assert torch.equal(
        motion.body_angular_velocities,
        torch.as_tensor(data["body_angular_velocities"], dtype=torch.float32),
    )


def test_g1_timing_and_amp_history():
    import numpy as np

    assert task_math.PHYSICS_DT == pytest.approx(1.0 / 200.0)
    assert task_math.CONTROL_DECIMATION == 4
    assert task_math.POLICY_DT == pytest.approx(1.0 / 50.0)
    assert task_math.EPISODE_LENGTH_S / task_math.POLICY_DT == pytest.approx(1000)
    assert task_math.AMP_HISTORY_STEPS * schema.AMP_OBSERVATION_SCHEMA.policy_dim == 404
    times = task_math.reference_history_times(np.array([1.0])).reshape(1, -1)
    assert times.shape == (1, 4)
    assert np.diff(times[0]) == pytest.approx([-1.0 / 50.0] * 3)


def test_twist_action_mapping_uses_default_pose_scale_and_soft_limits():
    limits = torch.tensor([[-1.0, 1.0], [0.0, 1.5]])
    defaults = torch.tensor([-0.2, 0.4])
    actions = torch.tensor([[-2.0, 0.25], [1.0, 3.0]])
    targets = task_math.twist_action_to_position(actions, defaults, limits)
    assert torch.allclose(targets, torch.tensor([[-1.0, 0.525], [0.3, 1.5]]))


def test_action_finite_difference_penalties():
    previous_previous = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    previous = torch.tensor([[1.0, 1.0, 1.0], [2.0, 4.0, 6.0]])
    current = torch.tensor([[3.0, 1.0, -1.0], [4.0, 8.0, 12.0]])
    action_rate, action_second_difference = task_math.action_finite_difference_penalties(
        current, previous, previous_previous
    )
    assert torch.allclose(action_rate, torch.tensor([8.0 / 3.0, 56.0 / 3.0]))
    assert torch.allclose(action_second_difference, torch.tensor([11.0 / 3.0, 14.0 / 3.0]))


def test_twist_action_mapping_rejects_invalid_scale():
    with pytest.raises(ValueError, match="positive"):
        task_math.twist_action_to_position(
            torch.zeros(1), torch.zeros(1), torch.tensor([[-1.0, 1.0]]), action_scale=0.0
        )


def test_policy_model_does_not_preclip_twist_actions():
    for path in (JOSE_DIR / "agents").glob("*.yaml"):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert config["models"]["policy"]["clip_actions"] is False


def test_observation_schemas_round_trip():
    assert schema.SCHEMA_VERSION == 2
    assert schema.AMP_OBSERVATION_SCHEMA.policy_dim == 101
    assert schema.AMP_OBSERVATION_SCHEMA.estimator_target_dim == 43
    assert schema.AMP_OBSERVATION_SCHEMA.estimator_target_indices == tuple(range(58, 101))
    assert len(schema.AMP_PRIVILEGED_NAMES) == 43
    assert schema.ObservationSchema.from_dict(schema.AMP_OBSERVATION_SCHEMA.to_dict()) == schema.AMP_OBSERVATION_SCHEMA


def test_ppo_walk_schema_matches_the_flattened_observation_layout():
    """The layout table must tile the policy group exactly, with no gaps or overlap."""
    offset = 0
    for name, dim, start, _scale in schema.PPO_WALK_TERM_LAYOUT:
        assert start == offset, f"{name} block starts at {start}, expected {offset}"
        offset += dim * schema.PPO_WALK_HISTORY_LENGTH
    assert offset == schema.PPO_WALK_OBSERVATION_SCHEMA.policy_dim == 495
    assert schema.PPO_WALK_OBSERVATION_SCHEMA.estimator_target_dim == 9
    assert (
        schema.ObservationSchema.from_dict(schema.PPO_WALK_OBSERVATION_SCHEMA.to_dict())
        == schema.PPO_WALK_OBSERVATION_SCHEMA
    )
    # The 9 schema indices address the newest frame of the three estimated terms.
    layout = {name: (dim, start) for name, dim, start, _ in schema.PPO_WALK_TERM_LAYOUT}
    newest = []
    for name in schema.PPO_WALK_ESTIMATOR_TERMS:
        dim, start = layout[name]
        base = start + (schema.PPO_WALK_HISTORY_LENGTH - 1) * dim
        newest.extend(range(base, base + dim))
    assert schema.PPO_WALK_OBSERVATION_SCHEMA.estimator_target_indices == tuple(newest)
    # joint_pos_rel / joint_vel_rel offsets also point at the newest frame.
    for attr, term in (("joint_position_start", "joint_pos_rel"), ("joint_velocity_start", "joint_vel_rel")):
        dim, start = layout[term]
        expected = start + (schema.PPO_WALK_HISTORY_LENGTH - 1) * dim
        assert getattr(schema.PPO_WALK_OBSERVATION_SCHEMA, attr) == expected


def test_ppo_walk_history_indices_cover_every_estimated_frame():
    indices = schema.ppo_walk_history_target_indices()
    assert len(indices) == 9 * schema.PPO_WALK_HISTORY_LENGTH == 45
    assert len(set(indices)) == len(indices)
    assert max(indices) < schema.PPO_WALK_OBSERVATION_SCHEMA.policy_dim
    # Newest frame last, so it coincides with the schema's own target indices.
    assert indices[-9:] == schema.PPO_WALK_OBSERVATION_SCHEMA.estimator_target_indices
    # Every slot must belong to one of the three estimated terms and nothing else.
    layout = {name: (dim, start) for name, dim, start, _ in schema.PPO_WALK_TERM_LAYOUT}
    allowed = set()
    for name in schema.PPO_WALK_ESTIMATOR_TERMS:
        dim, start = layout[name]
        allowed.update(range(start, start + dim * schema.PPO_WALK_HISTORY_LENGTH))
    assert set(indices) == allowed


def test_locomotion_summarize_normalizes_each_axis_by_its_command_range():
    """track_error_norm must be dimensionless: vx is m/s, yaw is rad/s."""
    locomotion = _load_module("jose_g1_locomotion_test", JOSE_DIR / "estimator" / "locomotion.py")

    def row(vx_rmse, vy_rmse, yaw_rmse):
        return {
            "command": {"vx": 0.0, "vy": 0.0, "yaw": 0.0},
            "measured": {"vx": 0.0, "vy": 0.0, "yaw": 0.0},
            "error": {
                "vx_rmse": vx_rmse, "vy_rmse": vy_rmse, "yaw_rmse": yaw_rmse,
                "vx_mean": 0.0, "vy_mean": 0.0, "yaw_mean": 0.0,
            },
            "survival_rate": 1.0, "fall_count": 0, "base_height_m": 0.78,
            "feet_air_time_reward": 0.1, "feet_slide_penalty": 0.0,
            "foot_lifts_per_s": 2.0, "double_stance_fraction": 0.3, "drift_speed_mps": 0.0,
        }

    metrics = locomotion.summarize({"a": row(0.2, 0.1, 0.4), "b": row(0.4, 0.3, 0.6)})
    assert metrics["track_vx_rmse"] == pytest.approx(0.3)
    assert metrics["track_vy_rmse"] == pytest.approx(0.2)
    assert metrics["track_yaw_rmse"] == pytest.approx(0.5)
    # each axis divided by its own range (1.0, 0.5, 1.0) before averaging
    assert locomotion.COMMAND_SCALES == (1.0, 0.5, 1.0)
    assert metrics["track_error_norm"] == pytest.approx((0.3 / 1.0 + 0.2 / 0.5 + 0.5 / 1.0) / 3.0)
    assert len(metrics["command_tracking"]) == 2


def test_locomotion_eval_commands_include_the_zero_command():
    """The zero row is the only one that exposes creeping or marching in place."""
    locomotion = _load_module("jose_g1_locomotion_commands_test", JOSE_DIR / "estimator" / "locomotion.py")
    assert (0.0, 0.0, 0.0) in locomotion.EVAL_COMMANDS
    # forward sweep must have at least two distinct nonzero speeds for the
    # monotonicity sanity check to mean anything
    forward = sorted({vx for vx, vy, yaw in locomotion.EVAL_COMMANDS if vy == 0.0 and yaw == 0.0})
    assert len(forward) >= 3 and forward[0] == 0.0
    # yaw must be probed in both directions, or a sign error passes unnoticed
    yaws = {yaw for vx, vy, yaw in locomotion.EVAL_COMMANDS}
    assert any(y > 0 for y in yaws) and any(y < 0 for y in yaws)


def test_report_table_picks_its_columns_from_the_metrics_present():
    amp_row = {
        "task": "amp_walk", "experiment": "JOSE", "successful": 3, "seeds": 3,
        "episode_length_mean_mean": 850.0, "mpjpe_g_mean": 40.0,
    }
    walk_row = {
        "task": "ppo_walk", "experiment": "JOSE", "successful": 3, "seeds": 3,
        "track_error_norm_mean": 0.25, "track_vx_rmse_mean": 0.15,
        "episode_length_mean_mean": 1000.0,
    }
    amp_only = reporting._markdown_table([amp_row])
    assert "MPJPE-G mm" in amp_only and "Track err" not in amp_only

    walk_only = reporting._markdown_table([walk_row])
    assert "Track err" in walk_only and "vx RMSE" in walk_only
    # MPJPE is dropped for locomotion: gait phase, not estimator error, dominates it
    assert "MPJPE-G mm" not in walk_only

    mixed = reporting._markdown_table([amp_row, walk_row])
    assert "MPJPE-G mm" in mixed and "Track err" in mixed
    assert mixed.count("| Task | Experiment |") == 2


def test_aggregate_adds_teacher_relative_tracking_deltas():
    rows = [
        {"task": "ppo_walk", "experiment": "TeacherGT", "status": "ok",
         "metrics": {"track_error_norm": 0.20, "track_vx_rmse": 0.10, "episode_length_mean": 1000.0}},
        {"task": "ppo_walk", "experiment": "JOSE-LSTM", "status": "ok",
         "metrics": {"track_error_norm": 0.25, "track_vx_rmse": 0.16, "episode_length_mean": 1000.0}},
    ]
    summary = {row["experiment"]: row for row in reporting.aggregate(rows)}
    jose = summary["JOSE-LSTM"]
    assert jose["track_error_norm_delta"] == pytest.approx(0.05)
    assert jose["track_vx_rmse_delta"] == pytest.approx(0.06)
    assert jose["track_error_ratio_percent"] == pytest.approx(125.0)
    # the teacher is its own baseline, so its delta is zero rather than absent
    assert summary["TeacherGT"]["track_error_norm_delta"] == pytest.approx(0.0)


def test_ppo_walk_target_scales_follow_the_observation_terms():
    """base_ang_vel is stored pre-scaled by 0.2, so estimates must be scaled to match."""
    scales = schema.ppo_walk_target_scales()
    assert scales == (1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 1.0, 1.0, 1.0)
    assert len(scales) == schema.PPO_WALK_OBSERVATION_SCHEMA.estimator_target_dim
    declared = {name: scale for name, _, _, scale in schema.PPO_WALK_TERM_LAYOUT}
    for i, name in enumerate(schema.PPO_WALK_ESTIMATOR_TERMS):
        assert scales[3 * i : 3 * i + 3] == (declared[name],) * 3


def test_ppo_walk_injection_replaces_the_whole_history_of_estimated_terms():
    """Leaving old frames untouched would keep ground truth visible to the policy."""
    indices = schema.ppo_walk_history_target_indices()
    observations = torch.zeros(2, schema.PPO_WALK_OBSERVATION_SCHEMA.policy_dim)
    observations[:] = 7.0
    # Offset past the sentinel so no estimated value can coincide with it.
    estimate = torch.arange(2 * len(indices), dtype=torch.float32).reshape(2, len(indices)) + 100.0
    injected = task_math.inject_observation_estimate(observations, estimate, indices)
    assert torch.equal(injected[:, list(indices)], estimate)
    untouched = [i for i in range(observations.shape[1]) if i not in set(indices)]
    assert torch.equal(injected[:, untouched], observations[:, untouched])
    # No frame of an estimated term keeps its original value.
    assert not (injected[:, list(indices)] == 7.0).any()
    # The source tensor is not mutated.
    assert torch.equal(observations, torch.full_like(observations, 7.0))


def test_amp_estimator_replaces_all_privileged_columns_only():
    observations = torch.zeros(2, 101)
    observations[:, :58] = 7.0
    estimate = torch.arange(86, dtype=torch.float32).reshape(2, 43)
    injected = task_math.inject_observation_estimate(
        observations, estimate, schema.AMP_OBSERVATION_SCHEMA.estimator_target_indices
    )
    assert torch.equal(injected[:, :58], observations[:, :58])
    assert torch.equal(injected[:, 58:], estimate)
    assert torch.equal(observations[:, 58:], torch.zeros(2, 43))


@pytest.mark.parametrize("model_type,window", (("LSTM", 50), ("TCN", 50), ("MLP", 1)))
def test_estimator_shapes(model_type, window):
    estimator = models.build_estimator(model_type, input_dim=58, output_dim=43)
    sample = torch.randn(3, window, 58)
    assert estimator(sample).shape == (3, 43)
    estimator.set_normalization(sample, torch.randn(3, 43))
    assert estimator.predict(sample).shape == (3, 43)


def test_default_estimator_matches_amp_schema_v2():
    estimator = models.build_estimator("LSTM")
    assert estimator.input_dim == 58
    assert estimator.output_dim == schema.AMP_OBSERVATION_SCHEMA.estimator_target_dim == 43


def test_history_mlp_estimator_and_distillation_dimensions():
    estimator = models.build_estimator("HISTORY_MLP", input_dim=58, output_dim=43, window=50)
    assert estimator(torch.randn(2, 50, 58)).shape == (2, 43)
    assert history.JOINT_FRAME_DIM == 87
    assert history.IMU_FRAME_DIM == 93
    joint_student = history.HistoryMLPStudent(history.JOINT_FRAME_DIM)
    imu_student = history.HistoryMLPStudent(history.IMU_FRAME_DIM)
    assert joint_student.input_dim == 1827
    assert imu_student.input_dim == 1953
    assert joint_student.config()["explicit_linear_velocity"] is False


def test_projected_gravity_reference_sign_invariance_and_faults():
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    gravity = imu.projected_gravity_from_quaternion(identity)
    assert torch.allclose(gravity, torch.tensor([[0.0, 0.0, -1.0]]), atol=1.0e-6)
    quarter_turn_x = torch.tensor([[2**-0.5, 2**-0.5, 0.0, 0.0]])
    assert torch.allclose(
        imu.projected_gravity_from_quaternion(quarter_turn_x),
        imu.projected_gravity_from_quaternion(-quarter_turn_x),
        atol=1.0e-6,
    )
    assert torch.allclose(
        imu.projected_gravity_from_quaternion(quarter_turn_x), torch.tensor([[0.0, -1.0, 0.0]]), atol=1.0e-5
    )
    spec = imu.IMUObservationSpec(stale_after_s=0.1)
    valid = spec.observe(identity[0], torch.zeros(3), timestamp_s=1.0, now_s=1.05)
    assert valid.valid and valid.projected_gravity.norm() == pytest.approx(1.0)
    assert spec.observe(identity[0], torch.zeros(3), timestamp_s=1.0, now_s=1.2).fault is imu.IMUFault.STALE_TIMESTAMP
    assert spec.observe(torch.zeros(4), torch.zeros(3), timestamp_s=1.0).fault is imu.IMUFault.INVALID_QUATERNION


def test_simulation_and_lowstate_imu_contract_match():
    spec = imu.IMUObservationSpec()
    quaternion = torch.tensor([0.9238795, 0.0, 0.3826834, 0.0])
    gyro = torch.tensor([0.1, -0.2, 0.3])
    simulation = spec.observe(quaternion, gyro, 2.0)
    lowstate = spec.from_low_state(
        {"imu_state": {"quaternion": quaternion.tolist(), "gyroscope": gyro.tolist(), "timestamp_s": 2.0}}
    )
    assert torch.allclose(simulation.angular_velocity, lowstate.angular_velocity)
    assert torch.allclose(simulation.projected_gravity, lowstate.projected_gravity)


def test_history_reset_and_sensor_corruption_contract():
    buffer = history.ObservationHistory(2, 3, history.JOINT_FRAME_DIM, "cpu")
    first = torch.randn(2, history.JOINT_FRAME_DIM)
    flattened = buffer.push(first)
    assert torch.equal(flattened.reshape(2, 3, -1), first[:, None, :].expand(-1, 3, -1))
    buffer.reset(torch.tensor([1]))
    second = torch.randn_like(first)
    buffer.push(second)
    assert torch.equal(buffer.values[1], second[1, None, :].expand(3, -1))
    cfg = imu.SensorCorruptionCfg(gyro_noise_std=0.0, gyro_bias_std=0.0, gravity_tilt_std_rad=0.0, max_latency_steps=2)
    corruptor = imu.SensorCorruptor(2, "cpu", cfg)
    gyro, gravity = corruptor(torch.ones(2, 3), torch.tensor([[0.0, 0.0, -1.0]]).expand(2, -1))
    assert torch.equal(gyro, torch.ones(2, 3))
    assert torch.allclose(gravity.norm(dim=-1), torch.ones(2))


def test_history_rollout_samples_are_snapshotted_before_buffer_mutation():
    buffer = history.ObservationHistory(1, 2, history.JOINT_FRAME_DIM, "cpu")
    first = buffer.push(torch.zeros(1, history.JOINT_FRAME_DIM)).detach().clone()
    second = buffer.push(torch.ones(1, history.JOINT_FRAME_DIM)).detach().clone()

    assert first.data_ptr() != second.data_ptr()
    assert torch.count_nonzero(first) == 0
    assert torch.count_nonzero(second) == history.JOINT_FRAME_DIM

    training_source = (JOSE_DIR / "train_history_student.py").read_text(encoding="utf-8")
    assert "collected_x.append(flattened.detach().clone())" in training_source
    assert "collected_y.append(teacher.detach().clone())" in training_source


def test_history_distillation_uses_student_rollouts_and_teacher_labels():
    training_source = (JOSE_DIR / "train_history_student.py").read_text(encoding="utf-8")
    assert "action = predicted" in training_source
    assert "collected_y.append(teacher.detach().clone())" in training_source
    assert "beta * teacher" not in training_source
    assert '"rollout_policy": "student"' in training_source


def test_dagger_student_normalizer_and_replay_buffer():
    student = models.DaggerStudent()
    assert student(torch.randn(4, 58)).shape == (4, 29)
    normalizer = models.RunningNormalizer(2, "cpu")
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    normalizer.update(values)
    restored = models.RunningNormalizer(2, "cpu")
    restored.load_state_dict(normalizer.state_dict())
    assert torch.allclose(normalizer.normalize(values), restored.normalize(values))
    replay = models.ReplayBuffer(3, 2, 1, "cpu")
    replay.add(torch.arange(10, dtype=torch.float32).reshape(5, 2), torch.arange(5, dtype=torch.float32)[:, None])
    assert replay.size == 3
    observations, actions = replay.sample(4)
    assert observations.shape == (4, 2) and actions.shape == (4, 1)


def test_dagger_beta_schedule():
    assert models.dagger_beta(1.0, 0.5, 0.2, 0) == pytest.approx(1.0)
    assert models.dagger_beta(1.0, 0.5, 0.2, 1) == pytest.approx(0.5)
    assert models.dagger_beta(1.0, 0.5, 0.2, 10) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        models.dagger_beta(1.2, 0.5, 0.2, 0)


def test_dagger_v2_checkpoint_round_trip(tmp_path):
    student = models.DaggerStudent()
    observation_normalizer = models.RunningNormalizer(58, "cpu")
    action_normalizer = models.RunningNormalizer(29, "cpu")
    observation_normalizer.update(torch.randn(8, 58))
    action_normalizer.update(torch.randn(8, 29))
    checkpoint = {
        "jose_schema_version": schema.SCHEMA_VERSION,
        "kind": "dagger_student",
        "model_config": student.config(),
        "model_state_dict": student.state_dict(),
        "observation_normalizer": observation_normalizer.state_dict(),
        "action_normalizer": action_normalizer.state_dict(),
    }
    path = tmp_path / "student.pt"
    torch.save(checkpoint, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert loaded["jose_schema_version"] == 2
    assert loaded["kind"] == "dagger_student"
    restored = models.DaggerStudent(**{
        "input_dim": loaded["model_config"]["input_dim"],
        "action_dim": loaded["model_config"]["action_dim"],
        "hidden_dims": tuple(loaded["model_config"]["hidden_dims"]),
    })
    restored.load_state_dict(loaded["model_state_dict"])
    sample = torch.randn(4, 58)
    assert torch.allclose(student(sample), restored(sample))


def test_dagger_rounds_are_distinguishable_and_existing_slugs_are_stable():
    experiments = ablation_catalog.DAGGER_EXPERIMENTS
    slugs = [experiment.slug for experiment in experiments]
    assert len(set(slugs)) == len(slugs)
    # 0 through 10, every round count, since a round-K readout from a single
    # 10-round run is not the same experiment as an independent K-round run.
    assert sorted(experiment.dagger_rounds for experiment in experiments) == list(range(11))
    # The default-rounds arm keeps the bare slug, so results already in the
    # catalog under that name stay addressable and keep grouping together.
    assert ablation_catalog.AblationExperiment("LSTM", 25, "all").slug == "lstm_w25_all"
    assert ablation_catalog.AblationExperiment("LSTM", 25, "all", 0).slug == "lstm_w25_all_r00"
    for experiments in (ablation_catalog.ARCHITECTURE_EXPERIMENTS, ablation_catalog.JOINT_SCOPE_EXPERIMENTS):
        assert all("_r" not in experiment.slug.rsplit("_", 1)[-1] for experiment in experiments)


def test_jose_pipeline_entry_points():
    for name in (
        "train_state_estimator.py", "train_history_student.py", "train_imu_distillation.py",
        "train_joint_only_distillation.py", "play_teacher_with_estimator.py", "play_dagger.py",
        "play_history_student.py",
        "run_architecture_ablation.py", "run_window_ablation.py", "run_joint_scope_ablation.py",
        "run_dagger_ablation.py",
        "run_method_comparison.py", "run_all_ablation.py", "ablation_catalog.py", "ablation_common.py",
    ):
        assert (JOSE_DIR / name).is_file()


def test_ablation_factors_are_isolated_and_window_is_configurable(tmp_path):
    assert "default=250000" in (JOSE_DIR / "ablation_runner.py").read_text(encoding="utf-8")
    assert "default=250000" in (JOSE_DIR / "train_state_estimator.py").read_text(encoding="utf-8")
    common = [
        "--teacher_checkpoint", str(tmp_path / "teacher.pt"), "--dry-run", "--fast", "--seeds", "1",
        "--output-dir", str(tmp_path / "ablation"),
    ]
    architecture = subprocess.run(
        [sys.executable, str(JOSE_DIR / "run_architecture_ablation.py"), *common],
        check=True, capture_output=True, text=True,
    ).stdout
    assert all(token in architecture for token in ("--est_type LSTM --window 25 --joint_preset all", "--est_type TCN --window 25 --joint_preset all", "--est_type HISTORY_MLP --window 25 --joint_preset all"))
    window = subprocess.run(
        [sys.executable, str(JOSE_DIR / "run_window_ablation.py"), "--windows", "20", "1", "5", "5", *common],
        check=True, capture_output=True, text=True,
    ).stdout
    commands = [line for line in window.splitlines() if "train_state_estimator.py" in line]
    assert len(commands) == 3
    assert [next(token for token in line.split() if token.startswith("Isaac-")) for line in commands]
    assert {line.split("--window ", 1)[1].split()[0] for line in commands} == {"1", "5", "20"}
    assert all("--est_type LSTM" in line and "--joint_preset all" in line for line in commands)
    scope = subprocess.run(
        [sys.executable, str(JOSE_DIR / "run_joint_scope_ablation.py"), *common],
        check=True, capture_output=True, text=True,
    ).stdout
    scope_commands = [line for line in scope.splitlines() if "train_state_estimator.py" in line]
    assert {line.split("--joint_preset ", 1)[1].split()[0] for line in scope_commands} == {"all", "legs", "upper"}
    assert all("--est_type LSTM --window 25" in line for line in scope_commands)


def test_window_defaults_and_validation():
    module = _load_module("jose_window_ablation_test", JOSE_DIR / "run_window_ablation.py")
    assert module.DEFAULT_WINDOWS == (1, 5, 10, 25, 50)
    assert module.parse_windows([50, 1, 5, 5]) == (1, 5, 50)
    with pytest.raises(ValueError, match="positive"):
        module.parse_windows([1, 0])


def test_teacher_catalog_is_content_scoped_and_hierarchical(tmp_path):
    first = tmp_path / "teacher_run_a" / "checkpoints" / "best_agent.pt"
    second = tmp_path / "copied_teacher" / "best_agent.pt"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"same teacher")
    second.write_bytes(b"same teacher")
    fingerprint = {"size": first.stat().st_size, "sha256": "teacher-sha"}
    root = tmp_path / "ablation"
    catalog_a = ablation_catalog.TeacherCatalog.open(root, first, fingerprint, create=True)
    catalog_b = ablation_catalog.TeacherCatalog.open(root, second, fingerprint, create=True)
    assert catalog_a.teacher_root == catalog_b.teacher_root
    entry = catalog_a.entry_path(
        "amp_walk", 42, "lstm_w50_all", "spec-id", estimator="LSTM", window=50, joint_preset="all"
    )
    assert entry.relative_to(catalog_a.teacher_root).parts == (
        "amp_walk", "catalog", "estimators", "lstm", "window_50", "joints_all", "seed_42",
        "variants", "spec-id",
    )
    study = catalog_a.study_path("amp_walk", "architecture", "2026-08-27_12-00-00")
    assert study.relative_to(catalog_a.teacher_root).parts == (
        "amp_walk", "studies", "architecture", "2026-08-27_12-00-00",
    )


def test_catalog_requires_complete_artifacts(tmp_path):
    entry = tmp_path / "entry"
    artifact = tmp_path / "artifact" / "training.json"
    artifact.parent.mkdir()
    artifact.write_text("{}", encoding="utf-8")
    record = {"status": "ok", "artifact": str(artifact), "experiment": "lstm_w50_all", "metrics": {}}
    ablation_catalog.TeacherCatalog.write_attempt(entry, "run", record, make_current=True)
    assert ablation_catalog.TeacherCatalog.read_complete(entry, require_checkpoint=True) is None
    (artifact.parent / "best_estimator.pt").write_bytes(b"checkpoint")
    assert ablation_catalog.TeacherCatalog.read_complete(entry, require_checkpoint=True) == record


def test_catalog_reuses_results_after_the_tree_moves(tmp_path):
    """A catalog copied from another machine must still be reusable.

    Records store the artifact as an absolute path, so before relocation every
    entry in a copied tree pointed at a directory that does not exist locally
    and the whole catalog silently re-trained.
    """
    entry = tmp_path / "entry"
    artifact = entry / "attempts" / "run" / "artifact" / "training.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}", encoding="utf-8")
    (artifact.parent / "best_estimator.pt").write_bytes(b"checkpoint")
    stale = "/home/someone-else/checkout/entry/attempts/run/artifact/training.json"
    record = {"status": "ok", "artifact": stale, "experiment": "lstm_w25_all", "metrics": {}}
    ablation_catalog.TeacherCatalog.write_attempt(entry, "run", record, make_current=True)

    found = ablation_catalog.TeacherCatalog.read_complete(entry, require_checkpoint=True)
    assert found is not None
    # The returned record carries the local path, so callers that open it work.
    assert found["artifact"] == str(artifact)
    # A path that cannot be rebuilt under the entry is still rejected.
    missing = dict(record, artifact="/nowhere/attempts/run/artifact/training.json")
    other = tmp_path / "other"
    ablation_catalog.TeacherCatalog.write_attempt(other, "run", missing, make_current=True)
    assert ablation_catalog.TeacherCatalog.read_complete(other, require_checkpoint=True) is None


def test_complete_catalog_finalizes_study_without_rerunning(tmp_path):
    teacher = tmp_path / "teacher_run" / "checkpoints" / "best_agent.pt"
    teacher.parent.mkdir(parents=True)
    teacher.write_bytes(b"teacher")
    output = tmp_path / "ablation"
    common = [
        "--teacher_checkpoint", str(teacher), "--fast", "--seeds", "1", "--output-dir", str(output),
    ]
    dry_run = subprocess.run(
        [sys.executable, str(JOSE_DIR / "run_architecture_ablation.py"), "--dry-run", *common],
        check=True, capture_output=True, text=True,
    )
    attempts = [
        line.split("--output-dir ", 1)[1].split()[0]
        for line in dry_run.stdout.splitlines()
        if "--output-dir " in line and "/variants/" in line
    ]
    names = ("teacher_gt", "lstm_w25_all", "tcn_w25_all", "history_mlp_w25_all")
    assert len(attempts) == len(names)
    for attempt, name in zip(attempts, names):
        entry = Path(attempt).parent.parent
        artifact = tmp_path / f"{name}_artifact" / "training.json"
        artifact.parent.mkdir()
        metrics = {
            "episode_length_mean": 100.0,
            "episode_length_std": 0.0,
            "return_mean": 10.0,
            "death_rate": 0.0,
            "timeout_rate": 100.0,
            "rmse": 0.1,
            "r2": 0.9,
            "inference_ms_per_sample": 0.01,
        }
        if name == "lstm_w25_all":
            metrics.update({
                "target_rmse": [0.1, 0.2],
                "rounds": [{"round": 0, "training": {"best_validation_mse": 0.25}}],
                "trace_target": [[0.0, 1.0], [1.0, 0.0]],
                "trace_prediction": [[0.1, 0.9], [0.9, 0.1]],
            })
        artifact.write_text(json.dumps({"metrics": metrics}))
        if name != "teacher_gt":
            (artifact.parent / "best_estimator.pt").write_bytes(b"checkpoint")
        record = {
            "task": "amp_walk", "task_id": "Task", "experiment": name, "seed": 42,
            "status": "ok", "artifact": str(artifact),
            "metrics": metrics,
        }
        ablation_catalog.TeacherCatalog.write_attempt(entry, "fixture", record, make_current=True)

    completed = subprocess.run(
        [sys.executable, str(JOSE_DIR / "run_architecture_ablation.py"), *common],
        check=True, capture_output=True, text=True,
    )
    assert completed.stdout.count("REUSE_RESULT") == 4
    assert "teacher_gt/seed42 REUSE_RESULT eplen=100.00" in completed.stdout
    manifests = list(output.glob("*/amp_walk/studies/architecture/*/manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["status"] == "complete" and manifest["catalog_complete"] is True
    assert all(job["eplen"] == 100.0 for job in manifest["jobs"] if job["status"] == "complete")
    teacher_job = next(job for job in manifest["jobs"] if job["experiment"] == "teacher_gt")
    assert teacher_job["eplen"] == 100.0
    assert Path(manifest["report"]).is_file()
    intermediate = json.loads((manifests[0].parent / "intermediate_results.json").read_text())
    assert intermediate["status"] == "complete"
    assert intermediate["progress"] == {"total": 4, "complete": 4, "failed": 0, "missing": 0}
    assert all(job["eplen"] == 100.0 for job in intermediate["jobs"] if job["status"] == "complete")
    assert manifest["jobs"] == intermediate["jobs"]
    lstm = next(row for row in intermediate["results"] if row["experiment"] == "lstm_w25_all")
    assert lstm["metrics"]["target_rmse"] == [0.1, 0.2]
    assert lstm["metrics"]["rounds"][0]["training"]["best_validation_mse"] == 0.25


def test_method_comparison_accepts_one_to_three_cases(tmp_path):
    script = JOSE_DIR / "run_method_comparison.py"
    result = subprocess.run(
        [sys.executable, str(script), "--case", "walk", str(tmp_path / "walk.pt"), "--case", "jump", str(tmp_path / "jump.pt"), "--seeds", "7", "--output-dir", str(tmp_path / "ablation"), "--run-name", "paper_baseline", "--dry-run", "--fast"],
        check=True, capture_output=True, text=True,
    )
    assert "2 task(s), 1 seed(s), 8 jobs" in result.stdout
    assert all(method in result.stdout for method in ("PrivilegedTeacher", "IMU-BasedDistillation", "Joint-OnlyDistillation", "JOSE"))
    assert "/studies/method_comparison/paper_baseline" in result.stdout
    assert "/methods/imu_based_distillation/window_25/joints_all/seed_7" in result.stdout
    assert "/amp_walk/" in result.stdout
    assert "/methods/jose/window_25/joints_all/seed_7" in result.stdout
    assert "/amp_jump/" in result.stdout
    assert "/methods/jose/window_25/joints_all/seed_7" in result.stdout
    assert "/methods/jose/window_50/" not in result.stdout
    walk_jose = next(
        line for line in result.stdout.splitlines()
        if "train_state_estimator.py" in line and "Isaac-G1-AMP-Walk" in line
    )
    assert "--estimator LSTM" in walk_jose
    assert "--window 25 --joint-preset all" in walk_jose
    assert "logs/jose_g1/method_comparison" not in result.stdout
    failure = subprocess.run(
        [sys.executable, str(script), "--case", "walk", "x", "--case", "dance", "x", "--case", "jump", "x", "--case", "walk", "x", "--dry-run"],
        capture_output=True, text=True,
    )
    assert failure.returncode != 0


def test_default_estimator_is_lstm_w25_all_for_every_task():
    training_source = (JOSE_DIR / "train_state_estimator.py").read_text(encoding="utf-8")
    assert 'default="LSTM"' in training_source
    assert 'default="all"' in training_source
    assert '"--window", type=int, default=25' in training_source
    for task_id, _, _ in ablation_catalog.TASKS.values():
        assert ablation_common.default_estimator_window(task_id) == 25


def test_rollout_diagnostics_npz_and_plot(tmp_path):
    joint_count = 3
    data = SimpleNamespace(
        joint_names=("j0", "j1", "j2"),
        joint_pos=torch.zeros(1, joint_count),
        computed_torque=torch.ones(1, joint_count),
        applied_torque=torch.full((1, joint_count), 0.5),
        joint_effort_limits=torch.full((1, joint_count), 2.0),
    )
    core = SimpleNamespace(
        action_offset=torch.zeros(joint_count),
        action_scale=torch.full((joint_count,), 0.5),
        action_soft_limits=torch.tensor([[-0.75, 0.75]] * joint_count),
        robot=SimpleNamespace(data=data),
    )
    recorder = rollout_diagnostics.RolloutDiagnostics(core)
    recorder.record(torch.tensor([[2.0, 0.0, -2.0]]))
    artifacts = recorder.save(tmp_path, 1.0 / 30.0)
    assert Path(artifacts["data"]).is_file()
    assert Path(artifacts["plot"]).is_file()
    assert len(artifacts["joint_plots"]) == joint_count
    assert all(Path(path).is_file() for path in artifacts["joint_plots"])
    saved = np.load(artifacts["data"])
    assert np.array_equal(saved["action_applied"][0], np.array([2.0, 0.0, -2.0]))
    assert np.array_equal(saved["position_target"][0], np.array([0.75, 0.0, -0.75]))


def test_amp_environment_and_implicit_actuator_source():
    env_source = (JOSE_DIR / "g1_amp_env_cfg.py").read_text(encoding="utf-8")
    env_impl_source = (JOSE_DIR / "g1_amp_env.py").read_text(encoding="utf-8")
    loader_source = (JOSE_DIR / "motions" / "motion_loader.py").read_text(encoding="utf-8")
    robot_source = (JOSE_DIR / "g1_cfg.py").read_text(encoding="utf-8")
    assert 'reset_strategy = "random"' in env_source
    assert "vel_window_min_vx" in env_source
    assert "termination_min_vel_x" not in env_source
    assert "motion_speed_scale" not in env_source
    assert "speed_scale" not in loader_source
    assert "too_slow_instant" not in env_impl_source
    assert "upright_weight" not in env_source
    assert "height_weight" not in env_source
    assert "target_velocity" in env_source
    assert "velocity_reward_weight = 0.5" in env_source
    assert "action_rate_penalty_weight = 0.0" in env_source
    assert "action_second_difference_penalty_weight = 0.1" in env_source
    assert "velocity_reward = torch.exp" in env_impl_source
    assert "previous_previous_actions" in env_impl_source
    assert "action_finite_difference_penalties" in env_impl_source
    assert '"reward/action_rate_penalty_weighted"' in env_impl_source
    assert '"reward/action_second_difference_penalty_weighted"' in env_impl_source
    assert '"episode/height_deaths"' in env_impl_source
    assert '"episode/slow_velocity_deaths"' in env_impl_source
    assert "env_spacing=4.0" in env_source
    assert "GroundPlaneCfg" in env_impl_source
    assert "spawn_ground_plane" in env_impl_source
    assert "DCMotorCfg" not in robot_source
    assert "action_scale = TWIST_ACTION_SCALE" in env_source
    assert "twist_action_to_position" in env_impl_source
    assert 'effort_limit_sim={".*_hip_.*": 88.0, ".*_knee_joint": 139.0}' in robot_source
    assert "effort_limit_sim=50.0" in robot_source
    assert 'stiffness={".*_hip_.*": 100.0, ".*_knee_joint": 150.0}' in robot_source
    assert 'damping={".*_hip_.*": 2.0, ".*_knee_joint": 4.0}' in robot_source
    assert "stiffness=150.0" in robot_source
    assert "damping=4.0" in robot_source
    assert '".*_wrist_.*": 20.0' in robot_source
    assert '".*_wrist_.*": 1.0' in robot_source


def test_skrl_2_yaml_and_style_scale():
    obsolete = set(compat.OBSOLETE_KEYS)
    for path in (JOSE_DIR / "agents").glob("*.yaml"):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert not obsolete.intersection(config["agent"])
        compat.validate_skrl_config(config)
        assert config["agent"]["gae_lambda"] == pytest.approx(0.95)
        assert config["agent"]["rollouts"] == 16
        assert config["agent"]["learning_epochs"] == 6
        assert config["agent"]["mini_batches"] == 2
        assert config["agent"]["learning_rate"] == pytest.approx(5.0e-5)
        assert config["agent"]["learning_rate_scheduler"] is None
        expected_entropy = 0.0 if config["agent"]["class"] == "AMP" else 0.005
        assert config["agent"]["entropy_loss_scale"] == pytest.approx(expected_entropy)
        if config["agent"]["class"] == "AMP":
            assert config["agent"]["time_limit_bootstrap"] is True
            assert config["models"]["policy"]["min_log_std"] == pytest.approx(-3.5)
            assert config["models"]["policy"]["initial_log_std"] == pytest.approx(-1.2)
            assert config["models"]["policy"]["fixed_log_std"] is False
            assert config["trainer"]["timesteps"] == 100000
            assert config["agent"]["task_reward_scale"] == pytest.approx(0.5)
            assert config["agent"]["style_reward_scale"] == pytest.approx(2.0)
            assert config["agent"]["discriminator_loss_scale"] == pytest.approx(6.0)
            assert config["models"]["policy"]["network"][0]["layers"] == [512, 256]
            assert config["models"]["discriminator"]["network"][0]["layers"] == [1024, 512, 256]
        else:
            assert config["agent"]["time_limit_bootstrap"] is True
            assert config["models"]["policy"]["min_log_std"] == pytest.approx(-3.5)
            assert config["models"]["policy"]["initial_log_std"] == pytest.approx(-1.2)
            assert config["trainer"]["timesteps"] == 80000
    raw_task, raw_style = torch.tensor([1.5]), torch.tensor([0.25])
    task, style, total = compat.scaled_reward(raw_task, raw_style)
    assert task.item() == pytest.approx(0.0)
    assert style.item() == pytest.approx(0.5)
    assert total.item() == pytest.approx(0.5)


def test_amp_effective_reward_tracking_preserves_raw_rollout_reward():
    pytest.importorskip("skrl")
    from skrl.agents.torch import Agent as TorchAgent

    class IdentityPreprocessor:
        def __call__(self, value, *args, **kwargs):
            return value

    class Discriminator:
        def act(self, inputs, role):
            # sigmoid(0) produces raw style reward -log(0.5).
            observations = inputs["observations"]
            return torch.zeros((observations.shape[0], 1)), {}

    class Config:
        task_reward_scale = 0.0
        style_reward_scale = 2.0

    class FakeAmp:
        discriminator = Discriminator()
        _amp_observation_preprocessor = IdentityPreprocessor()
        cfg = Config()
        write_interval = 1
        tracking_data = defaultdict(list)
        _cumulative_rewards = None
        _cumulative_timesteps = None
        _track_rewards = deque(maxlen=100)
        _track_timesteps = deque(maxlen=100)

        def __init__(self):
            self.raw_rewards = []

        def record_transition(self, **transition):
            TorchAgent.record_transition(self, **transition)
            self.raw_rewards.append(transition["rewards"].clone())

        def track_data(self, tag, value):
            self.tracking_data[tag].append(value)

    agent = FakeAmp()
    assert compat.install_amp_reward_tracking(agent)
    assert compat.install_amp_reward_tracking(agent)

    transition = {
        "observations": torch.zeros((1, 1)),
        "states": None,
        "actions": torch.zeros((1, 1)),
        "rewards": torch.zeros((1, 1)),
        "next_observations": torch.zeros((1, 1)),
        "next_states": None,
        "terminated": torch.ones((1, 1), dtype=torch.bool),
        "truncated": torch.zeros((1, 1), dtype=torch.bool),
        "infos": {"amp_obs": torch.zeros((1, 4))},
        "timestep": 0,
        "timesteps": 1,
    }
    agent.record_transition(**transition)

    assert agent.raw_rewards[0].item() == pytest.approx(0.0)
    expected = 2.0 * -torch.log(torch.tensor(0.5)).item()
    assert agent.tracking_data["Reward / Total reward (mean)"][-1] == pytest.approx(expected)
    assert agent.tracking_data["Reward / AMP effective reward (mean)"][-1] == pytest.approx(expected)


@pytest.mark.parametrize("key", tuple(compat.OBSOLETE_KEYS))
def test_obsolete_skrl_keys_have_migration_hints(key):
    with pytest.raises(ValueError, match="Obsolete"):
        compat.validate_skrl_config({"agent": {key: 1}})


def test_force_skrl_reset_reenables_nested_reset_once_wrappers():
    inner = SimpleNamespace(_reset_once=False, _env=None)
    outer = SimpleNamespace(_reset_once=False, _env=inner)
    compat.force_skrl_isaaclab_reset(outer)
    assert outer._reset_once is True
    assert inner._reset_once is True


def test_evaluation_disables_only_velocity_termination():
    env_cfg = SimpleNamespace(
        early_termination=True,
        termination_height=0.55,
        vel_window_min_vx=0.01,
    )
    threshold = compat.disable_velocity_termination_for_evaluation(env_cfg)
    assert threshold == pytest.approx(0.01)
    assert env_cfg.vel_window_min_vx == 0.0
    assert env_cfg.early_termination is True
    assert env_cfg.termination_height == pytest.approx(0.55)


def test_report_artifacts_and_failed_run(tmp_path):
    raw = tmp_path / "raw.jsonl"
    rows = [
        {
            "task": "amp_walk",
            "experiment": "LSTM",
            "seed": 42,
            "status": "ok",
            "metrics": {
                "rmse": 0.2,
                "r2": 0.8,
                "return_mean": 10.0,
                "episode_length_mean": 580.0,
                "episode_length_std": 25.0,
                "death_rate": 3.0,
                "timeout_rate": 97.0,
                "target_rmse": [0.1] * 9,
                "trace_target": [[0.0] * 9, [1.0] * 9],
                "trace_prediction": [[0.1] * 9, [0.9] * 9],
                "rounds": [
                    {"round": 0, "training": {"best_validation_mse": 0.4}},
                    {"round": 1, "training": {"best_validation_mse": 0.2}},
                ],
            },
        },
        {
            "task": "amp_walk", "experiment": "LSTM", "seed": 43, "status": "ok",
            "metrics": {
                "rmse": 0.3, "r2": 0.7, "return_mean": 12.0,
                "episode_length_mean": 600.0, "episode_length_std": 0.0,
                "death_rate": 0.0, "timeout_rate": 100.0,
            },
        },
        {"task": "amp_walk", "experiment": "TCN", "seed": 42, "status": "failed", "error": "synthetic failure"},
    ]
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    result = reporting.generate_report(raw, tmp_path / "report")
    assert result["runs"] == 3 and result["failures"] == 1
    assert result["required_files"] == list(reporting.REQUIRED_REPORT_FILES)
    for name in reporting.REQUIRED_REPORT_FILES:
        assert (tmp_path / "report" / name).is_file()
    # The report directory is deliberately small: the five files above and below
    # are the whole contract, and nothing else should reappear in it.
    produced = {path.name for path in (tmp_path / "report").iterdir()}
    assert not produced - set(reporting.REQUIRED_REPORT_FILES) - set(reporting.REQUIRED_PLOT_FILES)
    assert not any(name.endswith(".pdf") or name.endswith(".csv") for name in produced)
    assert "synthetic failure" in (tmp_path / "report" / "report.md").read_text(encoding="utf-8")
    table = (tmp_path / "report" / "table.md").read_text(encoding="utf-8")
    assert "Episode steps" in table and "590.0 ± 14.1" in table
    assert "Death %" in table and "Timeout %" in table
    if result["plots"]:
        # The fixture's rows carry metrics.rounds but no metrics.learning_curve,
        # so only the DAgger figure has data to draw.
        assert (tmp_path / "report" / "dagger_learning_curve.png").is_file()
        assert "learning_curves" in result["skipped_plots"]


def test_paired_motion_fidelity_measures_divergence_and_restores_rng():
    pipeline = _load_pipeline_module()

    class FakeEnv:
        """Two bodies that drift apart at a fixed rate per step, per policy."""

        def __init__(self):
            self.drift = 0.0
            self.step_index = 0

        def reset(self):
            self.step_index = 0
            return torch.zeros(2, 3), {}

        def step(self, action):
            self.step_index += 1
            self.drift = float(action.mean())
            return torch.zeros(2, 3), torch.zeros(2), torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool), {}

    class FakeAdapter:
        def __init__(self, env):
            self.env = env

        def body_positions(self):
            # One body at the root, one offset by the drift the policy caused.
            offset = self.env.drift * self.env.step_index
            return torch.tensor([[[0.0, 0.0, 0.0], [offset, 0.0, 0.0]]]).repeat(2, 1, 1)

        def root_position(self):
            return torch.zeros(2, 3)

        def body_names(self):
            return ["root_link", "drifting_link"]

        def action(self, agent, observations):
            return torch.zeros(2, 1)

    env = FakeEnv()
    adapter = FakeAdapter(env)
    pipeline.force_skrl_isaaclab_reset = lambda _env: None

    torch.manual_seed(1234)
    before = torch.rand(4)
    torch.manual_seed(1234)

    result = pipeline.evaluate_paired_motion_fidelity(
        env, adapter, object(), lambda observations: torch.full((2, 1), 0.001), seed=7, horizon=10,
    )

    # The borrowed RNG stream must be handed back untouched, or a caller's next
    # draw silently changes and every result after it diverges.
    assert torch.allclose(torch.rand(4), before)

    # Teacher holds still, student drifts 0.001 per step on one of two bodies,
    # so the mean over both bodies at step t is 0.001*t/2 metres.
    expected = sum(0.001 * step / 2 for step in range(10)) / 10 * 1000.0
    assert result["mpjpe_g"] == pytest.approx(expected, rel=1e-6)
    assert result["mpjpe_horizon"] == 10
    assert result["root_position_error"] == pytest.approx(0.0)
    assert set(result["mpjpe_per_body"]) == {"root_link", "drifting_link"}
    assert result["mpjpe_per_body"]["root_link"] == pytest.approx(0.0)
    assert len(result["mpjpe_by_step"]) == 10
    assert result["mpjpe_by_step"][0]["mpjpe_g"] == pytest.approx(0.0)
    assert result["mpjpe_by_step"][-1]["mpjpe_g"] > result["mpjpe_by_step"][1]["mpjpe_g"]
    with pytest.raises(ValueError):
        pipeline.evaluate_paired_motion_fidelity(
            env, adapter, object(), lambda observations: torch.zeros(2, 1), seed=7, horizon=0,
        )


def test_step_metrics_are_computed_identically_for_every_caller():
    package = "jose_pipeline_test_package"
    metrics_module = sys.modules[f"{package}.estimator.metrics"]

    class FakeData:
        applied_torque = torch.full((2, 4), 2.0)
        computed_torque = torch.full((2, 4), 99.0)
        joint_vel = torch.full((2, 4), 3.0)
        joint_effort_limits = torch.full((2, 4), 10.0)
        body_lin_vel_w = torch.tensor([[[3.0, 4.0, 0.0]], [[3.0, 4.0, 0.0]]])
        body_ang_vel_w = torch.tensor([[[0.0, 0.0, 2.0]], [[0.0, 0.0, 2.0]]])

    class FakeCore:
        robot = type("R", (), {"data": FakeData()})()
        step_dt = 0.5
        extras: dict = {}
        ref_body_index = 0

    core = FakeCore()
    result = metrics_module.step_metrics(
        core, object(), object(),
        action=torch.zeros(2, 4), previous_action=torch.zeros(2, 4), rewards=torch.ones(2),
        policy_action=torch.ones(2, 4), teacher_action=torch.zeros(2, 4),
    )
    # applied_torque wins over computed_torque: the post-clamp value is what the
    # robot really used, and the two loops used to disagree about which to read.
    assert result["torque_rms"] == pytest.approx(2.0)
    # Energy sums power over joints before averaging over envs. Summing vs
    # averaging here was a factor-of-29 discrepancy between the teacher's row
    # and the students' in the comparison table.
    assert result["energy"] == pytest.approx(2.0 * 3.0 * 4 * 0.5)
    assert result["base_linear_speed"] == pytest.approx(5.0)
    assert result["base_angular_speed"] == pytest.approx(2.0)
    assert result["teacher_action_mse"] == pytest.approx(1.0)
    assert result["torque_saturation"] == pytest.approx(0.0)
    assert "amp_raw_style" not in result

    accumulator = metrics_module.MetricAccumulator()
    accumulator.add({"a": 1.0, "b": 10.0})
    accumulator.add({"a": 3.0})
    # Per-key counts: "b" appearing on one step of two must not be halved.
    assert accumulator.mean() == {"a": 2.0, "b": 10.0}


def test_frozen_rng_keeps_metric_work_out_of_the_random_stream():
    pipeline = _load_pipeline_module()

    def draw():
        return torch.rand(3).tolist()

    torch.manual_seed(99)
    expected = [draw(), draw()]

    torch.manual_seed(99)
    first = draw()
    # Stands in for skrl's policy act(), which rsamples on every call and throws
    # the sample away when the caller asks for the mean -- an extra evaluation
    # done purely to measure something must not move the stream.
    with pipeline.frozen_rng():
        torch.rand(64)
        torch.randn(128)
    second = draw()
    assert [first, second] == expected


def test_checkpoint_sweep_seed_translation_and_checkpoint_resolution(tmp_path):
    sweep = _load_module("jose_g1_checkpoint_sweep_test", JOSE_DIR / "run_all_ablation.py")

    # method_comparison passes any seed set straight through.
    assert sweep._seed_args("method_comparison", [42, 44, 50]) == ["--seeds", "42", "44", "50"]
    assert sweep._seed_args("architecture", None) == []

    # The other four studies only accept a contiguous range.
    assert sweep._seed_args("architecture", [42, 43, 44]) == ["--seed-start", "42", "--seeds", "3"]
    assert sweep._seed_args("dagger", [5]) == ["--seed-start", "5", "--seeds", "1"]
    with pytest.raises(ValueError):
        sweep._seed_args("window", [42, 44])

    checkpoint = tmp_path / "best_agent.pt"
    checkpoint.write_bytes(b"stub")
    other = tmp_path / "sub" / ".." / "best_agent.pt"
    # Same path reached two ways collapses to one resolved checkpoint.
    assert sweep.resolve_checkpoints([str(checkpoint), str(other)]) == [checkpoint.resolve()]
    with pytest.raises(ValueError):
        sweep.resolve_checkpoints([])


def test_run_all_ablation_builds_translated_task_and_headless_flags(tmp_path):
    sweep = _load_module("jose_g1_run_all_ablation_build_test", JOSE_DIR / "run_all_ablation.py")
    args = sweep._parser().parse_args([
        "--checkpoints", "ckpt.pt", "--task", "locomotion", "--seeds", "7", "--no-headless",
    ])
    checkpoint = Path("ckpt.pt")

    architecture_command = sweep.build_command("architecture", args, checkpoint)
    assert "--task" in architecture_command and "locomotion" in architecture_command
    assert architecture_command[-1] == "--no-headless"
    assert "--seed-start" in architecture_command

    # run_method_comparison.py's own vocabulary, not ablation_catalog.py's.
    comparison_command = sweep.build_command("method_comparison", args, checkpoint)
    assert "--case" in comparison_command
    case_index = comparison_command.index("--case")
    assert comparison_command[case_index + 1 : case_index + 3] == ["locomotion", "ckpt.pt"]
    assert "--seeds" in comparison_command and "7" in comparison_command
