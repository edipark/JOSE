# JOSE G1

### State estimation from robot joint observations

JOSE is a standalone Isaac Lab package: task modules, motions, assets, train/play entry points, and experiment code
live directly in the installable `jose` package. It trains 29-DOF Unitree G1 walk,
dance, and jump policies with an
[Adversarial Motion Prior (AMP)](https://xbpeng.github.io/projects/AMP/), then replaces policy-only base information
with a learned state estimator. Its AMP objective mirrors the local SOLO task, while its policy-to-actuator interface
uses TWIST-style default-centered actions and PD gains. The
original G1 assets and reference motions derive from
[`linden713/humanoid_amp`](https://github.com/linden713/humanoid_amp). All task code, assets, motions, estimator code,
tools, and tests are physically contained here; it is not an IsaacLab fork and no external research repository is
imported or used at runtime.

The default estimator uses every G1 joint: 29 joint positions plus the 29 simulator joint velocities (58D). For AMP it
predicts the complete 43D privileged suffix: height, tangent/normal basis, root velocities, and ten key-body relative
positions. PPO keeps its policy-specific 9D base-state target. No finite-difference velocity, encoder quantization, or
EMA filtering is used.

Install JOSE into the same Python environment as Isaac Lab:

```bash
python -m pip install -e /home/usd/jose_ws/JOSE
python -c "import jose; print('JOSE tasks registered')"
```

## Pipeline

### Phase 1 — privileged policy

Train an AMP walk, dance, or jump teacher with the 101D G1 motion observation. The AMP objective follows SOLO, while
joint targets use `q_target = q_default + 0.5 * action` and TWIST PD gains. Final targets are clamped to each joint's
soft limit; raw policy actions are not clipped to `[-1, 1]`. JOSE retains all 29 actions, including the wrists, rather
than adopting TWIST's 23-action interface with fixed wrists. The policy uses trainable Gaussian exploration from
`log_std=-1.2`, and the same policy/value/discriminator networks. Walk combines velocity tracking with a
second-order finite-difference action penalty
(`task_reward_scale: 0.5`) with AMP style reward (`style_reward_scale: 2.0`); Dance and Jump use only the style term.
Physics runs at 200 Hz with decimation 4 (50 Hz policy control), matching Unitree's G1 RL/deployment timing. The
discriminator receives four policy-rate AMP frames (4 x 101D = 404D), and episodes last 20 seconds. JOSE's estimator,
distillation, metrics, and replay interfaces
are layered on this teacher without changing its AMP objective.

```bash
# Walk AMP
python -m jose.train \
  --task Isaac-G1-AMP-Walk-JOSE-Direct-v0 --algorithm AMP \
  --experiment_name amp_walk --headless

# Video also writes action/position/torque diagnostics next to the recording
python -m jose.play \
  --task Isaac-G1-AMP-Walk-JOSE-Direct-v0 --algorithm AMP \
  --checkpoint <best_agent.pt> --video --video_length 300 --print-base-velocity

# Dance AMP
python -m jose.train \
  --task Isaac-G1-AMP-Dance-JOSE-Direct-v0 --algorithm AMP --headless

# Jump AMP: random reference reset, pure style reward, 0.20 m termination
python -m jose.train \
  --task Isaac-G1-AMP-Jump-JOSE-Direct-v0 --algorithm AMP --headless

# Development audit of the shared AMP objective/assets (TWIST action/PD differences are excluded)
python -m jose.tools.check_solo_amp_parity \
  --reference /home/usd/jose_ws/SOLO

# A policy-only SKRL PPO walking example using the same 29-DOF interface
python -m jose.train \
  --task Isaac-G1-PPO-Walk-JOSE-Direct-v0 --algorithm PPO --headless
```

### Phase 2 — estimator and distillation

Collect frozen-teacher rollouts and train the default two-layer, hidden-256 LSTM with history 50. Add
`--joint-preset legs` or `--joint-preset upper` for the input ablations, or select `--estimator TCN`/`MLP`.

```bash
python -m jose.train_state_estimator \
  --teacher_checkpoint <best_agent.pt> \
  --task Isaac-G1-AMP-Walk-JOSE-Direct-v0 \
  --estimator LSTM --window 50 --joint-preset all --dagger-rounds 10 --headless
```

Two direct-policy baselines use the same deterministic `256-256-128` history MLP, normalizers, optimizer, replay,
DAgger beta schedule, and training budget. Both consume the current frame plus 20 prior frames:

- Joint-only: `q(29) + qdot(29) + previous_action(29)`, 87D/frame and 1,827D total.
- IMU-based: joint-only plus body gyro(3) and quaternion-derived projected gravity(3), 93D/frame and 1,953D total.

```bash
python -m jose.train_joint_only_distillation \
  --teacher-checkpoint <best_agent.pt> --num-iterations 300 --headless

python -m jose.train_imu_distillation \
  --teacher-checkpoint <best_agent.pt> --num-iterations 300 --headless
```

### 왜 explicit linear velocity를 제외하는가

이 history student 설계는 [GMT](https://arxiv.org/abs/2506.14770)의 현재+과거 20프레임 proprioceptive history와
DAgger 구성을 주 근거로 삼고, [OmniH2O](https://arxiv.org/abs/2406.08858)를 보조 근거로 삼는다. OmniH2O는
실기에서 직접 신뢰하기 어려운 global/base linear velocity를 관측으로 주는 대신 proprioceptive history가 속도를
암묵적으로 추론하도록 하며, history student의 closed-loop distribution shift에 DAgger가 특히 유효함을 보고한다.
따라서 두 direct student에는 base linear velocity나 linear acceleration을 넣지 않는다. 반면 JOSE estimator는
joint history로 teacher의 43D privileged suffix를 추정하므로, 그 suffix 안에 linear velocity 추정값이 존재하는 것은
실기 센서가 explicit velocity를 policy에 제공하는 것과 다르다.

Projected gravity는 real-world 입력으로 유지한다. Unitree G1 LowState quaternion에서
`g_B = R_WB^T [0, 0, -1]`로 계산하며 raw accelerometer를 정규화하지 않는다. 학습 시 gyro noise/bias,
gravity tilt, 0--2 step latency를 적용한다. quaternion order/frame, IMU-to-pelvis extrinsic, norm, NaN, timestamp
freshness는 `IMUObservationSpec` 하나로 simulation과 G1 adapter가 공유한다. 실제 G1 standing/walking/jump-landing
log 검증은 배포 acceptance gate이며 synthetic log는 parser 검증만 통과시킨다.

### Phase 3 — estimated-state inference

Teacher checkpoints are loaded through SKRL's public `Runner`/`agent.load()` API. The player uses the SKRL 2.x
evaluation API and can produce an action/estimate CSV alongside a video.

```bash
python -m jose.play_teacher_with_estimator \
  --teacher-checkpoint <best_agent.pt> --estimator-checkpoint <best_estimator.pt> \
  --task Isaac-G1-AMP-Walk-JOSE-Direct-v0 --csv-output logs/rollout.csv --video

python -m jose.play_dagger \
  --checkpoint <student_best_eval.pt> --video

python -m jose.play_history_student \
  --checkpoint <imu_or_joint_student_best_eval.pt> --video
```

## Motion tools

The bundled `G1_walk.npz`, `G1_dance.npz`, and `G1_jump.npz` follow the 29-joint reference schema. Available tools include schema
validation, matplotlib visualization, Isaac Sim replay/recording, conversion, pelvis alignment, and reference tracking.

```bash
# Inspect schema and values
python -m jose.motions.verify_motion \
  --file motions/G1_walk.npz

# Isaac Sim replay; --record-output writes another compatible NPZ
python -m jose.motions.motion_replayer \
  --file motions/G1_dance.npz \
  --speed 1.0 --video --print-base-velocity

# Optional side-by-side skeleton view (requires a desktop session)
python -m jose.motions.motion_replayer \
  --file motions/G1_walk.npz --matplotlib
```

Unrelated AX18/hardware/system-identification code is intentionally excluded.

## Ablation and four-way comparison

The three estimator factors have separate executable entry points. They share cache/report infrastructure but never
mix factors in one study. Window defaults are `1 5 10 20 50`; `--windows` accepts positive integers, removes duplicates,
and sorts them.

```bash
python -m jose.run_architecture_ablation \
  --teacher_checkpoint <walk_amp.pt> --task amp_walk --seeds 3 --seed_start 42 --headless
python -m jose.run_window_ablation \
  --teacher_checkpoint <walk_amp.pt> --windows 1 5 10 20 50 --headless
python -m jose.run_joint_scope_ablation \
  --teacher_checkpoint <walk_amp.pt> --headless

python -m jose.run_method_comparison \
  --case walk <walk_teacher.pt> --case jump <jump_teacher.pt> --seeds 42 43 44 --headless
```

Estimator ablations keep the standalone defaults (2,000 collection steps, 500k samples, 50 initial epochs, 10 epochs
per DAgger round, and 10 rounds). Direct-policy baselines are intentionally excluded from these three factor studies
and live only in the four-way comparison. Initial teacher rollouts are cached per
teacher/task/seed: the 50-frame all-joint dataset is projected to shorter windows and joint subsets and shared for the
initial supervised phase. Model-dependent DAgger rollouts are not shared.
Runs stay sequential on a single GPU to avoid multiple Isaac Sim processes competing for VRAM, while subprocess output
is streamed live to the terminal and `process_logs/`.
Use `--experiments LSTM_DAgger_w50_all --skip-student` for a targeted estimator rerun.

Each argument/checkpoint/code combination is isolated under `ablation/sessions/<run-signature>/`. Repeated attempts have
separate artifact and TensorBoard directories; `raw_results.jsonl` remains an audit log while reports use only the latest
record for each task/seed/experiment. Dataset cache keys include the teacher checkpoint contents and estimator/environment
implementation, and an output-directory lock rejects overlapping ablation launchers.

Outputs include raw JSONL, aggregate JSON, tidy/summary CSV, Markdown and LaTeX tables, `report.md`, and PNG/PDF
plots with mean, standard deviation, and 95% confidence intervals. Estimator error, target-specific error, closed-loop
return/termination, action agreement/smoothness, dynamics/energy, AMP rewards, parameter count, and inference latency
are supported metric fields.

## SKRL compatibility

- Supported dependency: `skrl>=2.0,<3.0`.
- Configs use `observation_preprocessor`, `amp_observation_preprocessor`, `gae_lambda`, `task_reward_scale`, and
  `style_reward_scale`.
- Legacy keys such as `amp_state_preprocessor`, `*_reward_weight`, `discriminator_reward_scale`, `lambda`, and
  `clip_predicted_values` are rejected with migration hints.
- Checkpoint loaders accept only the current schema-v2 estimator and DAgger formats. Retrain older G1 checkpoints.

## Layout

```text

├── g1_amp_env.py, g1_amp_env_cfg.py     # AMP walk/dance/jump
├── g1_ppo_env.py                        # extensible PPO walk example
├── schema.py, skrl_compat.py             # public policy/estimator contracts
├── estimator/                           # models, adapters, collection and training
├── distillation/                        # history students and deploy IMU ABI
├── agents/                              # SKRL 2.x AMP/PPO YAML
├── motions/                             # reference data and conversion/view tools
├── tools/                               # replay, tracking, rollout diagnostics
├── train_state_estimator.py
├── train_joint_only_distillation.py, train_imu_distillation.py
├── play_teacher_with_estimator.py, play_dagger.py, play_history_student.py
└── run_*_ablation.py, run_method_comparison.py, reporting.py
```

To adapt another policy, implement the environment's `get_estimator_joint_state()` and `get_estimator_target()`
methods, define its `ObservationSchema`, and add a `PolicyAdapter`. The collection, estimator, DAgger, evaluation, and
reporting code then remains unchanged.

## License and attribution

This repository is based on [Isaac Lab](https://github.com/isaac-sim/IsaacLab), licensed under BSD-3-Clause. See
[LICENSE](LICENSE). The bundled G1 motion and
USD resources retain attribution to [`linden713/humanoid_amp`](https://github.com/linden713/humanoid_amp); details are
in `usd/README.md`.
