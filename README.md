# JOSE G1

JOSE is an Isaac Lab package for the Unitree G1 robot (29 DOF). You can use it to train walk, dance, and jump policies. It also includes state estimators, DAgger students, ablation studies, and motion tools.

## Requirements

- Linux
- Python 3.10 or newer
- A working Isaac Lab environment
- `skrl>=2.0,<3.0`
- A CUDA GPU is recommended

Activate the Python environment that contains Isaac Lab. Then install JOSE in the same environment.

Replace `/path/to/JOSE` with the path to this repository.

```bash
cd /path/to/JOSE
python -m pip install -e .
```

Check the installation:

```bash
python -c "import jose; print('JOSE tasks registered')"
```

The `-e` option installs the package in editable mode. Code changes are available without reinstalling the package.

## Quick start

The basic workflow is:

1. Train an AMP teacher.
2. Find the saved checkpoint.
3. Play the trained policy.

### 1. Train a walking teacher

```bash
python -m jose.train \
  --task Isaac-G1-AMP-Walk-JOSE-Direct-v0 \
  --algorithm AMP \
  --experiment_name walk_example \
  --headless
```

The training output is saved in a directory like this:

```text
logs/skrl/g1_jose_amp_walk/<date>_AMP_torch_walk_example/
├── checkpoints/
├── params/
└── ...
```

List the saved checkpoints:

```bash
find logs/skrl/g1_jose_amp_walk -path '*/checkpoints/*.pt' -print
```

### 2. Play the teacher

Set `TEACHER` to the real checkpoint path:

```bash
export TEACHER=/absolute/path/to/best_agent.pt

python -m jose.play \
  --task Isaac-G1-AMP-Walk-JOSE-Direct-v0 \
  --algorithm AMP \
  --checkpoint "$TEACHER"
```

To record a video without opening the simulator window:

```bash
python -m jose.play \
  --task Isaac-G1-AMP-Walk-JOSE-Direct-v0 \
  --algorithm AMP \
  --checkpoint "$TEACHER" \
  --video \
  --video_length 600 \
  --headless
```

Videos and rollout diagnostics are saved under `videos/play/` in the training run directory. Add `--print-base-velocity` to print the base velocity in the terminal.

## Tasks

| Purpose | Task ID | Algorithm |
|---|---|---|
| AMP walk | `Isaac-G1-AMP-Walk-JOSE-Direct-v0` | `AMP` |
| AMP dance | `Isaac-G1-AMP-Dance-JOSE-Direct-v0` | `AMP` |
| AMP jump | `Isaac-G1-AMP-Jump-JOSE-Direct-v0` | `AMP` |
| PPO walk example | `Isaac-G1-PPO-Walk-JOSE-Direct-v0` | `PPO` |

Use a different task ID to train dance, jump, or PPO walk:

```bash
# Dance
python -m jose.train \
  --task Isaac-G1-AMP-Dance-JOSE-Direct-v0 \
  --algorithm AMP \
  --headless

# Jump
python -m jose.train \
  --task Isaac-G1-AMP-Jump-JOSE-Direct-v0 \
  --algorithm AMP \
  --headless

# PPO walk
python -m jose.train \
  --task Isaac-G1-PPO-Walk-JOSE-Direct-v0 \
  --algorithm PPO \
  --headless
```

Common training options:

| Option | Description | Example |
|---|---|---|
| `--num_envs` | Number of parallel environments | `--num_envs 2048` |
| `--max_iterations` | Maximum training iterations | `--max_iterations 5000` |
| `--seed` | Random seed | `--seed 42` |
| `--device` | GPU device | `--device cuda:0` |
| `--video` | Record videos during training | `--video --video_interval 5000` |
| `--headless` | Run without a GUI | `--headless` |

To continue training from a checkpoint:

```bash
python -m jose.train \
  --task Isaac-G1-AMP-Walk-JOSE-Direct-v0 \
  --algorithm AMP \
  --checkpoint "$TEACHER" \
  --experiment_name walk_resume \
  --headless
```

## State estimator

The state estimator learns from a fixed teacher. It uses a history of joint positions and simulator joint velocities to estimate the privileged state.

### Train an estimator

The default model is a two-layer LSTM with a 50-frame history.

```bash
python -m jose.train_state_estimator \
  --teacher-checkpoint "$TEACHER" \
  --task Isaac-G1-AMP-Walk-JOSE-Direct-v0 \
  --estimator LSTM \
  --window 50 \
  --joint-preset all \
  --dagger-rounds 10 \
  --run-name walk_lstm_w50 \
  --headless
```

The result is saved in:

```text
logs/jose_g1/estimators/walk_lstm_w50/
```

Use `best_estimator.pt` for evaluation and playback.

For a short test run, use less data and fewer training rounds:

```bash
python -m jose.train_state_estimator \
  --teacher-checkpoint "$TEACHER" \
  --collect-steps 100 \
  --epochs 2 \
  --dagger-rounds 1 \
  --num-envs 16 \
  --run-name estimator_smoke \
  --headless
```

Available model and input options:

- `--estimator`: `LSTM`, `TCN`, `MLP`, or `HISTORY_MLP`
- `--joint-preset`: `all`, `legs`, or `upper`
- `--window`: any positive history length

### Play the teacher with an estimator

```bash
export ESTIMATOR=logs/jose_g1/estimators/walk_lstm_w50/best_estimator.pt

python -m jose.play_teacher_with_estimator \
  --teacher-checkpoint "$TEACHER" \
  --estimator-checkpoint "$ESTIMATOR" \
  --task Isaac-G1-AMP-Walk-JOSE-Direct-v0 \
  --steps 1000 \
  --csv-output logs/rollout.csv \
  --video \
  --headless
```

Videos are saved in `logs/jose_g1/videos/teacher_estimator/` by default. The `--csv-output` option saves actions and state estimates for each step.

## DAgger policies and history students

### Train a 58D joint-state DAgger student

```bash
python -m jose.train_dagger \
  --teacher-checkpoint "$TEACHER" \
  --task Isaac-G1-AMP-Walk-JOSE-Direct-v0 \
  --num-iterations 300 \
  --headless
```

Checkpoints are saved under:

```text
logs/jose_g1/dagger/<date>/checkpoints/
```

The best checkpoint is named `student_best_eval.pt`.

Play the trained student:

```bash
export DAGGER=/absolute/path/to/student_best_eval.pt

python -m jose.play_dagger \
  --checkpoint "$DAGGER" \
  --steps 1000 \
  --video \
  --headless
```

### Train 21-frame direct-policy baselines

The joint-only and IMU students use the same training setup but different inputs.

```bash
# Joint position, joint velocity, and previous action
python -m jose.train_joint_only_distillation \
  --teacher-checkpoint "$TEACHER" \
  --num-iterations 300 \
  --headless

# Joint inputs, body gyro, and projected gravity
python -m jose.train_imu_distillation \
  --teacher-checkpoint "$TEACHER" \
  --num-iterations 300 \
  --headless
```

The outputs are saved in:

```text
logs/jose_g1/distillation/joint_only/<date>/checkpoints/
logs/jose_g1/distillation/imu/<date>/checkpoints/
```

Both student types use the same playback command:

```bash
python -m jose.play_history_student \
  --checkpoint /absolute/path/to/student_best_eval.pt \
  --steps 1000 \
  --video \
  --headless
```

## Ablation studies

All ablation studies need a teacher checkpoint. The default runs can take a long time. Use `--dry-run` to check the run plan or `--fast` for a smaller test.

### Compare estimator architectures

```bash
python -m jose.run_architecture_ablation \
  --teacher-checkpoint "$TEACHER" \
  --task amp_walk \
  --seeds 3 \
  --seed-start 42 \
  --headless
```

### Compare history windows

```bash
python -m jose.run_window_ablation \
  --teacher-checkpoint "$TEACHER" \
  --task amp_walk \
  --windows 1 5 10 25 50 \
  --headless
```

### Compare joint groups

Joint-scope comparisons use the LSTM estimator with a fixed window of 25.

```bash
python -m jose.run_joint_scope_ablation \
  --teacher-checkpoint "$TEACHER" \
  --task amp_walk \
  --headless
```

To run only one experiment, pass its canonical name:

```bash
python -m jose.run_architecture_ablation \
  --teacher-checkpoint "$TEACHER" \
  --experiments lstm_w25_all \
  --dry-run
```

Use `--rerun` to run completed ablation jobs again.

### Compare all four methods

This command compares the teacher, estimator, joint-only student, and IMU student:

```bash
python -m jose.run_method_comparison \
  --case walk /absolute/path/to/walk_teacher.pt \
  --case jump /absolute/path/to/jump_teacher.pt \
  --seeds 42 43 44 \
  --headless
```

Default output directories:

```text
logs/jose_g1/ablation/
logs/jose_g1/method_comparison/
```

Completed jobs are reused when you restart the same study. Architecture, window, joint-scope, and method comparisons all write the same report bundle under their run's `report/` directory: JSON and CSV results, Markdown and LaTeX tables, and PNG and PDF plots. Check `intermediate_results.json` while a catalog study is running; every completed `jobs[]` entry exposes its mean episode length directly as `eplen`, matching the final `manifest.json`.

## Motion tools

JOSE includes these motion files:

- `motions/G1_walk.npz`
- `motions/G1_dance.npz`
- `motions/G1_jump.npz`

### Check an NPZ motion file

```bash
python -m jose.motions.verify_motion \
  --file motions/G1_walk.npz
```

### Replay a motion in Isaac Sim

```bash
python -m jose.motions.motion_replayer \
  --file motions/G1_dance.npz \
  --speed 1.0 \
  --video \
  --print-base-velocity \
  --headless
```

Add `--matplotlib` to show a skeleton plot on a desktop. Use `--record-output` to save the replay as another compatible NPZ file.

```bash
python -m jose.motions.motion_replayer \
  --file motions/G1_walk.npz \
  --loops 1 \
  --record-output logs/recorded_walk.npz
```

### Convert a CSV motion to NPZ

The input CSV must contain:

1. Seven root columns: `x`, `y`, `z`, `qx`, `qy`, `qz`, `qw`
2. Twenty-nine G1 joint-position columns

```bash
python -m jose.motions.data_convert \
  --csv data/walk.csv \
  --output motions/custom_walk.npz \
  --input-fps 30 \
  --output-fps 60
```

## Tests

Run a short simulator test for one task:

```bash
python -m jose.tools.smoke_test \
  --task amp_walk \
  --steps 12 \
  --num-envs 1 \
  --headless
```

Available smoke-test tasks are `amp_walk`, `amp_dance`, `amp_jump`, and `ppo_walk`. Run one task per process because Isaac Sim uses one simulation context.

Run the Python tests:

```bash
python -m pytest test_jose.py -q
```

Use `--help` to see all options for a command:

```bash
python -m jose.train_state_estimator --help
python -m jose.run_window_ablation --help
python -m jose.motions.motion_replayer --help
```

## Tips

- Use `--headless` when you do not need the simulator window.
- You can use `--video --headless` together to record a video without a GUI.
- If GPU memory is low, reduce `--num_envs` or `--num-envs`.
- Some commands use underscores in option names, while others use hyphens. Check `--help` if an option is not accepted.
- Absolute checkpoint paths are easier to use when you run commands from different directories.
- Ablation jobs run one at a time to avoid multiple Isaac Sim processes using the same GPU.
- Old estimator or DAgger checkpoints may not work with the current schema-v2 loader. Train them again with the current code if needed.

## Project layout

```text
├── g1_amp_env.py, g1_amp_env_cfg.py       # AMP walk, dance, and jump environments
├── g1_ppo_env.py                          # PPO walk example
├── agents/                                # SKRL AMP and PPO settings
├── motions/                               # Motion data and motion tools
├── estimator/                             # Estimator models and training code
├── distillation/                          # History students and IMU input code
├── tools/                                 # Smoke tests and diagnostics
├── train.py, play.py                      # Teacher training and playback
├── train_state_estimator.py               # State-estimator training
├── train_dagger.py                        # 58D DAgger student training
├── train_*_distillation.py                # Joint-only and IMU student training
└── run_*_ablation.py                      # Ablations and method comparison
```

JOSE runs policy control at 50 Hz and uses all 29 G1 joints. The default estimator input contains 29 joint positions and 29 simulator joint velocities. Direct history students do not use explicit base linear velocity or raw accelerometer data.

## License and attribution

This repository is based on [Isaac Lab](https://github.com/isaac-sim/IsaacLab), which uses the BSD-3-Clause license. See [LICENSE](LICENSE) for details.

The G1 motion and USD files keep attribution to [`linden713/humanoid_amp`](https://github.com/linden713/humanoid_amp). See [usd/README.md](usd/README.md) for details.
