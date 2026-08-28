# JOSE G1

JOSE is an Isaac Lab package for the Unitree G1 robot (29 DOF). You can use it to train walk, dance, and jump policies. It also includes state estimators, DAgger students, ablation studies, and motion tools.

## Requirements

- Linux
- Python 3.10 or newer
- A working Isaac Lab environment (Isaac Sim 5.1.x, Isaac Lab 2.3.x or newer)
- `skrl>=2.0,<3.0` for the AMP and Direct-PPO tasks
- `rsl-rl-lib` (5.0.1, already pinned by `isaaclab_rl`) for the
  [velocity-tracking PPO walk](#velocity-tracking-ppo-walk) task
- A CUDA GPU is recommended

Both RL libraries ship with a standard Isaac Lab install, so there is nothing to
install beyond JOSE itself.

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
| **Velocity-tracking PPO walk** | `Isaac-G1-PPO-Walk-JOSE-v0` | `PPO` (rsl-rl) |
| **PPO walk, estimator variant** | `Isaac-G1-PPO-Walk-Estimator-JOSE-v0` | `PPO` (rsl-rl) |

The first four tasks end in `-Direct-v0`. They are Direct-workflow environments
trained with `skrl` through `jose.train`. The last two are manager-based
environments trained with `rsl-rl` through `train_ppo_walk.py`; see
[Velocity-tracking PPO walk](#velocity-tracking-ppo-walk) below.

Use a different task ID to train dance, jump, or the Direct PPO walk:

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

## Velocity-tracking PPO walk

`Isaac-G1-PPO-Walk-JOSE-v0` is a velocity-tracking locomotion task ported into
JOSE from [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab)
(Apache License 2.0). Unlike the `-Direct-v0` tasks it is a manager-based Isaac
Lab environment trained with `rsl-rl`, and it follows a full locomotion recipe:
generated terrain with a difficulty curriculum, a command curriculum that grows
the commanded velocity range, contact sensors, gait and foot-clearance rewards,
domain randomization, and a privileged critic.

**The training recipe is unchanged from upstream.** Every reward weight, command
range, event, termination, curriculum term, network size and PPO hyperparameter
is carried over verbatim. You can verify this yourself by diffing
`ppo_walk/mdp/*.py`, `ppo_walk/walk_env_cfg.py` and
`ppo_walk/agents/rsl_rl_ppo_cfg.py` against the upstream repository.

### Setup

Nothing extra to install. Activate the environment that has Isaac Lab and
install JOSE in it, exactly as in [Requirements](#requirements):

```bash
conda activate env_isaaclab      # or whichever env has Isaac Lab
cd /path/to/JOSE
python -m pip install -e .
```

Every dependency this task needs is already part of an Isaac Lab install:

| Package | Needed version | Why |
|---|---|---|
| `isaacsim` | 5.1.x | Matches the version the recipe was tuned on |
| `isaaclab`, `isaaclab_rl`, `isaaclab_tasks` | 2.3.x or newer | Environment, rsl-rl wrapper, shared MDP terms |
| `rsl-rl-lib` | 5.0.1 | Pinned by `isaaclab_rl`; the recipe needs ≥ 2.3.1, so the pin already satisfies it |
| `torch`, `gymnasium`, `tensordict`, `psutil`, `onnxscript` | as installed by Isaac Lab | Training, ONNX export |

`skrl` is untouched, so the AMP and Direct-PPO pipelines keep working alongside
this task. Do **not** downgrade `rsl-rl-lib`: `isaaclab_rl` pins it to 5.0.1, and
that version satisfies the recipe's own minimum.

Confirm the tasks are registered without starting the simulator:

```bash
python -c "import jose, gymnasium as gym; print([k for k in gym.registry if 'PPO-Walk' in k])"
```

### Robot asset

The task spawns the G1 29-DOF USD that JOSE already bundles at
`usd/g1_29dof_rev_1_0.usd`, so it runs standalone with no extra download. That
file is a conversion of the official `unitree_ros` `g1_29dof_rev_1_0.urdf`, which
upstream also recommends: link masses (33.3411 kg total), all 29 joint limits and
the collision shapes (convex hulls, four r=0.005 spheres per foot, r=0.03/h=0.05
shoulder cylinders) all match the URDF exactly.

To spawn the published USD dataset instead, download
[unitree_model](https://huggingface.co/datasets/unitreerobotics/unitree_model)
and point `JOSE_G1_MODEL_DIR` at it:

```bash
export JOSE_G1_MODEL_DIR=/path/to/unitree_model
```

### Train

```bash
python train_ppo_walk.py --task Isaac-G1-PPO-Walk-JOSE-v0 --headless
```

That uses the upstream defaults: 4096 environments and 50000 iterations. Output
goes to `logs/rsl_rl/isaac_g1_ppo_walk_jose_v0/<timestamp>/`, containing
checkpoints, TensorBoard events, and a `params/` directory with `env.yaml`,
`agent.yaml`, and `deploy.yaml` (the joint ordering, gains, and observation
scales needed for deployment).

For a quick check that everything works before committing to a full run:

```bash
python train_ppo_walk.py --task Isaac-G1-PPO-Walk-JOSE-v0 --headless \
  --num_envs 64 --max_iterations 5
```

Useful options: `--num_envs`, `--max_iterations`, `--seed`, `--device`,
`--resume`, `--load_run`, `--checkpoint`, `--video`, `--logger wandb`.

### Play and export

```bash
python play_ppo_walk.py --task Isaac-G1-PPO-Walk-JOSE-v0 --num_envs 32
```

This loads the latest checkpoint, writes `exported/policy.pt` (TorchScript) and
`exported/policy.onnx` next to it, and then runs the policy. Playback uses 32
environments on easier terrain with the command range opened to its full limits.

### What was adapted, and why

The upstream scripts target an older `rsl-rl` API. Three compatibility fixes were
needed; none of them change a training hyperparameter.

| Change | Reason |
|---|---|
| `obs_groups = {"actor": ["policy"], "critic": ["critic"]}` in the runner config | `rsl-rl` ≥ 4.0 requires this mapping explicitly. It is the same routing the old wrapper did implicitly. Without it `rsl-rl` falls back to `critic: ["policy"]` and the privileged critic observations are silently dropped. |
| `handle_deprecated_rsl_rl_cfg()` called in the train and play scripts | The runner config keeps upstream's `policy` / `empirical_normalization` form; this Isaac Lab helper translates it into the `actor` / `critic` model configs that `rsl-rl` 5.x expects. |
| Policy export goes through `runner.export_policy_to_jit/onnx` | `rsl-rl` ≥ 4.0 stores the actor and critic as separate models, so the runner owns the export. |
| `argcomplete` task autocompletion dropped | It is not a JOSE dependency, and `--task` accepts any registered task ID. |

You can confirm the routing worked: at startup the actor prints 480 input
features (96 observations × 5 history steps) and the critic prints 495, because
the critic additionally sees the privileged `base_lin_vel`.

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

#### Attaching an estimator to the velocity-tracking PPO walk teacher

Use `--adapter ppo_walk` with the estimator variant of the task. The training
method, number of rounds, and ablation axes are the same as for the AMP teachers;
only the teacher stack differs.

```bash
# 1. Train the teacher on the estimator variant of the task
python train_ppo_walk.py --task Isaac-G1-PPO-Walk-Estimator-JOSE-v0 --headless

# 2. Train an estimator against it
python -m jose.train_state_estimator \
  --teacher-checkpoint logs/rsl_rl/isaac_g1_ppo_walk_estimator_jose_v0/<run>/model_<n>.pt \
  --task Isaac-G1-PPO-Walk-Estimator-JOSE-v0 \
  --adapter ppo_walk \
  --run-name ppo_walk_lstm_w50 \
  --headless
```

`--adapter ppo_walk` selects the `rsl_rl_cfg_entry_point` config automatically, so
you do not have to pass `--agent_cfg_entry_point`.

Two things to know about this variant:

- **Why a separate task.** The velocity-tracking walk task keeps base linear
  velocity privileged: the critic sees it, the policy does not. That leaves the
  estimator nothing to write into. `Isaac-G1-PPO-Walk-Estimator-JOSE-v0` adds
  `base_lin_vel` to the policy observation group and changes nothing else, so
  `Isaac-G1-PPO-Walk-JOSE-v0` stays a verbatim copy of the upstream recipe.
- **What gets injected.** The estimator predicts the same 9-D privileged state as
  the other JOSE teachers (base linear velocity, base angular velocity, projected
  gravity), so estimator models, schedules, and ablation axes are reused
  unchanged. Because the policy observation carries a 5-step history per term,
  injection overwrites the *entire* history block of those three terms rather
  than only the newest frame; leaving the older frames intact would keep
  ground-truth privileged state visible to the policy and understate the
  closed-loop cost of estimation error. Angular velocity is rescaled by 0.2 on
  the way in, matching the observation term's own scale.

The same `--adapter ppo_walk` flag works for `evaluate_teacher.py` and
`play_teacher_with_estimator.py`, and `ablation_runner.py` exposes the task as
`--task ppo_walk_manager`.

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

Completed jobs are reused when you restart the same study. A finished report contains JSON and CSV results, Markdown and LaTeX tables, and PNG and PDF plots. Check `intermediate_results.json` while a study is running; every completed `jobs[]` entry exposes its mean episode length directly as `eplen`, matching the final `manifest.json`.

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
├── ppo_walk/                              # Velocity-tracking PPO walk task (rsl-rl)
├── train_ppo_walk.py, play_ppo_walk.py    # PPO walk training and playback
├── teacher_setup.py                       # Shared SKRL / rsl-rl teacher construction
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

The velocity-tracking PPO walk task in [ppo_walk/](ppo_walk/) is ported from [`unitreerobotics/unitree_rl_lab`](https://github.com/unitreerobotics/unitree_rl_lab), which uses the Apache 2.0 license. Ported files carry a header naming the source.
