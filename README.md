# JOSE G1

JOSE is an Isaac Lab package for the Unitree G1 robot (29 DOF). You can use it to train walk, dance, and jump policies. It also includes state estimators, DAgger students, ablation studies, and motion tools.

## Requirements

| Requirement | Version | Notes |
|---|---|---|
| OS | Linux | |
| Python | 3.10+ | |
| Isaac Sim | 5.1.x | |
| Isaac Lab | 2.3.x+ | provides `skrl>=2.0,<3.0` and `rsl-rl-lib` — nothing extra to install for those |
| GPU | CUDA GPU | recommended |

### 1. Set up Isaac Lab

Skip this if you already have an Isaac Lab environment — jump to step 2. Otherwise, follow the
[official Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/quickstart.html)
to create a Python environment (conda or venv) with Isaac Sim and Isaac Lab installed in it. That
install already includes `skrl` and `rsl-rl-lib`, so there is nothing extra to add for those.

### 2. Install JOSE

Activate that environment, then install JOSE into it in editable mode. Replace `/path/to/JOSE` with the path to this repository.

```bash
conda activate <your-isaac-lab-env>
cd /path/to/JOSE
python -m pip install -e .
```

Check the installation:

```bash
python -c "import jose; print('JOSE tasks registered')"
```

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
| **Velocity-tracking PPO walk** | `Isaac-G1-PPO-Walk-JOSE-v0` | `PPO` (rsl-rl) |
| **PPO walk, estimator variant** | `Isaac-G1-PPO-Walk-Estimator-JOSE-v0` | `PPO` (rsl-rl) |

The first three tasks end in `-Direct-v0`. They are Direct-workflow environments
trained with `skrl` through `jose.train`. The last two are manager-based
environments trained with `rsl-rl` through `train_ppo_walk.py`; see
[Velocity-tracking PPO walk](#velocity-tracking-ppo-walk) below.

Use a different task ID to train dance or jump:

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
conda activate <your-isaac-lab-env>
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

Use `train_ppo_walk.py`. This is the rsl-rl entry point; do **not** use
`jose.train`, which drives the SKRL `-Direct-v0` tasks and cannot load this task's
config.

```bash
conda activate <your-isaac-lab-env>
cd /path/to/JOSE

python train_ppo_walk.py --task Isaac-G1-PPO-Walk-JOSE-v0 --headless
```

That single command runs the complete recipe at its defaults — 4096 environments
for 50000 iterations — and needs no other arguments.

Before committing to a full run, a short smoke run confirms the whole stack
works (this finishes in well under a minute):

```bash
python train_ppo_walk.py --task Isaac-G1-PPO-Walk-JOSE-v0 --headless \
  --num_envs 64 --max_iterations 5
```

#### Defaults

These are the values already in `ppo_walk/agents/rsl_rl_ppo_cfg.py` and
`ppo_walk/walk_env_cfg.py`. You do not need to pass any of them; they are listed
so you know what a bare `train_ppo_walk.py` run is actually doing.

**Runner / PPO** (`ppo_walk/agents/rsl_rl_ppo_cfg.py`)

| Setting | Default | Setting | Default |
|---|---|---|---|
| `max_iterations` | `50000` | `learning_rate` | `1.0e-3` |
| `num_steps_per_env` | `24` | `schedule` | `adaptive` |
| `save_interval` | `100` | `desired_kl` | `0.01` |
| `num_learning_epochs` | `5` | `gamma` | `0.99` |
| `num_mini_batches` | `4` | `lam` | `0.95` |
| `clip_param` | `0.2` | `max_grad_norm` | `1.0` |
| `entropy_coef` | `0.01` | `value_loss_coef` | `1.0` |
| `actor_hidden_dims` | `[512, 256, 128]` | `use_clipped_value_loss` | `True` |
| `critic_hidden_dims` | `[512, 256, 128]` | `init_noise_std` | `1.0` |
| `activation` | `elu` | `empirical_normalization` | `False` |
| `obs_groups` | `{"actor": ["policy"], "critic": ["critic"]}` | `clip_actions` | `None` (no clipping) |
| `seed` | `42` | `device` | `cuda:0` |
| `logger` | `tensorboard` | `experiment_name` | derived from the task ID |

With the defaults, one iteration collects `4096 envs x 24 steps = 98304`
transitions and takes 5 epochs over 4 minibatches.

**Environment** (`ppo_walk/walk_env_cfg.py`)

| Setting | Default | Note |
|---|---|---|
| `scene.num_envs` | `4096` | `64` for the play config |
| `sim.dt` | `0.005` | 200 Hz physics |
| `decimation` | `4` | 50 Hz policy, so `step_dt = 0.02` |
| `episode_length_s` | `20.0` | 1000 policy steps per episode |
| action scale | `0.25` | joint position offsets from the default pose |
| `history_length` | `5` | on both the policy and critic groups |
| command resample | every `10.0` s | no curriculum; the full range is sampled from step 0 |
| command range | `lin_x [0, 1.0]`, `lin_y [-0.5, 0.5]`, `ang_z [-1.0, 1.0]` | official Isaac Lab G1 flat ranges |
| terrain | flat plane | no generator, no height scanner |
| terminations | timeout, base height `< 0.2` m, tilt `> 0.8` rad | |

To change any of them, edit the config file rather than passing flags. Note that
the reward weights and command ranges are not free parameters: the combination in
the file is the one that was measured to walk, and several nearby combinations
were measured to produce a statue. See
[Flat-terrain command tracking](#flat-terrain-command-tracking) before retuning.

#### Command-line options

| Option | Default | Purpose |
|---|---|---|
| `--task` | `Isaac-G1-PPO-Walk-JOSE-v0` | Task ID to train |
| `--headless` | off | No GUI; use this for real runs |
| `--num_envs` | from config (`4096`) | Override the environment count |
| `--max_iterations` | from config (`50000`) | Override the iteration count |
| `--seed` | from config (`42`) | Seed; `-1` picks a random one |
| `--device` | `cuda:0` | GPU to use |
| `--run_name` | none | Suffix appended to the run directory name |
| `--resume` | off | Continue from a saved checkpoint |
| `--load_run` | latest | Which run directory to resume from |
| `--checkpoint` | latest | Which checkpoint file to resume from |
| `--logger` | `tensorboard` | `tensorboard`, `wandb`, or `neptune` |
| `--video` | off | Record training clips (needs `--enable_cameras`) |
| `--distributed` | off | Multi-GPU training via `torchrun` |

#### Output layout

```text
logs/rsl_rl/isaac_g1_ppo_walk_jose_v0/<YYYY-MM-DD_HH-MM-SS>[_<run_name>]/
├── model_0.pt, model_100.pt, ...   # checkpoints, every save_interval iterations
├── events.out.tfevents.*           # TensorBoard scalars
├── git/                            # repository state at launch
└── params/
    ├── env.yaml                    # resolved environment config
    ├── agent.yaml                  # resolved runner config
    ├── deploy.yaml                 # joint order, gains, observation scales for sim2sim/sim2real
    └── walk_env_cfg.py             # copy of the config source
```

#### Watching a run

```bash
tensorboard --logdir logs/rsl_rl/isaac_g1_ppo_walk_jose_v0
```

**`Train/mean_reward` is not a health signal here.** The tracking kernels use
`std = 0.5`, so a perfectly motionless robot already collects
`exp(-|cmd|^2 / 0.25)` every step — 0.795 reward/s under this task's command
ranges. A policy can raise total reward while standing perfectly still, which is
exactly how the previous recipe failed.

Watch these instead:

- `Episode_Reward/feet_air_time` — **zero means the robot is not stepping at
  all**, whatever the total reward says. Its ceiling is 0.30/s (weight 0.75 ×
  threshold 0.4); a walking policy reaches a few percent of that within a few
  hundred iterations and keeps climbing.
- `Train/mean_episode_length` should climb toward 1000. If it *peaks early and
  then decays* while reward rises, the policy is learning to fall on purpose.
- `Metrics/base_velocity/error_vel_xy` and `error_vel_yaw` should fall.
- `Episode_Termination/bad_orientation` should fall as the robot stops toppling.

Confirm with `eval_ppo_walk.py` before concluding anything; see
[Flat-terrain command tracking](#flat-terrain-command-tracking).

#### Resuming

```bash
python train_ppo_walk.py --task Isaac-G1-PPO-Walk-JOSE-v0 --headless \
  --resume --load_run 2026-08-28_22-25-37 --checkpoint model_1000.pt
```

Omit `--load_run` / `--checkpoint` to pick up the most recent run and checkpoint.

Note that `--checkpoint` means different things in the two scripts: in
`train_ppo_walk.py` it is a file name looked up inside `--load_run`, while in
`play_ppo_walk.py` it is a full path to the checkpoint file.

#### Multi-GPU

```bash
python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 \
  train_ppo_walk.py --task Isaac-G1-PPO-Walk-JOSE-v0 --headless --distributed
```

### Play and export

```bash
python play_ppo_walk.py --task Isaac-G1-PPO-Walk-JOSE-v0 --num_envs 32
```

This loads the latest checkpoint, writes `exported/policy.pt` (TorchScript) and
`exported/policy.onnx` next to it, and then runs the policy. Playback uses 32
environments on easier terrain with the command range opened to its full limits.

To play a specific checkpoint, and to run at wall-clock speed:

```bash
python play_ppo_walk.py --task Isaac-G1-PPO-Walk-JOSE-v0 \
  --checkpoint logs/rsl_rl/isaac_g1_ppo_walk_jose_v0/<run>/model_5000.pt \
  --num_envs 4 --real-time
```

The play loop runs until you close it, so stop it with Ctrl+C. The exported files
are written before the loop starts, so you can interrupt it as soon as they
appear if you only wanted the export.

> Recording with `--video` in headless mode has to initialize the RTX renderer,
> which can take several minutes the first time while shaders compile. Leave
> `--video` off unless you need the clip.

### Flat-terrain command tracking

`ppo_walk/walk_env_cfg.py` no longer matches the upstream unitree_rl_lab reward
set: the stock `foot_clearance_reward` term rewards standing still at its
maximum, so the original recipe converges to a policy that ignores the
velocity command. The current reward set is the one measured to actually walk.
See [docs/DESIGN_NOTES.md](docs/DESIGN_NOTES.md#flat-terrain-command-tracking)
for the bug, the five configurations that were measured, and the known
remaining defects.

#### Warm-starting the actor only

Resuming a checkpoint trained under the old reward set would drag its stale value
function into the new MDP. `--load_actor_only` loads the actor weights and leaves
the critic and the optimizer freshly initialised:

```bash
python train_ppo_walk.py --task Isaac-G1-PPO-Walk-JOSE-v0 --headless \
  --load_actor_only logs/rsl_rl/isaac_g1_ppo_walk_jose_v0/<run>/model_2999.pt
```

The actor now takes 495 inputs against the old 480. Because `base_lin_vel` is
declared *first* in the observation group, every old input keeps its relative
offset, so the old weight matrix is copied into the right-hand columns and the 15
new columns are zeroed. The flag is mutually exclusive with `--resume`.

#### Evaluating command tracking

Watching the viewer cannot distinguish a walking policy from a statue that
survives. `eval_ppo_walk.py` holds each of eight fixed commands across every
environment and reports tracking error, survival, air time, foot-lift rate,
double-stance fraction and zero-command drift:

```bash
python eval_ppo_walk.py --task Isaac-G1-PPO-Walk-JOSE-v0 --headless \
  --num_envs 64 \
  --checkpoint logs/rsl_rl/isaac_g1_ppo_walk_jose_v0/<run>/model_2999.pt \
  --output eval.json
```

It also runs six sanity checks, including that `feet_air_time` never fires in
double stance and that measured `vx` rises with commanded `vx`.

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
`--task ppo_walk`.

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

### Evaluate the teacher baseline

`evaluate_teacher.py` runs the teacher on ground-truth privileged observations
(no estimator involved) and reports the deterministic baseline that every
estimator ablation compares against. It is the collection step every ablation
script drives internally, so you rarely need to run it by hand outside of a
sanity check.

```bash
# AMP teacher
python -m jose.evaluate_teacher \
  --teacher-checkpoint "$TEACHER" \
  --task Isaac-G1-AMP-Walk-JOSE-Direct-v0 \
  --agent skrl_amp_cfg_entry_point \
  --adapter amp \
  --num-envs 256 \
  --collect-steps 2000 \
  --headless

# ppo_walk teacher
python -m jose.evaluate_teacher \
  --teacher-checkpoint "$TEACHER" \
  --task Isaac-G1-PPO-Walk-Estimator-JOSE-v0 \
  --agent rsl_rl_cfg_entry_point \
  --adapter ppo_walk \
  --num-envs 256 \
  --collect-steps 2000 \
  --headless
```

Do not confuse this with `eval_ppo_walk.py`: that script holds a fixed
velocity command and reports tracking error/gait statistics on the *plain*
`Isaac-G1-PPO-Walk-JOSE-v0` task (see
[Evaluating command tracking](#evaluating-command-tracking)) and never touches
an estimator. `evaluate_teacher.py` is the estimator-pipeline baseline;
`eval_ppo_walk.py` is a standalone policy-quality check.

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

The `ppo_walk` variant works the same way, against the estimator task and its own checkpoint:

```bash
python -m jose.play_teacher_with_estimator \
  --teacher-checkpoint "$TEACHER" \
  --estimator-checkpoint "$ESTIMATOR" \
  --task Isaac-G1-PPO-Walk-Estimator-JOSE-v0 \
  --steps 1000 \
  --csv-output logs/rollout.csv \
  --video \
  --headless
```

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

All ablation studies need a teacher checkpoint. The default runs can take a long time. Use `--dry-run` to check the run plan or `--fast` for a smaller test. `--task` accepts `amp_walk`, `amp_dance`, `amp_jump`, or `ppo_walk` for `run_architecture_ablation.py`/`run_window_ablation.py`/`run_joint_scope_ablation.py`/`run_dagger_ablation.py` (default `amp_walk`); `run_method_comparison.py` only covers the three AMP tasks.

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

### Measure what DAgger is worth

JOSE trains in two phases: a supervised warm start on teacher-driven data, then
on-policy DAgger rounds. This study varies only the number of DAgger rounds, so
the `lstm_w25_all_r00` arm stops after the warm start and is the "no DAgger"
baseline.

```bash
python -m jose.run_dagger_ablation \
  --teacher-checkpoint "$TEACHER" \
  --task amp_walk \
  --headless
```

The 10-round arm keeps the plain `lstm_w25_all` slug, so it is reused from the
catalog if another study already ran it -- only the 5- and 0-round arms actually
train. Read the result from `dagger_learning_curve.png`: the middle panel is
closed-loop episode length per round, which is what says whether the rounds
helped. The top panel (validation MSE) does not track it.

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

All comparison and ablation studies are grouped by the content identity of the
teacher checkpoint:

```text
logs/jose_g1/ablation/<teacher>__<checkpoint>/<task>/studies/
  architecture/<date>/
  window/<date>/
  joint_scope/<date>/
  method_comparison/<date>/methods/<method>/window_<N>/joints_<scope>/seed_<N>/
```

Pass `--run-name <name>` to `run_method_comparison` when a stable,
human-readable study name is preferred or when resuming that exact study.

Completed jobs are reused when you restart the same named study. Architecture,
window, joint-scope, DAgger, and method comparisons all write the same report
bundle under their run's `report/` directory: `summary.json`, `table.md`,
`report.md`, and two PNG figures. Per-run raw values stay in `results.jsonl`
next to it. For estimator ablations, check
`intermediate_results.json` while a catalog study is running; every completed
`jobs[]` entry exposes its mean episode length directly as `eplen`, matching the
final `manifest.json`.

### Motion-fidelity metrics

Once every method reaches the performance ceiling, `return_mean` and
`episode_length_mean` stop separating them and fidelity metrics do the work.
Every method reports these, computed by one shared implementation
(`estimator/metrics.py`) so the columns mean the same thing across rows:

| Metric | What it says |
|---|---|
| `mpjpe_g`, `mpjpe_l` | Mean per-body position error against the teacher, global and root-relative, in mm |
| `root_position_error` | Root drift from the teacher's trajectory, in mm |
| `teacher_action_mse` | Action-level imitation error. For JOSE this is the same teacher policy driven by the estimate instead of the true state |
| `amp_raw_style` | The AMP discriminator's own score for how reference-like the motion is |
| `action_smoothness`, `torque_rms`, `energy` | Jitter and effort, the `E_acc` analogues |

MPJPE rolls the teacher and the student separately from the *same* seeded
reset and compares body positions frame by frame, so `--mpjpe-horizon` (default
100 steps) is part of the measurement: the two trajectories diverge chaotically
once the policies differ, and a number without its horizon is meaningless.
`PrivilegedTeacher` rows carry the teacher measured against itself — the
determinism floor, which reads 0.000 mm and is what makes every other row
interpretable. `mpjpe_per_body` and `mpjpe_by_step` are kept in `results.jsonl`
for plotting the divergence curve later without re-running anything.

### Catalog layout

Each study run directory (`studies/<study>/<date>/`) contains:

- `manifest.json` — study identity (teacher, task, config) and per-job status; `status` becomes `"complete"` once every job has a result.
- `intermediate_results.json` — the same information plus every currently-available result row and its aggregated `summary`; safe to read while the study is still running.
- `results.jsonl` — one JSON object per completed job, written once the study finishes; this is what `generate_report.py`/`reporting.aggregate()` consume.
- `report/` — `summary.json` (every aggregated metric), `table.md`, `report.md`, and, if matplotlib is installed, `dagger_learning_curve.png` and `learning_curves.png`. Both figures are mean ± std bands across seeds. Anything you want to plot yourself comes from `results.jsonl`, which is unfiltered.

Below the study directories, `catalog/` holds the content-addressed jobs and
datasets themselves (keyed by a digest of the job's full configuration), which
is what lets two different studies reuse an identical estimator run instead of
retraining it.

### Regenerate a report from existing results

If you edit `results.jsonl` by hand, or change `reporting.py` and want updated
tables/plots without retraining anything, regenerate the report bundle directly:

```bash
python -m jose.generate_report \
  logs/jose_g1/ablation/<teacher>__<checkpoint>/amp_walk/studies/architecture/<date>/results.jsonl \
  logs/jose_g1/ablation/<teacher>__<checkpoint>/amp_walk/studies/architecture/<date>/report \
  --require-plots
```

`--require-plots` fails the command if matplotlib is not importable; it does
not require every individual plot to have qualifying data — a plot skipped for
that reason is reported in `report.md` instead of raising an error.

### Sweeping across checkpoints

`run_architecture_ablation.py`, `run_window_ablation.py`,
`run_joint_scope_ablation.py`, `run_dagger_ablation.py`, and
`run_method_comparison.py` each take one
teacher checkpoint per invocation. `run_checkpoint_sweep.py` repeats one of
them across every checkpoint in a training run (e.g. `agent_10000.pt` ..
`agent_100000.pt`, plus `best_agent.pt`) and merges the resulting
`results.jsonl` files into one comparison report, so you can see how an
ablation's outcome changes over the course of training rather than only at the
final checkpoint.

```bash
python -m jose.run_checkpoint_sweep \
  --ablation-script architecture \
  --checkpoint-dir logs/skrl/g1_jose_amp_walk/<run>/checkpoints \
  --glob "agent_*.pt" \
  --include-best \
  --sweep-output-dir logs/jose_g1/ablation/sweeps/amp_walk_architecture \
  -- --task amp_walk --seeds 3 --fast --headless
```

Everything after the bare `--` is forwarded unchanged to each per-checkpoint
invocation, exactly like the flags shown for `run_architecture_ablation.py`
above. `--ablation-script` selects the target (`architecture`, `window`,
`joint_scope`, or `method_comparison`); `method_comparison` additionally
requires `--case-task {walk,dance,jump}` in place of passthrough `--task`,
since that script's own `--case` flag bundles a task with its checkpoint:

```bash
python -m jose.run_checkpoint_sweep \
  --ablation-script method_comparison \
  --checkpoint-dir logs/skrl/g1_jose_amp_walk/<run>/checkpoints \
  --case-task walk \
  -- --seeds 42 43 44 --fast --headless
```

Use `--dry-run` first to check the resolved checkpoint list and the exact
command each one will run. Each checkpoint still lands in the normal
content-addressed catalog under `--output-dir`, so a failed or interrupted
sweep can just be re-run — completed checkpoints are reused, not repeated;
pass `--continue-on-error` to keep sweeping past a checkpoint whose run fails
instead of aborting. The merged output goes to `--sweep-output-dir`
(default `<output-dir>/sweeps/<script>_<date>/`): `combined_results.jsonl`,
`sweep_manifest.json` (per-checkpoint status), and a `report/` bundle exactly
like a single study's, with an added `teacher_id` column identifying which
checkpoint each row came from.

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

### Visualize a motion file

Plots joint trajectories and base orientation from an NPZ motion file on a desktop (no Isaac Sim needed).

```bash
python -m jose.motions.visualize_motion --file motions/G1_walk.npz
```

### Copy pelvis data between motion files

Copies pelvis position/velocity/rotation trajectories from one NPZ file into another (e.g. to substitute a corrected pelvis trace into an otherwise-good motion capture). Add `--output` to write a new file instead of overwriting `--target`; `--dry-run` reports what would change without writing anything.

```bash
python -m jose.motions.update_pelvis_data \
  --source motions/corrected_pelvis.npz \
  --target motions/G1_walk.npz \
  --output motions/G1_walk_fixed.npz
```

## Diagnostics and parity checks

These compare JOSE's implementation against reference code or hardware logs; they are not needed for day-to-day training.

```bash
# Diff AMP hyperparameters/config against a humanoid_amp checkout
python -m jose.tools.check_humanoid_amp_parity --reference /path/to/humanoid_amp

# Diff AMP config against a SOLO reference checkout
python -m jose.tools.check_solo_amp_parity --reference /path/to/SOLO

# Inspect a G1 IMU log (or a synthetic one) for sanity
python -m jose.tools.analyze_g1_imu_log --input logs/imu_capture.json
python -m jose.tools.analyze_g1_imu_log --synthetic --output logs/imu_synthetic.json

# Compare a recorded rollout against a reference motion
python -m jose.tools.reference_tracking \
  --motion motions/G1_walk.npz \
  --tracked-motion logs/recorded_walk.npz \
  --output reference_tracking.json
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
├── ppo_walk/                              # Velocity-tracking PPO walk task (rsl-rl)
├── train_ppo_walk.py, play_ppo_walk.py    # PPO walk training and playback
├── eval_ppo_walk.py                       # Fixed-command tracking evaluation (no estimator)
├── teacher_setup.py                       # Shared SKRL / rsl-rl teacher construction
├── schema.py                              # Observation-layout schemas (AMP + ppo_walk)
├── skrl_compat.py                         # SKRL runner-config compatibility shims
├── task_math.py                           # Shared reward/observation math helpers
├── agents/                                # SKRL AMP settings
├── motions/                               # Motion data and motion tools
├── estimator/                             # Estimator models and training code
├── distillation/                          # History students and IMU input code
├── tools/                                 # Smoke tests and diagnostics
├── train.py, play.py                      # Teacher training and playback
├── train_state_estimator.py               # State-estimator training
├── evaluate_teacher.py                    # Deterministic teacher-only baseline eval
├── train_dagger.py                        # 58D DAgger student training
├── train_*_distillation.py                # Joint-only and IMU student training
├── ablation_catalog.py                    # Content-addressed teacher/job catalog primitives
├── ablation_common.py                     # Shared subprocess/fingerprint helpers
├── ablation_runner.py                     # Shared ablation execution engine
├── reporting.py                           # Aggregation, tables, and plots for ablation studies
├── generate_report.py                     # Regenerate a report from an existing results.jsonl
├── run_architecture_ablation.py, run_window_ablation.py,
│   run_joint_scope_ablation.py,
│   run_dagger_ablation.py                 # Estimator architecture/window/joint-scope/DAgger studies
├── run_method_comparison.py               # Four-way Teacher/IMU/Joint-only/JOSE comparison
└── run_checkpoint_sweep.py                # Repeat any of the above across many checkpoints
```

JOSE runs policy control at 50 Hz and uses all 29 G1 joints. The default estimator input contains 29 joint positions and 29 simulator joint velocities. Direct history students do not use explicit base linear velocity or raw accelerometer data.

## License and attribution

This repository is based on [Isaac Lab](https://github.com/isaac-sim/IsaacLab), which uses the BSD-3-Clause license. See [LICENSE](LICENSE) for details.

The G1 motion and USD files keep attribution to [`linden713/humanoid_amp`](https://github.com/linden713/humanoid_amp). See [usd/README.md](usd/README.md) for details.

The velocity-tracking PPO walk task in [ppo_walk/](ppo_walk/) is ported from [`unitreerobotics/unitree_rl_lab`](https://github.com/unitreerobotics/unitree_rl_lab), which uses the Apache 2.0 license. Ported files carry a header naming the source.
