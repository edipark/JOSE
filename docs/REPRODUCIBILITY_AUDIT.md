# Seed and reproducibility audit

Written 2026-09-09, while adding the terrain and training-protocol studies. This
records every seeding hazard found in the training and evaluation paths, what was
done about each, and — for the ones deliberately left alone — what it would cost
to fix them and what they affect.

Kept in `docs/` rather than in `logs/paper_data/` because `logs/` is gitignored
(`.gitignore:6`), and this needs to travel with the code it describes.

## Fixed

### H1 — the distillation trainer seeded only torch

`train_history_student.py` called `torch.manual_seed` and nothing else, while the
two methods it is compared against seed NumPy as well
(`train_state_estimator.py:115-116`, `train_set_baseline.py:108-109`). Any NumPy
or `random` draw made during environment construction or during the 300×250
rollout therefore came from a process-dependent stream for the students and a
seeded one for JOSE and SET.

**Fixed:** `random.seed` and `np.random.seed` added beside the existing call.

This mattered more than it first appeared. Isaac Lab seeds terrain generation from
`np.random.get_state()[1][0]` when the generator's own seed is unset
(`terrain_generator.py:139-144`), so on the new sloped task an unseeded trainer
would have drawn a *different terrain on every run*. See H10.

### H2 — the IMU arm consumed more randomness than the joint-only arm

`train_history_student.py` runs a second, clean evaluation every eval interval,
but only when `--method imu`. Evaluations step the environment and advance the
global torch stream, and nothing froze it, so from the first evaluation onward the
two arms trained on divergent streams *at an identical seed* — the one difference
that was supposed to be the six IMU channels and nothing else.

**Fixed:** both evaluations wrapped in `frozen_rng()` (`estimator/pipeline.py:26`),
the remedy the estimator pipeline already applied to the same problem.

### H10 — terrain generation was unseeded by construction

`TerrainGeneratorCfg.seed` defaults to `None`. Left there, the sloped task would
have given each training seed and each method a different terrain, so survival
would not have been measured on a common test set and seed-to-seed spread would
have mixed terrain variation into training variation.

**Fixed:** `SLOPE_TERRAINS_CFG` pins `seed=42`, making the terrain a property of
the task in the same way the reference clip is for the AMP teachers. Pinning it
also makes `use_cache=True` safe, which Isaac Lab otherwise warns against.

### Regression coverage

`test_jose.py` gained tests for the above, plus two for the normalizer bug fixed
in `e0ce15e`: one asserting that `best_state` carries both normalizers, and one
demonstrating *why* the clone in `_snapshot` is load-bearing —
`RunningNormalizer.state_dict()` ends in `.cpu()`, a no-op for a CPU normalizer
that hands back the live tensor which `update()` then mutates in place.

`train_dagger.py` was checked for the same class of bug and does not have it: it
keeps no in-memory best state and measures nothing after its loop, so its reported
metrics were taken live at the best iteration with the matching normalizers. A
comment now says so, and says what would reintroduce the bug.

## Left alone, deliberately

### H3 — training-time command grids are not re-seeded per command

`estimator/locomotion.py:443` re-seeds only when `command_seed_base` is passed,
and the three trainers do not pass it. Each command's initial condition therefore
depends on how much randomness the *previous* commands consumed. That is not
neutral across methods: the IMU corruptor draws two `randn_like` per step
(`distillation/imu.py:231`), so a method that reads an IMU shifts every later
command's reset relative to one that does not.

**This reaches a published number.** Table I(b)'s tracking column comes from
`eval_sensor_robustness.py` (the `robustness_cmdfix` study at scale 0), which does
not re-seed per command. Only `eval_friction_robustness.py` does — it was built
that way so its forward/reverse order gate could pass.

**Not fixed here**, because `estimator/locomotion.py` is in both
`TRAINING_IMPLEMENTATION` and the friction sweep's `SOURCE_FILES`, and changing it
moves Table I(b). Instead:

* the new studies pass `command_seed_base` and are clean from the start;
* the effect size is measurable without retraining anything — evaluate one
  recorded checkpoint with and without per-command re-seeding and difference the
  two. That measurement should be made before deciding whether Table I(b) needs
  re-measuring.

### H4, H5 — three different evaluation-seed conventions

The student trainer evaluates at `seed`, SET at `seed + 1000`
(`train_set_baseline.py:60`), JOSE at `seed + 10000`
(`train_state_estimator.py:63`). The friction sweep uses a fourth namespace
(10042/10043/10044) and asserts it is disjoint from the checkpoint seeds
(`run_friction_sweep.py:337-338`).

Left as is. Unifying them mid-study would misalign each new arm from the recorded
arm it is compared against, which is a worse problem than the inconsistency. The
new arms each inherit the convention of their own comparison: JOSE+IMU uses
`+10000`, SET+DAgger uses `+1000`.

### H6, H7, H9 — smaller ones

* **H6**, `train_history_student.py:257,271`: `previous = action` followed by
  `previous[done] = 0.0` writes into the tensor the policy just returned. Harmless
  as written — the environment has already stepped and `action` is rebound on the
  next iteration — but it is the pattern that broke rollout collection once before
  (`test_jose.py::test_history_rollout_samples_are_snapshotted_before_buffer_mutation`).
* **H7**: `ReplayBuffer.sample` and `.add` draw from the global device generator,
  interleaved with the rollout, so changing `--train-steps` changes the trajectory
  and not only the optimisation. `_fit_model` avoids this with a local generator
  (`estimator/pipeline.py:554`); the student trainer does not.
* **H9**: the student model is constructed after the environment, so its weight
  initialisation consumes whatever stream position the environment build left.
  Deterministic for a fixed stack, sensitive to an Isaac Lab upgrade.

## What determinism this project actually claims

`torch.use_deterministic_algorithms` is never called. `cudnn.deterministic` is set
in one place and to `False` (`train_ppo_walk.py:110`). `CUBLAS_WORKSPACE_CONFIG`
and `PYTHONHASHSEED` are unset.

So bitwise reproducibility holds *within one GPU and one software stack*, and the
project has demonstrated it there: walk's `lstm_w25_all`, retrained 21 hours later
under a different implementation digest, reproduced `10.939131736755371` to the
last digit. It does not hold across machines, which is why the friction gate uses
non-zero tolerances on everything except survival, and why the terrain study
trains its own teachers rather than shipping one.

## Cross-references

* `logs/paper_data/README.md` — the audit trail for the published numbers.
* `verify_build_inertness.py` — the gate that decides whether recorded rows may be
  reused beside newly trained ones.
* `docs/REMOTE_TERRAIN_STUDY.md` — the determinism caveats restated for the
  machine the terrain study runs on.
