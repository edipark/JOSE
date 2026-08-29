# Design notes

Background on non-obvious configuration decisions. This is not a usage guide —
see [README.md](../README.md) for that.

## Flat-terrain command tracking

The original unitree_rl_lab recipe converged to a policy that stands still and
ignores the velocity command. The cause is `foot_clearance_reward`:

```python
reward = exp(-sum((foot_z - target)^2 * tanh(k * |v_foot_xy|)) / std)
```

`tanh(k * |v_foot_xy|)` is zero whenever a foot is not moving, so the sum inside
`exp` is zero and the term returns `exp(0) = 1` — its **maximum** — for a robot
standing on two feet. In run `2026-08-28_23-06-25` it saturated at `0.787` of a
`0.81` ceiling, 69% of the net episode reward. Upstream tracks the same bug in
[unitree_rl_lab#80](https://github.com/unitreerobotics/unitree_rl_lab/pull/80).

`ppo_walk/walk_env_cfg.py` now carries the configuration that was measured to
walk. Its module docstring derives every change; the short version is that
removing the degenerate term was necessary but **not sufficient**. Five
configurations were trained and measured with `eval_ppo_walk.py`, and only the
last one produces a gait:

| Configuration | `feet_air_time` @500 | Walks? |
|---|---|---|
| air-time reward, original command ranges | 0.0001 | no — also no at 3000 iterations |
| + official G1 command ranges | 0.0003 | no |
| + posture penalties at official weights | 0.0004 | no |
| **+ drops `joint_vel` and `alive`** ← shipped | **0.0117** | **yes** |
| + a zero-command standing reward | 0.0020 | no — it rebuilt the statue |

`joint_vel_l2` (-0.001, absent from official G1) was the largest single penalty in
every run at -0.13 to -0.17/s, and it taxes exactly what a gait needs. Dropping it
together with the `alive` bonus is what unlocked stepping.

Measured at 500 iterations, 64 environments: commanded `vx` 0.0 / 0.3 / 0.6 gives
0.13 / 0.27 / 0.38 m/s, yaw ±0.3 gives ±0.24 rad/s, ~11 foot lifts/s, zero falls.
The known remaining defect is that it marches in place and creeps at 0.13 m/s
under a zero command — fix that by fine-tuning a checkpoint that already walks,
with a weight ≤0.05 or Isaac Lab's `stand_still_joint_deviation_l1`, and re-run
the evaluation to confirm the gait survived. The naive version of that fix
(weight 0.5) destroyed the gait outright.
