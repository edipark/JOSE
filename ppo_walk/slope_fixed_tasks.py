"""Gym registrations for the sloped task with pinned levels, one id per ``K``.

``Isaac-G1-PPO-Walk-SlopeFixed-L<K>-Estimator-JOSE-v0`` runs the student
pipeline on levels ``0..K``. Registered from its own module for the same reason
as :mod:`terrain_tasks` and :mod:`push_tasks`: no recorded digest moves.
"""

import gymnasium as gym

_PKG = __name__.rsplit(".", 2)[0]

#: Mirrors ``slope_fixed_env_cfg.NUM_LEVELS`` without importing Isaac Lab here.
NUM_LEVELS = 10


def task_id(max_level: int) -> str:
    return f"Isaac-G1-PPO-Walk-SlopeFixed-L{max_level}-Estimator-JOSE-v0"


for _level in range(NUM_LEVELS):
    gym.register(
        id=task_id(_level),
        entry_point=f"{_PKG}.ppo_walk.walk_estimator_env:G1WalkEstimatorEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{_PKG}.ppo_walk.slope_fixed_env_cfg:G1WalkSlopeFixedL{_level}EstimatorEnvCfg",
            "play_env_cfg_entry_point": f"{_PKG}.ppo_walk.slope_fixed_env_cfg:G1WalkSlopeFixedL{_level}EstimatorEnvCfg",
            "rsl_rl_cfg_entry_point": f"{_PKG}.ppo_walk.agents.rsl_rl_ppo_cfg:G1WalkEstimatorPPORunnerCfg",
        },
    )
del _level
