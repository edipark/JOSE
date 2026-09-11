"""Gym registrations for the push-disturbed locomotion task.

A separate module for the same reason as :mod:`terrain_tasks`: the package
``__init__.py`` is hashed into every ``TASK_IMPLEMENTATION`` tuple, and
``terrain_tasks.py`` into the two terrain tuples, so registering here moves no
recorded digest. Imported by ``ppo_walk/__init__.py``, which the study runners'
child processes import through their bootstrap (``run_method_comparison.py``).
"""

import gymnasium as gym

#: Root package name, derived rather than hardcoded (see ``terrain_tasks``).
_PKG = __name__.rsplit(".", 2)[0]


gym.register(
    id="Isaac-G1-PPO-Walk-Push-JOSE-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PKG}.ppo_walk.push_env_cfg:G1WalkPushEnvCfg",
        "play_env_cfg_entry_point": f"{_PKG}.ppo_walk.push_env_cfg:G1WalkPushPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PKG}.ppo_walk.agents.rsl_rl_ppo_cfg:G1WalkPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-PPO-Walk-Push-Estimator-JOSE-v0",
    entry_point=f"{_PKG}.ppo_walk.walk_estimator_env:G1WalkEstimatorEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PKG}.ppo_walk.push_env_cfg:G1WalkPushEstimatorEnvCfg",
        "play_env_cfg_entry_point": f"{_PKG}.ppo_walk.push_env_cfg:G1WalkPushEstimatorPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PKG}.ppo_walk.agents.rsl_rl_ppo_cfg:G1WalkEstimatorPPORunnerCfg",
    },
)
