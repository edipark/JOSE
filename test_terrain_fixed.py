"""Pinned slope levels: the assignment rule, the level choice, and the wiring.

Pure Python -- no Isaac Sim. The terrain importer is stood in for by the three
tensors ``place_on_levels`` touches.
"""

from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch

from jose.ppo_walk.fixed_levels import level_assignment, place_on_levels, select_max_level


def _terrain(num_envs=40, num_rows=10, num_cols=4):
    origins = torch.stack(
        torch.meshgrid(torch.arange(num_rows).float(), torch.arange(num_cols).float(), indexing="ij"), dim=-1
    )
    origins = torch.cat([origins, torch.zeros(num_rows, num_cols, 1)], dim=-1)  # (row, col, z=0)
    types = torch.div(torch.arange(num_envs), num_envs / num_cols, rounding_mode="floor").long()
    return SimpleNamespace(
        terrain_origins=origins,
        terrain_types=types,
        terrain_levels=torch.zeros(num_envs, dtype=torch.long),
        env_origins=torch.zeros(num_envs, 3),
    )


def test_levels_are_dealt_round_robin_and_stay_in_range():
    levels = level_assignment(23, 4)
    assert levels.tolist()[:7] == [0, 1, 2, 3, 4, 0, 1]
    assert int(levels.min()) == 0 and int(levels.max()) == 4


def test_every_column_holds_every_level_when_a_block_is_large_enough():
    terrain = _terrain(num_envs=40, num_cols=4)  # blocks of 10 envs per column
    levels = level_assignment(40, 4)
    for column in range(4):
        assert set(levels[terrain.terrain_types == column].tolist()) == set(range(5))


def test_place_on_levels_moves_each_env_to_its_own_row_and_column():
    terrain = _terrain()
    levels = level_assignment(40, 3)
    place_on_levels(terrain, levels)
    assert torch.equal(terrain.terrain_levels, levels)
    assert torch.equal(terrain.env_origins[:, 0], levels.float())  # row index
    assert torch.equal(terrain.env_origins[:, 1], terrain.terrain_types.float())  # column untouched


def test_place_on_levels_rejects_a_level_the_terrain_does_not_have():
    with pytest.raises(ValueError):
        place_on_levels(_terrain(num_rows=10), torch.full((40,), 10))


def test_selected_level_is_cumulative_not_the_deepest_survivor():
    rows = [{"level": 0, "death_rate": 0.0}, {"level": 1, "death_rate": 0.0},
            {"level": 2, "death_rate": 0.4}, {"level": 3, "death_rate": 0.0}]
    assert select_max_level(rows, 0.0) == 1
    assert select_max_level(rows, 0.5) == 3
    assert select_max_level([{"level": 0, "death_rate": 0.2}], 0.0) == -1


def test_every_pinned_level_is_registered_and_catalogued():
    import jose.ppo_walk  # noqa: F401  -- registers the ids
    from jose.ablation_catalog import TASK_IMPLEMENTATION, TASKS
    from jose.ppo_walk.slope_fixed_tasks import NUM_LEVELS, task_id

    for level in range(NUM_LEVELS):
        key = f"locomotion_slope_fixed_l{level}"
        assert TASKS[key][0] == task_id(level)
        assert task_id(level) in gym.registry
        assert "ppo_walk/fixed_levels.py" in TASK_IMPLEMENTATION[key]
