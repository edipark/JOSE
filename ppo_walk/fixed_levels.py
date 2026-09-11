"""Pin every environment of a curriculum terrain to one level, for good.

The sloped task trains its teacher under ``terrain_levels_vel``, which moves an
environment one row up or down at every reset according to how far it walked.
That is right for PPO and wrong for everything measured afterwards: under it a
method that falls is sent to gentler slopes, one that walks well to steeper
ones, so no two methods are scored on the same terrain. With these helpers the
level becomes a property of the environment index -- the same for every method,
every seed, collection and evaluation alike.

Only ``torch`` is imported, and the module sits outside ``ppo_walk/mdp`` (whose
package init pulls in Isaac Lab), so the rule is testable without Isaac Sim.
"""

from __future__ import annotations

import torch


def level_assignment(num_envs: int, max_level: int, device=None) -> torch.Tensor:
    """Levels ``0..max_level`` dealt round-robin over the environment index.

    Isaac Lab assigns the terrain *type* (the column) in contiguous blocks of
    ``num_envs / num_cols`` (``TerrainImporter._compute_env_origins_curriculum``),
    so a round-robin level puts every level in every column once a block holds at
    least ``max_level + 1`` environments.
    """
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if max_level < 0:
        raise ValueError("max_level must be non-negative")
    return torch.arange(num_envs, device=device) % (max_level + 1)


def place_on_levels(terrain, levels: torch.Tensor) -> None:
    """Move every environment's origin to row ``levels[i]`` of its own column."""
    if terrain.terrain_origins is None:
        raise RuntimeError("fixed terrain levels need a generated (curriculum) terrain")
    num_rows = terrain.terrain_origins.shape[0]
    if int(levels.min()) < 0 or int(levels.max()) >= num_rows:
        raise ValueError(f"levels must lie in [0, {num_rows - 1}], got [{int(levels.min())}, {int(levels.max())}]")
    levels = levels.to(device=terrain.terrain_levels.device, dtype=terrain.terrain_levels.dtype)
    terrain.terrain_levels[:] = levels
    terrain.env_origins[:] = terrain.terrain_origins[levels, terrain.terrain_types]


def fix_terrain_levels(env, env_ids, max_level: int) -> None:
    """Startup event: deal levels ``0..max_level`` round-robin, once, at scene build.

    Paired with the curriculum term removed, nothing moves them afterwards: every
    reset returns an environment to the tile it started on.
    """
    terrain = env.scene.terrain
    place_on_levels(terrain, level_assignment(terrain.env_origins.shape[0], max_level, terrain.env_origins.device))


def select_max_level(levels: list[dict], max_death_rate: float) -> int:
    """Largest ``K`` with every level ``0..K`` at or below ``max_death_rate`` (%).

    Cumulative on purpose: a steep level that survives after a gentler one failed
    is a lucky draw, not a capability. Returns -1 when level 0 already fails.
    """
    by_level = {int(row["level"]): float(row["death_rate"]) for row in levels}
    chosen = -1
    for level in range(len(by_level)):
        if level not in by_level or by_level[level] > max_death_rate:
            break
        chosen = level
    return chosen
