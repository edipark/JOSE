"""The sloped task with the terrain curriculum removed and the levels pinned.

For the *student* pipeline only. The teacher keeps training under
``terrain_levels_vel`` (:mod:`terrain_env_cfg`), which PPO needs: it cannot learn
a steep slope before a gentle one. Nothing after the teacher needs it. DAgger's
labels come from a teacher that is already trained, and an evaluation run under
the curriculum scores every method on terrain it chose for itself -- a method
that falls is demoted to gentler slopes, one that walks well promoted to steeper
ones -- so the numbers of two methods were never measured on the same slopes.
The first slope study (``locomotion_slope``) ran that way; its logs show the
curriculum term active in all twelve comparison jobs.

Here every environment is dealt a level ``0..K`` by index at scene build
(:func:`fixed_levels.fix_terrain_levels`) and keeps it: through collection,
through every evaluation, for every method and seed. The terrain itself is the
sloped task's, byte for byte -- same generator, same pinned seed -- so the rows
mean the same slopes they meant to the teacher.

``K`` is not chosen here. ``eval_slope_levels.py`` measures the teacher level by
level in the teacher's own survival protocol and picks the largest ``K`` at which
the teacher's death rate stays at or below 0.5 % on every level ``0..K``; the
study script then runs the pipeline on ``locomotion_slope_fixed_l<K>``. One configuration per ``K`` is generated below
so the level is part of the task id, and therefore of every recorded row.

Nothing in :mod:`terrain_env_cfg` is edited: that file is hashed into the two
terrain tuples, and the studies already recorded under them stay resumable.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from . import fixed_levels
from .terrain_env_cfg import SLOPE_TERRAINS_CFG, G1WalkSlopeEstimatorEnvCfg
from .walk_env_cfg import EventCfg

#: One configuration per possible ``K``: the generator has this many rows.
NUM_LEVELS = SLOPE_TERRAINS_CFG.num_rows


@configclass
class FixedLevelEventCfg(EventCfg):
    """The base events plus the one-time level assignment."""

    fix_terrain_levels = EventTerm(
        func=fixed_levels.fix_terrain_levels,
        mode="startup",
        params={"max_level": NUM_LEVELS - 1},
    )


@configclass
class G1WalkSlopeFixedEstimatorEnvCfg(G1WalkSlopeEstimatorEnvCfg):
    """Estimator task on the sloped terrain, levels ``0..MAX_LEVEL`` pinned by index."""

    events: FixedLevelEventCfg = FixedLevelEventCfg()

    #: Overridden per generated subclass below.
    MAX_LEVEL = NUM_LEVELS - 1

    def __post_init__(self):
        super().__post_init__()
        # No curriculum term: nothing may move an environment off its level.
        self.curriculum.terrain_levels = None
        # Keep the generator's curriculum *layout* -- rows sorted by difficulty --
        # so that "level" still means "slope band". It is the ``terrain_levels``
        # term above, not this flag, that moves environments between rows.
        self.scene.terrain.terrain_generator.curriculum = True
        self.events.fix_terrain_levels.params["max_level"] = self.MAX_LEVEL


def _pinned(max_level: int) -> type:
    @configclass
    class Pinned(G1WalkSlopeFixedEstimatorEnvCfg):
        MAX_LEVEL = max_level

    Pinned.__name__ = Pinned.__qualname__ = f"G1WalkSlopeFixedL{max_level}EstimatorEnvCfg"
    return Pinned


for _level in range(NUM_LEVELS):
    globals()[f"G1WalkSlopeFixedL{_level}EstimatorEnvCfg"] = _pinned(_level)
del _level
