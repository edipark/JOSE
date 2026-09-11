"""Measure the slope teacher level by level, and pick the student pipeline's levels.

The student pipeline on slopes runs with every environment pinned to one level
(``ppo_walk/slope_fixed_env_cfg.py``). Which levels to include is decided here,
before any student is trained, by one rule fixed in advance:

    K = the largest level such that the teacher's death rate is at most
        --max-death-rate (default 0.5 %) on every level 0..K.

Death rate is measured exactly the way the teacher row of every study measures
it (``evaluate_teacher.py``): ``collect_rollout`` driving the teacher from the
true privileged state, the environment's own command resampling, the same
episode length, low-velocity termination disabled as it is there -- 512
episodes per level at the defaults. Cumulative on purpose -- a steep level that
happens to pass after a gentler one failed is a lucky draw, not a capability.

Why 0.5 % and not 0: the first measurement of the slope teacher (2026-09-11)
found 1 death in 512 episodes already on level 0, the gentlest band, and 0-2 on
every level up to 6 -- a floor that does not grow with the slope -- before 4, 10
and 25 on levels 7, 8 and 9. A zero threshold would reject level 0 itself and
select nothing. 0.5 % (at most 2 of 512) sits above that floor and below the
first level where the rate rises.

Every level is measured on the same terrain the students will see: the sloped
task's generator with its pinned seed, rows sorted by difficulty. All
environments are moved to one row per pass, so a pass measures one slope band
across all twenty columns (flat, pyramid and inverted-pyramid tiles).

Writes ``<output>`` (JSON) and prints ``SELECTED_MAX_LEVEL=<K>`` for the driver.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--teacher-checkpoint", required=True)
parser.add_argument("--task", default="Isaac-G1-PPO-Walk-SlopeFixed-L9-Estimator-JOSE-v0")
parser.add_argument("--agent", default=None)
parser.add_argument("--num-envs", type=int, default=256, help="Same as the study's teacher row.")
parser.add_argument("--collect-steps", type=int, default=2000, help="Same as the study's teacher row.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--max-death-rate", type=float, default=0.5,
    help="Largest teacher death rate (%%) a level may have and still be kept.",
)
parser.add_argument("--output", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

from jose.teacher_setup import resolve_agent_entry_point  # noqa: E402

args_cli.agent = resolve_agent_entry_point("ppo_walk", args_cli.agent)
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402

import jose.ppo_walk  # noqa: F401, E402  -- registers the SlopeFixed ids
from jose.estimator.adapters import make_policy_adapter  # noqa: E402
from jose.estimator.pipeline import collect_rollout  # noqa: E402
from jose.ppo_walk.fixed_levels import place_on_levels, select_max_level  # noqa: E402
from jose.ppo_walk.slope_fixed_tasks import task_id  # noqa: E402
from jose.skrl_compat import disable_velocity_termination_for_evaluation  # noqa: E402
from jose.teacher_setup import build_env_and_teacher  # noqa: E402


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed
    disable_velocity_termination_for_evaluation(env_cfg)
    env, teacher_agent = build_env_and_teacher(
        args_cli.task, "ppo_walk", env_cfg, agent_cfg, args_cli.teacher_checkpoint, args_cli.device,
        seed=args_cli.seed,
    )
    adapter = make_policy_adapter("ppo_walk", env, "all")
    terrain = adapter.core_env.scene.terrain
    num_rows = terrain.terrain_origins.shape[0]

    levels = []
    for level in range(num_rows):
        torch.manual_seed(args_cli.seed)
        np.random.seed(args_cli.seed)
        place_on_levels(terrain, torch.full_like(terrain.terrain_levels, level))
        _, stats = collect_rollout(env, adapter, teacher_agent, args_cli.collect_steps, window=1)
        row = {
            "level": level,
            "deaths": stats["deaths"],
            "timeouts": stats["timeouts"],
            "death_rate": stats["death_rate"],
            "episode_length_mean": stats["episode_length_mean"],
        }
        levels.append(row)
        print(
            f"[levels] level {level}: {row['deaths']} deaths / {row['deaths'] + row['timeouts']} episodes"
            f"  death_rate {row['death_rate']:.2f}%  eplen {row['episode_length_mean']:.1f}",
            flush=True,
        )

    chosen = select_max_level(levels, args_cli.max_death_rate)
    report = {
        "created_at": datetime.now().isoformat(),
        "teacher_checkpoint": str(Path(args_cli.teacher_checkpoint).resolve()),
        "task": args_cli.task,
        "seed": args_cli.seed,
        "num_envs": args_cli.num_envs,
        "collect_steps": args_cli.collect_steps,
        "rule": f"largest K with teacher death_rate <= {args_cli.max_death_rate}% on every level 0..K",
        "max_death_rate": args_cli.max_death_rate,
        "levels": levels,
        "selected_max_level": chosen,
        "selected_task": task_id(chosen) if chosen >= 0 else None,
    }
    output = Path(args_cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"SELECTED_MAX_LEVEL={chosen}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
