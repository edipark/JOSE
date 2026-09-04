"""Prove SET's evaluator measures the same thing JOSE's does.

``set_baseline/evaluate.py`` is a copy of ``evaluate_estimator_closed_loop``.
A copy exists because the original is fingerprinted and cannot be taught to call
``SETEstimator.reset`` on its autoregressive ring. Copies drift, and a drifted
evaluator would make SET's row in the comparison table meaningless while looking
entirely plausible.

So: take a JOSE estimator whose closed-loop numbers are already logged, run it
through SET's loop, and require the numbers back. The two evaluators differ in
exactly two calls -- ``predict_step(sequence, pass_through)`` versus
``predict(sequence)``, and ``reset(done)`` versus nothing -- so a shim of a dozen
lines is enough to make a JOSE estimator satisfy SET's interface. Everything
else on the path (history buffer, injection, episode accounting, metric
accumulation, seeding) is exercised as-is.

A pass means SET's reported episode length, death rate and return are produced
by the same protocol as every other row. A failure localises the drift.

Usage:

    python -m JOSE.verify_set_protocol \
        --estimator <study>/methods/jose/window_25/joints_all/seed_42/best_estimator.pt \
        --expect    <study>/methods/jose/window_25/joints_all/seed_42/training.json \
        --teacher-checkpoint <walk teacher .pt> --seed 42 --headless
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Validate the SET evaluator against JOSE's")
parser.add_argument("--estimator", required=True, help="A JOSE best_estimator.pt")
parser.add_argument(
    "--expect", default=None,
    help="training.json from the same run. Omit to just print the measurement.",
)
parser.add_argument("--teacher-checkpoint", required=True)
parser.add_argument("--task", default="Isaac-G1-PPO-Walk-Estimator-JOSE-v0")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--adapter", choices=("amp", "ppo_walk"), default="ppo_walk")
parser.add_argument(
    "--num-envs", type=int, default=256,
    help="Must match the run being reproduced: the episode accounting depends on it.",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--eval-seed-offset", type=int, default=10000,
    help="train_state_estimator.py evaluates at seed + this. The default is the "
    "value recorded in the method_comparison runs.",
)
parser.add_argument("--episodes", type=int, default=200)
parser.add_argument("--max-episode-steps", type=int, default=1000)
parser.add_argument(
    "--tolerance", type=float, default=0.02,
    help="Relative tolerance on episode length and return. Not zero: the two loops "
    "issue their RNG draws in the same order but through different call sites, and "
    "the closed loop is chaotic. Drift that matters is orders of magnitude larger.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

from jose.teacher_setup import resolve_agent_entry_point  # noqa: E402

args_cli.agent = resolve_agent_entry_point(args_cli.adapter, args_cli.agent)

sys.argv = [sys.argv[0], *hydra_args]
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402

from jose.estimator.adapters import make_policy_adapter  # noqa: E402
from jose.estimator.pipeline import load_estimator  # noqa: E402
from jose.set_baseline.evaluate import evaluate_set_closed_loop  # noqa: E402
from jose.teacher_setup import build_env_and_teacher  # noqa: E402


#: Keys compared against the logged run. Deliberately the closed-loop ones only:
#: `rmse`/`r2` in training.json come from the offline validation split, not from
#: this rollout, so they are not this script's to reproduce.
COMPARED = (
    "episode_length_mean",
    "episode_length_std",
    "death_rate",
    "timeout_rate",
    "return_mean",
    "episodes",
)


class JoseAsSET:
    """A JOSE sequence estimator behind SET's inference interface.

    JOSE's estimator is stateless across steps -- the LSTM is run over the whole
    window on every call -- so ``reset`` has nothing to clear, and it predicts
    every target dimension, so there is no pass-through half.
    """

    def __init__(self, estimator):
        self._estimator = estimator

    def eval(self):
        self._estimator.eval()
        return self

    def reset(self, env_ids=None) -> None:
        return None

    def predict_step(self, sequence: torch.Tensor, pass_through=None) -> torch.Tensor:
        if pass_through is not None:
            raise ValueError("A JOSE estimator predicts every target dimension")
        return self._estimator.predict(sequence)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    estimator_path = Path(args_cli.estimator).resolve()
    if not estimator_path.is_file():
        raise FileNotFoundError(estimator_path)

    torch.manual_seed(args_cli.seed)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed

    env, teacher_agent = build_env_and_teacher(
        args_cli.task, args_cli.adapter, env_cfg, agent_cfg,
        args_cli.teacher_checkpoint, args_cli.device, seed=args_cli.seed,
    )
    print(f"[verify] env and teacher ready ({args_cli.num_envs} envs)", flush=True)
    teacher_agent.enable_training_mode(False, apply_to_models=True)

    estimator, payload = load_estimator(estimator_path, args_cli.device)
    config = payload["model_config"]
    if config["type"].upper() == "MLP":
        raise ValueError(
            "An MLP estimator consumes the newest frame, not the window; SET's loop "
            "always feeds a sequence, so this check needs a sequence model."
        )
    window = int(payload.get("window") or config.get("window"))
    adapter = make_policy_adapter(args_cli.adapter, env, payload["joint_preset"])
    print(
        f"[verify] restored {config['type']} window={window} preset={payload['joint_preset']}",
        flush=True,
    )

    print("[verify] running SET's closed-loop evaluator", flush=True)
    try:
        measured = evaluate_set_closed_loop(
            env, adapter, teacher_agent, JoseAsSET(estimator),
            context=window,
            pass_through_fn=None,
            episodes=args_cli.episodes,
            max_episode_steps=args_cli.max_episode_steps,
            seed=args_cli.seed + args_cli.eval_seed_offset,
        )
    finally:
        env.close()

    if args_cli.expect is None:
        print(json.dumps({key: measured[key] for key in COMPARED}, indent=2))
        return

    expected = json.loads(Path(args_cli.expect).read_text(encoding="utf-8")).get("metrics", {})
    width = max(len(key) for key in COMPARED)
    failures = []
    print(f"\n{'metric'.ljust(width)}  {'logged':>12}  {'measured':>12}  {'rel':>9}")
    for key in COMPARED:
        if key not in expected:
            print(f"{key.ljust(width)}  {'(absent)':>12}")
            continue
        want, got = float(expected[key]), float(measured[key])
        scale = max(abs(want), 1e-9)
        relative = abs(got - want) / scale
        ok = relative <= args_cli.tolerance
        failures.append(key) if not ok else None
        print(f"{key.ljust(width)}  {want:12.4f}  {got:12.4f}  {relative:8.2%}  {'ok' if ok else 'MISMATCH'}")

    if failures:
        raise SystemExit(
            f"\nSET's evaluator does not reproduce JOSE's on {failures}. Its rows are not "
            "comparable with the rest of the table until this is explained."
        )
    print("\n[verify] PASS -- SET's evaluator reproduces JOSE's protocol")


if __name__ == "__main__":
    import traceback

    try:
        main()
    except SystemExit as exit_code:
        print(exit_code)
        simulation_app.close()
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
    simulation_app.close()
