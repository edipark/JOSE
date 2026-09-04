"""Closed-loop evaluation for SET, mirroring JOSE's protocol exactly.

``estimator/pipeline.py``'s ``evaluate_estimator_closed_loop`` is the function
every other row of the comparison is measured with, and it cannot be reused
here: it calls ``estimator.predict(...)`` on a stateless model, whereas SET
carries a ring of its own past predictions that has to be cleared whenever an
environment resets. That file is fingerprinted -- editing it would invalidate the
JOSE dataset caches and re-label already-logged results as a different variant --
so the loop is mirrored instead.

Mirrored means mirrored: same 200 completed episodes, same 1000-step episode cap,
same derived ``max_steps``, same partial-episode accounting, same metric names.
``test_set_baseline.py`` drives a JOSE LSTM checkpoint through this loop and
checks it reproduces that seed's logged numbers, which is what makes SET's
survival column comparable with the rest of the table rather than merely
similar-looking.

The two intended differences:

* ``predict_step`` replaces ``predict``, so the privileged tokens come from the
  model's own earlier outputs -- never the simulator.
* ``estimator.reset(done)`` runs alongside ``history.reset(done)``.

``predict_step`` returns the *complete* target vector (estimated dimensions plus
any read straight off the IMU), so ``target_rmse`` spans the same dimensions
JOSE reports and the per-dimension breakdown shows which ones SET measured
rather than inferred.
"""

from __future__ import annotations

import numpy as np
import torch

from jose.estimator.adapters import PolicyAdapter
from jose.distillation.command_eval import reset_ids
from jose.estimator.metrics import MetricAccumulator, step_metrics
from jose.estimator.pipeline import HistoryBuffer, frozen_rng
from jose.skrl_compat import force_skrl_isaaclab_reset


@torch.no_grad()
def evaluate_set_closed_loop(
    env,
    adapter: PolicyAdapter,
    teacher_agent,
    estimator,
    context: int,
    pass_through_fn=None,
    episodes: int = 200,
    max_episode_steps: int = 1000,
    seed: int | None = None,
) -> dict:
    """Drive the teacher through SET's estimate over completed episodes.

    ``pass_through_fn()`` returns the target dimensions SET measures rather than
    estimates, ordered to match the estimator's ``pass_through_indices``. It is
    required whenever that list is non-empty.
    """
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    force_skrl_isaaclab_reset(env)
    observations, _ = env.reset()
    num_envs = observations.shape[0]
    device = observations.device

    history = HistoryBuffer(num_envs, context, adapter.input_dim, device)
    estimator.eval()
    estimator.reset()
    teacher_agent.enable_training_mode(False, apply_to_models=True)

    returns = torch.zeros(num_envs, device=device)
    lengths = torch.zeros_like(returns)
    completed_returns: list[float] = []
    completed_lengths: list[float] = []
    deaths = timeouts = 0
    squared_error = torch.zeros(adapter.schema.estimator_target_dim, dtype=torch.float64)
    sample_count = 0
    metrics = MetricAccumulator()
    previous_action = torch.zeros((num_envs, adapter.schema.action_dim), device=device)

    max_steps = max_episode_steps * max(1, (episodes + num_envs - 1) // num_envs + 1)
    for _ in range(max_steps):
        if getattr(adapter, "imu_corruptor", None) is not None:
            adapter.invalidate_imu()
        frame = adapter.estimator_input()
        target = adapter.estimator_target()
        sequence = history.push(frame)
        estimate = estimator.predict_step(
            sequence, None if pass_through_fn is None else pass_through_fn()
        )
        squared_error += (estimate - target).double().square().sum(dim=0).cpu()
        sample_count += target.shape[0]

        action = adapter.action(teacher_agent, adapter.inject_estimate(observations, estimate))
        driving_observations = observations
        observations, rewards, terminated, truncated, _ = env.step(action)
        returns += rewards.flatten()
        lengths += 1
        with frozen_rng():
            teacher_action = adapter.action(teacher_agent, driving_observations)
            metrics.add(
                step_metrics(
                    adapter.core_env, adapter, teacher_agent,
                    action=action, previous_action=previous_action, rewards=rewards,
                    policy_action=action, teacher_action=teacher_action,
                )
            )
        previous_action = action
        done = (terminated | truncated).flatten()
        if done.any():
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            remaining = episodes - len(completed_lengths)
            selected = done_ids[:remaining]
            completed_returns.extend(returns[selected].cpu().tolist())
            completed_lengths.extend(lengths[selected].cpu().tolist())
            deaths += int(terminated[selected].sum())
            timeouts += int((truncated[selected] & ~terminated[selected]).sum())
            returns[done_ids] = 0.0
            lengths[done_ids] = 0.0
            history.reset(done)
            estimator.reset(done)
            if getattr(adapter, "imu_corruptor", None) is not None:
                adapter.imu_corruptor.reset(reset_ids(done))
            previous_action[done] = 0.0
            if len(completed_lengths) >= episodes:
                break

    if not completed_lengths:
        raise RuntimeError("Closed-loop evaluation completed no episodes")
    target_rmse = (squared_error / max(sample_count, 1)).sqrt().float()
    return {
        "episodes": len(completed_lengths),
        "episode_length_mean": sum(completed_lengths) / len(completed_lengths),
        "episode_length_std": float(np.std(completed_lengths)),
        "return_mean": sum(completed_returns) / len(completed_returns),
        "deaths": deaths,
        "timeouts": timeouts,
        "death_rate": 100.0 * deaths / len(completed_lengths),
        "timeout_rate": 100.0 * timeouts / len(completed_lengths),
        "success_rate": 100.0 * timeouts / len(completed_lengths),
        "rmse": float(target_rmse.square().mean().sqrt()),
        "target_rmse": target_rmse.tolist(),
        **metrics.mean(),
    }
