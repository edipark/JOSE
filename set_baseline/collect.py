"""Offline dataset collection for SET, as the paper specifies it.

SET trains on trajectories recorded from an already-trained expert: the estimator
never influences what states are visited. That is the whole methodological
difference from JOSE's DAgger, and it is preserved deliberately -- a baseline is
implemented as published. What the on-policy protocol is worth is already
measured by JOSE's own DAgger ablation, and does not require modifying SET.

The sample budget matches JOSE's ``--max-dataset-size``, so only the
*distribution* differs, not the amount of data.

``estimator/pipeline.py``'s ``collect_rollout`` cannot be reused as-is: its
``RolloutDataset`` carries the target only for the current step, while SET's
``o'`` tokens need a *history* of past privileged vectors. The packing convention
below is the load-bearing detail.

Packing
-------
A row is ``concat(observation_history, privileged_history)`` where

* ``observation_history[:, i]`` is ``o_{t-H+1+i}``  -- newest at ``-1``
* ``privileged_history[:, i]``  is ``o'_{t-H+i}``   -- **lagged by one**

so slot ``i`` pairs ``o_k`` with ``o'_{k-1}`` and nothing at slot ``i`` reveals
the answer at slot ``i``. This must match what ``SETEstimator.predict_step``
assembles at inference, where the privileged half comes from the model's own ring
buffer -- written *after* each prediction, so its newest entry is likewise
``o'_{t-1}``. ``test_set_baseline.py`` asserts the two agree; if they diverge the
model trains on one problem and is evaluated on another, and the baseline looks
bad for the wrong reason.
"""

from __future__ import annotations

import torch

from jose.estimator.adapters import PolicyAdapter
from jose.estimator.pipeline import HistoryBuffer, RolloutDataset
from jose.skrl_compat import force_skrl_isaaclab_reset


def pack(observation_history: torch.Tensor, privileged_history: torch.Tensor) -> torch.Tensor:
    """The single definition of SET's packed input, shared by training and inference."""
    return torch.cat((observation_history, privileged_history), dim=-1)


@torch.no_grad()
def collect_expert_rollout(
    env,
    adapter: PolicyAdapter,
    teacher_agent,
    steps: int,
    context: int,
    estimated_indices: tuple[int, ...],
    max_samples: int | None = None,
) -> tuple[RolloutDataset, dict]:
    """Record ``steps`` of the frozen teacher, packed for SET.

    Returns a ``RolloutDataset`` whose ``histories`` are the packed inputs and
    whose ``targets`` are the *estimated* dimensions only -- the ones SET is
    actually asked to predict. That is exactly the shape ``train_estimator``
    expects, so SET is fitted by the same optimizer, epoch budget and
    best-validation selection as JOSE's LSTM.
    """
    force_skrl_isaaclab_reset(env)
    observations, _ = env.reset()
    num_envs = observations.shape[0]
    device = observations.device
    target_dim = adapter.schema.estimator_target_dim
    estimated = torch.as_tensor(estimated_indices, device=device, dtype=torch.long)

    observation_history = HistoryBuffer(num_envs, context, adapter.input_dim, device)
    privileged_history = HistoryBuffer(num_envs, context, target_dim, device)
    previous_target = torch.zeros(num_envs, target_dim, device=device)

    packed_rows, target_rows, frame_rows, action_rows = [], [], [], []
    deaths = timeouts = 0
    lengths = torch.zeros(num_envs, device=device)
    completed_lengths: list[float] = []

    teacher_agent.enable_training_mode(False, apply_to_models=True)
    per_step = num_envs if max_samples is None else max(1, min(num_envs, max_samples // max(steps, 1)))

    corruptor = getattr(adapter, "imu_corruptor", None)

    for _ in range(steps):
        # One sensor read per step. Harmless when the adapter is not corrupting.
        if corruptor is not None:
            adapter.invalidate_imu()
        frame = adapter.estimator_input()
        target = adapter.estimator_target()
        # Push the *previous* target: slot -1 of the privileged history holds
        # o'_{t-1}, never o'_t.
        sequence = observation_history.push(frame)
        privileged = privileged_history.push(previous_target)
        packed = pack(sequence, privileged)

        action = adapter.action(teacher_agent, observations)

        if per_step < num_envs:
            sample_ids = torch.randperm(num_envs, device=device)[:per_step]
        else:
            sample_ids = slice(None)
        packed_rows.append(packed[sample_ids].cpu().clone())
        target_rows.append(target.index_select(1, estimated)[sample_ids].cpu().clone())
        frame_rows.append(frame[sample_ids].cpu().clone())
        action_rows.append(action[sample_ids].cpu().clone())

        previous_target = target
        observations, _, terminated, truncated, _ = env.step(action)
        lengths += 1
        done = (terminated | truncated).flatten()
        deaths += int(terminated.sum())
        timeouts += int((truncated & ~terminated).sum())
        if done.any():
            completed_lengths.extend(lengths[done].cpu().tolist())
            lengths[done] = 0.0
            observation_history.reset(done)
            privileged_history.reset(done)
            if corruptor is not None:
                corruptor.reset(done)
            previous_target = previous_target.clone()
            previous_target[done] = 0.0

    dataset = RolloutDataset(
        torch.cat(packed_rows), torch.cat(target_rows), torch.cat(frame_rows), torch.cat(action_rows)
    )
    completed = deaths + timeouts
    stats = {
        "samples": len(dataset.targets),
        "deaths": deaths,
        "timeouts": timeouts,
        "death_rate": 100.0 * deaths / completed if completed else 0.0,
        "episode_length_mean": (
            sum(completed_lengths) / len(completed_lengths) if completed_lengths else 0.0
        ),
        "collection_protocol": "offline_expert_rollout",
        "estimator_ratio": 0.0,
    }
    return dataset, stats
