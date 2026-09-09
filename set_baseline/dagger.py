"""On-policy dataset aggregation for the SET architecture.

This is not a correction to the published baseline. ``train_set_baseline.py``
implements SET as specified -- one offline fit on expert rollouts -- and Table I
reports that. This module adds the arm the paper is missing: SET's *architecture*
trained under *JOSE's* protocol.

Why it is needed. JOSE and SET differ on two axes at once, what they read (joint
encoders alone against encoders plus an IMU) and how they are trained (dataset
aggregation against a single offline fit), so neither comparison in Table I can
say which axis produced the difference. Filling the two missing cells --
SET+DAgger here, JOSE+IMU via ``--imu-input`` on the estimator trainer -- makes
each axis identifiable.

What is held fixed. Everything except the state distribution. The model, the
target split, the packing convention, the loss, the optimizer, the epoch budget
and the dataset cap are the ones ``train_set_baseline.py`` already uses; the
round schedule, the estimator ratio and its ramp are the ones
``train_state_estimator.py`` already uses. Nothing here is tuned.

One methodological choice is worth stating, because it could reasonably have gone
the other way. The packed input keeps the *ground-truth* lagged privileged history
that offline SET trains on, rather than the estimator's own past predictions. SET
is teacher-forced during training and autoregressive at inference; that gap is a
property of the published method. Feeding its own predictions during the on-policy
rounds would close it -- and would change the input convention at the same time as
the state distribution, leaving a difference that could not be attributed to
either. So only the states move, which is what DAgger is.
"""

from __future__ import annotations

import torch

from jose.distillation.command_eval import reset_ids
from jose.estimator.adapters import PolicyAdapter
from jose.estimator.pipeline import HistoryBuffer, RolloutDataset
from jose.skrl_compat import force_skrl_isaaclab_reset

from .collect import pack


@torch.no_grad()
def collect_dagger_rollout(
    env,
    adapter: PolicyAdapter,
    teacher_agent,
    estimator,
    steps: int,
    context: int,
    estimated_indices: tuple[int, ...],
    estimator_ratio: float,
    pass_through_fn=None,
    max_samples: int | None = None,
) -> tuple[RolloutDataset, dict]:
    """Record ``steps`` of a rollout the estimator partly drives.

    A per-environment Bernoulli draw at ``estimator_ratio`` decides whose action
    is stepped -- the teacher reading the estimate, or the teacher reading ground
    truth. Identical to how ``estimator/pipeline.py:collect_rollout`` mixes for
    JOSE, so the two methods aggregate over comparably shifted distributions.

    Labels are always ground truth, taken at the state actually reached.
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

    estimator.eval()
    estimator.reset()
    teacher_agent.enable_training_mode(False, apply_to_models=True)
    per_step = num_envs if max_samples is None else max(1, min(num_envs, max_samples // max(steps, 1)))
    corruptor = getattr(adapter, "imu_corruptor", None)
    estimator_steps = 0

    for _ in range(steps):
        if corruptor is not None:
            adapter.invalidate_imu()
        frame = adapter.estimator_input()
        target = adapter.estimator_target()
        sequence = observation_history.push(frame)
        privileged = privileged_history.push(previous_target)
        packed = pack(sequence, privileged)

        teacher_action = adapter.action(teacher_agent, observations)
        action = teacher_action
        if estimator_ratio > 0.0:
            # ``predict_step`` advances the estimator's own privileged ring, so it
            # is called once per step whatever the mask decides. Calling it only
            # for the selected environments would leave the ring of the others
            # stale and make the next round's rollout depend on this round's draw.
            # `sequence`, not `packed`. predict_step takes the observation history
            # alone and concatenates the privileged half from its *own* ring of
            # past outputs (set_baseline/model.py:218) -- that is what makes the
            # deployed estimator autoregressive. Handing it the packed tensor
            # would append a second privileged block and change the input width.
            estimate = estimator.predict_step(
                sequence, None if pass_through_fn is None else pass_through_fn()
            )
            estimated_obs = adapter.inject_estimate(observations, estimate)
            estimator_action = adapter.action(teacher_agent, estimated_obs)
            use_estimator = torch.rand(num_envs, device=device) < estimator_ratio
            action = torch.where(use_estimator[:, None], estimator_action, teacher_action)
            estimator_steps += int(use_estimator.sum())

        if per_step < num_envs:
            sample_ids = torch.randperm(num_envs, device=device)[:per_step]
        else:
            sample_ids = slice(None)
        packed_rows.append(packed[sample_ids].cpu().clone())
        target_rows.append(target.index_select(1, estimated)[sample_ids].cpu().clone())
        frame_rows.append(frame[sample_ids].cpu().clone())
        action_rows.append(teacher_action[sample_ids].cpu().clone())

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
            estimator.reset(done)
            if corruptor is not None:
                corruptor.reset(reset_ids(done))
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
        "collection_protocol": "on_policy_dagger_rollout",
        "estimator_ratio": estimator_ratio,
        "estimator_driven_steps": estimator_steps,
    }
    return dataset, stats


def estimator_ratio_for(round_index: int, rounds: int, initial: float, final: float, schedule: str) -> float:
    """The ratio for ``round_index`` (1-based), matching JOSE's schedule.

    Reproduces ``train_state_estimator.py``'s ramp so the two methods aggregate
    under the same mixing, and so a SET+DAgger round is comparable to the JOSE
    round of the same index.
    """
    if schedule == "constant" or rounds <= 1:
        return initial
    progress = (round_index - 1) / (rounds - 1)
    return initial + (final - initial) * progress
