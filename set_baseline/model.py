"""State Estimation Transformer (Yu et al., IROS 2024), ported to JOSE's targets.

The paper casts state estimation as conditional sequence modelling: a trajectory
is written as ``tau = (o_1, o'_1, o_2, o'_2, ...)``, embeddings of the
non-privileged ``o`` and privileged ``o'`` are stacked as separate tokens with a
per-timestep positional embedding, and a causally masked GPT predicts the
privileged entry from the history. Deployment supplies only ``o``; the ``o'``
slots are filled autoregressively by the model's own earlier outputs, which is
what :meth:`SETEstimator.predict_step` and its ring buffer implement.

One ambiguity is worth recording, because it is a judgement call and not a fact
from the paper. The trajectory definition pairs ``o`` and ``o'`` tokens, but the
training equation is written ``o'_t = SET(o^H_t)``, conditioning on the
non-privileged history alone. We implement the pair-token reading -- prediction
of ``o'_t`` conditioned on ``o_{t-H+1..t}`` and ``o'_{t-H+1..t-1}`` -- because the
deployment section describes generation as autoregressive, which only means
something if past ``o'`` are inputs. The resulting ``2H`` token count is our
reading; the paper does not state a sequence length.

What the paper fixes: **6 blocks**, **H = 20**, **MSE**. Everything else --
model width, head count, feed-forward ratio, dropout, optimizer, learning rate,
batch size -- is unspecified there and chosen here. The width is picked so the
parameter count lands next to JOSE's LSTM (1,079,397), so the comparison is not
confounded by capacity.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from jose.estimator.models import NormalizedEstimator


SET_BLOCKS = 6
SET_CONTEXT = 20


class _CausalBlock(nn.Module):
    """Pre-norm transformer block with a causal self-attention mask."""

    def __init__(self, width: int, heads: int, feedforward_ratio: int, dropout: float):
        super().__init__()
        self.norm_attention = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm_feedforward = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, feedforward_ratio * width),
            nn.GELU(),
            nn.Linear(feedforward_ratio * width, width),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        normed = self.norm_attention(tokens)
        attended, _ = self.attention(normed, normed, normed, attn_mask=mask, need_weights=False)
        tokens = tokens + attended
        return tokens + self.feedforward(self.norm_feedforward(tokens))


class SETEstimator(NormalizedEstimator):
    """GPT-style estimator over interleaved ``(o', o)`` tokens.

    The packed input is ``(batch, context, observation_dim + target_dim)`` where
    slot ``t`` holds ``concat(o_t, o'_{t-1})`` -- the privileged entry is lagged
    by one so that nothing at slot ``t`` reveals the answer at slot ``t``. Slot 0
    carries a zero privileged part, the start-of-sequence case.

    Subclassing :class:`NormalizedEstimator` is what lets this drop into the
    existing training path: ``set_normalization`` / ``normalized_targets`` /
    ``predict`` all work unchanged, so ``estimator/pipeline.py``'s
    ``train_estimator`` fits it with the same optimizer, epoch budget and
    best-validation selection JOSE gets.
    """

    def __init__(
        self,
        observation_dim: int,
        target_dim: int,
        output_dim: int | None = None,
        context: int = SET_CONTEXT,
        width: int = 128,
        heads: int = 4,
        blocks: int = SET_BLOCKS,
        feedforward_ratio: int = 4,
        dropout: float = 0.1,
        estimated_indices: tuple[int, ...] | None = None,
        pass_through_indices: tuple[int, ...] = (),
    ):
        estimated_indices = (
            tuple(range(target_dim)) if estimated_indices is None else tuple(estimated_indices)
        )
        pass_through_indices = tuple(pass_through_indices)
        if set(estimated_indices) & set(pass_through_indices):
            raise ValueError("A target dimension cannot be both estimated and passed through")
        if sorted(estimated_indices + pass_through_indices) != list(range(target_dim)):
            raise ValueError("Estimated and passed-through indices must partition the target")
        output_dim = len(estimated_indices) if output_dim is None else output_dim
        super().__init__(observation_dim + target_dim, output_dim)
        if context <= 0 or width <= 0 or blocks <= 0:
            raise ValueError("SET context, width and block count must be positive")
        if width % heads:
            raise ValueError(f"width {width} must be divisible by heads {heads}")

        self.observation_dim = observation_dim
        self.target_dim = target_dim
        self.context = context
        self.width = width

        self.observation_embedding = nn.Linear(observation_dim, width)
        self.privileged_embedding = nn.Linear(target_dim, width)
        self.position_embedding = nn.Embedding(context, width)
        self.embedding_norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            _CausalBlock(width, heads, feedforward_ratio, dropout) for _ in range(blocks)
        )
        self.head_norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, output_dim)

        # (2 * context) causal mask, built once. True == "may not attend".
        tokens = 2 * context
        mask = torch.triu(torch.ones(tokens, tokens, dtype=torch.bool), diagonal=1)
        self.register_buffer("attention_mask", mask, persistent=False)
        # Autoregressive feedback at deploy time. The ring holds the *full*
        # privileged vector -- estimated dimensions plus the ones read straight
        # off the IMU -- because that is what the next step conditions on and
        # what the policy is handed.
        self.register_buffer(
            "estimated_indices", torch.tensor(estimated_indices, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "pass_through_indices", torch.tensor(pass_through_indices, dtype=torch.long), persistent=False
        )
        self.register_buffer("privileged_ring", torch.zeros(0, context, target_dim), persistent=False)

    def _tokens(self, packed: torch.Tensor) -> torch.Tensor:
        """Interleave each slot into ``(o'_{t-1}, o_t)``, oldest slot first."""
        if packed.ndim != 3 or packed.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected packed input (batch, {self.context}, {self.input_dim}), got {tuple(packed.shape)}"
            )
        if packed.shape[1] != self.context:
            raise ValueError(f"Expected context {self.context}, got {packed.shape[1]}")
        batch = packed.shape[0]
        observations = packed[..., : self.observation_dim]
        privileged = packed[..., self.observation_dim :]

        positions = torch.arange(self.context, device=packed.device)
        position = self.position_embedding(positions)[None]
        # Privileged first inside a slot: o'_{t-1} precedes o_t in real time.
        stacked = torch.stack(
            (self.privileged_embedding(privileged) + position, self.observation_embedding(observations) + position),
            dim=2,
        )
        return self.dropout(self.embedding_norm(stacked.reshape(batch, 2 * self.context, self.width)))

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        tokens = self._tokens(self.normalize(packed))
        mask = self.attention_mask
        for block in self.blocks:
            tokens = block(tokens, mask)
        # The final token is o_H: the newest observation, which is the only
        # position that has seen the whole history and no future.
        return self.head(self.head_norm(tokens[:, -1]))

    # -- autoregressive inference ------------------------------------------

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Clear remembered predictions for environments that just reset."""
        if self.privileged_ring.numel() == 0:
            return
        if env_ids is None:
            self.privileged_ring.zero_()
            return
        ids = env_ids.nonzero(as_tuple=True)[0] if env_ids.dtype == torch.bool else env_ids
        if ids.numel():
            self.privileged_ring[ids] = 0.0

    @torch.no_grad()
    def predict_step(
        self, observation_history: torch.Tensor, pass_through: torch.Tensor | None = None
    ) -> torch.Tensor:
        """One closed-loop step: predict, reassemble, remember, return.

        ``observation_history`` is ``(batch, context, observation_dim)``, newest
        last, as ``HistoryBuffer`` produces it. ``pass_through`` carries the
        target dimensions SET measures rather than estimates, ordered to match
        ``pass_through_indices``; it is required whenever that list is non-empty.

        The privileged half of the packed input comes from this object's own past
        outputs -- never from the simulator -- which is what makes the deployed
        estimator honest. The return value is the complete target vector, so it
        can be handed straight to ``adapter.inject_estimate``.
        """
        batch = observation_history.shape[0]
        if self.privileged_ring.shape[0] != batch:
            self.privileged_ring = torch.zeros(
                batch, self.context, self.target_dim,
                device=observation_history.device, dtype=observation_history.dtype,
            )
        packed = torch.cat((observation_history, self.privileged_ring), dim=-1)
        estimate = self.predict(packed)

        full = torch.zeros(batch, self.target_dim, device=estimate.device, dtype=estimate.dtype)
        full[:, self.estimated_indices] = estimate
        if self.pass_through_indices.numel():
            if pass_through is None:
                raise ValueError(
                    f"This SET estimator reads {self.pass_through_indices.numel()} target "
                    "dimensions from its input; pass them to predict_step"
                )
            full[:, self.pass_through_indices] = pass_through.to(full.dtype)

        self.privileged_ring = torch.roll(self.privileged_ring, -1, dims=1)
        self.privileged_ring[:, -1] = full
        return full

    def config(self) -> dict:
        return {
            "type": "SET",
            "observation_dim": self.observation_dim,
            "target_dim": self.target_dim,
            "output_dim": self.output_dim,
            "estimated_indices": self.estimated_indices.tolist(),
            "pass_through_indices": self.pass_through_indices.tolist(),
            "context": self.context,
            "width": self.width,
            "blocks": len(self.blocks),
            "heads": self.blocks[0].attention.num_heads,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "paper_specified": {"blocks": SET_BLOCKS, "context": SET_CONTEXT, "loss": "mse"},
        }
