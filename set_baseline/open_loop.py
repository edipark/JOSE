"""Open-loop error for SET, evaluated in chunks.

The evaluation batch is sized at run time from what is actually free, and
never falls back to the CPU: a silent device switch would put a CPU timing into
``inference_ms_per_sample`` beside GPU ones with nothing marking it.

``estimator/pipeline.py``'s ``evaluate_predictions`` pushes the whole validation
set through the model in a single forward pass. That is fine for the LSTM, whose
activations are one hidden state per sample, and fatal for SET, which expands
every sample into ``2H = 40`` tokens of width 128: at the 250k-sample budget the
embedding alone asks for 2.38 GiB on top of the ~8 GiB the simulator is already
holding, and the run dies *after* training, at evaluation, with nothing saved.

The smoke test did not catch it because it runs at a 20k cap, an order of
magnitude below where the allocation bites.

This is a chunked copy rather than a fix to the original because
``estimator/pipeline.py`` is fingerprinted: editing it changes the identity of
every JOSE variant and detaches the ablation catalog from results already
logged. Chunking is arithmetic-neutral -- every metric below is computed from
the same predictions, just accumulated -- with one exception noted on
``inference_ms_per_sample``.
"""

from __future__ import annotations

import time

import torch

from jose.estimator.models import NormalizedEstimator
from jose.estimator.pipeline import RolloutDataset


#: Never go below this, however little is free -- a smaller batch buys nothing
#: and makes the pass crawl.
MIN_CHUNK = 128
#: Never go above this. Past a few tens of thousands the pass is bounded by
#: memory bandwidth rather than launch overhead, so a larger batch is no faster
#: and only widens the blast radius if the estimate is off.
MAX_CHUNK = 32768
#: Fraction of free memory the batch is allowed to claim. The rest absorbs
#: allocator fragmentation and whatever the simulator does while this runs.
SAFETY = 0.5


def peak_bytes_per_sample(estimator) -> int:
    """Concurrent activation bytes one sample needs inside a block.

    Read off the model rather than assumed, so a re-tuned width, head count or
    feed-forward ratio re-sizes the batch instead of silently invalidating it.
    """
    tokens = 2 * estimator.context
    width = estimator.width
    block = estimator.blocks[0]
    ratio = block.feedforward[0].out_features // width
    heads = block.attention.num_heads
    token_map = tokens * width * 4
    return (
        2 * token_map          # the two slot embeddings, live together in _tokens
        + ratio * token_map    # the feed-forward intermediate -- the dominant term
        + heads * tokens * tokens * 4  # attention matrices
    )


def auto_chunk(estimator, device: str) -> int:
    """The largest batch that fits in what is free *right now*, with margin.

    Sizing this at run time rather than hardcoding it is what makes the same code
    correct on a 12 GB card with a simulator resident (a few hundred MiB free,
    so about a thousand samples) and on a 24 GB one (tens of thousands). The
    first attempt at this evaluation asked for the whole 250k-sample dataset in
    one pass and died with 426 MiB free.
    """
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return MAX_CHUNK
    # mem_get_info reports what the driver sees free, which does not include
    # blocks PyTorch is holding in its cache. Releasing them first is the
    # difference between sizing against real headroom and against whatever the
    # allocator happened to keep.
    torch.cuda.empty_cache()
    free, _total = torch.cuda.mem_get_info(torch.device(device).index or 0)
    per_sample = peak_bytes_per_sample(estimator)
    fits = int(free * SAFETY / per_sample)
    chunk = max(MIN_CHUNK, min(MAX_CHUNK, fits))
    print(
        f"[set] open-loop batch {chunk} ({free / 2**20:.0f} MiB free, "
        f"{per_sample / 1024:.0f} KB/sample)",
        flush=True,
    )
    return chunk


@torch.no_grad()
def evaluate_predictions_chunked(
    estimator: NormalizedEstimator,
    dataset: RolloutDataset,
    device: str,
    target_names: tuple[str, ...] | list[str] | None = None,
    chunk: int | None = None,
) -> dict:
    """``evaluate_predictions``'s metric dict, computed without a 250k-wide pass."""
    inputs = dataset.histories
    estimator.eval().to(device)
    if chunk is None:
        chunk = auto_chunk(estimator, device)

    def run(target_device: str, size: int) -> tuple[torch.Tensor, float]:
        estimator.to(target_device)
        outputs = []
        started = time.perf_counter()
        for begin in range(0, len(inputs), size):
            outputs.append(estimator.predict(inputs[begin : begin + size].to(target_device)).cpu())
        return torch.cat(outputs), time.perf_counter() - started

    # The batch above is sized from an activation estimate, and estimates are
    # wrong sometimes. Being wrong here is expensive: this evaluation runs
    # *after* training and before anything is saved, so an unhandled OOM throws
    # the model away. Halving costs seconds.
    #
    # There is deliberately no CPU fallback. A silent device switch would make
    # inference_ms_per_sample a CPU number sitting in a column of GPU ones, with
    # nothing in the output to say so. At MIN_CHUNK the pass needs about 18 MB,
    # so exhausting the loop means something other than this batch is wrong, and
    # that should be read rather than worked around.
    prediction = elapsed = None
    while chunk >= MIN_CHUNK:
        try:
            prediction, elapsed = run(device, chunk)
            break
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            chunk //= 2
            print(f"[set] open-loop batch too large; retrying at {chunk}", flush=True)
    if prediction is None:
        free = torch.cuda.mem_get_info(torch.device(device).index or 0)[0] if torch.cuda.is_available() else 0
        raise RuntimeError(
            f"Open-loop evaluation ran out of memory even at batch {MIN_CHUNK} "
            f"({free / 2**20:.0f} MiB free, {peak_bytes_per_sample(estimator) / 1024:.0f} KB/sample). "
            "Something other than the evaluation batch is holding the GPU -- check for a "
            "second run sharing the card, and lower --num-envs if there is not."
        )

    error = prediction - dataset.targets
    mse = error.square().mean(dim=0)
    variance = dataset.targets.var(dim=0).clamp_min(1.0e-8)
    metrics = {
        "mae": float(error.abs().mean()),
        "rmse": float(error.square().mean().sqrt()),
        "r2": float((1.0 - mse / variance).mean()),
        "target_mae": error.abs().mean(dim=0).tolist(),
        "target_rmse": mse.sqrt().tolist(),
        # Carries the per-chunk launch overhead the single-pass version does not,
        # so it reads slightly high. It was never a deployment latency -- the
        # closed-loop path reports the per-step cost that is -- and the closed
        # loop is where SET's inference number for the paper comes from.
        "inference_ms_per_sample": elapsed * 1000.0 / len(inputs),
        "parameters": sum(parameter.numel() for parameter in estimator.parameters()),
        "trace_target": dataset.targets[:200].tolist(),
        "trace_prediction": prediction[:200].tolist(),
    }
    if target_names is not None:
        metrics["target_names"] = list(target_names)
    return metrics
