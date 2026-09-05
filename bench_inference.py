"""Per-sample forward-pass cost for every estimator in the paper, at one batch.

The number in ``training.json`` is not comparable across estimators. It times a
single ``predict`` call over the whole 250k-sample evaluation set, with no
warm-up and no CUDA synchronisation, and it includes the device-to-host copy.
Worse for the table, SET cannot run that call at all: a 250k single pass does not
fit in memory on any card we have, so its cost would have to be measured at a
different batch from everyone else's and put in the same column.

This measures all four the same way instead: the same batch, the same warm-up,
an explicit synchronise around each timed repeat, and the median over repeats.
The batch is the environment count the closed loop actually steps, so the number
is a relative cost per sample under the conditions the policy runs in -- not a
single-robot deployment latency, which would be a batch of one and dominated by
launch overhead.

Usage:
    python -m jose.bench_inference [--batch 256] [--repeats 50]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
CATALOG = ROOT / "logs/jose_g1/ablation/2026-08-29_23-46-41__model_4999/locomotion/catalog/estimators"


def newest(pattern: str) -> Path | None:
    hits = sorted(CATALOG.glob(pattern))
    return hits[-1] if hits else None


def load_jose(kind: str, device):
    path = newest(f"{kind}/window_25/joints_all/seed_42/variants/*/attempts/*/artifact/best_estimator.pt")
    if path is None:
        return None
    from jose.estimator.models import build_estimator

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    window = config.get("window", 25)
    estimator = build_estimator(
        config["type"], config["input_dim"], config["output_dim"], window=window,
        hidden_size=config.get("hidden_size", 256), num_layers=config.get("num_layers", 2),
        tcn_channels=tuple(config.get("channels", (64, 128, 128))),
    ).to(device)
    estimator.load_state_dict(checkpoint["model_state_dict"])
    estimator.eval()
    shape = (window, config["input_dim"])
    params = sum(p.numel() for p in estimator.parameters())
    return estimator, shape, params, str(path)


def load_set(device):
    study = sorted((ROOT / "logs/jose_g1/set_baseline").glob("*/locomotion"))
    if not study:
        return None
    path = study[-1] / "methods/set/context_20/seed_42/set_estimator.pt"
    if not path.is_file():
        return None
    from jose.set_baseline.model import SETEstimator

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = checkpoint["model_config"]
    estimator = SETEstimator(
        observation_dim=config["observation_dim"], target_dim=config["target_dim"],
        output_dim=config["output_dim"], context=config["context"], width=config["width"],
        blocks=config["blocks"], heads=config["heads"],
        estimated_indices=tuple(config["estimated_indices"]),
        pass_through_indices=tuple(config["pass_through_indices"]),
    ).to(device)
    estimator.load_state_dict(checkpoint["model_state_dict"])
    estimator.eval()
    # SET's input is the packed pair sequence: each timestep carries the
    # non-privileged observation and the privileged entry beside it, which is
    # what the two token embeddings consume.
    shape = (config["context"], config["observation_dim"] + config["target_dim"])
    params = sum(p.numel() for p in estimator.parameters())
    return estimator, shape, params, str(path)


@torch.no_grad()
def time_forward(estimator, shape, batch: int, repeats: int, warmup: int, device) -> float:
    """Median microseconds per sample."""
    inputs = torch.randn(batch, *shape, device=device)
    for _ in range(warmup):
        estimator.predict(inputs)
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        estimator.predict(inputs)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1.0e6 / batch)
    return st.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=256, help="Environments the closed loop steps")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--out", default="logs/jose_g1/inference_cost.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device; the table's numbers are GPU numbers.")
    device = torch.device("cuda:0")
    print(f"device: {torch.cuda.get_device_name(0)}  batch={args.batch}  repeats={args.repeats}")

    results = {}
    for label, loader in (("LSTM", lambda: load_jose("lstm", device)),
                          ("History MLP", lambda: load_jose("history_mlp", device)),
                          ("TCN", lambda: load_jose("tcn", device)),
                          ("SET", lambda: load_set(device))):
        loaded = loader()
        if loaded is None:
            print(f"  {label:12s} checkpoint not found")
            continue
        estimator, shape, params, path = loaded
        micros = time_forward(estimator, shape, args.batch, args.repeats, args.warmup, device)
        results[label] = {"us_per_sample": micros, "parameters": params,
                          "input_shape": list(shape), "checkpoint": path}
        print(f"  {label:12s} {micros:7.2f} us/sample   {params/1e6:.2f} M params   input {tuple(shape)}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"batch": args.batch, "repeats": args.repeats,
                               "device": torch.cuda.get_device_name(0),
                               "estimators": results}, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
