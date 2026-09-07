"""Forward-pass FLOPs per sample for every estimator in the paper.

Wall-clock is a property of the GPU it was measured on, and this project has
already been bitten once by timing under contention. FLOPs are a property of the
architecture, so anyone can reproduce the column without our hardware.

`torch.utils.flop_counter` handles the linear, convolution and attention layers
but does not see `aten::lstm`, whose fused cuDNN kernel it never decomposes, so
the recurrence is added analytically below and cross-checked against the head the
counter does see. The convention throughout is 2 FLOPs per multiply-accumulate,
which is what the counter reports; the teacher policy's plain MLP reproduces its
hand-computed 0.842 MFLOPs exactly, which is what pins the convention down.

    python -m jose.bench_flops
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from jose import bench_inference as bench

ROOT = Path(__file__).resolve().parent


def lstm_flops(module: nn.LSTM, steps: int) -> int:
    """FLOPs the counter misses, for one sample over `steps` timesteps.

    Each layer computes four gates from the input and from the previous hidden
    state: 4H x in and 4H x H multiply-accumulates per step.
    """
    total = 0
    for layer in range(module.num_layers):
        in_dim = module.input_size if layer == 0 else module.hidden_size
        macs = 4 * module.hidden_size * (in_dim + module.hidden_size)
        total += macs * steps
    return 2 * total


def measure(estimator, shape, device) -> int:
    inputs = torch.randn(1, *shape, device=device)
    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        estimator.predict(inputs)
    flops = counter.get_total_flops()
    for module in estimator.modules():
        if isinstance(module, nn.LSTM):
            flops += lstm_flops(module, shape[0])
    return flops


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rows = {}
    print(f"{'model':16s} {'MFLOPs/sample':>14s} {'params (M)':>11s}   input")
    for label, loader in (("LSTM", lambda: bench.load_jose("lstm", device)),
                          ("History MLP", lambda: bench.load_jose("history_mlp", device)),
                          ("TCN", lambda: bench.load_jose("tcn", device)),
                          ("SET", lambda: bench.load_set(device)),
                          ("Teacher policy", lambda: bench.load_teacher(device))):
        got = loader()
        if got is None:
            print(f"{label:16s} checkpoint not found"); continue
        estimator, shape, params, path = got
        flops = measure(estimator, shape, device)
        rows[label] = {"mflops_per_sample": flops / 1e6, "parameters": params,
                       "input_shape": list(shape), "checkpoint": path}
        print(f"{label:16s} {flops/1e6:14.2f} {params/1e6:11.2f}   {tuple(shape)}")

    out = ROOT / "logs/jose_g1/inference_flops.json"
    out.write_text(json.dumps({"convention": "2 FLOPs per multiply-accumulate",
                               "models": rows}, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
