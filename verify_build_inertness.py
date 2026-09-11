"""Gate: does the current build still reproduce the recorded rows exactly?

Why this exists
---------------
The JOSE+IMU and SET+DAgger arms needed new options in ``estimator/adapters.py``,
``train_state_estimator.py`` and ``train_set_baseline.py``. All three files are
hashed into implementation fingerprints (``ablation_catalog.TRAINING_IMPLEMENTATION``
and ``run_set_baseline.SET_IMPLEMENTATION``), so adding a flag changes the digest
of every JOSE and SET study even though the flag defaults to off and the executed
path is untouched.

That matters because the new arms are meant to be compared against the *recorded*
JOSE and SET rows rather than against fresh ones. A digest mismatch is exactly the
signal this project uses to say "these numbers came from different code" -- it is
what the jump discrepancy in ``logs/paper_data/README.md`` turned out to be. So
the digest cannot be waved away; it has to be answered.

It is answered by measurement. Estimator training here is deterministic to the
last bit -- walk's ``lstm_w25_all``, retrained 21 hours later under a different
digest, reproduced ``10.939131736755371`` exactly -- so if a re-run of an existing
arm under the current build returns the recorded numbers to full precision, the
byte change provably did not reach the executed path, and reusing the recorded
rows is sound. If it does not, that must be found before six GPU-hours are spent
on arms that cannot be compared to anything.

This is the same idea as ``verify_set_protocol.py``, which gates SET training on
reproducing JOSE's logged numbers before it starts.

Usage
-----
    python verify_build_inertness.py --arm jose --run-dir <fresh JOSE seed-42 run>
    python verify_build_inertness.py --arm set  --run-dir <fresh SET  seed-42 run>

Each ``--run-dir`` is the output of re-running that arm at seed 42 on the
locomotion task with the flags recorded in ``EXPECTED[...]["command"]`` below.
Exit code 1 on any mismatch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


#: What each arm must reproduce, and the command that produces it.
#:
#: Values are copied from logs/paper_data/table1_deployment_strategies at full
#: precision. They are the seed-42 locomotion rows the paper reports.
EXPECTED = {
    "jose": {
        "source": "logs/paper_data/table1_deployment_strategies/locomotion/results.jsonl"
                  "  (experiment=JOSE, seed=42)",
        "command": (
            "python train_state_estimator.py --task Isaac-G1-PPO-Walk-Estimator-JOSE-v0 "
            "--adapter ppo_walk --teacher-checkpoint <flat teacher model_4999.pt> "
            "--estimator LSTM --window 25 --joint-preset all --dagger-rounds 10 "
            "--dagger-epochs 10 --max-dataset-size 250000 --num-envs 256 --seed 42 --headless"
        ),
        "metrics": {
            "rmse": 0.013094832189381123,
            "grid_rmse": 0.021027546375989914,
            "track_error_norm": 0.050354464600483574,
            "episode_length_mean": 997.925,
            "grid_survival_rate": 1.0,
            "death_rate": 0.5,
            "best_round": 10,
        },
    },
    "set": {
        # `track_error_norm` below is NOT the value in the 2026-09-04 run that
        # produced the paper's SET row. That run predates 59d5e17, which fixed
        # three CommandEvaluator bugs and, in passing, swapped the order of the
        # survivor mask and the column slice:
        #
        #     old:  rmse = sqrt(...)[valid]   then  float(rmse[:, 0].mean())
        #     new:  survivor_mean(rmse_all[:, 0])  ==  rmse_all[:, 0][valid].mean()
        #
        # Same numbers, same count -- but one reduction runs over a stride-3 view
        # and the other over a contiguous one, so torch splits the sum differently
        # and the result moves by one float32 ulp. Measured on 2026-09-09: 18 of
        # the 45 per-command RMSE fields moved by exactly +-1 ulp; all 45
        # per-command *bias* fields did not (their subtraction builds a contiguous
        # temporary either way); the trained weights are bit-identical across all
        # 1,204,013 parameters. Nothing dies in this run (grid_survival_rate 1.0),
        # so the bug fixes are semantic no-ops here and only the rounding differs.
        #
        # Gating on the 09-04 value would therefore fail for every build after
        # 2026-09-07 whether or not this study existed, which is not the question
        # this gate asks. The reference is the build immediately BEFORE this study
        # (e0ce15e = b1509b4^, re-run from a worktree on 2026-09-09), so a mismatch
        # means *this* study's changes moved something.
        #
        # Pre-59d5e17 value, kept for provenance:
        #     track_error_norm = 0.06190768314732445
        #     from logs/jose_g1/set_baseline/2026-09-04_11-44-46/locomotion
        #          /methods/set/context_20/seed_42/training.json
        "source": "logs/jose_g1/training_imu_study/gate_set_oldbuild/training.json"
                  "  (e0ce15e = b1509b4^, the build before this study -- see note above)",
        "command": (
            "python train_set_baseline.py --task Isaac-G1-PPO-Walk-Estimator-JOSE-v0 "
            "--adapter ppo_walk --teacher-checkpoint <flat teacher model_4999.pt> "
            "--context 20 --blocks 6 --width 128 --heads 4 --dropout 0.1 "
            "--imu-noise-scale 0.0 --dagger-rounds 0 --num-envs 256 --seed 42 --headless"
            "  (the reference run predates --dagger-rounds and simply omits it;"
            " its default is 0, which is the same executed path)"
        ),
        "metrics": {
            "rmse": 0.020385736599564552,
            "track_error_norm": 0.061907683312892904,
            "episode_length_mean": 1000.0,
            "grid_survival_rate": 1.0,
            "death_rate": 0.0,
            "open_loop_rmse": 0.004247531294822693,
            "best_validation_mse": 0.0010490843560546637,
            "total_gradient_steps": 11750,
            "parameters": 1203843,
        },
    },
}

#: Exact equality is the claim being tested, so there is no tolerance to tune.
#: A float that differs in the last bit means the executed path changed, which is
#: precisely what this gate is for.
TOLERANCE = 0.0


def load_metrics(run_dir: Path) -> dict:
    for name in ("training.json", "results.json"):
        candidate = run_dir / name
        if candidate.exists():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if "metrics" in payload:
                return payload["metrics"]
            return payload
    raise SystemExit(f"no training.json under {run_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", choices=sorted(EXPECTED), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="Write the verdict as JSON")
    args = parser.parse_args()

    spec = EXPECTED[args.arm]
    actual = load_metrics(args.run_dir)

    print(f"=== build inertness gate: {args.arm} ===")
    print(f"reference : {spec['source']}")
    print(f"fresh run : {args.run_dir}\n")
    print(f"{'metric':24s} {'recorded':>26s} {'fresh':>26s}   verdict")

    rows, failures = [], []
    for key, expected in spec["metrics"].items():
        got = actual.get(key)
        if got is None:
            ok = False
            note = "MISSING"
        elif isinstance(expected, (int, float)) and isinstance(got, (int, float)):
            ok = abs(float(got) - float(expected)) <= TOLERANCE
            note = "match" if ok else f"differs by {float(got) - float(expected):+.3e}"
        else:
            ok = got == expected
            note = "match" if ok else "differs"
        rows.append({"metric": key, "expected": expected, "actual": got, "ok": ok})
        if not ok:
            failures.append(key)
        print(f"{key:24s} {expected!r:>26} {got!r:>26}   {note}")

    verdict = {
        "arm": args.arm,
        "run_dir": str(args.run_dir),
        "reference": spec["source"],
        "tolerance": TOLERANCE,
        "checks": rows,
        "failures": failures,
        "passed": not failures,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")

    if failures:
        print(
            f"\nFAIL: {len(failures)} of {len(rows)} metrics moved -- "
            f"{', '.join(failures)}.\n"
            "The current build does not reproduce the recorded row, so the change is not\n"
            "inert and the recorded JOSE/SET rows cannot be reused beside the new arms.\n"
            "Stop here and report it: the choice is to re-run every arm under one build,\n"
            "which is a separate decision about GPU time, not something to work around."
        )
        return 1

    print(
        f"\nPASS: all {len(rows)} metrics reproduce exactly.\n"
        "The implementation digest changed, the executed path did not. Comparing the\n"
        "new arms against the recorded JOSE and SET rows is sound; cite this verdict\n"
        "wherever that comparison is reported."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
