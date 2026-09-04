"""Train any of our own methods with encoder-noise randomization.

The encoder axis compares methods that have seen encoder noise at training time
against the same methods that have not. Giving that treatment to JOSE alone would
make the comparison meaningless -- our method hardened, the baselines not -- so
this one entry point serves all three methods we implement, and they are degraded
by the same code at the same magnitude.

SET is included, and the distinction that admits it is worth stating. Adding
DAgger to SET, or changing its architecture or its loss, would be modifying a
published baseline and is not ours to do. Collecting its offline rollouts in an
environment whose encoders are noisy changes no line of the method -- the
procedure is still "roll out the frozen expert, fit a causal transformer to the
privileged state" -- it changes only the data that procedure is handed. Leaving
SET out would have made its encoder-axis result unreadable: a reviewer could
attribute it to the method being sensitive, or to SET being the one arm that did
not get the treatment. Table I still reports SET trained exactly as published;
this produces the additional hardened arm, labelled as our addition since the
paper specifies no randomization.

The trainers themselves are untouched on disk, so their fingerprints are
unchanged and their existing runs stay comparable. See ``robustness/patch.py``.

Usage: the target trainer's own arguments, plus --trainer and
--encoder-noise-scale.

    python -m JOSE.train_encoder_noise --trainer jose --encoder-noise-scale 1.0 \
        --teacher-checkpoint <...> --estimator LSTM --window 25 ...
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


#: Which script each method's clean arm was trained by. The noisy arm must be
#: trained by the same one, or the pair differs in more than the noise.
TRAINERS = {
    "jose": "train_state_estimator.py",
    "imu_distillation": "train_imu_distillation.py",
    "joint_only_distillation": "train_joint_only_distillation.py",
    "set": "train_set_baseline.py",
}


def _pop(argv: list[str], flag: str, required: bool = True) -> str | None:
    if flag not in argv:
        if required:
            raise SystemExit(f"{flag} is required")
        return None
    index = argv.index(flag)
    if index + 1 >= len(argv):
        raise SystemExit(f"{flag} needs a value")
    value = argv[index + 1]
    del argv[index : index + 2]
    return value


if __name__ == "__main__":
    trainer = _pop(sys.argv, "--trainer")
    if trainer not in TRAINERS:
        raise SystemExit(f"--trainer must be one of {sorted(TRAINERS)}, got {trainer!r}")
    try:
        scale = float(_pop(sys.argv, "--encoder-noise-scale"))
    except ValueError:
        raise SystemExit("--encoder-noise-scale needs a number") from None
    if scale <= 0.0:
        raise SystemExit(
            f"--encoder-noise-scale={scale} is the clean arm, which already exists. Run "
            f"{TRAINERS[trainer]} directly so that arm keeps its recorded identity."
        )

    # Armed, not applied: the environment classes need a live simulator, and
    # the trainer launches it. See robustness/patch.py.
    from jose.robustness.patch import arm_encoder_noise

    arm_encoder_noise(scale)
    runpy.run_path(str(Path(__file__).with_name(TRAINERS[trainer])), run_name="__main__")
