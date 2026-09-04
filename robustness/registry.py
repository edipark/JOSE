"""Which arms exist on each degradation axis, and where their checkpoints live.

Kept apart from ``eval_sensor_robustness.py`` so it can be imported -- and
therefore tested -- without a running simulator. The driver imports isaaclab at
module scope, which needs a live Omniverse app; this file is plain data and path
arithmetic, and the invariants that matter (every axis entry is a known method,
every hardened arm has an unhardened partner, no two arms share a directory) are
exactly the ones worth checking before an eleven-hour queue starts.
"""

from __future__ import annotations

from pathlib import Path


#: method name -> (loader kind, directory slug under ``methods/``).
#:
#: Written out rather than derived from a suffix rule. The difference between an
#: arm that saw sensor noise during training and one that did not is the entire
#: point of both axes, and a convention that made the two look interchangeable
#: would be an efficient way to plot the wrong pair.
METHOD_SPECS: dict[str, tuple[str, str | None]] = {
    "teacher": ("teacher", None),
    "jose": ("estimator", "jose"),
    "jose_enc": ("estimator", "jose_encoder_noise"),
    "joint_only": ("student", "joint_only_distillation"),
    "joint_only_enc": ("student", "joint_only_distillation_encoder_noise"),
    "imu_clean": ("student", "imu_based_distillation_clean"),
    "imu_clean_enc": ("student", "imu_based_distillation_clean_encoder_noise"),
    "imu_dr": ("student", "imu_based_distillation"),
    "set": ("set", "set"),
    "set_enc": ("set_enc", "set_encoder_noise"),
}

#: Arms that saw noise during training, paired with the arm that did not.
#: Every entry here is a "does randomization help" comparison; a hardened arm
#: with no partner would be a number with nothing to be measured against.
RANDOMIZATION_PAIRS: dict[str, str] = {
    "jose_enc": "jose",
    "joint_only_enc": "joint_only",
    "imu_clean_enc": "imu_clean",
    "set_enc": "set",
    "imu_dr": "imu_clean",
}

#: What each axis can actually move.
#:
#: IMU axis -- ``imu_dr`` trained against a noisy IMU and ``imu_clean`` did not,
#: which is the randomization pair. JOSE and the joint-only student read no IMU,
#: so their curves are flat by construction and are stated in the paper rather
#: than simulated. The teacher is measured despite having no IMU either, so that
#: flat line is checked rather than asserted.
#:
#: Encoder axis -- everything reads joints, so everything moves, the teacher
#: included, and its curve is the ceiling. Every trained method appears twice,
#: SET included: hardening some arms and not others would show which arm got the
#: treatment rather than which method tolerates bad encoders.
AXIS_METHODS: dict[str, tuple[str, ...]] = {
    "imu": ("teacher", "imu_dr", "imu_clean", "set"),
    "encoder": (
        "teacher",
        "jose", "jose_enc",
        "joint_only", "joint_only_enc",
        "imu_clean", "imu_clean_enc",
        "set", "set_enc",
    ),
}

#: Filenames each loader kind expects, relative to the seed directory.
CHECKPOINT_NAMES = {
    "estimator": "best_estimator.pt",
    "student": "checkpoints/student_best_eval.pt",
    "set": "set_estimator.pt",
    "set_enc": "set_estimator.pt",
}


def resolve(
    method: str, study: Path, set_study: Path | None, seed: int, context: int = 20, window: int = 25
) -> tuple[str, Path | None]:
    """``(loader kind, checkpoint path)``. ``None`` for the teacher, or no SET study."""
    if method not in METHOD_SPECS:
        raise KeyError(f"Unknown method {method!r}; choose from {sorted(METHOD_SPECS)}")
    kind, slug = METHOD_SPECS[method]
    if kind == "teacher":
        return kind, None
    if kind in ("set", "set_enc"):
        if set_study is None:
            return kind, None
        base = set_study / "methods" / slug / f"context_{context}" / f"seed_{seed}"
    else:
        base = study / "methods" / slug / f"window_{window}" / "joints_all" / f"seed_{seed}"
    return kind, base / CHECKPOINT_NAMES[kind]
