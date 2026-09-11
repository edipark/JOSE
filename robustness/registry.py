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
    "set_imu_dr": ("set", "set_imu_noise"),
    # The two cells the training-protocol x IMU study added. They do not live
    # under a method_comparison ``methods/`` tree -- they are their own study --
    # so ``resolve`` reads them from ``imu_study`` instead. See the note there.
    "jose_imu": ("estimator_imu", "jose_imu"),
    "jose_imu_dr": ("estimator_imu", "jose_imu_dr"),
    "set_dagger": ("set_dagger", "set_dagger"),
    "set_dagger_dr": ("set_dagger", "set_dagger_dr"),
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
    "set_imu_dr": "set",
    "jose_imu_dr": "jose_imu",
    "set_dagger_dr": "set_dagger",
}

#: What each axis can actually move.
#:
#: IMU axis -- ``imu_dr`` trained against a noisy IMU and ``imu_clean`` did not,
#: which is the randomization pair, and ``set_imu_dr`` is SET collected against
#: the same noisy IMU. Both methods that read an IMU are hardened, for the reason
#: the encoder axis hardens everything: a gap between a treated arm and an
#: untreated one shows which arm got the treatment, not which method tolerates a
#: bad sensor. JOSE and the joint-only student read no IMU,
#: so they are flat by construction; they are measured anyway, because a bar the
#: reader can see should come from a run and not from an argument, and because a
#: flatness that is asserted is a flatness nobody checked. The teacher is
#: measured for the same reason.
#:
#: Encoder axis -- everything reads joints, so everything moves, the teacher
#: included, and its curve is the ceiling. Every trained method appears twice,
#: SET included: hardening some arms and not others would show which arm got the
#: treatment rather than which method tolerates bad encoders.
AXIS_METHODS: dict[str, tuple[str, ...]] = {
    "imu": (
        "teacher", "jose", "joint_only", "imu_clean", "set", "imu_dr", "set_imu_dr",
        # JOSE+IMU is the first arm on this axis that beat joint-only JOSE in the
        # clean condition (0.0481 against 0.0499), so unlike every earlier IMU
        # method its curve has somewhere to fall from: the scale at which it
        # crosses JOSE's flat line is the price of reading an inertial unit.
        # SET+DAgger is here because SET's pass-through is an *architecture*
        # choice whose cost only exists under noise -- six sensor dimensions go
        # into the teacher's observation unfiltered, where JOSE+IMU regresses
        # them. Clean evaluation cannot see that difference at all.
        "jose_imu", "jose_imu_dr", "set_dagger", "set_dagger_dr",
    ),
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
    "estimator_imu": "best_estimator.pt",
    "set_dagger": "set_estimator.pt",
}


def resolve(
    method: str, study: Path, set_study: Path | None, seed: int, context: int = 20, window: int = 25,
    imu_study: Path | None = None,
) -> tuple[str, Path | None]:
    """``(loader kind, checkpoint path)``. ``None`` for the teacher, or a missing study.

    ``imu_study`` is the training-protocol x IMU study directory. Its arms are
    laid out ``<imu_study>/<slug>/seed_N/`` -- flat, because that study varies
    neither window nor joint set nor context, and mirroring the deeper
    ``methods/window_25/joints_all/`` path would assert three constants as if
    they were choices.
    """
    if method not in METHOD_SPECS:
        raise KeyError(f"Unknown method {method!r}; choose from {sorted(METHOD_SPECS)}")
    kind, slug = METHOD_SPECS[method]
    if kind == "teacher":
        return kind, None
    if kind in ("estimator_imu", "set_dagger"):
        if imu_study is None:
            return kind, None
        return kind, imu_study / slug / f"seed_{seed}" / CHECKPOINT_NAMES[kind]
    if kind in ("set", "set_enc"):
        if set_study is None:
            return kind, None
        base = set_study / "methods" / slug / f"context_{context}" / f"seed_{seed}"
    else:
        base = study / "methods" / slug / f"window_{window}" / "joints_all" / f"seed_{seed}"
    return kind, base / CHECKPOINT_NAMES[kind]
