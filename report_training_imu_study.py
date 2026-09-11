"""Build the training-protocol x IMU table from the four arms.

Two of the arms are read from the recorded study directories (JOSE, SET) and two
from the runs made by ``logs/jose_g1/run_training_imu_study.sh`` (JOSE+IMU,
SET+DAgger). Reusing the recorded pair is licensed by ``verify_build_inertness.py``
and this script refuses to emit a table until that verdict is on disk and passing.

Reading the columns
-------------------
**Estimation error is not comparable arm-to-arm as a single number.** SET copies
the six inertial dimensions from the sensor instead of predicting them, so its
error on those is a sensor reading rather than a prediction and its overall RMSE
is flattered by six free dimensions. The paper concedes this in prose; here it is
measured. ``rmse_v`` is the base-linear-velocity group alone -- the only part of
the target every arm must actually predict -- and it is the honest head-to-head.

**The training-protocol axis is read within a family, never across one.** The two
families name their metrics differently for the same reason they measure
differently: for both, ``rmse`` is closed-loop error over 200 episodes, which is
comparable; but JOSE's ``open_loop_rmse`` is prediction error on states the
estimator itself induces, while SET's is error on its own training set. Those are
not the same quantity, so the offline-to-closed-loop gap is reported per family
and not differenced between them. The DAgger effect is therefore read down the
columns of each family -- JOSE against JOSE (K=0), SET against SET+DAgger -- which
is what makes it identifiable at all.

The JOSE (K=0) column is not a new run. It is the ``lstm_w25_all_r00`` arm of the
aggregation-rounds ablation behind Fig. 3, recorded under JOSE's own
instrumentation, which is exactly the offline corner this table needs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent

#: Component groups of the 9-D locomotion estimator target, in schema order
#: (schema.py: base_lin_vel xyz, base_ang_vel xyz, projected_gravity xyz).
GROUPS = {"v": slice(0, 3), "w": slice(3, 6), "g": slice(6, 9)}

ARMS = {
    "JOSE (K=0)": {
        "reads": "encoders",
        "predicts": "9 of 9",
        "training": "offline",
        "recorded": "logs/paper_data/fig3_dagger_rounds/locomotion/results.jsonl",
        "experiment": "lstm_w25_all_r00",
    },
    "JOSE": {
        "reads": "encoders",
        "predicts": "9 of 9",
        "training": "DAgger K=10",
        "recorded": "logs/paper_data/table1_deployment_strategies/locomotion/results.jsonl",
        "experiment": "JOSE",
    },
    "JOSE+IMU": {
        "reads": "encoders + IMU",
        "predicts": "9 of 9",
        "training": "DAgger K=10",
        "run_dir": "logs/jose_g1/training_imu_study/jose_imu",
    },
    "SET": {
        "reads": "encoders + IMU",
        "predicts": "3 of 9 (6 passed through)",
        "training": "offline",
        "recorded": "logs/paper_data/table1_deployment_strategies/"
                    "set__walk_jump_locomotion/results.jsonl",
        "task_key": "locomotion",
    },
    "SET+DAgger": {
        "reads": "encoders + IMU",
        "predicts": "3 of 9 (6 passed through)",
        "training": "DAgger K=10",
        "run_dir": "logs/jose_g1/training_imu_study/set_dagger",
    },
}


def group_rmse(per_component, group: str):
    """Root-mean-square over one component group.

    Combining per-component RMSEs means squaring, averaging and rooting -- not
    averaging the RMSEs, which would understate the group.
    """
    if not per_component:
        return None
    values = list(per_component)[GROUPS[group]]
    if not values or any(value is None for value in values):
        return None
    return (sum(float(value) ** 2 for value in values) / len(values)) ** 0.5


def summarize(metrics: dict) -> dict:
    # `target_rmse` is the closed-loop episodic per-component vector, and both
    # families emit it from the same 200-episode protocol. `grid_target_rmse` is
    # JOSE-only (SET runs no command grid inside its trainer), so the group
    # columns are built from the vector both have.
    per_component = metrics.get("target_rmse")
    row = {
        # -- closed-loop control
        "survival": metrics.get("grid_survival_rate"),
        # Table I(b)'s `Surv. (%)`, recovered on its own terms: the fraction of
        # episodes that did not terminate early, from the same 200-episode
        # rollout `ep_len` comes from, where the task's sampler re-draws the
        # command mid-episode. Reported beside the grid figure rather than
        # instead of it -- the grid holds one command for four seconds and cannot
        # see a failure that arrives at step 582, which is where JOSE's single
        # lost episode out of 600 actually ended.
        "surv_ep": None if metrics.get("death_rate") is None else 100.0 - metrics["death_rate"],
        "ep_len": metrics.get("episode_length_mean"),
        "track": metrics.get("track_error_norm"),
        "track_vx": metrics.get("track_vx_rmse"),
        "track_vy": metrics.get("track_vy_rmse"),
        "track_yaw": metrics.get("track_yaw_rmse"),
        "falls": metrics.get("grid_fall_count"),
        # -- teacher fidelity
        "action_mse": metrics.get("teacher_action_mse"),
        # -- estimation
        "rmse_closed": metrics.get("rmse"),
        "rmse_grid": metrics.get("grid_rmse"),
        "rmse_open": metrics.get("open_loop_rmse"),
        "rmse_v": group_rmse(per_component, "v"),
        "rmse_w": group_rmse(per_component, "w"),
        "rmse_g": group_rmse(per_component, "g"),
        # -- training
        "best_round": metrics.get("best_round"),
        "grad_steps": metrics.get("total_gradient_steps"),
        "wall_min": (metrics.get("wall_time_s") or 0) / 60 or None,
        # -- cost
        "params": metrics.get("estimator_parameters") or metrics.get("parameters"),
        "ms_est": metrics.get("estimator_inference_ms_per_sample"),
        "ms_step": metrics.get("inference_ms_per_sample"),
        # -- gait
        "energy": metrics.get("energy"),
        "torque_rms": metrics.get("torque_rms"),
        "smoothness": metrics.get("action_smoothness"),
        "slide": metrics.get("feet_slide_penalty"),
        "double_stance": metrics.get("double_stance_fraction"),
    }
    return row


def load_recorded(spec: dict) -> dict[int, dict]:
    path = ROOT / spec["recorded"]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    out = {}
    for row in rows:
        if "experiment" in spec and row.get("experiment") != spec["experiment"]:
            continue
        if "task_key" in spec and row.get("task_key") != spec["task_key"]:
            continue
        metrics = row.get("metrics")
        if metrics is None:
            # The SET study keeps metrics in the artifact rather than the row.
            artifact = Path(row["artifact"])
            if not artifact.exists():
                continue
            metrics = json.loads(artifact.read_text(encoding="utf-8"))["metrics"]
        out[row["seed"]] = metrics
    return out


def load_runs(spec: dict) -> dict[int, dict]:
    base = ROOT / spec["run_dir"]
    out = {}
    if not base.exists():
        return out
    for run in sorted(base.glob("seed_*")):
        artifact = run / "training.json"
        if artifact.exists():
            out[int(run.name.split("_")[1])] = json.loads(artifact.read_text(encoding="utf-8"))["metrics"]
    return out


def aggregate(per_seed: dict[int, dict]) -> dict:
    rows = [summarize(metrics) for metrics in per_seed.values()]
    if not rows:
        return {}
    out = {}
    for key in rows[0]:
        values = [row[key] for row in rows if row.get(key) is not None]
        if not values:
            out[key] = (None, None)
        else:
            out[key] = (
                statistics.fmean(values),
                statistics.stdev(values) if len(values) > 1 else 0.0,
            )
    out["_seeds"] = sorted(per_seed)
    return out


def cell(pair, digits: int) -> str:
    if not pair or pair[0] is None:
        return "--"
    mean, sd = pair
    return f"{mean:.{digits}f}" if not sd else f"{mean:.{digits}f}±{sd:.{digits}f}"


def check_gates(strict: bool) -> None:
    base = ROOT / "logs/jose_g1/training_imu_study"
    missing, failed = [], []
    for arm in ("jose", "set"):
        verdict = base / f"gate_{arm}_verdict.json"
        if not verdict.exists():
            missing.append(verdict.name)
        elif not json.loads(verdict.read_text(encoding="utf-8")).get("passed"):
            failed.append(verdict.name)
    if failed:
        raise SystemExit(
            f"inertness gate FAILED ({', '.join(failed)}). The recorded JOSE/SET rows were "
            "produced by a build this one does not reproduce, so they cannot be placed beside "
            "the new arms. Re-run every arm under one build instead."
        )
    if missing and strict:
        raise SystemExit(
            f"inertness gate not run ({', '.join(missing)}). Two of the four arms here are "
            "recorded rather than freshly trained, and that is only sound once the gate has "
            "shown the current build reproduces them. Run:\n"
            "    STAGE=gate bash logs/jose_g1/run_training_imu_study.sh\n"
            "or pass --no-require-gate to render a provisional table."
        )
    if missing:
        print(f"WARNING: provisional -- inertness gate not run ({', '.join(missing)})\n")


#: The IMU degradation axis. Two files: the seven arms measured by
#: run_robustness_cmdfix.sh (the paper's Fig. 4/5 source, never written to here)
#: and the four cells this study added, in a sibling file.
AXIS_FILES = (
    ROOT / "logs/jose_g1/robustness_cmdfix/imu_axis.jsonl",
    ROOT / "logs/jose_g1/robustness_cmdfix/imu_axis_study2.jsonl",
)
AXIS_SCALES = (0.0, 1.0, 2.0, 4.0)
#: Ordered for reading, not alphabetically: the two flat references first, then
#: each IMU arm beside the randomized version of itself.
AXIS_ARMS = (
    ("teacher", "Teacher (privileged)"),
    ("jose", "JOSE (joint-only)"),
    ("joint_only", "Joint-only distill."),
    ("jose_imu", "JOSE+IMU"),
    ("jose_imu_dr", "JOSE+IMU + DR"),
    ("set_dagger", "SET+DAgger"),
    ("set_dagger_dr", "SET+DAgger + DR"),
    ("set", "SET"),
    ("set_imu_dr", "SET + DR"),
    ("imu_clean", "Distillation"),
    ("imu_dr", "Distillation + DR"),
)


def imu_axis_rows():
    """``{(method, scale): {metric: [per-seed values]}}`` across both axis files."""
    out = {}
    for path in AXIS_FILES:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            bucket = out.setdefault((row["method"], row["scale"]), {"track": [], "surv": []})
            metrics = row["metrics"]
            if metrics.get("track_error_norm") is not None:
                bucket["track"].append(metrics["track_error_norm"])
            if metrics.get("grid_survival_rate") is not None:
                bucket["surv"].append(metrics["grid_survival_rate"])
    return out


def imu_axis_block(axis):
    """The degradation table, or a note saying why there isn't one."""
    if not axis:
        return ["", "## IMU degradation axis", "",
                "Not measured: no rows under logs/jose_g1/robustness_cmdfix.", ""]

    def cell(method, scale, key, fmt):
        values = axis.get((method, scale), {}).get(key, [])
        return fmt % statistics.mean(values) if values else "--"

    lines = ["", "## IMU degradation axis", "",
        "Command tracking as the inertial unit degrades. 1x is the nominal model in",
        "`distillation/imu.py` -- gyro white noise, a per-episode gyro bias, an attitude",
        "tilt, and up to two steps of staleness -- and it is the scale the recorded",
        "`set_imu_noise` arm trained against, not a level chosen here. `+ DR` arms were",
        "trained against that same 1x model; the others saw a clean IMU.",
        "",
        "JOSE and the joint-only student read no inertial channels, so their rows are flat",
        "by construction. They are measured anyway rather than asserted.",
        ""]
    header = "| arm | " + " | ".join(f"{s:g}x" for s in AXIS_SCALES) + " | vs JOSE at 4x |"
    lines += [header, "|---" * (len(AXIS_SCALES) + 2) + "|"]
    base = axis.get(("jose", 4.0), {}).get("track", [])
    base = statistics.mean(base) if base else None
    for method, label in AXIS_ARMS:
        cells = [cell(method, s, "track", "%.4f") for s in AXIS_SCALES]
        far = axis.get((method, 4.0), {}).get("track", [])
        delta = ("%+.0f%%" % (100 * (statistics.mean(far) - base) / base)
                 if far and base else "--")
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {delta} |")

    lines += ["", "Grid survival on the same rollouts.", "",
              "| arm | " + " | ".join(f"{s:g}x" for s in AXIS_SCALES) + " |",
              "|---" * (len(AXIS_SCALES) + 1) + "|"]
    for method, label in AXIS_ARMS:
        cells = [cell(method, s, "surv", "%.3f") for s in AXIS_SCALES]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "logs/jose_g1/training_imu_study/report.md")
    parser.add_argument("--require-gate", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    check_gates(args.require_gate)

    data = {}
    for name, spec in ARMS.items():
        per_seed = load_recorded(spec) if "recorded" in spec else load_runs(spec)
        data[name] = aggregate(per_seed)

    teacher_track = None
    teacher = load_recorded({"recorded": ARMS["JOSE"]["recorded"], "experiment": "PrivilegedTeacher"})
    if teacher:
        teacher_track = statistics.fmean(
            m["track_error_norm"] for m in teacher.values() if m.get("track_error_norm") is not None
        )

    lines = ["# Training protocol x IMU, clean locomotion", ""]
    if teacher_track is not None:
        lines += [f"Teacher command-tracking error: **{teacher_track:.4f}** (the reference for Δ).", ""]

    lines += [
        "| arm | reads | predicts | training | seeds |",
        "|---|---|---|---|---|",
    ]
    for name, spec in ARMS.items():
        seeds = data[name].get("_seeds", [])
        lines.append(
            f"| {name} | {spec['reads']} | {spec['predicts']} | {spec['training']} | "
            f"{', '.join(map(str, seeds)) or 'MISSING'} |"
        )

    blocks = [
        ("Closed-loop control", [
            # Two rollouts, not one. Everything here except `Ep. len.` comes from
            # the command grid: 15 held commands, one second to settle and four to
            # measure, scored by `estimator/locomotion.py:summarize`. `Ep. len.`
            # comes from the episode evaluation -- 200 episodes of up to 1000
            # steps, with the task's own sampler re-drawing the command mid-episode.
            #
            # So `Survival` 1.0000 beside an `Ep. len.` under 1000 is not a
            # contradiction: the policy held every grid command through a four
            # second window and still lost one episode out of 600 in a rollout
            # five times longer that kept changing what it asked for.
            #
            # The combination that IS impossible is a full `Ep. len.` beside a
            # `Survival` of 0.0000, and it is worth recognising: that is what a
            # broken grid evaluation looks like, and it is how the frozen-IMU-cache
            # bug in train_set_baseline.py was found.
            ("Surv. (%) [as Table I(b)]", "surv_ep", 2),
            ("Grid survival", "survival", 4), ("Ep. len.", "ep_len", 1),
            ("Track err", "track", 4), ("Δ teacher", "_delta", 4),
            ("vx", "track_vx", 4), ("vy", "track_vy", 4), ("yaw", "track_yaw", 4),
            ("Falls", "falls", 1),
        ]),
        ("Fidelity to the teacher", [("Action MSE", "action_mse", 6)]),
        ("Estimation error (closed loop, 200 episodes)", [
            ("RMSE, all 9", "rmse_closed", 5),
            ("RMSE v (the comparable one)", "rmse_v", 5),
            ("RMSE w", "rmse_w", 5), ("RMSE g", "rmse_g", 5),
            ("RMSE, command grid", "rmse_grid", 5),
            ("open-loop RMSE (within family only)", "rmse_open", 5),
        ]),
        ("Training", [
            ("Best round", "best_round", 1), ("Gradient steps", "grad_steps", 0),
            ("Wall clock (min)", "wall_min", 1),
        ]),
        ("Cost", [
            ("Parameters", "params", 0),
            ("ms/sample (estimator)", "ms_est", 5),
            ("ms/sample (whole step)", "ms_step", 5),
        ]),
        ("Gait", [
            ("Energy", "energy", 4), ("Torque RMS", "torque_rms", 3),
            ("Action smoothness", "smoothness", 4), ("Foot slide", "slide", 4),
            ("Double stance", "double_stance", 4),
        ]),
    ]

    #: Printed under a block heading. Only where a reader has a real chance of
    #: reading two rows as one measurement.
    BLOCK_NOTES = {
        "Closed-loop control":
            "`Ep. len.` is the only row here from the episode evaluation: 200 "
            "episodes of up to 1000 steps, with the task's sampler re-drawing the "
            "command mid-episode. Every other row comes from the command grid -- "
            "15 held commands, one second to settle and four to measure.\n\n"
            "`Grid survival` is deliberately not the paper's `Surv. (%)`. Table I(b) "
            "reports `100 - death_rate` from the episode evaluation; this row is "
            "`grid_survival_rate`, the fraction of environments that hold a command "
            "through its four-second measurement window. JOSE reads 99.2 there and "
            "1.0000 here, and both are correct: it never lost a grid command and "
            "lost one episode in 600. Both are given, named apart, because one name "
            "for two quantities is how a reader concludes the numbers disagree. "
            "`Surv. (%) [as Table I(b)]` is the paper's definition and reproduces "
            "its JOSE cell exactly (99.17 -> 99.2).",
    }

    for title, rows in blocks:
        lines += ["", f"## {title}", ""]
        if title in BLOCK_NOTES:
            lines += [BLOCK_NOTES[title], ""]
        lines += ["| metric | " + " | ".join(ARMS) + " |",
                  "|---" * (len(ARMS) + 1) + "|"]
        for label, key, digits in rows:
            cells = []
            for name in ARMS:
                if key == "_delta":
                    track = data[name].get("track", (None, None))
                    cells.append(
                        "--" if track[0] is None or teacher_track is None
                        else f"{track[0] - teacher_track:+.4f}"
                    )
                else:
                    cells.append(cell(data[name].get(key), digits))
            lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Reading the estimation block",
        "",
        "SET and SET+DAgger copy angular velocity and projected gravity from the IMU",
        "instead of predicting them, so their `RMSE w` and `RMSE g` are sensor readings,",
        "not predictions, and their overall RMSE is flattered by six free dimensions.",
        "**`RMSE v` is the comparable column**: base linear velocity is the only part of",
        "the target every arm has to predict.",
        "",
        "Both families compute this identically -- the same float64 per-dimension sum of",
        "squares over 200 episodes, reduced the same way (`estimator/pipeline.py:343`,",
        "`set_baseline/evaluate.py:145`) -- so the columns are comparable in *how* the",
        "number is formed. They are not evaluated on the same initial conditions: JOSE",
        "evaluates at `seed + 10000` and SET at `seed + 1000`, a pre-existing convention",
        "difference each new arm inherits from the arm it is compared against rather than",
        "having it changed mid-study. Over 200 episodes that should be small next to the",
        "gaps here, but it is a reason to read a narrow difference cautiously.",
        "",
        "The open-loop row is not comparable between the two families and is printed for",
        "completeness only: JOSE measures it on states the estimator induces, SET on its",
        "own training set. Read the DAgger effect down each family instead -- JOSE against",
        "JOSE (K=0), SET against SET+DAgger.",
    ]

    lines += imu_axis_block(imu_axis_rows())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
