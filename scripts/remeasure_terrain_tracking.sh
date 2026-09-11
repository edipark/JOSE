#!/usr/bin/env bash
# Re-measure a finished terrain study's command tracking the way Table I(b) is
# measured: eval_sensor_robustness.py at IMU scale 0 (clean sensors), on the
# study's own checkpoints and teacher, in the study's own environment.
#
# Why it is needed. A study's own rows carry a tracking error from the grid each
# trainer runs at the end of training, and that grid is not Table I(b)'s: the
# distillation students run it with their training-time IMU corruption switched
# on (train_history_student.py builds the grid policy from frame(True)). On flat
# ground that puts IMU-based distillation at 0.065 against Table I(b)'s 0.054,
# and joint-only distillation at 0.054 against 0.052. Survival needs no redo:
# Table I(b) takes it from the same study rows (100 - death_rate).
#
#   JOSE_PY   interpreter with Isaac Sim / Isaac Lab (required)
#   VARIANT   friction | slope | push | slope_fixed   (required)
#   SEEDS     checkpoint seeds (default: "42 43 44")
#   DRY_RUN   1 to print what would be measured, check every checkpoint exists, and stop
#
# Writes logs/jose_g1/terrain_<variant>/track_sweep0/imu_axis.jsonl, which
# collect_terrain_results.sh packs with the driver logs. Rerunnable: the output
# is replaced, not appended to.
set -euo pipefail

: "${JOSE_PY:?set JOSE_PY to the python that has Isaac Sim and Isaac Lab}"
: "${VARIANT:?set VARIANT to friction, slope, push or slope_fixed}"
ROOT="${JOSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)}"
SEEDS="${SEEDS:-42 43 44}"
cd "$ROOT"

case "$VARIANT" in
  friction)    EXPERIMENT="isaac_g1_ppo_walk_friction_jose_v0"; CASE_GLOB="locomotion_friction" ;;
  slope)       EXPERIMENT="isaac_g1_ppo_walk_slope_jose_v0";    CASE_GLOB="locomotion_slope" ;;
  push)        EXPERIMENT="isaac_g1_ppo_walk_push_jose_v0";     CASE_GLOB="locomotion_push" ;;
  slope_fixed) EXPERIMENT="isaac_g1_ppo_walk_slope_jose_v0";    CASE_GLOB="locomotion_slope_fixed_l*" ;;
  *) echo "VARIANT must be friction, slope, push or slope_fixed" >&2; exit 2 ;;
esac

latest() { ls -d $1 2>/dev/null | sort | tail -1; }
TEACHER=$(ls logs/rsl_rl/$EXPERIMENT/*/model_*.pt | sed 's/.*model_\([0-9]*\)\.pt/\1 &/' | sort -rn | head -1 | cut -d' ' -f2-)
STUDY=$(latest "logs/jose_g1/ablation/*/$CASE_GLOB/studies/method_comparison/*")
SET_STUDY=$(latest "logs/jose_g1/set_baseline/*/$CASE_GLOB")
[ -n "$STUDY" ] || { echo "no method_comparison study for $CASE_GLOB" >&2; exit 1; }
CASE=$(basename "$(dirname "$(dirname "$(dirname "$STUDY")")")")
TASK=$("$JOSE_PY" -c "from jose.ablation_catalog import TASKS; print(TASKS['$CASE'][0])")

echo "=== re-measuring tracking: $VARIANT ==="
echo "    case       $CASE ($TASK)"
echo "    teacher    $TEACHER"
echo "    study      $STUDY"
echo "    set study  ${SET_STUDY:-(none)}"
missing=0
for seed in $SEEDS; do
  for f in "$STUDY/methods/jose/window_25/joints_all/seed_$seed/best_estimator.pt" \
           "$STUDY/methods/joint_only_distillation/window_25/joints_all/seed_$seed/checkpoints/student_best_eval.pt" \
           "$STUDY/methods/imu_based_distillation/window_25/joints_all/seed_$seed/checkpoints/student_best_eval.pt" \
           ${SET_STUDY:+"$SET_STUDY/methods/set/context_20/seed_$seed/set_estimator.pt"}; do
    [ -f "$f" ] || { echo "    MISSING    $f"; missing=1; }
  done
done
[ "$missing" = 0 ] || { echo "checkpoints missing; nothing measured" >&2; exit 1; }
echo "    checkpoints all present"
[ "${DRY_RUN:-0}" = 1 ] && { echo "DRY_RUN=1: stopping before evaluation"; exit 0; }

OUT="logs/jose_g1/terrain_${VARIANT}/track_sweep0"
mkdir -p "$OUT"
rm -f "$OUT/imu_axis.jsonl"

# imu_dr is the distillation arm the studies train (IMU corruption on, the
# runners' default); at scale 0 it is measured with clean sensors, as Table I(b).
"$JOSE_PY" -u eval_sensor_robustness.py --axis imu --scale 0 \
    --methods teacher jose joint_only imu_dr ${SET_STUDY:+set} --seeds $SEEDS \
    --study "$STUDY" ${SET_STUDY:+--set-study "$SET_STUDY"} \
    --teacher-checkpoint "$TEACHER" --task "$TASK" --num-envs 256 \
    --out "$OUT/imu_axis.jsonl" --headless 2>&1 | tee "$OUT/run.log"

"$JOSE_PY" - "$OUT/imu_axis.jsonl" <<'EOF'
import json, statistics as st, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
by = {}
for row in rows:
    by.setdefault(row["method"], []).append(row.get("metrics", row)["track_error_norm"])
for method, values in by.items():
    spread = f" +-{st.stdev(values):.4f}" if len(values) > 1 else ""
    print(f"  {method:12s} track {st.mean(values):.4f}{spread}  (n={len(values)})")
EOF
