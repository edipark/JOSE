#!/usr/bin/env bash
# Terrain study: train a teacher on one terrain variant, then run the whole
# Table I pipeline on it.
#
# Lives in scripts/ (tracked) rather than logs/jose_g1/ (gitignored), so that a
# `git pull` on another machine brings it along with the code it drives.
#
# Unlike every other driver in this directory, nothing here is hardcoded to one
# machine. This one is meant to run somewhere else, so paths come from the
# environment and the repository root is discovered from git:
#
#   JOSE_PY    interpreter with Isaac Sim / Isaac Lab (required)
#   JOSE_ROOT  repository checkout          (default: this file's git root)
#   VARIANT    friction | slope | push | slope_fixed   (default: slope)
#
#   slope_fixed reuses the slope teacher and runs the student pipeline with the
#   terrain curriculum removed and every environment pinned to a level 0..K. K is
#   chosen by eval_slope_levels.py before any student trains: the largest level
#   such that the teacher's death rate is at most MAX_DEATH_RATE (%) on every
#   level 0..K.
#   MAX_DEATH_RATE  slope_fixed level rule    (default: 0.5 -- see eval_slope_levels.py)
#   SEEDS      estimator seeds              (default: "42 43 44")
#   ITERATIONS teacher PPO iterations       (default: 5000, what the flat teacher used)
#   NUM_ENVS   teacher envs                 (default: 4096, what the flat teacher used)
#
# Usage:
#   JOSE_PY=~/miniconda3/envs/jose/bin/python VARIANT=slope bash run_terrain_study.sh
#
# Resumable at every stage: an existing teacher checkpoint is reused rather than
# retrained, and both study runners take --resume by default.
set -euo pipefail

: "${JOSE_PY:?set JOSE_PY to the python that has Isaac Sim and Isaac Lab}"
ROOT="${JOSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)}"
VARIANT="${VARIANT:-slope}"
SEEDS="${SEEDS:-42 43 44}"
ITERATIONS="${ITERATIONS:-5000}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_DEATH_RATE="${MAX_DEATH_RATE:-0.5}"

case "$VARIANT" in
  friction) TEACHER_TASK="Isaac-G1-PPO-Walk-Friction-JOSE-v0"; CASE="locomotion_friction" ;;
  slope)    TEACHER_TASK="Isaac-G1-PPO-Walk-Slope-JOSE-v0";    CASE="locomotion_slope" ;;
  push)     TEACHER_TASK="Isaac-G1-PPO-Walk-Push-JOSE-v0";     CASE="locomotion_push" ;;
  # CASE is filled in after the level measurement below.
  slope_fixed) TEACHER_TASK="Isaac-G1-PPO-Walk-Slope-JOSE-v0"; CASE="" ;;
  *) echo "VARIANT must be 'friction', 'slope', 'push' or 'slope_fixed', got '$VARIANT'" >&2; exit 2 ;;
esac

EXPERIMENT=$(echo "$TEACHER_TASK" | tr '[:upper:]-' '[:lower:]_')
TEACHER_ROOT="$ROOT/logs/rsl_rl/$EXPERIMENT"
LOGS="$ROOT/logs/jose_g1/terrain_${VARIANT}"
mkdir -p "$LOGS"

# One study at a time per checkout: two would fight over the GPU and over the
# study directories. Directory-create is the atomic primitive here, as in the
# other drivers in this directory.
LOCK="$LOGS/.running.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "another run holds $LOCK; remove it if no process is alive" >&2
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

cd "$ROOT"
echo "=== terrain study: $VARIANT ==="
echo "    root       $ROOT"
echo "    git        $(git rev-parse HEAD)$(git diff --quiet || echo ' (DIRTY)')"
echo "    python     $JOSE_PY"
echo "    task       $TEACHER_TASK"
echo "    case       ${CASE:-(chosen after the level measurement)}"
echo "    seeds      $SEEDS"

# --- 1. teacher ------------------------------------------------------------
# Reuse an existing checkpoint rather than retraining: this stage is ~2.5 h on an
# RTX 4070 and the stages after it are the ones likely to need a second attempt.
#
# But reuse only a *finished* one. A smoke test (`--max_iterations 5`) leaves a
# perfectly well-formed `model_4.pt` in exactly this tree, and reusing it would
# hand every downstream stage a teacher that cannot walk -- silently, since
# nothing below inspects the checkpoint's provenance. Pick the highest iteration
# available and require it to have reached the requested budget.
CKPT=$(ls "$TEACHER_ROOT"/*/model_*.pt 2>/dev/null |
       sed 's/.*model_\([0-9]*\)\.pt/\1 &/' | sort -rn | head -1 | cut -d' ' -f2- || true)
if [ -n "$CKPT" ]; then
  ITER=$(basename "$CKPT" .pt | sed 's/model_//')
  if [ "$ITER" -lt "$((ITERATIONS - 1))" ]; then
    echo "[1/4] ignoring $CKPT: iteration $ITER < $((ITERATIONS - 1)), looks like a smoke run"
    CKPT=""
  fi
fi
if [ -z "$CKPT" ]; then
  echo "[1/4] training teacher ($ITERATIONS iterations, $NUM_ENVS envs)"
  "$JOSE_PY" train_ppo_walk.py --task "$TEACHER_TASK" --headless \
      --num_envs "$NUM_ENVS" --max_iterations "$ITERATIONS" \
      2>&1 | tee "$LOGS/teacher_train.log"
  CKPT=$(ls "$TEACHER_ROOT"/*/model_*.pt |
         sed 's/.*model_\([0-9]*\)\.pt/\1 &/' | sort -rn | head -1 | cut -d' ' -f2-)
else
  echo "[1/4] reusing teacher $CKPT"
fi
echo "    teacher    $CKPT"

# --- 2. teacher gate -------------------------------------------------------
# Everything below this line is measured against this teacher, so a teacher that
# ignores its command makes every number underneath it meaningless. The upstream
# recipe has converged to a stationary policy before, which is why this gate
# exists rather than being assumed.
echo "[2/4] evaluating teacher"
"$JOSE_PY" eval_ppo_walk.py --task "$TEACHER_TASK" --checkpoint "$CKPT" \
    --output "$LOGS/teacher_eval.json" 2>&1 | tee "$LOGS/teacher_eval.log"
#
# One check is allowed to fail: "zero command does not lift feet repeatedly".
# The flat teacher behind every published number fails it too (2.245 / 2.245 /
# 2.242 lifts/s over seeds 42-44 in cmdfix_v1's privileged_teacher rows), as do
# the slope (1.506) and friction (2.199) teachers -- it marches in place when
# told to stand, which the recipe never penalised. It is a property of the
# recipe every variant shares, not a regression, so it is reported but does not
# stop the run. Every other check still does. Override with KNOWN_GATE_FAILURES
# (a "|"-separated list of check names; empty to allow none).
KNOWN_GATE_FAILURES="${KNOWN_GATE_FAILURES-zero command does not lift feet repeatedly}"
KNOWN_GATE_FAILURES="$KNOWN_GATE_FAILURES" "$JOSE_PY" - "$LOGS/teacher_eval.json" <<'EOF'
import json, os, sys
report = json.load(open(sys.argv[1]))
checks = report.get("sanity_checks", [])
known = {name for name in os.environ.get("KNOWN_GATE_FAILURES", "").split("|") if name}
failed = [c["name"] for c in checks if not c["passed"]]
blocking = [name for name in failed if name not in known]
print(f"sanity checks: {len(checks) - len(failed)}/{len(checks)} passed")
for check in checks:
    tag = "ok  " if check["passed"] else ("KNOWN" if check["name"] in known else "FAIL")
    print(f"  {tag} {check['name']}: {check['detail']}")
if not checks:
    raise SystemExit("teacher gate: no sanity checks in the report")
if blocking:
    raise SystemExit(f"teacher gate failed: {blocking}")
EOF

# --- 2b. pinned slope levels (slope_fixed only) ----------------------------
# The measurement is made once per teacher and reused on a rerun; the rule is
# applied to it afresh each time, so changing MAX_DEATH_RATE never re-measures.
if [ "$VARIANT" = "slope_fixed" ]; then
  LEVELS_JSON="$LOGS/teacher_levels.json"
  if [ -f "$LEVELS_JSON" ] && "$JOSE_PY" - "$LEVELS_JSON" "$CKPT" <<'EOF'
import json, os, sys
report = json.load(open(sys.argv[1]))
sys.exit(0 if os.path.realpath(report["teacher_checkpoint"]) == os.path.realpath(sys.argv[2]) else 1)
EOF
  then
    echo "[2b] reusing level measurement $LEVELS_JSON"
  else
    echo "[2b] measuring the teacher level by level"
    "$JOSE_PY" eval_slope_levels.py --teacher-checkpoint "$CKPT" --output "$LEVELS_JSON" \
        --max-death-rate "$MAX_DEATH_RATE" --headless \
        2>&1 | tee "$LOGS/teacher_levels.log"
  fi
  K=$("$JOSE_PY" - "$LEVELS_JSON" "$MAX_DEATH_RATE" <<'EOF'
import json, sys
from jose.ppo_walk.fixed_levels import select_max_level
report = json.load(open(sys.argv[1]))
for row in report["levels"]:
    print(f"    level {row['level']}: death_rate {row['death_rate']:.2f}%", file=sys.stderr)
print(select_max_level(report["levels"], float(sys.argv[2])))
EOF
)
  if [ "$K" -lt 0 ]; then
    echo "teacher exceeds MAX_DEATH_RATE=$MAX_DEATH_RATE% already on level 0; nothing to run" >&2
    exit 1
  fi
  CASE="locomotion_slope_fixed_l${K}"
  echo "    rule       death_rate <= $MAX_DEATH_RATE% on every level 0..K"
  echo "    levels     0..$K  -> case $CASE"
fi

# --- 3. the four-way comparison -------------------------------------------
echo "[3/4] method comparison (~5 h)"
"$JOSE_PY" run_method_comparison.py --case "$CASE" "$CKPT" --seeds $SEEDS \
    2>&1 | tee "$LOGS/method_comparison.log"

# --- 4. SET ----------------------------------------------------------------
echo "[4/4] SET baseline (~1 h)"
"$JOSE_PY" run_set_baseline.py --case "$CASE" "$CKPT" --seeds $SEEDS \
    2>&1 | tee "$LOGS/set_baseline.log"

echo "=== done. collect results with: VARIANT=$VARIANT bash scripts/collect_terrain_results.sh ==="
