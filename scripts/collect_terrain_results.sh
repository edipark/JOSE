#!/usr/bin/env bash
# Pack a finished terrain study into one tarball to ship back.
#
# `logs/` is gitignored (.gitignore:6), so results cannot travel by git. What
# travels is small -- jsonl, reports, manifests, the driver logs, and the teacher
# checkpoint so the study can be re-evaluated on the other machine without
# retraining. The full logs/ tree is tens of GB and must never be copied
# wholesale; per-round estimator checkpoints and dataset caches are the bulk of it.
#
#   JOSE_ROOT  repository checkout (default: this file's git root)
#   VARIANT    friction | slope | push | slope_fixed | all  (default: all)
#   OUT        output directory                              (default: $HOME)
#
# Study directories are matched by their exact case directory
# (`locomotion_<variant>/`), not by substring: "slope" is a prefix of
# "slope_fixed", and a substring match would pack one variant's rows into the
# other's bundle.
set -euo pipefail

ROOT="${JOSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)}"
VARIANT="${VARIANT:-all}"
OUT="${OUT:-$HOME}"
STAMP=$(date +%Y-%m-%d_%H-%M-%S)
BUNDLE="$OUT/jose_terrain_${VARIANT}_${STAMP}"
cd "$ROOT"

case "$VARIANT" in
  friction|slope|push|slope_fixed) VARIANTS="$VARIANT" ;;
  all) VARIANTS=$(for v in friction slope push slope_fixed; do
                    [ -d "logs/jose_g1/terrain_$v" ] && echo -n "$v "; done) ;;
  *) echo "VARIANT must be friction, slope, push, slope_fixed or all" >&2; exit 2 ;;
esac
[ -n "$VARIANTS" ] || { echo "no terrain study found under logs/jose_g1/terrain_*" >&2; exit 1; }
mkdir -p "$BUNDLE"

# Provenance first. Without these the numbers cannot be tied to the code that
# produced them, which is the whole point of the fingerprint machinery.
{
  echo "collected_at   $(date -Is)"
  echo "host           $(hostname)"
  echo "git_head       $(git rev-parse HEAD)"
  echo "git_dirty      $(git diff --quiet && echo no || echo YES)"
  echo "variants       $VARIANTS"
  echo
  echo "--- git status --porcelain ---"
  git status --porcelain
  echo
  echo "--- versions ---"
  "${JOSE_PY:-python}" - <<'EOF' 2>/dev/null || echo "(JOSE_PY not set; versions omitted)"
import importlib.metadata as md
for name in ("isaacsim", "isaaclab", "isaaclab_tasks", "isaaclab_rl", "torch",
             "rsl-rl-lib", "skrl", "numpy", "gymnasium"):
    try:
        print(f"{name:16s} {md.version(name)}")
    except Exception:
        print(f"{name:16s} (not installed)")
EOF
} > "$BUNDLE/PROVENANCE.txt"

copy_results() {  # copy_results <source dir> <dest root>
  find "$1" \( -name 'results.jsonl' -o -name 'report.md' -o -name 'table.md' -o -name 'manifest.json' \
       -o -name 'training.json' -o -name 'config.json' -o -name 'summary.json' \) -print0 2>/dev/null |
    while IFS= read -r -d '' file; do
      target="$2/${file#logs/jose_g1/}"
      mkdir -p "$(dirname "$target")"
      cp "$file" "$target"
    done
}

for variant in $VARIANTS; do
  case "$variant" in
    friction)    EXPERIMENT="isaac_g1_ppo_walk_friction_jose_v0"; CASE_GLOB="locomotion_friction" ;;
    slope)       EXPERIMENT="isaac_g1_ppo_walk_slope_jose_v0";    CASE_GLOB="locomotion_slope" ;;
    push)        EXPERIMENT="isaac_g1_ppo_walk_push_jose_v0";     CASE_GLOB="locomotion_push" ;;
    slope_fixed) EXPERIMENT="isaac_g1_ppo_walk_slope_jose_v0";    CASE_GLOB="locomotion_slope_fixed_l*" ;;
  esac
  dest="$BUNDLE/$variant"
  mkdir -p "$dest"

  # Teacher: the final checkpoint and the dumped configs, not the intermediate
  # ones. params/ carries the env cfg source the run actually used.
  for run in "logs/rsl_rl/$EXPERIMENT"/*/; do
    [ -d "$run" ] || continue
    name=$(basename "$run")
    mkdir -p "$dest/teacher/$name"
    cp -r "$run/params" "$dest/teacher/$name/" 2>/dev/null || true
    latest=$(ls "$run"/model_*.pt 2>/dev/null | sort -V | tail -1 || true)
    [ -n "$latest" ] && cp "$latest" "$dest/teacher/$name/"
  done

  # Method comparison: logs/jose_g1/ablation/<teacher>/<case>/studies/...
  for study in logs/jose_g1/ablation/*/$CASE_GLOB; do
    [ -d "$study" ] && copy_results "$study" "$dest/studies"
  done
  # SET: logs/jose_g1/set_baseline/<run>/<case>/..., with the run's results.jsonl
  # and report at <run>/ -- so a run is taken whole when it holds this case.
  for run in logs/jose_g1/set_baseline/*/; do
    compgen -G "${run}${CASE_GLOB}" >/dev/null && copy_results "${run%/}" "$dest/studies"
  done

  cp -r "logs/jose_g1/terrain_${variant}" "$dest/driver_logs" 2>/dev/null || true
done

# Checksums, so a truncated transfer is detected rather than silently analysed.
( cd "$BUNDLE" && find . -type f ! -name SHA256SUMS -print0 | sort -z |
    xargs -0 sha256sum > SHA256SUMS )

tar -czf "$BUNDLE.tar.gz" -C "$(dirname "$BUNDLE")" "$(basename "$BUNDLE")"
rm -rf "$BUNDLE"
echo "wrote $BUNDLE.tar.gz  ($(du -h "$BUNDLE.tar.gz" | cut -f1))"
echo
echo "Send it with:"
echo "  rsync -avP $BUNDLE.tar.gz <user>@<host>:~/"
echo "On arrival, verify before analysing:"
echo "  tar -xzf $(basename "$BUNDLE.tar.gz") && cd $(basename "$BUNDLE") && sha256sum -c SHA256SUMS"
