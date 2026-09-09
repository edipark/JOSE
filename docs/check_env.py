#!/usr/bin/env python3
"""Compare this machine's environment against the reference the paper was produced on.

    python docs/check_env.py

Reads ``docs/reference_env_jose.txt`` and reports, per package, whether the local
version matches. It also runs the checks that a version table cannot cover: that
the JOSE package imports, that the terrain task ids register, and that the frozen
495-dimensional observation layout is intact.

Exit codes
----------
0  everything matches, or only advisory differences
1  a blocking difference -- something the study depends on is missing or the
   wrong major/minor version

What counts as blocking
-----------------------
Isaac Sim, Isaac Lab and its sub-packages, torch, rsl-rl and skrl decide how the
simulator steps and how the policy is loaded, so a mismatch there changes results
rather than merely annoying you. numpy, gymnasium and the rest are reported but
not fatal: the project already does not claim bitwise reproducibility across
machines (`torch.use_deterministic_algorithms` is never called and
`cudnn.deterministic` is False), so the goal here is to catch a *different stack*,
not to chase the last digit.
"""

from __future__ import annotations

import importlib.metadata as md
from pathlib import Path
import platform
import subprocess
import sys

BLOCKING = {
    "isaacsim", "isaaclab", "isaaclab_assets", "isaaclab_rl", "isaaclab_tasks",
    "torch", "rsl-rl-lib", "skrl",
}

ROOT = Path(__file__).resolve().parent.parent
REFERENCE = Path(__file__).resolve().parent / "reference_env_jose.txt"


def parse_reference(path: Path) -> dict[str, str]:
    versions, section = {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.rstrip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        if not line or line.startswith("#"):
            continue
        if section in ("packages", "isaacsim wheels"):
            name, _, version = line.partition(" ")
            versions[name.strip()] = version.strip()
    return versions


def local_version(name: str) -> str:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "MISSING"


def main() -> int:
    if not REFERENCE.exists():
        print(f"reference file not found: {REFERENCE}")
        return 1
    reference = parse_reference(REFERENCE)

    print("=" * 72)
    print("JOSE environment check")
    print("=" * 72)
    print(f"python    {platform.python_version()}")
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip()
        print(f"gpu       {gpu}")
    except Exception:
        print("gpu       (nvidia-smi unavailable)")
    print()

    print(f"{'package':22s} {'reference':16s} {'local':16s} verdict")
    print("-" * 72)
    blocking, advisory = [], []
    for name, want in reference.items():
        if want == "MISSING":
            continue
        got = local_version(name)
        if got == want:
            verdict = "ok"
        elif got == "MISSING":
            verdict = "MISSING"
        else:
            verdict = "differs"
        if verdict != "ok":
            (blocking if name in BLOCKING else advisory).append((name, want, got))
        print(f"{name:22s} {want:16s} {got:16s} {verdict}")

    print()
    print("-" * 72)
    print("functional checks")
    print("-" * 72)
    failures = []

    def check(label: str, fn) -> None:
        try:
            detail = fn()
            print(f"  ok    {label}{'  ' + detail if detail else ''}")
        except Exception as exc:                                   # noqa: BLE001
            print(f"  FAIL  {label}: {type(exc).__name__}: {exc}")
            failures.append(label)

    def imports_jose() -> str:
        sys.path.insert(0, str(ROOT.parent))
        import jose                                                # noqa: F401
        return f"from {ROOT}"

    def registers_terrain_tasks() -> str:
        import gymnasium as gym
        import jose.ppo_walk.terrain_tasks                          # noqa: F401
        ids = sorted(k for k in gym.registry if "Friction" in k or "Slope" in k)
        assert len(ids) == 4, f"expected 4 terrain task ids, found {ids}"
        return f"{len(ids)} ids"

    def observation_layout_frozen() -> str:
        from jose.schema import PPO_WALK_OBSERVATION_SCHEMA as s
        assert s.policy_dim == 495, s.policy_dim
        assert s.estimator_target_dim == 9, s.estimator_target_dim
        return "policy 495, target 9"

    def assets_present() -> str:
        usd = ROOT / "usd" / "g1_29dof_rev_1_0.usd"
        assert usd.exists(), f"missing {usd} -- copy the usd/ directory across"
        size_mb = sum(f.stat().st_size for f in (ROOT / "usd").rglob("*") if f.is_file()) / 1e6
        return f"usd/ {size_mb:.0f} MB"

    def worktree_clean() -> str:
        out = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                             capture_output=True, text=True).stdout.strip()
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
        assert not out, "worktree is dirty; commit before running a production study"
        return f"HEAD {head}"

    check("jose imports", imports_jose)
    check("terrain task ids register", registers_terrain_tasks)
    check("observation layout frozen", observation_layout_frozen)
    check("robot assets present", assets_present)
    check("git worktree clean", worktree_clean)

    print()
    print("=" * 72)
    if blocking:
        print(f"BLOCKING: {len(blocking)} package(s) differ on something the study depends on")
        for name, want, got in blocking:
            print(f"  {name}: reference {want}, local {got}")
        print("\nFix these before training. A different simulator or RL library does not")
        print("just shift the last digit -- it changes what the teacher learns.")
    if failures:
        print(f"BLOCKING: {len(failures)} functional check(s) failed: {', '.join(failures)}")
    if advisory:
        print(f"\nAdvisory ({len(advisory)} differ, not fatal):")
        for name, want, got in advisory:
            print(f"  {name}: reference {want}, local {got}")
        print("Record these in the results bundle so a later discrepancy is traceable.")
    if not blocking and not failures:
        print("PASS -- this environment matches the reference closely enough to train on.")
        print("\nNote: bitwise reproducibility across machines is not claimed and not")
        print("expected. Both teachers are trained here from scratch, and every method")
        print("is compared against the teacher trained beside it, so that is fine.")
    print("=" * 72)
    return 1 if (blocking or failures) else 0


if __name__ == "__main__":
    sys.exit(main())
