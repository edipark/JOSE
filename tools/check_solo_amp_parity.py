"""Audit JOSE's shared AMP objective and assets against the current SOLO checkout.

The audit is development-only. JOSE never reads SOLO during training, play,
estimation, or deployment. JOSE's TWIST-style action mapping, default pose,
and PD gains are intentional differences and are excluded from this audit.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path

import yaml


JOSE_ROOT = Path(__file__).resolve().parents[1]
SOLO_TASK_RELATIVE = Path("source/isaaclab_tasks/isaaclab_tasks/direct/SOLO")
ENV_METHODS = (
    "_setup_scene",
    "_pre_physics_step",
    "_current_amp_observation",
    "get_estimator_target",
    "get_estimator_joint_state",
    "_get_observations",
    "_get_rewards",
    "_get_dones",
    "_log_completed_episode_metrics",
    "_reset_idx",
    "_sample_motion_reset",
    "collect_reference_motions",
)
ENV_FUNCTIONS = ("quaternion_to_tangent_and_normal", "compute_amp_observation")
TASK_MATH_FUNCTIONS = (
    "inject_observation_estimate",
    "reference_history_times",
)
TASK_MATH_CONSTANTS = (
    "PHYSICS_DT",
    "CONTROL_DECIMATION",
    "POLICY_DT",
    "EPISODE_LENGTH_S",
    "AMP_HISTORY_STEPS",
    "WALK_TARGET_VELOCITY",
)


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"Class {name!r} was not found")


def _functions(nodes: list[ast.stmt]) -> dict[str, ast.FunctionDef]:
    return {node.name: node for node in nodes if isinstance(node, ast.FunctionDef)}


def _assignments(nodes: list[ast.stmt]) -> dict[str, ast.AST]:
    result = {}
    for node in nodes:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            result[node.targets[0].id] = node.value
    return result


def _same_ast(left: ast.AST, right: ast.AST) -> bool:
    return ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False)


def _assert_env(reference: Path, target: Path) -> None:
    source_tree, target_tree = _tree(reference), _tree(target)
    source_methods = _functions(_class(source_tree, "G1AmpEnv").body)
    target_methods = _functions(_class(target_tree, "G1AmpEnv").body)
    for name in ENV_METHODS:
        if not _same_ast(source_methods[name], target_methods[name]):
            raise AssertionError(f"SOLO AMP method differs: G1AmpEnv.{name}")
    source_functions, target_functions = _functions(source_tree.body), _functions(target_tree.body)
    for name in ENV_FUNCTIONS:
        if not _same_ast(source_functions[name], target_functions[name]):
            raise AssertionError(f"SOLO AMP function differs: {name}")


def _assert_task_math(reference: Path, target: Path) -> None:
    source_tree, target_tree = _tree(reference), _tree(target)
    source_assignments, target_assignments = _assignments(source_tree.body), _assignments(target_tree.body)
    for name in TASK_MATH_CONSTANTS:
        if not _same_ast(source_assignments[name], target_assignments[name]):
            raise AssertionError(f"SOLO timing constant differs: {name}")
    source_functions, target_functions = _functions(source_tree.body), _functions(target_tree.body)
    for name in TASK_MATH_FUNCTIONS:
        if not _same_ast(source_functions[name], target_functions[name]):
            raise AssertionError(f"SOLO shared helper differs: {name}")


def _assert_env_cfg(reference: Path, target: Path) -> None:
    source_tree, target_tree = _tree(reference), _tree(target)
    for class_name in ("G1AmpEnvCfg", "G1AmpWalkEnvCfg", "G1AmpDanceEnvCfg"):
        source = _assignments(_class(source_tree, class_name).body)
        actual = _assignments(_class(target_tree, class_name).body)
        for name, value in source.items():
            if name not in actual or not _same_ast(value, actual[name]):
                raise AssertionError(f"SOLO environment config differs: {class_name}.{name}")


def _assert_agent(reference: Path, target: Path) -> None:
    source = yaml.safe_load(reference.read_text(encoding="utf-8"))
    actual = yaml.safe_load(target.read_text(encoding="utf-8"))
    source["agent"]["experiment"]["directory"] = "<package log directory>"
    actual["agent"]["experiment"]["directory"] = "<package log directory>"
    if source != actual:
        raise AssertionError(f"SOLO SKRL agent configuration differs: {reference.name}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=JOSE_ROOT.parent / "SOLO")
    args = parser.parse_args()
    task = args.reference.resolve() / SOLO_TASK_RELATIVE
    if not (task / "g1_amp_env.py").is_file():
        raise FileNotFoundError(f"Invalid SOLO checkout: {args.reference}")

    _assert_env(task / "g1_amp_env.py", JOSE_ROOT / "g1_amp_env.py")
    _assert_task_math(task / "task_math.py", JOSE_ROOT / "task_math.py")
    _assert_env_cfg(task / "g1_amp_env_cfg.py", JOSE_ROOT / "g1_amp_env_cfg.py")
    for task_name in ("walk", "dance"):
        name = f"skrl_g1_{task_name}_amp_cfg.yaml"
        _assert_agent(task / "agents" / name, JOSE_ROOT / "agents" / name)
    for relative in (
        "motions/G1_walk.npz",
        "motions/G1_dance.npz",
        "assets/g1_29dof_rev_1_0.usd",
    ):
        target_relative = relative.replace("assets/", "usd/")
        if _sha256(task / relative) != _sha256(JOSE_ROOT / target_relative):
            raise AssertionError(f"SOLO asset differs byte-for-byte: {relative}")

    print("PASS: SOLO AMP objective, timing, rewards, agents, motions, and USD match")
    print("NOTE: TWIST-style action mapping, default pose, and PD gains are intentional JOSE differences")
    print("NOTE: JOSE uses the same GroundPlane implementation and material as SOLO")
    print("NOTE: JOSE additionally registers Jump and exposes read-only estimator/distillation sensors")


if __name__ == "__main__":
    main()
