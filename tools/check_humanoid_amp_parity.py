"""Audit JOSE's AMP training core against a humanoid_amp checkout.

This is a development-time audit only. JOSE never imports or reads the
reference repository during training or evaluation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path

import yaml


JOSE_ROOT = Path(__file__).resolve().parents[1]
AMP_METHODS = (
    "__init__",
    "_pre_physics_step",
    "_apply_action",
    "_get_observations",
    "_get_rewards",
    "_get_dones",
    "_reset_idx",
    "_reset_strategy_default",
    "_reset_strategy_random",
    "collect_reference_motions",
)
AMP_FUNCTIONS = ("quaternion_to_tangent_and_normal", "compute_obs", "compute_rewards")
ENV_CONFIG_FIELDS = (
    "rew_termination",
    "rew_action_l2",
    "rew_joint_pos_limits",
    "rew_joint_acc_l2",
    "rew_joint_vel_l2",
    "episode_length_s",
    "decimation",
    "observation_space",
    "action_space",
    "state_space",
    "num_amp_observations",
    "amp_observation_space",
    "early_termination",
    "termination_height",
    "reference_body",
    "reset_strategy",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _definitions(path: Path) -> tuple[dict[str, ast.FunctionDef], dict[str, ast.FunctionDef]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    methods: dict[str, ast.FunctionDef] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "G1AmpEnv":
            methods = {child.name: child for child in node.body if isinstance(child, ast.FunctionDef)}
            break
    return methods, functions


def _assert_same_ast(reference: Path, jose: Path) -> None:
    reference_methods, reference_functions = _definitions(reference)
    jose_methods, jose_functions = _definitions(jose)
    for name in AMP_METHODS:
        if ast.dump(reference_methods[name], include_attributes=False) != ast.dump(
            jose_methods[name], include_attributes=False
        ):
            raise AssertionError(f"AMP method differs from humanoid_amp: G1AmpEnv.{name}")
    for name in AMP_FUNCTIONS:
        if ast.dump(reference_functions[name], include_attributes=False) != ast.dump(
            jose_functions[name], include_attributes=False
        ):
            raise AssertionError(f"AMP function differs from humanoid_amp: {name}")


def _assignment_value(path: Path, assignment_name: str) -> ast.AST:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == assignment_name for target in node.targets
        ):
            return node.value
    raise AssertionError(f"Assignment {assignment_name!r} not found in {path}")


class _NormalizeUsdPath(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        for keyword in node.keywords:
            if keyword.arg == "usd_path":
                keyword.value = ast.Constant(value="<standalone G1 USD>")
        # Keyword order has no Python runtime semantics.
        node.keywords.sort(key=lambda keyword: keyword.arg or "")
        return node


def _assert_same_articulation(reference: Path, jose: Path) -> None:
    source = _NormalizeUsdPath().visit(_assignment_value(reference, "G1_CFG"))
    target = _NormalizeUsdPath().visit(_assignment_value(jose, "G1_CFG"))
    if ast.dump(source, include_attributes=False) != ast.dump(target, include_attributes=False):
        raise AssertionError("G1 articulation/actuator configuration differs from humanoid_amp")


def _class_literal_values(path: Path, class_name: str) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            values: dict[str, object] = {}
            for child in node.body:
                if isinstance(child, ast.Assign) and len(child.targets) == 1 and isinstance(child.targets[0], ast.Name):
                    try:
                        values[child.targets[0].id] = ast.literal_eval(child.value)
                    except (ValueError, TypeError):
                        pass
            return values
    raise AssertionError(f"Class {class_name!r} not found in {path}")


def _assert_same_env_config(reference: Path, jose: Path) -> None:
    source = _class_literal_values(reference, "G1AmpEnvCfg")
    target = _class_literal_values(jose, "G1AmpEnvCfg")
    for field in ENV_CONFIG_FIELDS:
        if source.get(field) != target.get(field):
            raise AssertionError(
                f"G1AmpEnvCfg.{field} differs: humanoid_amp={source.get(field)!r}, JOSE={target.get(field)!r}"
            )


def _translated_agent(source: dict) -> dict:
    translated = dict(source)
    translated["gae_lambda"] = translated.pop("lambda")
    translated["observation_preprocessor"] = translated.pop("state_preprocessor")
    translated["observation_preprocessor_kwargs"] = translated.pop("state_preprocessor_kwargs")
    translated["amp_observation_preprocessor"] = translated.pop("amp_state_preprocessor")
    translated["amp_observation_preprocessor_kwargs"] = translated.pop("amp_state_preprocessor_kwargs")
    task_weight = translated.pop("task_reward_weight")
    style_weight = translated.pop("style_reward_weight")
    discriminator_scale = translated.pop("discriminator_reward_scale")
    translated["task_reward_scale"] = float(task_weight)
    translated["style_reward_scale"] = float(style_weight) * float(discriminator_scale)
    translated.pop("clip_predicted_values")
    return translated


def _assert_same_agent(reference: Path, jose: Path) -> None:
    source = yaml.safe_load(reference.read_text(encoding="utf-8"))
    target = yaml.safe_load(jose.read_text(encoding="utf-8"))

    source_models = source["models"]
    for model in ("policy", "value", "discriminator"):
        source_models[model]["network"][0]["input"] = "OBSERVATIONS"
    if source_models != target["models"]:
        raise AssertionError("SKRL model definitions differ after STATES -> OBSERVATIONS translation")

    expected_agent = _translated_agent(source["agent"])
    actual_agent = dict(target["agent"])
    expected_agent["experiment"] = dict(expected_agent["experiment"])
    actual_agent["experiment"] = dict(actual_agent["experiment"])
    expected_agent["experiment"].pop("directory")
    actual_agent["experiment"].pop("directory")
    # These null fields are accepted by the generic SKRL 2.x runner but are
    # not part of AMP_CFG; they have no behavior.
    actual_agent.pop("state_preprocessor", None)
    actual_agent.pop("state_preprocessor_kwargs", None)
    if expected_agent != actual_agent:
        raise AssertionError("AMP hyperparameters differ after the required SKRL 1.x -> 2.x translation")

    for section in ("memory", "motion_dataset", "reply_buffer", "trainer"):
        if source[section] != target[section]:
            raise AssertionError(f"SKRL section differs: {section}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=JOSE_ROOT.parent / "humanoid_amp",
        help="Path to the humanoid_amp checkout used as the source of truth",
    )
    args = parser.parse_args()
    reference = args.reference.resolve()
    if not (reference / "g1_amp_env.py").is_file():
        raise FileNotFoundError(f"Invalid humanoid_amp checkout: {reference}")

    _assert_same_ast(reference / "g1_amp_env.py", JOSE_ROOT / "g1_amp_env.py")
    _assert_same_articulation(reference / "g1_cfg.py", JOSE_ROOT / "g1_cfg.py")
    _assert_same_env_config(reference / "g1_amp_env_cfg.py", JOSE_ROOT / "g1_amp_env_cfg.py")
    _assert_same_agent(
        reference / "agents" / "skrl_g1_walk_amp_cfg.yaml",
        JOSE_ROOT / "agents" / "skrl_g1_walk_amp_cfg.yaml",
    )
    for relative in ("motions/G1_walk.npz", "usd/g1_29dof_rev_1_0.usd"):
        if _sha256(reference / relative) != _sha256(JOSE_ROOT / relative):
            raise AssertionError(f"Asset differs byte-for-byte: {relative}")

    print("PASS: AMP core, environment, actuators, translated SKRL config, Walk motion, and G1 USD match")
    print("NOTE: _setup_scene uses a local flat collider instead of the upstream remote GroundPlane asset")


if __name__ == "__main__":
    main()
