"""Teacher-scoped catalog primitives for JOSE estimator ablations.

Human-facing runs are dated studies. Reuse is decided independently from the
date by canonical job specifications stored below the teacher checkpoint that
produced the rollout policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable


CATALOG_FORMAT_VERSION = 1


# Rounds every study but the DAgger ablation uses; also the slug's implicit value.
DEFAULT_DAGGER_ROUNDS = 10


@dataclass(frozen=True)
class AblationExperiment:
    estimator: str
    window: int
    joint_preset: str
    dagger_rounds: int = DEFAULT_DAGGER_ROUNDS

    @property
    def slug(self) -> str:
        model = self.estimator.lower()
        slug = f"{model}_w{self.window:02d}_{self.joint_preset}"
        # Only non-default round counts are spelled out, so every slug that
        # existed before the DAgger ablation keeps its name -- results already
        # in the catalog stay addressable and reports still group them together.
        # Without the suffix, rounds=0 and rounds=10 would share a slug and be
        # averaged into one row despite being different experiments.
        return slug if self.dagger_rounds == DEFAULT_DAGGER_ROUNDS else f"{slug}_r{self.dagger_rounds:02d}"

    def to_runner_tuple(self) -> tuple[str, str, int, str, int]:
        return self.slug, self.estimator, self.window, self.joint_preset, self.dagger_rounds


TEACHER_EXPERIMENT = AblationExperiment("TEACHER", 1, "all", 0)
ARCHITECTURE_EXPERIMENTS = (
    AblationExperiment("LSTM", 25, "all"),
    AblationExperiment("TCN", 25, "all"),
    AblationExperiment("HISTORY_MLP", 25, "all"),
)
JOINT_SCOPE_EXPERIMENTS = (
    AblationExperiment("LSTM", 25, "all"),
    AblationExperiment("LSTM", 25, "legs"),
    AblationExperiment("LSTM", 25, "upper"),
)
# How much the on-policy DAgger phase is worth over the supervised warm start
# alone. rounds=0 stops after the warm start, so its metrics.rounds holds just
# the round-0 record -- that arm is the "no DAgger" baseline.
DAGGER_EXPERIMENTS = (
    AblationExperiment("LSTM", 25, "all", 10),
    AblationExperiment("LSTM", 25, "all", 5),
    AblationExperiment("LSTM", 25, "all", 0),
)

# Task registry shared by every ablation/comparison entry point: task key ->
# (Gym task id, estimator adapter, default agent config entry point).
TASKS = {
    "amp_walk": ("Isaac-G1-AMP-Walk-JOSE-Direct-v0", "amp", "skrl_amp_cfg_entry_point"),
    "amp_dance": ("Isaac-G1-AMP-Dance-JOSE-Direct-v0", "amp", "skrl_amp_cfg_entry_point"),
    "amp_jump": ("Isaac-G1-AMP-Jump-JOSE-Direct-v0", "amp", "skrl_amp_cfg_entry_point"),
    # Manager-based walk teacher trained with rsl-rl; same estimator methodology.
    # (The SKRL Direct PPO walk task it replaced was removed.)
    "ppo_walk": (
        "Isaac-G1-PPO-Walk-Estimator-JOSE-v0",
        "ppo_walk",
        "rsl_rl_cfg_entry_point",
    ),
}

TRAINING_IMPLEMENTATION = (
    "train_state_estimator.py",
    "estimator/pipeline.py",
    "estimator/adapters.py",
    "estimator/models.py",
    "schema.py",
    "skrl_compat.py",
    "teacher_setup.py",
    "g1_amp_env.py",
    "g1_amp_env_cfg.py",
    "task_math.py",
)
TEACHER_IMPLEMENTATION = (
    "evaluate_teacher.py",
    "estimator/adapters.py",
    "schema.py",
    "skrl_compat.py",
    "teacher_setup.py",
    "g1_amp_env.py",
    "g1_amp_env_cfg.py",
    "task_math.py",
)
TASK_IMPLEMENTATION = {
    "amp_walk": ("__init__.py", "g1_cfg.py", "motions/motion_loader.py", "motions/G1_walk.npz", "agents/skrl_g1_walk_amp_cfg.yaml"),
    "amp_dance": ("__init__.py", "g1_cfg.py", "motions/motion_loader.py", "motions/G1_dance.npz", "agents/skrl_g1_dance_amp_cfg.yaml"),
    "amp_jump": ("__init__.py", "g1_cfg.py", "motions/motion_loader.py", "motions/G1_jump.npz", "agents/skrl_g1_jump_amp_cfg.yaml"),
    "ppo_walk": (
        "__init__.py",
        "ppo_walk/g1_asset.py",
        "ppo_walk/walk_env_cfg.py",
        "ppo_walk/walk_estimator_env_cfg.py",
        "ppo_walk/walk_estimator_env.py",
        "ppo_walk/agents/rsl_rl_ppo_cfg.py",
        "ppo_walk/mdp/rewards.py",
    ),
}


def window_experiments(windows: Iterable[int]) -> tuple[AblationExperiment, ...]:
    return tuple(AblationExperiment("LSTM", window, "all") for window in windows)


def normalize_experiments(
    experiments: Iterable[AblationExperiment | tuple[str, str, int, str, int]],
) -> tuple[AblationExperiment, ...]:
    normalized = []
    for item in experiments:
        if isinstance(item, AblationExperiment):
            normalized.append(item)
        else:
            _, estimator, window, preset, dagger_rounds = item
            normalized.append(AblationExperiment(estimator, window, preset, dagger_rounds))
    return tuple(normalized)


def canonical_digest(payload: dict, length: int = 16) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return value or "checkpoint"


def teacher_display_id(checkpoint: str | Path) -> str:
    checkpoint = Path(checkpoint)
    run_name = checkpoint.parent.parent.name if checkpoint.parent.name == "checkpoints" else checkpoint.parent.name
    return _slug(f"{run_name}__{checkpoint.stem}")


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


class TeacherCatalog:
    """Filesystem catalog rooted at one teacher checkpoint's content identity."""

    def __init__(self, output_root: Path, teacher_root: Path, teacher: dict):
        self.output_root = output_root
        self.teacher_root = teacher_root
        self.teacher = teacher

    @classmethod
    def open(
        cls,
        output_root: str | Path,
        checkpoint: str | Path,
        fingerprint: dict,
        *,
        create: bool,
    ) -> "TeacherCatalog":
        output_root = Path(output_root).resolve()
        sha256 = fingerprint.get("sha256")
        if output_root.exists() and sha256:
            for manifest in sorted(output_root.glob("*/teacher.json")):
                try:
                    existing = json.loads(manifest.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if existing.get("fingerprint", {}).get("sha256") == sha256:
                    return cls(output_root, manifest.parent, existing)

        display_id = teacher_display_id(checkpoint)
        teacher_root = output_root / display_id
        manifest_path = teacher_root / "teacher.json"
        if manifest_path.exists():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {}
            existing_sha = existing.get("fingerprint", {}).get("sha256")
            if sha256 and (not existing_sha or existing_sha != sha256):
                teacher_root = output_root / f"{display_id}__{sha256[:8]}"
                manifest_path = teacher_root / "teacher.json"

        teacher = {
            "catalog_format_version": CATALOG_FORMAT_VERSION,
            "teacher_id": teacher_root.name,
            "checkpoint": str(Path(checkpoint).resolve()),
            "fingerprint": fingerprint,
            "registered_at": datetime.now().isoformat(),
        }
        if create:
            teacher_root.mkdir(parents=True, exist_ok=True)
            if not manifest_path.exists():
                _atomic_json(manifest_path, teacher)
            else:
                teacher = json.loads(manifest_path.read_text(encoding="utf-8"))
        return cls(output_root, teacher_root, teacher)

    def task_root(self, task: str) -> Path:
        return self.teacher_root / _slug(task)

    def catalog_root(self, task: str) -> Path:
        return self.task_root(task) / "catalog"

    def entry_path(
        self,
        task: str,
        seed: int,
        experiment: str,
        spec_digest: str,
        *,
        estimator: str | None = None,
        window: int | None = None,
        joint_preset: str | None = None,
    ) -> Path:
        if estimator is None:
            hierarchy = Path("jobs") / _slug(experiment) / f"seed_{seed}"
        else:
            if window is None or joint_preset is None:
                raise ValueError("Estimator catalog entries require window and joint_preset")
            hierarchy = (
                Path("estimators")
                / _slug(estimator.lower())
                / f"window_{window}"
                / f"joints_{_slug(joint_preset)}"
                / f"seed_{seed}"
            )
        return self.catalog_root(task) / hierarchy / "variants" / spec_digest

    def dataset_path(
        self, task: str, seed: int, window: int, joint_preset: str, spec_digest: str
    ) -> Path:
        return (
            self.catalog_root(task)
            / "datasets"
            / f"seed_{seed}"
            / f"window_{window}"
            / f"joints_{_slug(joint_preset)}"
            / f"{spec_digest}.pt"
        )

    def study_path(self, task: str, study: str, run_id: str) -> Path:
        return self.task_root(task) / "studies" / _slug(study) / run_id

    def next_run_id(self, task: str, study: str) -> str:
        base = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        parent = self.task_root(task) / "studies" / _slug(study)
        candidate = base
        suffix = 1
        while (parent / candidate).exists():
            candidate = f"{base}_{suffix:02d}"
            suffix += 1
        return candidate

    @staticmethod
    def read_complete(entry: Path, *, require_checkpoint: bool) -> dict | None:
        current = entry / "current.json"
        if not current.is_file():
            return None
        try:
            payload = json.loads(current.read_text(encoding="utf-8"))
            record = payload["record"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            return None
        artifact = Path(record.get("artifact") or "")
        if record.get("status") != "ok" or not artifact.is_file():
            return None
        try:
            json.loads(artifact.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(record.get("metrics"), dict):
            return None
        checkpoint = artifact.parent / "best_estimator.pt"
        if require_checkpoint and (not checkpoint.is_file() or checkpoint.stat().st_size == 0):
            return None
        return record

    @staticmethod
    def write_attempt(entry: Path, run_id: str, record: dict, *, make_current: bool) -> Path:
        attempt = entry / "attempts" / run_id
        result_path = attempt / "result.json"
        _atomic_json(result_path, record)
        if make_current:
            _atomic_json(
                entry / "current.json",
                {
                    "catalog_format_version": CATALOG_FORMAT_VERSION,
                    "updated_at": datetime.now().isoformat(),
                    "result": str(result_path),
                    "record": record,
                },
            )
        return attempt


def write_json(path: Path, payload: dict) -> None:
    _atomic_json(path, payload)
