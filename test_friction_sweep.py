"""CPU-only invariants for the friction queue and its reporting."""

from __future__ import annotations

import json

import pytest

from jose import report_friction_robustness as report
from jose import run_friction_sweep as sweep
from jose.check_friction_reproducibility import compare


def _row(method, friction, checkpoint_seed, eval_seed, track, survival=1.0, slide=0.0):
    return {
        "friction": {"effective": friction},
        "method": method,
        "checkpoint_seed": checkpoint_seed,
        "eval_seed": eval_seed,
        "metrics": {
            "track_error_norm": track,
            "grid_survival_rate": survival,
            "feet_slide_penalty": slide,
            "command_tracking": [{} for _ in range(sweep.COMMAND_COUNT)],
        },
    }


def test_default_matrix_has_273_unique_cells():
    keys = sweep.expected_keys(
        sweep.DEFAULT_FRICTIONS,
        sweep.PAPER_METHODS,
        sweep.DEFAULT_CHECKPOINT_SEEDS,
        sweep.DEFAULT_EVAL_SEEDS,
    )
    assert len(keys) == 273


def test_checkpoint_and_evaluation_seed_namespaces_are_distinct():
    assert set(sweep.DEFAULT_CHECKPOINT_SEEDS).isdisjoint(sweep.DEFAULT_EVAL_SEEDS)
    assert [sweep.command_seed_base(seed) for seed in sweep.DEFAULT_EVAL_SEEDS] == [
        1004200,
        1004300,
        1004400,
    ]


@pytest.mark.parametrize(
    "values",
    [(0.2, 0.2), (0.4, 0.3), (0.0, 1.0), (float("nan"), 1.0)],
)
def test_invalid_friction_grids_are_rejected(values):
    with pytest.raises(ValueError):
        sweep.validate_frictions(values)


def test_worker_validation_requires_every_method_seed_and_command():
    rows = [_row("teacher", 1.0, None, 10042, 0.05)]
    for method in sweep.PAPER_METHODS[1:]:
        rows.append(_row(method, 1.0, 42, 10042, 0.1))
    sweep._validate_worker_rows(rows, 1.0, 10042, sweep.PAPER_METHODS, (42,))
    rows[-1]["metrics"]["command_tracking"].pop()
    with pytest.raises(RuntimeError):
        sweep._validate_worker_rows(rows, 1.0, 10042, sweep.PAPER_METHODS, (42,))


def test_report_averages_eval_seeds_before_checkpoint_seeds():
    rows = [
        _row("teacher", 1.0, None, 10042, 1.0),
        _row("teacher", 1.0, None, 10043, 3.0),
        _row("jose", 1.0, 42, 10042, 2.0),
        _row("jose", 1.0, 42, 10043, 4.0),
        _row("jose", 1.0, 43, 10042, 10.0),
        _row("jose", 1.0, 43, 10043, 10.0),
    ]
    summary = report.aggregate(rows)["jose|1"]
    # checkpoint 42 -> 3, checkpoint 43 -> 10, then mean over checkpoints.
    assert summary["track"]["mean"] == pytest.approx(6.5)
    assert summary["track"]["min"] == pytest.approx(3.0)
    assert summary["track"]["max"] == pytest.approx(10.0)
    # Paired regrets: checkpoint 42 -> mean(1, 1), checkpoint 43 -> mean(9, 7).
    assert summary["track_regret"]["mean"] == pytest.approx(4.5)


def test_jsonl_reader_rejects_duplicate_cells(tmp_path):
    path = tmp_path / "results.jsonl"
    row = _row("teacher", 1.0, None, 10042, 0.05)
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Duplicate"):
        report.read_rows(path)


def test_successful_resume_clears_stale_failure_metadata(tmp_path):
    manifest = {"status": "failed", "error": "old failure", "updated_at": "earlier"}
    sweep._mark_manifest_complete(
        manifest,
        completed_cells=5,
        expected_cells=5,
        results_path=tmp_path / "results.jsonl",
        run_dir=tmp_path,
    )
    assert manifest["status"] == "complete"
    assert manifest["expected_cells"] == manifest["completed_cells"] == 5
    assert "error" not in manifest
    assert "updated_at" not in manifest


def test_reproducibility_comparison_is_order_independent(tmp_path):
    forward = tmp_path / "forward.jsonl"
    reverse = tmp_path / "reverse.jsonl"
    rows = [
        _row("teacher", 0.6, None, 10042, 0.05),
        _row("jose", 0.6, 42, 10042, 0.06),
    ]
    for row in rows:
        row["metrics"]["command_tracking"] = [{"measured": 1.0}]
    forward.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    reverse.write_text(
        "".join(json.dumps(row) + "\n" for row in reversed(rows)), encoding="utf-8"
    )
    assert compare(forward, reverse)["passed"]


def test_reproducibility_comparison_rejects_large_drift(tmp_path):
    forward = tmp_path / "forward.jsonl"
    reverse = tmp_path / "reverse.jsonl"
    left = _row("jose", 0.6, 42, 10042, 0.05)
    right = _row("jose", 0.6, 42, 10042, 0.06)
    left["metrics"]["command_tracking"] = [{"measured": 1.0}]
    right["metrics"]["command_tracking"] = [{"measured": 1.0}]
    forward.write_text(json.dumps(left) + "\n", encoding="utf-8")
    reverse.write_text(json.dumps(right) + "\n", encoding="utf-8")
    assert not compare(forward, reverse)["passed"]
