from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from sublevel_detect.validation_common import (
    ALLOWED_STAGE_STATUSES,
    RunIdentity,
    assert_reusable,
    canonical_hash,
    resolve_baseline,
    sha256_file,
    wilson_interval,
    write_progress,
    write_stage_status,
)


LOCAL_FILES = {
    "config": "main/fullscan/config_used.json",
    "decision": "main/fullscan/decision.json",
    "model_selection": "main/fullscan/model_selection_table.csv",
    "forward_evidence": "main/fullscan/forward_evidence.json",
    "selected_scorecard": "main/k_selected_full/scorecard.json",
    "selected_checkpoint": "main/k_selected_full/checkpoint_best.pt",
    "prediction_points": "main/k_selected_full/prediction_points.csv",
}

PACKAGE_FILES = {
    "config": "output_results/main/fullscan/config_used.json",
    "decision": "output_results/main/fullscan/decision.json",
    "model_selection": "output_results/main/fullscan/model_selection_table.csv",
    "forward_evidence": "output_results/main/fullscan/forward_evidence.json",
    "selected_scorecard": "run_records/k_selected_full/scorecard.json",
    "selected_checkpoint": "run_records/k_selected_full/checkpoint_best.pt",
    "prediction_points": "run_records/k_selected_full/prediction_points.csv",
    "channel_parameters": "manuscript_source_tables/channel_parameters.csv",
}


def _write_files(root: Path, files: dict[str, str]) -> Path:
    for key, relative in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{key}\n".encode("utf-8"))
    return root


def test_resolver_prefers_complete_same_output_baseline(tmp_path: Path) -> None:
    output_root = _write_files(tmp_path / "output", LOCAL_FILES)
    package_root = _write_files(tmp_path / "source_data_package", PACKAGE_FILES)

    baseline = resolve_baseline(output_root, package_root=package_root)

    assert baseline.kind == "local"
    assert baseline.root == output_root.resolve()
    assert set(baseline.files) == set(LOCAL_FILES)
    assert all(str(path).startswith(str(output_root.resolve())) for path in baseline.files.values())


def test_incomplete_local_baseline_falls_back_as_a_whole(tmp_path: Path) -> None:
    incomplete = dict(LOCAL_FILES)
    incomplete.pop("selected_checkpoint")
    output_root = _write_files(tmp_path / "output", incomplete)
    package_root = _write_files(tmp_path / "source_data_package", PACKAGE_FILES)

    baseline = resolve_baseline(output_root, package_root=package_root)

    assert baseline.kind == "package"
    assert set(baseline.files) == set(PACKAGE_FILES)
    assert all(str(path).startswith(str(package_root.resolve())) for path in baseline.files.values())


def test_resolver_reports_missing_packaged_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="selected_checkpoint"):
        resolve_baseline(tmp_path / "output", package_root=tmp_path / "source_data_package")


def test_hashes_are_uppercase_and_canonical_hash_is_key_order_independent(tmp_path: Path) -> None:
    path = tmp_path / "input.txt"
    path.write_text("argon", encoding="utf-8")

    assert sha256_file(path) == sha256_file(path).upper()
    assert len(sha256_file(path)) == 64
    assert canonical_hash({"b": 2, "a": [1, 3]}) == canonical_hash({"a": [1, 3], "b": 2})


def test_existing_manifest_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "experiment_manifest.json"
    identity = RunIdentity("abc", False, "base", "matrix", "seeds")
    path.write_text(json.dumps({"identity": asdict(identity)}), encoding="utf-8")

    assert_reusable(path, identity)
    with pytest.raises(ValueError, match="refusing to reuse"):
        assert_reusable(path, RunIdentity("def", False, "base", "matrix", "seeds"))


def test_stage_status_and_progress_enforce_frozen_vocabulary_and_denominator(tmp_path: Path) -> None:
    status_path = tmp_path / "stage_status.json"
    progress_path = tmp_path / "progress.json"

    assert ALLOWED_STAGE_STATUSES == {
        "not_started", "smoke_passed", "formal_running", "complete", "incomplete", "blocked"
    }
    write_stage_status(status_path, "incomplete", reason="one failed fit")
    write_progress(progress_path, completed=8, failed=2, total=10)

    assert json.loads(status_path.read_text(encoding="utf-8"))["status"] == "incomplete"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress == {"completed": 8, "failed": 2, "pending": 0, "total": 10}
    with pytest.raises(ValueError, match="Unsupported stage status"):
        write_stage_status(status_path, "success")
    with pytest.raises(ValueError, match="exceeds"):
        write_progress(progress_path, completed=9, failed=2, total=10)


def test_wilson_interval_counts_failures_in_the_denominator() -> None:
    low, high = wilson_interval(successes=8, total=10)
    optimistic_low, _ = wilson_interval(successes=8, total=8)

    assert 0.0 <= low <= 0.8 <= high <= 1.0
    assert low < optimistic_low
