from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from sublevel_detect import validation_holdout
from sublevel_detect.validation_common import Baseline, sha256_file
from sublevel_detect.validation_holdout import (
    HoldoutUnit,
    SparseNuisance,
    calibration_split,
    fit_sparse_nuisance,
    holdout_units,
    prediction_metrics,
    split_vr_fold,
)


def _curves() -> list[dict]:
    return [
        {
            "curve_id": index + 1,
            "curve_idx": index,
            "Va": np.arange(161, dtype=np.float32) * 0.5,
            "Vr": float(vr),
            "Vr_is_vector": False,
            "Ip": np.linspace(0.0, 1.0 + index, 161, dtype=np.float32),
        }
        for index, vr in enumerate((0.0, 4.0, 6.0, 8.0, 10.0))
    ]


def test_fold_training_excludes_heldout_vr() -> None:
    training, heldout = split_vr_fold(_curves(), heldout_vr=6.0)

    assert {float(curve["Vr"]) for curve in training} == {0.0, 4.0, 8.0, 10.0}
    assert len(heldout) == 1
    assert float(heldout[0]["Vr"]) == 6.0
    assert [curve["curve_idx"] for curve in training] == [0, 1, 2, 3]


def test_calibration_indices_are_fixed_and_disjoint() -> None:
    calibration, evaluation = calibration_split(161)

    assert calibration.tolist() == list(range(0, 161, 8))
    assert len(calibration) == 21
    assert len(evaluation) == 140
    assert not set(calibration).intersection(evaluation)


def test_formal_holdout_matrix_matches_five_folds_eight_k_three_seeds() -> None:
    units = holdout_units("fullscan")

    assert len(units) == 120
    assert {unit.heldout_vr for unit in units} == {0.0, 4.0, 6.0, 8.0, 10.0}
    assert {unit.n_levels for unit in units} == set(range(1, 9))
    assert {unit.seed for unit in units} == {0, 1, 2}
    assert holdout_units("smoke") == [
        HoldoutUnit(heldout_vr=0.0, n_levels=1, seed=0),
        HoldoutUnit(heldout_vr=0.0, n_levels=2, seed=0),
    ]


def test_prediction_metrics_use_full_heldout_range_for_both_subsets() -> None:
    full_observed = np.linspace(0.0, 10.0, 161)
    observed = np.array([2.0, 4.0, 6.0])
    predicted = np.array([1.0, 5.0, 8.0])

    metrics = prediction_metrics(observed, predicted, denominator_observed=full_observed)

    expected_rmse = np.sqrt((1.0 + 1.0 + 4.0) / 3.0)
    assert metrics["rmse"] == pytest.approx(expected_rmse)
    assert metrics["mae"] == pytest.approx(4.0 / 3.0)
    assert metrics["nrmse"] == pytest.approx(expected_rmse / 10.0)


def test_sparse_calibration_optimizes_only_three_nuisance_parameters() -> None:
    va = torch.linspace(0.0, 10.0, 21)
    target = 1.05 * (va + 0.4) + 0.03

    def frozen_core(shifted_va: torch.Tensor) -> torch.Tensor:
        return shifted_va

    result = fit_sparse_nuisance(
        core_prediction=frozen_core,
        va=va,
        target=target,
        epochs=300,
        learning_rate=0.05,
    )

    assert isinstance(result["module"], SparseNuisance)
    assert sum(parameter.numel() for parameter in result["module"].parameters()) == 3
    assert set(result["values"]) == {"gain", "bias", "delta_va"}
    assert result["loss"] < 1e-4


def test_smoke_runner_uses_excluded_fold_and_writes_prediction_outputs(monkeypatch, tmp_path) -> None:
    rows = []
    for curve_id, vr in enumerate((0.0, 4.0, 6.0, 8.0, 10.0), start=1):
        for index in range(161):
            rows.append(
                {
                    "curve_id": curve_id,
                    "Vr": vr,
                    "Va": index * 0.5,
                    "IuA": float(index) / 100.0 + vr / 100.0,
                }
            )
    input_path = tmp_path / "input.csv"
    pd.DataFrame(rows).to_csv(input_path, index=False)
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    forward_path = tmp_path / "forward.json"
    forward_path.write_text("{}\n", encoding="utf-8")
    baseline = Baseline(
        kind="local",
        root=tmp_path,
        files={"config": config_path, "forward_evidence": forward_path},
        hashes={"config": sha256_file(config_path), "forward_evidence": sha256_file(forward_path)},
    )
    configs = []

    def fake_scan(cfg):
        configs.append(cfg)
        return {"decision": {"selected_k": 2}, "scan_dir": cfg.out_dir}

    def fake_evaluate(**kwargs):
        unit = kwargs["unit"]
        return {
            "zero_metrics": {"rmse": 1.0, "mae": 0.8, "nrmse": 0.5, "n_points": 161},
            "calibrated_metrics": {"rmse": 0.7, "mae": 0.6, "nrmse": 0.35, "n_points": 140},
            "zero_predictions": [{"index": 0, "observed": 1.0, "predicted": 0.0}],
            "calibrated_predictions": [{"index": 1, "observed": 1.0, "predicted": 0.3}],
            "nuisance": {"gain": 1.0, "bias": 0.0, "delta_va": 0.0},
            "unit": unit,
        }

    monkeypatch.setattr(validation_holdout.model, "run_level_scan", fake_scan)
    monkeypatch.setattr(validation_holdout, "evaluate_candidate", fake_evaluate)
    output_dir = tmp_path / "validation" / "holdout"

    result = validation_holdout.run(
        mode="smoke",
        input_path=input_path,
        baseline=baseline,
        output_dir=output_dir,
        validation_root=output_dir.parent,
        device="cpu",
    )

    assert result["ok"] is True
    assert result["status"] == "smoke_passed"
    assert len(configs) == 1
    assert configs[0].exclude_vr_values == "0.0"
    assert configs[0].scan_seeds == "0"
    assert configs[0].level_scan_min == 1
    assert configs[0].level_scan_max == 2
    for name in (
        "fold_manifest.csv",
        "candidate_status.csv",
        "zero_shot_metrics.csv",
        "calibrated_metrics.csv",
        "zero_shot_predictions.csv",
        "calibrated_predictions.csv",
        "holdout_summary.json",
        "stage_status.json",
    ):
        assert (output_dir / name).is_file()
