from __future__ import annotations

import numpy as np
import pytest
import torch

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
