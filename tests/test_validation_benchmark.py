from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from sublevel_detect import validation_benchmark
from sublevel_detect.validation_benchmark import (
    BenchmarkUnit,
    BudgetMLP,
    benchmark_units,
    design_matrix,
    fit_standardizer,
    transform,
)


def test_budget_mlp_has_exactly_45_parameters_and_nonnegative_output() -> None:
    network = BudgetMLP()
    output = network(torch.zeros(3, 2))

    assert sum(parameter.numel() for parameter in network.parameters()) == 45
    assert tuple(output.shape) == (3, 1)
    assert torch.all(output >= 0.0)


def test_standardizer_uses_training_fold_only() -> None:
    training = np.array([[0.0, 0.0], [2.0, 4.0]])
    test = np.array([[100.0, 100.0]])

    standardizer = fit_standardizer(training)
    transformed = transform(test, standardizer)

    assert standardizer.mean == pytest.approx((1.0, 2.0))
    assert standardizer.scale == pytest.approx((1.0, 2.0))
    assert transformed[0, 0] == pytest.approx(99.0)
    assert transformed[0, 1] == pytest.approx(49.0)


def test_design_matrix_contains_only_va_and_vr() -> None:
    curve = {
        "curve_id": 99,
        "curve_idx": 7,
        "Va": np.array([0.0, 0.5, 1.0]),
        "Vr": 4.0,
        "Vr_is_vector": False,
        "Ip": np.array([1.0, 2.0, 3.0]),
    }

    features, target = design_matrix([curve])

    assert features.shape == (3, 2)
    np.testing.assert_allclose(features, [[0.0, 4.0], [0.5, 4.0], [1.0, 4.0]])
    assert target.tolist() == pytest.approx([1.0, 2.0, 3.0])


def test_benchmark_matrix_matches_holdout_folds_and_seeds() -> None:
    units = benchmark_units("fullscan")

    assert len(units) == 15
    assert {unit.heldout_vr for unit in units} == {0.0, 4.0, 6.0, 8.0, 10.0}
    assert {unit.seed for unit in units} == {0, 1, 2}
    assert benchmark_units("smoke") == [BenchmarkUnit(heldout_vr=0.0, seed=0)]


def test_smoke_runner_writes_fixed_budget_metrics(tmp_path) -> None:
    rows = []
    for curve_id, vr in enumerate((0.0, 4.0, 6.0, 8.0, 10.0), start=1):
        for index in range(161):
            rows.append(
                {
                    "curve_id": curve_id,
                    "Vr": vr,
                    "Va": index * 0.5,
                    "IuA": 0.02 * index + 0.01 * vr,
                }
            )
    input_path = tmp_path / "input.csv"
    pd.DataFrame(rows).to_csv(input_path, index=False)
    holdout_dir = tmp_path / "validation" / "holdout"
    holdout_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "heldout_vr": 0.0,
                "seed": 0,
                "model_label": "selected_k",
                "rmse": 1.0,
                "mae": 0.8,
                "nrmse": 0.3,
            }
        ]
    ).to_csv(holdout_dir / "zero_shot_metrics.csv", index=False)
    output_dir = tmp_path / "validation" / "benchmark"

    result = validation_benchmark.run(
        mode="smoke",
        input_path=input_path,
        holdout_dir=holdout_dir,
        output_dir=output_dir,
        validation_root=output_dir.parent,
        device="cpu",
    )

    assert result["ok"] is True
    assert result["status"] == "smoke_passed"
    assert result["parameter_count"] == 45
    for name in (
        "mlp_config.json",
        "fold_metrics.csv",
        "predictions.csv",
        "paired_comparison.csv",
        "benchmark_summary.json",
        "stage_status.json",
    ):
        assert (output_dir / name).is_file()
