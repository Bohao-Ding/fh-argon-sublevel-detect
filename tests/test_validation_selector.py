from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sublevel_detect import model, validation_pipeline
from sublevel_detect.validation_common import Baseline, sha256_file
from sublevel_detect.validation_selector import (
    COMPONENTS,
    competition_ranks,
    rank_correlation,
    run,
    run_scenarios,
    selector_scenarios,
    selected_k_distribution,
)


EXPECTED_SCENARIOS = [
    "production",
    "leave_one_rmse_out",
    "leave_one_legacy_summary_out",
    "leave_one_d1_out",
    "leave_one_d2_out",
    "leave_one_structure_out",
    "leave_one_physical_out",
    "leave_one_bic_out",
    "leave_one_aic_out",
    "leave_one_degeneracy_out",
    "remove_fit_group",
    "remove_shape_group",
    "remove_physical_group",
    "remove_complexity_group",
    "fit_complexity_only",
]


def _rows() -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for k in range(1, 9):
        rows.append(
            {
                "n_levels": k,
                "rmse_mean": float(abs(k - 4)),
                "cv_rmse_mean": float(abs(k - 4)),
                "d1_rmse_mean": float(abs(k - 8)),
                "d2_rmse_mean": float(abs(k - 8)),
                "structure_score": float(abs(k - 8)),
                "vr_physical_response_score": float(abs(k - 4)),
                "bic": float(abs(k - 4)),
                "aic": float(abs(k - 4)),
                "degeneracy_penalty": 1.0 if k in {1, 4, 6} else 0.0,
            }
        )
    return rows


def test_selector_scenarios_are_exactly_preregistered() -> None:
    assert [scenario.name for scenario in selector_scenarios()] == EXPECTED_SCENARIOS


def test_scenarios_do_not_mutate_or_renormalize_production_weights() -> None:
    original = dict(model.DEFAULT_SELECTOR_RANK_WEIGHTS)
    scenarios = {scenario.name: scenario for scenario in selector_scenarios()}

    assert model.DEFAULT_SELECTOR_RANK_WEIGHTS == original
    assert scenarios["leave_one_rmse_out"].weights["rmse"] == 0.0
    assert scenarios["leave_one_rmse_out"].weights["structure"] == 1.25
    assert scenarios["remove_fit_group"].weights["rmse"] == 0.0
    assert scenarios["remove_fit_group"].weights["summary"] == 0.0
    assert scenarios["fit_complexity_only"].weights == {
        "rmse": 1.0,
        "summary": 0.0,
        "d1": 0.0,
        "d2": 0.0,
        "structure": 0.0,
        "physical": 0.0,
        "bic": 1.0,
        "aic": 0.5,
        "degeneracy": 1.0,
    }


def test_competition_ranks_use_tolerance_and_place_nonfinite_last() -> None:
    ranks = competition_ranks([1.0, 1.0 + 5e-13, 2.0, np.nan, np.inf])

    assert ranks == [1.0, 1.0, 3.0, 4.0, 4.0]


def test_composite_tie_selects_lower_k() -> None:
    rows = _rows()[:2]
    for row in rows:
        for key in (
            "rmse_mean",
            "cv_rmse_mean",
            "d1_rmse_mean",
            "d2_rmse_mean",
            "structure_score",
            "vr_physical_response_score",
            "bic",
            "aic",
            "degeneracy_penalty",
        ):
            row[key] = 1.0

    result = run_scenarios(rows)

    assert all(row["selected_k"] == 1 for row in result)


def test_rank_correlation_has_symmetric_nine_by_nine_schema() -> None:
    correlations = rank_correlation(_rows())

    assert len(COMPONENTS) == 9
    assert len(correlations) == 81
    by_pair = {(row["component_x"], row["component_y"]): row["spearman_rho"] for row in correlations}
    for component in COMPONENTS:
        assert by_pair[(component, component)] == pytest.approx(1.0)
    for left in COMPONENTS:
        for right in COMPONENTS:
            left_right = by_pair[(left, right)]
            right_left = by_pair[(right, left)]
            if np.isnan(left_right):
                assert np.isnan(right_left)
            else:
                assert left_right == pytest.approx(right_left)


def test_scenario_and_distribution_totals_are_preserved() -> None:
    decisions = run_scenarios(_rows())
    distribution = selected_k_distribution(decisions)

    assert len(decisions) == 15
    assert sum(row["scenario_count"] for row in distribution) == 15
    assert sum(row["scenario_fraction"] for row in distribution) == pytest.approx(1.0)


def test_selector_stage_writes_complete_artifact_family(tmp_path: Path) -> None:
    selection_path = tmp_path / "model_selection_table.csv"
    pd.DataFrame(_rows()).to_csv(selection_path, index=False)
    baseline = Baseline(
        kind="local",
        root=tmp_path,
        files={"model_selection": selection_path},
        hashes={"model_selection": sha256_file(selection_path)},
    )
    output_dir = tmp_path / "validation" / "selector_audit"

    summary = run(baseline=baseline, output_dir=output_dir, mode="fullscan")

    assert summary["ok"] is True
    assert summary["scenario_count"] == 15
    assert summary["status"] == "complete"
    for name in (
        "selector_scenarios.csv",
        "rank_correlation.csv",
        "selected_k_distribution.csv",
        "selector_summary.json",
        "stage_status.json",
    ):
        assert (output_dir / name).is_file()


def test_selector_only_pipeline_resolves_package_and_writes_root_manifest(tmp_path: Path) -> None:
    package_root = tmp_path / "source_data_package"
    package_files = {
        "output_results/main/fullscan/config_used.json": "{}\n",
        "output_results/main/fullscan/decision.json": "{}\n",
        "output_results/main/fullscan/forward_evidence.json": "{}\n",
        "run_records/k_selected_full/scorecard.json": "{}\n",
        "run_records/k_selected_full/checkpoint_best.pt": "checkpoint\n",
        "run_records/k_selected_full/prediction_points.csv": "curve_id,residual\n1,0\n",
        "manuscript_source_tables/channel_parameters.csv": "k,channel,energy_v,weight\n4,1,11.5,1\n",
    }
    for relative, content in package_files.items():
        path = package_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    selection = package_root / "output_results/main/fullscan/model_selection_table.csv"
    selection.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(_rows()).to_csv(selection, index=False)
    output_root = tmp_path / "output"

    result = validation_pipeline.run(
        mode="fullscan",
        input_path=tmp_path / "unused.xlsx",
        output_root=output_root,
        device="cpu",
        requested_stage="selector",
        package_root=package_root,
    )

    assert result["ok"] is True
    assert result["baseline_kind"] == "package"
    assert (output_root / "validation" / "experiment_manifest.json").is_file()
    assert (output_root / "validation" / "progress.json").is_file()
    assert (output_root / "validation" / "selector_audit" / "selector_summary.json").is_file()
