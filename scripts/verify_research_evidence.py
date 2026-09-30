"""Verify published file bytes, input identity, and primary archived claims."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_SHA256 = "FF4B316B7D11C84B99E07B9AE0E32AFD03EEFD33698CF9DE3D02062D7F1902F3"


def verify(root: Path = ROOT) -> dict:
    evidence = root / "source_data_package" / "research_evidence"
    checked = 0
    for line in (evidence / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        actual = hashlib.sha256((evidence / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"evidence SHA-256 mismatch: {relative}")
        checked += 1
    for relative in ("data/argon/FHdata.xlsx", "FULL-REtry/data/FHdata.xlsx", "NewModel-TEST/data/FHdata.xlsx"):
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest().upper() == DATA_SHA256, relative
    selection_path = root / "source_data_package/output_results/main/fullscan/model_selection_table.csv"
    selection = pd.read_csv(selection_path).set_index("n_levels")
    assert round(selection.loc[1, "rmse_mean"], 5) == 0.08020
    assert round(selection.loc[4, "rmse_mean"], 5) == 0.07735
    selector = evidence / "neural_selector"
    receipt = json.loads((selector / "recalculation_receipt.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(selection_path.read_bytes()).hexdigest() == receipt["source_sha256"]
    scenarios = pd.read_csv(selector / "selector_scenarios.csv").set_index("scenario")
    assert scenarios.loc["production", "selected_k"] == 4
    assert scenarios.loc["remove_complexity_group", "selected_k"] == 8
    h4s = evidence / "h4s_holdout"
    units = pd.read_csv(h4s / "unit_status.csv")
    expected_units = {(float(vr), hypothesis, seed) for vr in (0, 4, 6, 8, 10)
                      for hypothesis in ("h1", "h1_background", "h4s", "h4s_background") for seed in (0, 1, 2)}
    assert len(units) == 60 and set(units.status) == {"complete"}
    assert set(zip(units.heldout_vr, units.hypothesis, units.seed)) == expected_units
    summary = pd.read_csv(h4s / "manuscript_tables/h4s_hypothesis_summary.csv").set_index("hypothesis")
    column = "nrmse_mean_of_fold_seed_medians"
    h1, h4 = float(summary.loc["h1", column]), float(summary.loc["h4s", column])
    assert round(h1, 5) == 0.12918 and round(h4, 5) == 0.12613
    transport = json.loads((evidence / "collision_history/v2l_experiment_summary.json").read_text(encoding="utf-8"))
    assert transport["h1"]["passed_checks"] == 15 and transport["h4s"]["posthoc_passed_checks"] == 12
    assert round(max(transport["h4s"]["channel_rate_fractions"]) * 100, 2) == 91.36
    bootstrap = pd.read_csv(root / "source_data_package/output_results/sensitivity/uncertainty/uncertainty_selection_summary.csv")
    counts = bootstrap[bootstrap.analysis == "residual_bootstrap"].set_index("selected_k").scenario_count.to_dict()
    assert counts == {5: 10, 6: 10, 7: 5, 8: 5}
    a15 = json.loads((evidence / "diagnostics_v2/a15_shift_equivalence/summary.json").read_text(encoding="utf-8"))["summary"]
    assert a15["identity_verified"] and a15["max_prediction_difference"] < a15["numerical_identity_tolerance"]
    return {"verified_files": checked, "dataset_sha256": DATA_SHA256, "h4s_complete_units": len(units),
            "h4s_mean_nrmse_improvement_percent": 100 * (h1 - h4) / h1,
            "residual_bootstrap_counts": counts, "A4": "NOT_EXECUTED"}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
