from pathlib import Path

import numpy as np

from fh_retry.analysis import (
    EXPECTED_SHA256,
    load_and_audit,
    make_retarding_bands,
    modal_analysis,
    run_numeric_analysis,
)


DATA = Path(__file__).resolve().parents[1] / "data" / "FHdata.xlsx"


def test_authority_dataset_and_retarding_identity() -> None:
    _, pivot, audit = load_and_audit(DATA)
    _, fractions, _, band_audit = make_retarding_bands(pivot)
    assert audit["sha256"] == EXPECTED_SHA256
    assert audit["monotonic_violations"] == 1
    assert np.isclose(audit["max_monotonic_violation_uA"], 0.001)
    assert np.isclose(band_audit["total_clipping_correction_uA"], 0.001)
    assert np.allclose(fractions.sum(axis=1), 1.0)


def test_modal_analysis_is_deterministic() -> None:
    _, pivot, _ = load_and_audit(DATA)
    va, fractions, _, _ = make_retarding_bands(pivot)
    first = modal_analysis(va, fractions)
    second = modal_analysis(va, fractions)
    assert first[0] == second[0]
    assert np.allclose(first[2].to_numpy(float), second[2].to_numpy(float))


def test_minimal_decision_is_internally_consistent() -> None:
    result = run_numeric_analysis(DATA)
    decision = result["decision"]
    assert decision["phase_a_positive_screen"] == all(decision["gates"].values())
    assert decision["cv_metrics"]["rank2_better_blocks"] <= 5
