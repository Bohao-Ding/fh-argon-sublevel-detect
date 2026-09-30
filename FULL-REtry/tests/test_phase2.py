from pathlib import Path

import numpy as np

from fh_retry.analysis import load_and_audit
from fh_retry.phase2 import (
    NIST_4S_EV,
    _analysis_arrays,
    _operator_cv,
    _template_basis,
    _template_operators,
    endpoint_exclusion,
    retarding_order_audit,
)


DATA = Path(__file__).resolve().parents[1] / "data" / "FHdata.xlsx"


def _pivot():
    return load_and_audit(DATA)[1]


def test_physical_retarding_order_is_unique_best() -> None:
    audit = retarding_order_audit(_pivot())
    best = audit.iloc[0]
    assert best["order"] == "0-4-6-8-10"
    assert best["violations"] == 1
    assert audit.iloc[1]["violations"] >= 100


def test_nist_harmonic_design_is_ill_conditioned() -> None:
    va, _ = _analysis_arrays(_pivot())
    _, singular = _template_basis(va, NIST_4S_EV)
    assert singular[0] / singular[-1] > 1e4


def test_template_operator_returns_five_finite_block_errors() -> None:
    va, response = _analysis_arrays(_pivot())
    q, _ = _template_basis(va, NIST_4S_EV)
    _, operators = _template_operators(va, q)
    mean_error, errors = _operator_cv(response, operators)
    assert len(errors) == 5
    assert np.isfinite(mean_error)
    assert np.all(np.asarray(errors) > 0)


def test_rank2_result_is_not_created_only_by_vr10() -> None:
    _, summary = endpoint_exclusion(_pivot())
    without_vr10 = summary[summary["levels"] == "0-4-6-8"].iloc[0]
    assert without_vr10["improved_blocks"] >= 4
    assert without_vr10["median_gain"] >= 0.10
