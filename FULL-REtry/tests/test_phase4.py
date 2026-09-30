from pathlib import Path

import numpy as np

from fh_retry.analysis import load_and_audit
from fh_retry.phase4 import (
    REFERENCE_LAG_V,
    apply_voltage_lag,
    make_equilibrium_pivot,
    run_phase4,
)


DATA = Path(__file__).resolve().parents[1] / "data" / "FHdata.xlsx"


def _pivot():
    _, pivot, _ = load_and_audit(DATA)
    return pivot


def test_linear_vr2_is_exact_midpoint_and_has_six_levels() -> None:
    pivot = _pivot()
    equilibrium, levels = make_equilibrium_pivot(pivot, "linear_vr2")
    assert levels == [0, 2, 4, 6, 8, 10]
    assert np.allclose(equilibrium[2], 0.5 * (pivot[0] + pivot[4]))


def test_zero_lag_keeps_equilibrium_and_directions_identical() -> None:
    equilibrium, _ = make_equilibrium_pivot(_pivot(), "linear_vr2")
    forward = apply_voltage_lag(equilibrium, 0.0, "forward")
    reverse = apply_voltage_lag(equilibrium, 0.0, "reverse")
    assert np.array_equal(forward.to_numpy(), equilibrium.to_numpy())
    assert np.array_equal(reverse.to_numpy(), equilibrium.to_numpy())


def test_positive_lag_creates_finite_direction_difference() -> None:
    equilibrium, _ = make_equilibrium_pivot(_pivot(), "linear_vr2")
    forward = apply_voltage_lag(equilibrium, REFERENCE_LAG_V, "forward")
    reverse = apply_voltage_lag(equilibrium, REFERENCE_LAG_V, "reverse")
    assert np.isfinite(forward.to_numpy()).all()
    assert np.isfinite(reverse.to_numpy()).all()
    assert np.max(np.abs(forward.to_numpy() - reverse.to_numpy())) > 0.0


def test_reference_output_contains_exactly_twelve_curves() -> None:
    result = run_phase4(DATA, workers=1, lag_grid_v=(0.0, REFERENCE_LAG_V))
    reference = result["reference_12_curves"]
    assert reference.groupby(["direction", "Vr"]).ngroups == 12
    assert len(reference) == 12 * 161
    assert result["analysis_summary"]["strict_G2_pass"].dtype == bool
    linear = result["partition_diagnostic"][
        result["partition_diagnostic"]["design"] == "linear_vr2"
    ]
    assert np.allclose(linear["D02_D24_correlation"], 1.0, atol=1e-12)
    assert linear["second_singular_variance_fraction"].max() < 1e-28
    measured = result["measured_partition_summary"]
    assert measured["levels"].nunique() == 7
    assert len(measured) == 7 * 4
    full = measured[
        (measured["levels"] == "0-4-6-8-10")
        & (measured["representation"] == "raw_fraction")
    ].iloc[0]
    assert full["improved_blocks"] == 5
    assert np.isclose(full["median_gain"], 0.42145969694115953)


def test_threaded_and_serial_summaries_match() -> None:
    serial = run_phase4(DATA, workers=1, lag_grid_v=(0.0, REFERENCE_LAG_V))
    threaded = run_phase4(DATA, workers=2, lag_grid_v=(0.0, REFERENCE_LAG_V))
    left = serial["analysis_summary"].drop(columns=[]).reset_index(drop=True)
    right = threaded["analysis_summary"].drop(columns=[]).reset_index(drop=True)
    assert left.equals(right)
