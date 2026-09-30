from pathlib import Path

import numpy as np

from fh_retry.analysis import load_and_audit
from fh_retry.phase2 import _full_rank1_fit
from fh_retry.phase3 import (
    GEOMETRY_REPRESENTATIONS,
    _injection_setup,
    _rank_detail,
    _rank_summary,
    _run_tasks,
    _stable_seed,
    _surrogate_worker,
    composition_representations,
    endpoint_gate_consistency,
    shifted_blocks,
)


DATA = Path(__file__).resolve().parents[1] / "data" / "FHdata.xlsx"


def _inputs():
    _, pivot, audit = load_and_audit(DATA)
    va, transforms, transform_audit = composition_representations(
        pivot, audit["current_quantization_uA"]
    )
    return pivot, va, transforms, transform_audit


def test_composition_transforms_are_finite_and_have_expected_dimensions() -> None:
    _, _, transforms, audit = _inputs()
    assert transforms["raw_fraction"].shape[1] == 5
    assert transforms["hellinger"].shape[1] == 5
    assert transforms["clr"].shape[1] == 5
    assert transforms["ilr"].shape[1] == 4
    assert np.allclose(transforms["clr"].sum(axis=1), 0.0, atol=1e-10)
    assert all(np.isfinite(value).all() for value in transforms.values())
    assert audit["finite"].all()


def test_shifted_blocks_keep_five_nonempty_disjoint_tests() -> None:
    _, va, _, _ = _inputs()
    for shift in (-0.50, -0.25, 0.0):
        masks = [(va >= low) & (va < high) for low, high in shifted_blocks(shift)]
        assert len(masks) == 5
        assert all(mask.sum() >= 22 for mask in masks)
        assert np.max(np.sum(np.column_stack(masks), axis=1)) == 1


def test_endpoint_recheck_applies_original_worst_block_rule() -> None:
    pivot, _, _, _ = _inputs()
    frame = endpoint_gate_consistency(pivot)
    without_vr10 = frame[frame["levels"] == "0-4-6-8"].iloc[0]
    assert without_vr10["improved_blocks"] == 4
    assert without_vr10["median_gain"] >= 0.10
    assert without_vr10["worst_gain"] < -0.05
    assert not bool(without_vr10["passes_original_G2_full_rule"])


def test_geometry_aware_rank_metrics_are_finite() -> None:
    _, va, transforms, _ = _inputs()
    for representation in GEOMETRY_REPRESENTATIONS:
        detail = _rank_detail(va, transforms[representation])
        summary = _rank_summary(detail)
        assert len(detail) == 5
        assert np.isfinite(detail[["rank1_rmse", "rank2_rmse", "relative_gain"]]).all().all()
        assert 0 <= summary["improved_blocks"] <= 5


def test_injection_signals_have_equal_rms_and_candidate_independent_noise() -> None:
    pivot, _, _, _ = _inputs()
    setup = _injection_setup(pivot)
    rms = [np.sqrt(np.mean(signal**2)) for signal in setup["signals"]]
    assert np.allclose(rms, setup["target_signal_rms"], rtol=1e-10, atol=1e-12)
    assert setup["common_residual"].shape == setup["signals"][0].shape
    assert np.isfinite(setup["common_residual"]).all()


def test_parallel_surrogate_tasks_match_serial_results() -> None:
    _, va, transforms, _ = _inputs()
    response = transforms["hellinger"]
    detail = _rank_detail(va, response)
    fitted, residual = _full_rank1_fit(va, response)
    observed = _rank_summary(detail)
    tasks = []
    for length in (3, 5):
        tasks.append(
            {
                "representation": "hellinger",
                "block_length": length,
                "replicates": 4,
                "seed": _stable_seed(7, f"test:{length}"),
                "va": va,
                "fitted": fitted,
                "residual": residual,
                "observed": observed,
            }
        )
    serial = _run_tasks(_surrogate_worker, tasks, workers=1)
    parallel = _run_tasks(_surrogate_worker, tasks, workers=2)
    for left, right in zip(serial, parallel, strict=True):
        assert left["samples"].drop(columns=[]).equals(right["samples"].drop(columns=[]))
        for key in (
            "observed_improved_blocks",
            "observed_median_gain",
            "observed_worst_gain",
            "surrogate_median_gain_q95",
            "surrogate_median_gain_q99",
            "conditional_p",
        ):
            assert left["summary"][key] == right["summary"][key]
