"""Known-answer and integration tests for EXPERIMENT_V2 (复查层).

These tests validate the NEW diagnostic code against synthetic data with known
truth (T1-T4, T6), the frozen phase-3 numbers (T7), and the injection
machinery end-to-end on the real archived dataset (T5). The frozen
`fh_retry` implementations themselves are NOT re-tested here.
Run: python -m pytest EXPERIMENT_V2/tests -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
if str(PACKAGE / "src") not in sys.path:
    sys.path.insert(0, str(PACKAGE / "src"))

from exp_v2.common import (  # noqa: E402
    BLOCK_DEFINITIONS,
    load_pivot,
    split_first_blocks,
    steady_blocks,
)


def _synthetic_va() -> np.ndarray:
    return np.arange(22.5, 80.0, 0.5)


def _band_response(va: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    time_axis = 2.0 * np.pi * va / 11.55
    scales = np.array([1.0, 0.8, 0.6, -0.7, -0.9])
    base = np.sin(time_axis + 0.3)[:, None] * scales[None, :]
    response = base
    if kind == "rank2":
        second = np.sin(time_axis + 1.5)[:, None] * np.array([0.6, -0.4, 0.5, -0.3, 0.2])[None, :]
        response = response + second
    trend = 0.4 * (va / 80.0)[:, None] ** 3
    return 0.5 + trend + response + 0.02 * rng.standard_normal((va.size, 5))


def _fold_errors(va: np.ndarray, response: np.ndarray) -> tuple[list[float], list[float]]:
    from fh_retry.analysis import BLOCKS, _predict_fold

    rank1_errors, rank2_errors = [], []
    for low, high in BLOCKS:
        test = (va >= low) & (va < high)
        train = ~test
        _, rank1_z, observed_z = _predict_fold(va, response, train, test, period=11.55, rank=1, trend_degree=3)
        _, rank2_z, _ = _predict_fold(va, response, train, test, period=11.55, rank=2, trend_degree=3)
        rank1_errors.append(float(np.sqrt(np.mean((rank1_z - observed_z) ** 2))))
        rank2_errors.append(float(np.sqrt(np.mean((rank2_z - observed_z) ** 2))))
    return rank1_errors, rank2_errors


def test_t1_rank_selection_direction_on_synthetic():
    rng = np.random.default_rng(1)
    va = _synthetic_va()
    rank1_e, rank2_e = _fold_errors(va, _band_response(va, "rank1", rng))
    assert np.mean(rank1_e) <= np.mean(rank2_e) * 1.02, "rank-1 data should not favor rank-2"
    rank1_e, rank2_e = _fold_errors(va, _band_response(va, "rank2", rng))
    assert np.mean(rank2_e) < 0.95 * np.mean(rank1_e), "rank-2 data must favor rank-2 clearly"


def test_t2_single_frequency_selection():
    from fh_retry.phase2 import _operator_cv, _template_basis, _template_operators

    va = _synthetic_va()
    rng = np.random.default_rng(2)
    true_energy = 4.886
    trend = 0.3 + 0.4 * (va / 80.0)
    response = (trend + 0.5 * np.sin(2.0 * np.pi * va / true_energy)
                + 0.01 * rng.standard_normal(va.size))[:, None]
    grid = np.array([4.0, 4.4, 4.6, 4.886, 5.1, 5.4, 6.0])
    errors = []
    for energy in grid:
        q, _ = _template_basis(va, np.array([energy]))
        _, operators = _template_operators(va, q)
        mean_error, _ = _operator_cv(response, operators)
        errors.append(mean_error)
    assert int(np.argmin(errors)) == int(np.flatnonzero(np.isclose(grid, true_energy))[0])


def test_t3_block_definitions_partition():
    pivot, _ = load_pivot()
    va = pivot.index.to_numpy(float)
    va = va[va >= 15.0]
    expected_union = {
        "original_5": (22.5, 80.0),
        "steady_4": (34.0, 80.0),
        "split_first_5": (28.25, 80.0),
    }
    for name, builder in BLOCK_DEFINITIONS.items():
        blocks = builder()
        masks = [(va >= low) & (va < high) for low, high in blocks]
        stacked = np.stack(masks)
        assert not (stacked.sum(axis=0) > 1).any(), f"{name}: blocks overlap"
        covered = stacked.any(axis=0)
        low_expected, high_expected = expected_union[name]
        target = (va >= low_expected) & (va < high_expected)
        assert (covered == target).all(), f"{name}: union of blocks != expected range"
        # all full-period blocks are 11.5 V wide except the split_first half-block
        widths = [high - low for low, high in blocks]
        expected_widths = [5.75] + [11.5] * (len(blocks) - 1) if name == "split_first_5" else [11.5] * len(blocks)
        assert widths == pytest.approx(expected_widths)


def test_t4_phase_extraction_roundtrip():
    from exp_v2.a2_endpoint_mechanism import _phases

    rng = np.random.default_rng(4)
    true_phase = rng.uniform(-np.pi, np.pi, 5)
    true_amplitude = rng.uniform(0.2, 2.0, 5)
    coef = np.vstack([true_amplitude * np.cos(true_phase), true_amplitude * np.sin(true_phase)])
    amplitude, phase = _phases(coef)
    assert np.allclose(amplitude, true_amplitude)
    assert np.allclose(np.angle(np.exp(1j * (phase - true_phase))), 0.0, atol=1e-10)


def test_t5_argmin_pathology_and_detection_instrument():
    from fh_retry.phase3 import (
        INJECTION_FACTORS,
        _batch_candidate_scores,
        _injection_setup,
        _moving_block_batch,
    )

    pivot, _ = load_pivot()
    setup = _injection_setup(pivot)
    candidates = setup["candidates"]
    factors = np.array([item["factor"] for item in candidates])
    rng = np.random.default_rng(5)
    length = 5

    def simulate(signal_index, replicates, scale=1.0):
        noise = _moving_block_batch(setup["common_residual"], length, replicates, rng)
        base = setup["trend_fit"][None, :, :] + noise
        if signal_index is None:
            return base
        return base + scale * setup["signals"][signal_index][None, :, :]

    # (a) pathology: with NO signal the most flexible candidate wins the argmin
    null_scores = _batch_candidate_scores(simulate(None, 200), candidates)
    marginal = np.bincount(np.argmin(null_scores, axis=1), minlength=len(candidates)) / 200
    assert marginal.max() >= 0.5 and factors[int(np.argmax(marginal))] == 12.0

    # (b) detection instrument: per-template null q05 threshold keeps FP near 5%
    threshold = np.quantile(null_scores, 0.05, axis=0)
    fresh = _batch_candidate_scores(simulate(None, 200), candidates)
    fp = (fresh < threshold[None, :]).mean(axis=0)
    assert fp.max() <= 0.15, "false-positive rates must stay near the nominal 5%"

    # (c) power must increase with SNR for the widest-span truth
    true_index = int(np.flatnonzero(np.isclose(factors, 12.0))[0])
    low = float((_batch_candidate_scores(simulate(true_index, 150, scale=0.5), candidates) < threshold[None, :])[:, true_index].mean())
    high = float((_batch_candidate_scores(simulate(true_index, 150, scale=6.0), candidates) < threshold[None, :])[:, true_index].mean())
    assert high > low, "detection power must be monotone in injected SNR"


def test_t6_logistic_threshold_recovery():
    from exp_v2.a5_injection_debias import _fit_threshold50

    x = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0])
    true_x50, slope = 2.5, 1.2
    rates = 1.0 / (1.0 + np.exp(-(np.log(x) - np.log(true_x50)) / slope))
    fit = _fit_threshold50(x, rates)
    assert fit["fit_ok"]
    assert fit["x50"] == pytest.approx(true_x50, rel=0.15)


def test_t7_a3_baseline_replicates_frozen_phase3_numbers():
    from exp_v2.a3_composition_truncation import _variant_responses

    pivot, audit = load_pivot()
    variants, _ = _variant_responses(pivot, audit["current_quantization_uA"])
    from fh_retry.phase3 import _rank_detail

    clr = _rank_detail(_va15(pivot), variants["v0_baseline_delta0p5"]["clr"])
    ilr = _rank_detail(_va15(pivot), variants["v0_baseline_delta0p5"]["ilr"])
    assert clr["relative_gain"].min() == pytest.approx(-0.1425, abs=1e-3)
    assert ilr["relative_gain"].min() == pytest.approx(-0.0908, abs=1e-3)


def _va15(pivot) -> np.ndarray:
    from fh_retry.phase2 import _retarding_fractions

    va, _, _ = _retarding_fractions(pivot, [0, 4, 6, 8, 10])
    return va


def test_t8_dataset_receipt_locked():
    pivot, audit = load_pivot()
    assert audit["sha256"] == "FF4B316B7D11C84B99E07B9AE0E32AFD03EEFD33698CF9DE3D02062D7F1902F3"
    assert pivot.shape == (161, 5)


def test_t9_single_energy_recovery_at_realistic_noise():
    """Distinct-frequency templates must stay argmin-identifiable at archived noise."""
    from fh_retry.phase2 import _template_basis, _template_operators, _trend
    from fh_retry.phase3 import (
        _analysis_arrays,
        _batch_candidate_scores,
        _injection_setup,
        _moving_block_batch,
    )

    pivot, _ = load_pivot()
    setup = _injection_setup(pivot)
    va = setup["va"]
    raw = _analysis_arrays(pivot)[1]
    response = (raw - raw.mean(axis=0)) / raw.std(axis=0, ddof=1)
    trend = _trend(va)
    detrended = response - trend @ np.linalg.lstsq(trend, response, rcond=None)[0]
    target_rms = setup["target_signal_rms"]
    energies = np.array([4.886 * f for f in (0.8, 0.9, 1.0, 1.1, 1.25)])
    true_index = 2
    candidates = []
    for energy in energies:
        q, _ = _template_basis(va, np.array([energy]))
        signal = q @ (q.T @ detrended)
        rms = float(np.sqrt(np.mean(signal**2)))
        _, operators = _template_operators(va, q)
        candidates.append({"operators": operators, "signal": signal * (target_rms / rms)})
    rng = np.random.default_rng(9)
    noise = _moving_block_batch(setup["common_residual"], 5, 150, rng)
    simulated = setup["trend_fit"][None, :, :] + candidates[true_index]["signal"][None, :, :] + noise
    scores = _batch_candidate_scores(simulated, candidates)
    win = float((np.argmin(scores, axis=1) == true_index).mean())
    assert win >= 0.6, f"distinct single frequencies should be selectable at teaching noise (win={win:.2f})"


def test_t10_determinism_and_env_receipt():
    """The program's core claim: closed-form code is deterministic. Assert it."""
    import importlib
    import json

    from exp_v2.common import sha256_frame

    a1 = importlib.import_module("exp_v2.a1_block1_diagnosis")
    r1 = a1.run()
    r2 = a1.run()
    assert sha256_frame(r1["grid"]) == sha256_frame(r2["grid"]), "outputs not bit-identical"

    summary = json.loads((r1["out_dir"] / "summary.json").read_text(encoding="utf-8"))
    assert "environment" in summary and "numpy" in summary["environment"]
