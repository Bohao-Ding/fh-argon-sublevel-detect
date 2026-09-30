"""A9 — Phase gradient versus period gradient (review M3 / Q4).

The per-band phases reported by A2 were measured under a common period prior
(11.55 V). A band whose true effective period differs from the prior acquires
an approximately constant phase offset under such a fit, so "phase rotates
with the retarding-voltage label" and "the effective period drifts with the
retarding-voltage label" are degenerate at this level. This experiment
estimates the best single-sinusoid period of each response band
independently, with moving-block bootstrap CIs, and tests whether the
per-band periods drift along the retarding-voltage label.

Interpretation (registered before running):
  - per-band periods flat within CI  -> phase-shift reading survives;
  - per-band periods drift monotonically -> restate the finding as systematic
    timing variation (phase and/or period); either way the rank-2 conclusion
    (bands are not one rescaled waveform) is unchanged.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from .common import load_pivot, timed, write_outputs

BAND_LABELS = ["Vr_0_4", "Vr_4_6", "Vr_6_8", "Vr_8_10", "Vr_gt_10"]
BAND_MIDPOINTS_V = np.array([2.0, 5.0, 7.0, 9.0, 11.5])
PERIOD_GRID = np.round(np.arange(8.0, 16.0 + 0.02, 0.02), 2)
BOOTSTRAP_REPLICATES = 400
BOOTSTRAP_BLOCK = 9


def _best_period(va: np.ndarray, series: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Best single-sinusoid period for one detrended band series."""
    total = float(np.sum(series**2))
    best_r2, best_period, best_coef = -np.inf, float("nan"), None
    for period in PERIOD_GRID:
        design = np.column_stack(
            [np.sin(2.0 * np.pi * va / period), np.cos(2.0 * np.pi * va / period)]
        )
        coef = np.linalg.lstsq(design, series, rcond=None)[0]
        residual = series - design @ coef
        r2 = 1.0 - float(np.sum(residual**2)) / total
        if r2 > best_r2:
            best_r2, best_period, best_coef = r2, float(period), coef
    return best_period, best_coef, best_r2


def run() -> dict:
    timings: dict[str, float] = {}
    with timed("total_seconds", timings):
        pivot, audit = load_pivot()
        from fh_retry.phase2 import _retarding_fractions

        va, fractions, _ = _retarding_fractions(pivot, [0, 4, 6, 8, 10])
        rng = np.random.default_rng(20260922)

        with timed("period_scan_seconds", timings):
            rows = []
            point_periods = []
            for band_index, band in enumerate(BAND_LABELS):
                series = fractions[:, band_index]
                trend_design = np.column_stack(
                    [((va - va.mean()) / (va.max() - va.min())) ** power for power in range(4)]
                )
                trend_coef = np.linalg.lstsq(trend_design, series, rcond=None)[0]
                detrended = series - trend_design @ trend_coef
                period, coef, r2 = _best_period(va, detrended)
                point_periods.append(period)
                # moving-block bootstrap of the residual around the best fit
                fit = (
                    np.column_stack(
                        [np.sin(2.0 * np.pi * va / period), np.cos(2.0 * np.pi * va / period)]
                    )
                    @ coef
                )
                residual = detrended - fit
                count = va.size
                nblocks = int(np.ceil(count / BOOTSTRAP_BLOCK))
                boot_periods = np.empty(BOOTSTRAP_REPLICATES)
                for replicate in range(BOOTSTRAP_REPLICATES):
                    starts = rng.integers(0, count, size=nblocks)
                    indices = (
                        (starts[:, None] + np.arange(BOOTSTRAP_BLOCK)[None, :]) % count
                    ).reshape(-1)[:count]
                    boot_detrended = detrended[indices]
                    boot_design = trend_design[indices]
                    boot_trend_coef = np.linalg.lstsq(boot_design, boot_detrended, rcond=None)[0]
                    boot_series = boot_detrended - boot_design @ boot_trend_coef
                    boot_periods[replicate] = _best_period(va[indices], boot_series)[0]
                rows.append(
                    {
                        "band": band,
                        "midpoint_V": float(BAND_MIDPOINTS_V[band_index]),
                        "best_period_V": period,
                        "period_ci_low_V": float(np.quantile(boot_periods, 0.025)),
                        "period_ci_high_V": float(np.quantile(boot_periods, 0.975)),
                        "single_sinusoid_r2": r2,
                    }
                )
            frame = pd.DataFrame(rows)

        with timed("gates_seconds", timings):
            periods = frame["best_period_V"].to_numpy()
            span = float(periods.max() - periods.min())
            flat = all(
                row["period_ci_low_V"] <= periods[0] <= row["period_ci_high_V"]
                for _, row in frame.iterrows()
            )
            slope = float(np.polyfit(frame["midpoint_V"], periods, 1)[0])
            gates = {
                "per_band_period_span_eV": span,
                "all_band_period_CIs_contain_first_band_period": bool(flat),
                "period_label_slope_V_per_V": slope,
                "period_gradient_present": bool(abs(slope) > 0.05 and not flat),
                "note": (
                    "period_grid resolution 0.02 V; bootstrap block 9 points; 400 replicates"
                ),
            }
            summary = {
                "experiment": "A9_band_periods",
                "dataset_sha256": audit["sha256"],
                "gates": gates,
                "gate_verdict": (
                    (
                        "PHASE READING SURVIVES: per-band effective periods are mutually consistent "
                        "within bootstrap CIs, so the band-to-band offsets are phase shifts under a "
                        "common timing scale, not per-band period drift"
                        if flat
                        else
                        "PERIOD READING REQUIRED: per-band effective periods drift along the "
                        "retarding-voltage label; the A2 finding must be restated as systematic timing "
                        "variation (phase and/or period), not phase rotation alone"
                    )
                    if flat
                    else "as above"
                ),
                "claim_boundary": [
                    "per-band periods are descriptive dominant scales of standardized difference bands",
                    "the retarding-voltage label cannot be separated from acquisition order (no scan metadata)",
                ],
                "timings": timings,
            }

    out_dir = write_outputs("a9_band_periods", {"band_periods": frame}, summary)
    return {"summary": summary, "out_dir": out_dir}


if __name__ == "__main__":
    result = run()
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    print(f"outputs -> {result['out_dir']}")
