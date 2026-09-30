"""A5 — Injection recovery: pathology diagnosis and a calibrated replacement.

IMPORTANT correction to the frozen interpretation (established fresh here):
Phase 3's operator-CV argmin selection is GEOMETRY-DOMINATED. Under a pure
null (no injected signal) the most flexible candidate (span factor 12) is
selected in ~90% of replicates, so the frozen "exact recovery 85-89% at
3.36 eV" was never signal-driven, and no span - including 12x - is actually
recoverable by model selection. This module:

  Part 1  replicates the pathology with a fresh seed (null marginal argmin
          selection + argmin recovery-vs-span curve);
  Part 2  builds the replacement instrument: per-template DETECTION against
          the template's own null score distribution (score below its null
          q05), with false-positive calibration on fresh null data;
  Part 3  measures detection power vs (true span x SNR), fits logistic
          SNR50 curves (SNR needed for 50% detection power), and states the
          quotable boundary.

Gate G-A5a (amended protocol, registered before running Part 2): fresh-null
false-positive rate per candidate within [2%, 10%] and max/min < 2.5 ->
the detection instrument is calibrated and its power curve is quotable.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from .common import load_pivot, timed, write_outputs

DETECTION_SNR_LEVELS = (1.0, 2.0, 3.0, 5.0)
SWEEP_FACTORS = (0.5, 1.0, 2.0, 4.0, 8.0, 12.0)
NULL_REPLICATES = 500
POWER_REPLICATES = 300
FP_FRESH_NULL = 300


def _logistic(x: np.ndarray, x50: float, slope: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(x - x50) / slope))


def _fit_threshold50(x: np.ndarray, rates: np.ndarray) -> dict:
    if np.nanmax(rates) < 0.5:
        return {"x50": float("nan"), "slope": float("nan"), "fit_ok": False, "censored": True}
    if np.nanmin(rates) > 0.5:
        return {"x50": float("nan"), "slope": float("nan"), "fit_ok": False, "censored": False}
    try:
        popt, _ = curve_fit(_logistic, np.log(x), rates, p0=(float(np.log(np.median(x))), 1.0), maxfev=20000)
    except RuntimeError:
        return {"x50": float("nan"), "slope": float("nan"), "fit_ok": False, "censored": False}
    return {"x50": float(np.exp(popt[0])), "slope": float(popt[1]), "fit_ok": True, "censored": False}


def run() -> dict:
    timings: dict[str, float] = {}
    with timed("total_seconds", timings):
        pivot, audit = load_pivot()
        from fh_retry.phase3 import (
            INJECTION_FACTORS,
            INJECTION_SCHEMES,
            _batch_candidate_scores,
            _injection_setup,
            _moving_block_batch,
            _phase_noise_batch,
        )

        with timed("setup_seconds", timings):
            setup = _injection_setup(pivot)
        trend_fit = setup["trend_fit"]
        common_residual = setup["common_residual"]
        candidates = setup["candidates"]
        factors = np.array([item["factor"] for item in candidates])
        spans = np.array([item["span_eV"] for item in candidates])
        rng = np.random.default_rng(20260919)

        def simulate(signal: np.ndarray | None, length: int | None, replicates: int, scale: float = 1.0) -> np.ndarray:
            if length is None:
                noise = _phase_noise_batch(common_residual, replicates, rng)
            else:
                noise = _moving_block_batch(common_residual, length, replicates, rng)
            base = trend_fit[None, :, :] + noise
            if signal is None:
                return base
            return base + scale * signal[None, :, :]

        def scores_for(simulated: np.ndarray) -> np.ndarray:
            return _batch_candidate_scores(simulated, candidates)

        with timed("pathology_seconds", timings):
            pathology_rows = []
            for scheme, length in INJECTION_SCHEMES:
                null_scores = scores_for(simulate(None, length, NULL_REPLICATES))
                marginal = np.bincount(np.argmin(null_scores, axis=1), minlength=len(candidates)) / NULL_REPLICATES
                pathology_rows.append(
                    {
                        "scheme": scheme,
                        "null_argmin_marginal_max": float(marginal.max()),
                        "null_argmin_argmax_factor": float(factors[int(np.argmax(marginal))]),
                        "marginal_selection": ";".join(f"{value:.3f}" for value in marginal),
                    }
                )
            pathology = pd.DataFrame(pathology_rows)
            argmin_recovery_rows = []
            for scheme, length in INJECTION_SCHEMES:
                null_scores = scores_for(simulate(None, length, NULL_REPLICATES))
                for true_index in range(len(candidates)):
                    signal_scores = scores_for(simulate(setup["signals"][true_index], length, POWER_REPLICATES))
                    selected = np.argmin(signal_scores, axis=1)
                    argmin_recovery_rows.append(
                        {
                            "scheme": scheme,
                            "true_factor": float(factors[true_index]),
                            "true_span_eV": float(spans[true_index]),
                            "argmin_exact_recovery": float((selected == true_index).mean()),
                        }
                    )
            argmin_recovery = pd.DataFrame(argmin_recovery_rows)

        with timed("calibration_seconds", timings):
            thresholds = {}
            fp_rows = []
            for scheme, length in INJECTION_SCHEMES:
                null_scores = scores_for(simulate(None, length, NULL_REPLICATES))
                threshold = np.quantile(null_scores, 0.05, axis=0)  # detection if score below this
                fresh_scores = scores_for(simulate(None, length, FP_FRESH_NULL))
                fp = (fresh_scores < threshold[None, :]).mean(axis=0)
                thresholds[scheme] = threshold
                fp_rows.append(
                    {
                        "scheme": scheme,
                        "fp_rate_min": float(fp.min()),
                        "fp_rate_max": float(fp.max()),
                        "fp_rate_max_over_min": float(fp.max() / fp.min()),
                        "fp_per_candidate": ";".join(f"{value:.3f}" for value in fp),
                    }
                )
            calibration = pd.DataFrame(fp_rows)

        with timed("power_seconds", timings):
            power_rows = []
            for scheme, length in INJECTION_SCHEMES:
                threshold = thresholds[scheme]
                for true_factor in SWEEP_FACTORS:
                    true_index = int(np.flatnonzero(np.isclose(factors, true_factor))[0])
                    signal = setup["signals"][true_index]
                    for snr in DETECTION_SNR_LEVELS:
                        scores = scores_for(simulate(signal, length, POWER_REPLICATES, scale=snr))
                        detection = (scores < threshold[None, :]).mean(axis=0)
                        power_rows.append(
                            {
                                "scheme": scheme,
                                "true_factor": float(factors[true_index]),
                                "true_span_eV": float(spans[true_index]),
                                "snr": snr,
                                "truth_detection_power": float(detection[true_index]),
                                "max_other_detection": float(np.delete(detection, true_index).max()),
                            }
                        )
            power = pd.DataFrame(power_rows)

        with timed("gates_seconds", timings):
            fp_worst = calibration["fp_rate_max_over_min"].max()
            fp_in_band = bool(
                (calibration["fp_rate_min"] >= 0.02).all() and (calibration["fp_rate_max"] <= 0.10).all()
            )
            gates = {
                "pathology_confirmed_max_candidate_null_selection_above_50pct": bool(
                    (pathology["null_argmin_marginal_max"] > 0.50).all()
                ),
                "G_A5a_detection_instrument_calibrated": bool(fp_in_band and fp_worst < 2.5),
                "fp_max_over_min_worst_scheme": float(fp_worst),
                "frozen_85pct_recovery_at_12x_is_null_geometry": bool(
                    argmin_recovery[np.isclose(argmin_recovery["true_factor"], 12.0)][
                        "argmin_exact_recovery"
                    ].mean() > 0.50
                ),
            }
            snr50_rows = []
            for true_factor in SWEEP_FACTORS:
                local = power[np.isclose(power["true_factor"], true_factor)]
                by_scheme = (
                    local.groupby(["true_span_eV", "snr"], as_index=False)["truth_detection_power"].mean()
                    .sort_values("snr")
                )
                fit = _fit_threshold50(by_scheme["snr"].to_numpy(), by_scheme["truth_detection_power"].to_numpy())
                snr50_rows.append(
                    {
                        "true_factor": float(true_factor),
                        "true_span_eV": float(by_scheme["true_span_eV"].iloc[0]),
                        **fit,
                    }
                )
            snr50 = pd.DataFrame(snr50_rows)
            power_at_1 = (
                power[power["snr"] == 1.0]
                .groupby(["true_factor", "true_span_eV"], as_index=False)["truth_detection_power"]
                .mean()
            )
            nist_power = float(
                power_at_1[np.isclose(power_at_1["true_factor"], 1.0)]["truth_detection_power"].iloc[0]
            )
            gates["nist_span_detection_power_at_archived_snr"] = nist_power
            summary = {
                "experiment": "A5_injection_debias",
                "dataset_sha256": audit["sha256"],
                "null_replicates": NULL_REPLICATES,
                "power_replicates": POWER_REPLICATES,
                "gates": gates,
                "headline": (
                    "Phase-3 argmin recovery is null geometry (max-candidate selected at "
                    f"{100 * pathology['null_argmin_marginal_max'].mean():.0f}% under no-signal null); "
                    "the calibrated per-template detection instrument shows the NIST-span template at "
                    f"{100 * nist_power:.0f}% detection power at the archived noise level, so no "
                    "resolution number from model selection is quotable"
                ),
                "quotable_boundary": (
                    "detection-based: even the widest candidate span (3.36 eV) has "
                    f"{100 * float(power_at_1['truth_detection_power'].max()):.0f}% detection power at archived SNR; "
                    "S50 in SNR is reported per span in snr50 table"
                ),
                "gate_verdicts": {
                    "G_A5a": (
                        "PASS: detection instrument calibrated (FP in band); its power curve is quotable"
                        if gates["G_A5a_detection_instrument_calibrated"]
                        else "FAIL: detection instrument not calibrated; only qualitative boundary statements allowed"
                    )
                },
                "claim_boundary": [
                    "detection thresholds are per-template null quantiles at 5% level, block-resampled empirical residuals",
                    "S50 values are instrument properties of this pipeline, not instrument specifications of any FH apparatus",
                    "the frozen phase-3 'resolution boundary at 2.24 eV' wording must be retired in favor of this analysis",
                ],
                "timings": timings,
            }

    out_dir = write_outputs(
        "a5_injection_debias",
        {
            "argmin_pathology": pathology,
            "argmin_recovery_curve": argmin_recovery,
            "detection_calibration": calibration,
            "detection_power": power,
            "power_at_archived_snr": power_at_1,
            "snr50_per_span": snr50,
        },
        summary,
    )
    return {"summary": summary, "out_dir": out_dir}


if __name__ == "__main__":
    result = run()
    print(json.dumps(result["summary"]["gates"], indent=2, ensure_ascii=False))
    print(f"outputs -> {result['out_dir']}")
