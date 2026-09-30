# -*- coding: utf-8 -*-
"""A11 — Direct undergraduate-style evidence pack (sinusoid fits, spectra,
phasors, Lissajous plots, template-waveform overlays).

Purpose: give the three core claims an evidence layer that reads entirely with
standard laboratory tools (linear least squares, FFT, scatter plots), so the
rank formalism is only needed for the holdout prediction test.

  A11a  per-band least-squares sinusoid fit at the common period (amplitude,
        phase, covariance-propagated uncertainties); pairwise phase
        differences with significance (rank-1 requires parallel phasors);
  A11b  per-band FFT amplitude spectra (detrended): dominant period per band;
  A11c  band-vs-band Lissajous trajectories and correlations after detrending
        (rank-1 predicts a straight line through the origin);
  A11d  template waveforms for NIST / equal-spacing / pseudo templates with
        pairwise correlations, and the width-to-spacing resolution ratio.

Gates (frozen):
  G-A11a  every pairwise band phase difference excludes zero at 3 sigma;
  G-A11b  per-band dominant periods lie in [10.5, 12.5] V;
  G-A11d  NIST/equal-spacing/pseudo template waveform correlations >= 0.999
          (visual and numerical degeneracy supporting C3).
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from .common import load_pivot, timed, write_outputs

BAND_LABELS = ["Vr_0_4", "Vr_4_6", "Vr_6_8", "Vr_8_10", "Vr_gt_10"]
PERIOD = 11.55
FFT_MIN_PERIOD = 8.0


def _detrend(va: np.ndarray, series: np.ndarray):
    x = (va - va.mean()) / (va.max() - va.min())
    design = np.column_stack([x**power for power in range(4)])
    coef = np.linalg.lstsq(design, series, rcond=None)[0]
    return series - design @ coef


def _fit_sin(va: np.ndarray, series: np.ndarray, period: float):
    design = np.column_stack(
        [np.sin(2.0 * np.pi * va / period), np.cos(2.0 * np.pi * va / period)]
    )
    coef, __, _, _ = np.linalg.lstsq(design, series, rcond=None)
    residual = series - design @ coef
    dof = max(len(series) - 2, 1)
    sigma2 = float(np.sum(residual**2)) / dof
    cov = sigma2 * np.linalg.inv(design.T @ design)
    amp = float(np.hypot(coef[0], coef[1]))
    phase = float(np.arctan2(coef[1], coef[0]))
    var_amp = float((coef[0] ** 2 * cov[0, 0] + coef[1] ** 2 * cov[1, 1]
                     + 2 * coef[0] * coef[1] * cov[0, 1]) / amp**2)
    var_phase = float((coef[0] ** 2 * cov[1, 1] + coef[1] ** 2 * cov[0, 0]
                       - 2 * coef[0] * coef[1] * cov[0, 1]) / amp**2)
    r2 = 1.0 - float(np.sum(residual**2)) / float(np.sum(series**2))
    return amp, phase, np.sqrt(max(var_amp, 0.0)), np.sqrt(max(var_phase, 0.0)), r2


def _phase_sigma_boot(va, series, period, n_boot=300, block=9, seed=20260926):
    """Moving-block bootstrap phase sigma (residuals are strongly autocorrelated,
    lag-1 ~0.95; the iid formula underestimates sigma by ~sqrt(n/n_eff))."""
    t = 2.0 * np.pi * va / period
    design = np.column_stack([np.sin(t), np.cos(t)])
    coef = np.linalg.lstsq(design, series, rcond=None)[0]
    fitted = design @ coef
    residual = series - fitted
    rng = np.random.default_rng(seed)
    count = len(series)
    nblocks = int(np.ceil(count / block))
    phases = np.empty(n_boot)
    for r in range(n_boot):
        starts = rng.integers(0, count, size=nblocks)
        idx = ((starts[:, None] + np.arange(block)[None, :]) % count).reshape(-1)[:count]
        boot = fitted + residual[idx]
        c = np.linalg.lstsq(design, boot, rcond=None)[0]
        phases[r] = np.arctan2(c[1], c[0])
    return float(np.std(phases, ddof=1))


def run() -> dict:
    timings: dict[str, float] = {}
    with timed("total_seconds", timings):
        pivot, audit = load_pivot()
        from fh_retry.phase2 import (
            NIST_4S_EV,
            _analysis_arrays,
            _retarding_fractions,
            _same_span_templates,
            _template_basis,
        )

        va, fractions, _ = _retarding_fractions(pivot, [0, 4, 6, 8, 10])

        with timed("fits_seconds", timings):
            fit_rows = []
            amps, phases, phase_sig = [], [], []
            detrended = {}
            for index, band in enumerate(BAND_LABELS):
                series = _detrend(va, fractions[:, index])
                detrended[band] = series
                amp, phase, sig_amp, sig_phase, r2 = _fit_sin(va, series, PERIOD)
                sig_phase_boot = _phase_sigma_boot(va, series, PERIOD)
                amps.append(amp)
                phases.append(phase)
                phase_sig.append(sig_phase_boot)
                fit_rows.append(
                    {
                        "band": band,
                        "amplitude": amp,
                        "amplitude_sigma": sig_amp,
                        "phase_rad": phase,
                        "phase_sigma_rad_iid": sig_phase,
                        "phase_sigma_rad_boot": sig_phase_boot,
                        "autocorr_inflation": sig_phase_boot / sig_phase if sig_phase else np.nan,
                        "sinusoid_r2": r2,
                    }
                )
            fits = pd.DataFrame(fit_rows)

            pair_rows = []
            for left in range(5):
                for right in range(left + 1, 5):
                    diff = phases[right] - phases[left]
                    diff = float(np.angle(np.exp(1j * diff)))
                    sigma = float(np.hypot(phase_sig[left], phase_sig[right]))
                    pair_rows.append(
                        {
                            "band_left": BAND_LABELS[left],
                            "band_right": BAND_LABELS[right],
                            "phase_diff_rad": diff,
                            "sigma_rad": sigma,
                            "significance_sigma": abs(diff) / sigma,
                            "excludes_zero_3sigma": bool(abs(diff) > 3 * sigma),
                        }
                    )
            pairs = pd.DataFrame(pair_rows)

        with timed("fft_seconds", timings):
            step = float(np.median(np.diff(va)))
            fft_rows = []
            for band in BAND_LABELS:
                series = detrended[band] - detrended[band].mean()
                spectrum = np.fft.rfft(series)
                freqs = np.fft.rfftfreq(len(series), d=step)
                periods = np.where(freqs > 0, 1.0 / np.maximum(freqs, 1e-12), np.inf)
                mask = (periods >= FFT_MIN_PERIOD) & (periods <= 16.0)
                amplitudes = np.abs(spectrum)
                peak = int(np.flatnonzero(mask)[np.argmax(amplitudes[mask])])
                fft_rows.append(
                    {
                        "band": band,
                        "fft_peak_period_V": float(periods[peak]),
                        "fft_peak_amplitude": float(amplitudes[peak]),
                    }
                )
            fft = pd.DataFrame(fft_rows)

        with timed("lissajous_seconds", timings):
            liss_rows = []
            for left in range(1, 5):
                x = detrended[BAND_LABELS[0]]
                y = detrended[BAND_LABELS[left]]
                r = float(np.corrcoef(x, y)[0, 1])
                slope = float(np.linalg.lstsq(x[:, None], y, rcond=None)[0][0])
                liss_rows.append(
                    {
                        "pair": f"{BAND_LABELS[0]}-vs-{BAND_LABELS[left]}",
                        "correlation": r,
                        "best_scale": slope,
                        "line_r2": r * r,
                    }
                )
            liss = pd.DataFrame(liss_rows)

        with timed("templates_seconds", timings):
            mean_energy = float(NIST_4S_EV.mean())
            span = float(np.ptp(NIST_4S_EV))
            equal = np.linspace(mean_energy - span / 2, mean_energy + span / 2, 4)
            pseudo = _same_span_templates(3, 20260831)
            waveforms = {}
            for name, energies in (
                ("NIST", NIST_4S_EV),
                ("equal", equal),
                ("pseudo_0", pseudo[0]),
                ("pseudo_1", pseudo[1]),
            ):
                q, _ = _template_basis(va, np.asarray(energies))
                waveforms[name] = q @ (q.T @ detrended["Vr_4_6"])  # project one band for shape
            names = list(waveforms)
            corr_rows = []
            for i, left in enumerate(names):
                for right in names[i + 1:]:
                    r = float(np.corrcoef(waveforms[left], waveforms[right])[0, 1])
                    corr_rows.append({"template_left": left, "template_right": right, "correlation": r})
            corr = pd.DataFrame(corr_rows)
            min_template_corr = float(corr["correlation"].min())
            response_width_V = 2.90
            ratio = response_width_V / span

        with timed("gates_seconds", timings):
            gates = {
                "G_A11a_all_pairwise_phase_diffs_exclude_zero_3sigma": bool(
                    pairs["excludes_zero_3sigma"].all()
                ),
                "G_A11b_fft_peak_periods_within_10p5_12p5": bool(
                    fft["fft_peak_period_V"].between(10.5, 12.5).all()
                ),
                "G_A11d_template_correlations_ge_0p999": bool(min_template_corr >= 0.999),
                "min_pairwise_phase_significance_sigma": float(pairs["significance_sigma"].min()),
                "fft_peak_period_range_V": [
                    float(fft["fft_peak_period_V"].min()),
                    float(fft["fft_peak_period_V"].max()),
                ],
                "min_template_correlation": min_template_corr,
                "response_width_to_spacing_ratio": float(ratio),
                "lissajous_correlations": {
                    row["pair"]: row["correlation"] for _, row in liss.iterrows()
                },
            }
            summary = {
                "experiment": "A11_direct_evidence",
                "dataset_sha256": audit["sha256"],
                "gates": gates,
                "gate_verdicts": {
                    "G_A11a": (
                        "PASS: every pair of bands has a phase difference significant at 3 sigma - "
                        "the five phasors are not parallel, the direct-reading form of rank-2"
                        if gates["G_A11a_all_pairwise_phase_diffs_exclude_zero_3sigma"]
                        else "FAIL: some band pairs are phase-consistent at 3 sigma"
                    ),
                    "G_A11b": (
                        "PASS: all bands share the same dominant period scale (common timing)"
                        if gates["G_A11b_fft_peak_periods_within_10p5_12p5"]
                        else "FAIL: band spectra disagree on the dominant period"
                    ),
                    "G_A11d": (
                        "PASS: candidate template waveforms are numerically interchangeable "
                        "(r >= 0.999) - the direct-reading form of the spacing boundary"
                        if gates["G_A11d_template_correlations_ge_0p999"]
                        else "FAIL: template waveforms are distinguishable"
                    ),
                },
                "claim_boundary": [
                    "phases and amplitudes are parameters of a least-squares sinusoid fit to difference bands",
                    "the Lissajous trajectories are descriptive; the quantitative test is the 3-sigma phase table",
                ],
                "timings": timings,
            }

    out = {}
    out_dir = write_outputs(
        "a11_direct_evidence",
        {
            "band_sinusoid_fits": fits,
            "pairwise_phase_differences": pairs,
            "fft_peaks": fft,
            "lissajous": liss,
            "template_correlations": corr,
        },
        summary,
    )
    out.update({"summary": summary, "out_dir": out_dir, "fits": fits, "pairs": pairs,
                "fft": fft, "liss": liss, "corr": corr, "detrended": detrended,
                "waveforms": waveforms, "va": va})
    return out


if __name__ == "__main__":
    result = run()
    print(json.dumps(result["summary"]["gates"], indent=2, ensure_ascii=False))
    print(f"outputs -> {result['out_dir']}")
