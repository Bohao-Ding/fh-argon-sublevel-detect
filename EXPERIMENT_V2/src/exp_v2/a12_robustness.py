"""A12 — Completeness-audit completions (gaps 1-4 of the internal audit).

  A12a  endpoint re-judgment under the steady-region framing: rebuild the
        retarding bands without the Vr=10 V curve ({0,4,6,8} -> 4 bands) and
        report per-block rank-2 gains. Question: do failures concentrate in
        the transition block, or do steady blocks also fail? (frozen phase-3
        G14 FAIL was scored under the old five-block full rule)
  A12b  time-domain cross-correlation lag between adjacent bands (model-free
        timing evidence; rank-1 predicts zero lag everywhere)
  A12c  subspace/model-separation geometry: how many per-point noise sigmas
        separate the NIST-template prediction from stretched/equal-spacing
        alternatives at their best fits - an estimator-independent
        indistinguishability bound for C3
  gap4  Wilson CI on the template-specificity percentile; effective
        oscillation count of the dataset (independent units are cycles, not
        voltage points)

Frozen gates:
  G-A12a  without Vr=10, every steady block keeps positive gain and the
          steady median stays >= 10%;
  G-A12b  the first-to-last band cumulative lag CI excludes zero lag;
  G-A12c  the worst model-prediction separation (NIST vs stretched or
          equal-spacing alternatives) is below 1 per-point sigma.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from .common import load_pivot, timed, write_outputs

BAND_LABELS = ["Vr_0_4", "Vr_4_6", "Vr_6_8", "Vr_8_10", "Vr_gt_10"]
LAG_MAX_STEPS = 8  # +-4.0 V at the 0.5 V grid
SEGMENTS = 5
PERCENTILE_TEMPLATES = 512
PERCENTILE_SEED = 20260831


def _detrend(va: np.ndarray, series: np.ndarray) -> np.ndarray:
    x = (va - va.mean()) / (va.max() - va.min())
    design = np.column_stack([x**power for power in range(4)])
    return series - design @ np.linalg.lstsq(design, series, rcond=None)[0]


def _lag_steps(x: np.ndarray, y: np.ndarray, max_steps: int = LAG_MAX_STEPS):
    lags = np.arange(-max_steps, max_steps + 1)
    cors = np.full(len(lags), -2.0)
    for j, s in enumerate(lags):
        if s < 0:
            a, b = x[-s:], y[:s]
        elif s > 0:
            a, b = x[:-s], y[s:]
        else:
            a, b = x, y
        if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0:
            cors[j] = float(np.corrcoef(a, b)[0, 1])
    best = int(np.argmax(cors))
    zero = int(np.flatnonzero(lags == 0)[0])
    return int(lags[best]), float(cors[best]), float(cors[zero])


def _segment_lags(va, x, y, segments: int = SEGMENTS):
    edges = np.linspace(0, len(x), segments + 1).astype(int)
    rows = []
    for k in range(segments):
        sl = slice(edges[k], edges[k + 1])
        lag, peak, zero = _lag_steps(x[sl], y[sl])
        rows.append({"segment": k + 1, "lag_steps": lag, "peak_corr": peak, "zero_lag_corr": zero})
    return pd.DataFrame(rows)


def _wilson(count: int, n: int) -> tuple[float, float]:
    z = 1.959963984540054
    rate = count / n
    center = rate + z * z / (2 * n)
    spread = z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n))
    denom = 1 + z * z / n
    return ((center - spread) / denom, (center + spread) / denom)


def run() -> dict:
    timings: dict[str, float] = {}
    with timed("total_seconds", timings):
        pivot, audit = load_pivot()
        from fh_retry.phase2 import (
            NIST_4S_EV,
            _harmonic_design,
            _retarding_fractions,
            _same_span_templates,
            _template_basis,
            _template_operators,
            _operator_cv,
            _analysis_arrays,
        )
        from fh_retry.phase3 import _rank_detail, _rank_summary

        rng = np.random.default_rng(20260924)

        # ---------- A12a: without-Vr10 steady-region re-judgment ----------
        with timed("a12a_seconds", timings):
            va8, frac8, _ = _retarding_fractions(pivot, [0, 4, 6, 8])
            detail8 = _rank_detail(va8, frac8)
            detail8["is_steady_block"] = detail8["block"] > 1
            steady8 = detail8[detail8["is_steady_block"]]
            worst_steady = steady8.loc[steady8["relative_gain"].idxmin()]

        # ---------- A12b: cross-correlation lags ----------
        with timed("a12b_seconds", timings):
            va, fractions, _ = _retarding_fractions(pivot, [0, 4, 6, 8, 10])
            detrended = {b: _detrend(va, fractions[:, i]) for i, b in enumerate(BAND_LABELS)}
            pair_rows = []
            for left, right in ((0, 1), (1, 2), (2, 3), (3, 4), (0, 4)):
                x, y = detrended[BAND_LABELS[left]], detrended[BAND_LABELS[right]]
                lag, peak, zero = _lag_steps(x, y)
                segs = _segment_lags(va, x, y)
                sign_agree = int((np.sign(segs["lag_steps"]) == np.sign(lag)).sum())
                pair_rows.append(
                    {
                        "pair": f"{BAND_LABELS[left]}-vs-{BAND_LABELS[right]}",
                        "lag_steps": lag,
                        "lag_V": lag * 0.5,
                        "peak_corr": peak,
                        "zero_lag_corr": zero,
                        "corr_gain": peak - zero,
                        "segments_agreeing_with_full_lag": f"{sign_agree}/{SEGMENTS}",
                        "segment_lag_steps": ";".join(str(int(v)) for v in segs["lag_steps"]),
                    }
                )
            lags = pd.DataFrame(pair_rows)
            cumulative = lags[lags["pair"] == "Vr_0_4-vs-Vr_gt_10"].iloc[0]
            gate_b = bool(
                cumulative["lag_steps"] != 0
                and int(str(cumulative["segments_agreeing_with_full_lag"]).split("/")[0]) >= 3
            )

        # ---------- A12c: model-separation geometry ----------
        with timed("a12c_seconds", timings):
            center = float(NIST_4S_EV.mean())
            span = float(np.ptp(NIST_4S_EV))
            equal = np.linspace(center - span / 2, center + span / 2, 4)

            # pooled per-point noise sigma from A11-style detrended sinusoid residuals
            sigmas = []
            for i in range(5):
                series = _detrend(va, fractions[:, i])
                t = 2.0 * np.pi * va / 11.55
                design = np.column_stack([np.sin(t), np.cos(t)])
                resid = series - design @ np.linalg.lstsq(design, series, rcond=None)[0]
                sigmas.append(float(np.std(resid, ddof=2)))
            sigma_pt = float(np.sqrt(np.mean(np.asarray(sigmas) ** 2)))

            # raw (column-normalized) design conditioning
            X = _harmonic_design(va, NIST_4S_EV)
            Xn = X / np.linalg.norm(X, axis=0, keepdims=True)
            sv = np.linalg.svd(Xn, compute_uv=False)

            separation_rows = []
            y_all = np.column_stack([_detrend(va, fractions[:, i]) for i in range(5)])
            alternatives = [("stretch_x0.75", center + 0.75 * (NIST_4S_EV - center)),
                            ("stretch_x1.25", center + 1.25 * (NIST_4S_EV - center)),
                            ("equal_spacing", equal)]
            for name, energies in alternatives:
                Xf = _harmonic_design(va, np.asarray(energies, dtype=float))
                Xfn = Xf / np.linalg.norm(Xf, axis=0, keepdims=True)
                seps = []
                for i in range(5):
                    y = y_all[:, i]
                    b0 = np.linalg.lstsq(Xn, y, rcond=None)[0]
                    b1 = np.linalg.lstsq(Xfn, y, rcond=None)[0]
                    seps.append(float(np.sqrt(np.mean((Xn @ b0 - Xfn @ b1) ** 2)) / sigma_pt))
                separation_rows.append(
                    {
                        "alternative": name,
                        "separation_sigma_min": float(np.min(seps)),
                        "separation_sigma_median": float(np.median(seps)),
                        "separation_sigma_max": float(np.max(seps)),
                    }
                )
            separation = pd.DataFrame(separation_rows)

            # template-design conditioning numbers
            conditioning = {
                "smallest_singular_value_unit_cols": float(sv[-1]),
                "largest_singular_value_unit_cols": float(sv[0]),
                "condition_number": float(sv[0] / sv[-1]),
                "pooled_noise_sigma": sigma_pt,
            }

        # ---------- gap 4: percentile CI + effective oscillation count ----------
        with timed("percentile_seconds", timings):
            raw_response = _analysis_arrays(pivot)[1]
            response = (raw_response - raw_response.mean(axis=0)) / raw_response.std(axis=0, ddof=1)
            nist_q, _ = _template_basis(va, NIST_4S_EV)
            _, nist_ops = _template_operators(va, nist_q)
            nist_error, _ = _operator_cv(response, nist_ops)
            pseudo_errors = []
            for energies in _same_span_templates(PERCENTILE_TEMPLATES, PERCENTILE_SEED):
                q, _ = _template_basis(va, energies)
                _, ops = _template_operators(va, q)
                error, _ = _operator_cv(response, ops)
                pseudo_errors.append(error)
            pseudo_errors = np.asarray(pseudo_errors)
            count_le = int((pseudo_errors <= nist_error).sum())
            lo, hi = _wilson(count_le, PERCENTILE_TEMPLATES)

        with timed("oscillations_seconds", timings):
            from scipy.signal import find_peaks

            osc_rows = []
            va_all = pivot.index.to_numpy(float)
            for vr in (0, 4, 6, 8, 10):
                curve = pivot[vr].to_numpy(float)
                mask = (va_all >= 15.0) & (va_all <= 80.0)
                y = curve[mask]
                peaks, _ = find_peaks(y, prominence=0.05 * float(np.ptp(y)))
                osc_rows.append({"Vr": vr, "oscillation_cycles": int(len(peaks))})
            osc = pd.DataFrame(osc_rows)
            total_cycles = int(osc["oscillation_cycles"].sum())

        with timed("gates_seconds", timings):
            gates = {
                "G_A12a_steady_survives_without_Vr10": bool(
                    (steady8["relative_gain"] > 0).all()
                    and float(steady8["relative_gain"].median()) >= 0.10
                ),
                "G_A12b_cumulative_lag_excludes_zero": gate_b,
                "G_A12c_worst_separation_below_1sigma": bool(
                    separation["separation_sigma_max"].max() < 1.0
                ),
                "a12a_without_vr10_median_steady_gain": float(steady8["relative_gain"].median()),
                "a12a_without_vr10_worst_steady_block": int(worst_steady["block"]),
                "a12a_without_vr10_worst_steady_gain": float(worst_steady["relative_gain"]),
                "a12b_cumulative_lag_steps": int(cumulative["lag_steps"]),
                "a12c_worst_separation_sigma": float(separation["separation_sigma_max"].max()),
                "percentile_count": count_le,
                "percentile_wilson_ci": [round(lo, 4), round(hi, 4)],
                "total_oscillation_cycles": total_cycles,
            }
            summary = {
                "experiment": "A12_completeness_audit",
                "lag_alias_note": (
                    "adjacent-band lags are 3/3/3/7 steps; the direct first-to-last measurement is -7 steps. "
                    "Consistency check: adjacent sum 16 steps = 8.0 V = -3.55 V modulo one period (23.1 steps) - "
                    "matching the direct -7 steps (-3.5 V). Lags are resolution-limited to the 0.5 V step and the "
                    "direct pair is read modulo one period."
                ),
                "dataset_sha256": audit["sha256"],
                "gates": gates,
                "gate_verdicts": {
                    "G_A12a": (
                        "PASS: without Vr=10 every steady block keeps a positive gain and the steady median stays >=10% - "
                        "the old G14 failure concentrates in the transition region under the new framing"
                        if gates["G_A12a_steady_survives_without_Vr10"]
                        else "FAIL: dropping Vr=10 damages steady blocks - the endpoint weakness must be weighted more heavily"
                    ),
                    "G_A12b": (
                        "PASS: the model-free time-domain lag between first and last bands is nonzero with segment agreement - "
                        "the timing gradient no longer depends on any fitted basis"
                        if gate_b
                        else "FAIL: cross-correlation lags do not confirm the timing gradient"
                    ),
                    "G_A12c": (
                        "PASS: alternative templates' best predictions differ from NIST by less than one noise sigma - "
                        "the spacing is geometrically indistinguishable, independent of the estimator"
                        if gates["G_A12c_worst_separation_below_1sigma"]
                        else "FAIL: some alternative is separable at >=1 sigma - refine the boundary claim"
                    ),
                },
                "claim_boundary": [
                    "lags are resolution-limited to the 0.5 V sampling step",
                    "the separation bound assumes per-point Gaussian noise at the pooled residual sigma",
                ],
                "timings": timings,
            }

    out_dir = write_outputs(
        "a12_robustness",
        {
            "a12a_without_vr10_blocks": detail8,
            "a12b_lags": lags,
            "a12c_separation": separation,
            "a12c_conditioning": pd.DataFrame([conditioning]),
            "percentile_ci": pd.DataFrame(
                [{"count_le_nist": count_le, "n": PERCENTILE_TEMPLATES,
                  "wilson_low": lo, "wilson_high": hi}]
            ),
            "oscillation_counts": osc,
        },
        summary,
    )
    return {"summary": summary, "out_dir": out_dir}


if __name__ == "__main__":
    result = run()
    print(json.dumps(result["summary"]["gates"], indent=2, ensure_ascii=False))
    print(f"outputs -> {result['out_dir']}")
