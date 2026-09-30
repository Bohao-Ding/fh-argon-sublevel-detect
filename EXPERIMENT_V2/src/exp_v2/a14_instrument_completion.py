"""A14 — Instrument completion (finishes A13b's skipped nested arm and the
audit's remaining disclosure items).

  A14a  nested-model detection power versus injected span, using
        DESIGN-CONSTRUCTED, H1-orthogonal, equal-weight signals (not the
        degenerate data projections of the frozen setup). Decides whether the
        "nested detection span-flat at 58-61%" wording survives
        non-degenerate signals.
  A14b  response-definition cross-check: the template-specificity percentile
        is normally computed on raw currents (C3 layer) while the rank layer
        uses retarding fractions (audit finding 5g). Compute both.
  A14c  a12b boundary disclosure: recompute the band-lag table with a wider
        search window and flag boundary-pinned segments.

Frozen gates:
  G-A14a  nested power at the NIST span (factor 1) is at least 30% with
          design-constructed signals (existence detection works at all);
  G-A14b  the two response definitions give percentiles within 10 points of
          each other (layer definitional consistency).
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from .common import load_pivot, timed, write_outputs

INJECTION_FACTORS = (0.5, 1.0, 2.0, 4.0, 8.0, 12.0)
NULL_REPLICATES = 300
POWER_REPLICATES = 250
LAG_MAX_STEPS_WIDE = 10


def run() -> dict:
    timings: dict[str, float] = {}
    with timed("total_seconds", timings):
        pivot, audit = load_pivot()
        from fh_retry.phase2 import (
            NIST_4S_EV,
            _analysis_arrays,
            _retarding_fractions,
            _operator_cv,
            _same_span_templates,
            _template_basis,
            _template_operators,
        )
        from fh_retry.phase3 import (
            _batch_candidate_scores,
            _injection_setup,
            _moving_block_batch,
            _phase_noise_batch,
        )

        rng = np.random.default_rng(20260927)

        # ---------- shared setup ----------
        setup = _injection_setup(pivot)
        trend_fit = setup["trend_fit"]
        common_residual = setup["common_residual"]
        target_rms = setup["target_signal_rms"]
        va = setup["va"]
        center = float(NIST_4S_EV.mean())

        q_h1, _ = _template_basis(va, np.array([center]))
        _, h1_ops = _template_operators(va, q_h1)
        h1_cand = [{"operators": h1_ops}]

        def noise_batch(length, replicates):
            if length is None:
                return _phase_noise_batch(common_residual, replicates, rng)
            return _moving_block_batch(common_residual, length, replicates, rng)

        # ---------- A14a: nested power on design-constructed signals ----------
        with timed("a14a_seconds", timings):
            signals = {}
            pair_rows = []
            for factor in INJECTION_FACTORS:
                energies = center + factor * (NIST_4S_EV - center)
                q_f, _ = _template_basis(va, energies)
                orth = q_f - q_h1 @ (q_h1.T @ q_f)
                # equal-weight physical signature: in-phase sum of orthogonal components
                signal = orth @ np.ones(orth.shape[1])
                signal = signal * (target_rms / float(np.sqrt(np.mean(signal**2))))
                signals[factor] = signal
            for i, f1 in enumerate(INJECTION_FACTORS):
                for f2 in INJECTION_FACTORS[i + 1:]:
                    pair_rows.append(
                        {
                            "factor_pair": f"{f1:g}-{f2:g}",
                            "correlation": float(
                                np.corrcoef(signals[f1], signals[f2])[0, 1]
                            ),
                        }
                    )
            signal_gram = pd.DataFrame(pair_rows)
            max_offdiag = float(signal_gram["correlation"].abs().max())

            def delta_statistic(larger_ops, simulated):
                err_small = _batch_candidate_scores(simulated, h1_cand)[:, 0]
                err_large = _batch_candidate_scores(simulated, [{"operators": larger_ops}])[:, 0]
                return err_small - err_large

            power_rows = []
            scheme = ("block5", 5)
            length = scheme[1]
            for factor in INJECTION_FACTORS:
                energies = center + factor * (NIST_4S_EV - center)
                q_f, _ = _template_basis(va, energies)
                _, larger_ops = _template_operators(va, q_f)
                d_null = delta_statistic(
                    larger_ops, trend_fit[None, :, :] + noise_batch(length, NULL_REPLICATES)
                )
                threshold = float(np.quantile(d_null, 0.95))
                simulated = (
                    trend_fit[None, :, :]
                    + np.repeat(signals[factor][:, None], 5, axis=1)[None, :, :]
                    + noise_batch(length, POWER_REPLICATES)
                )
                d_truth = delta_statistic(larger_ops, simulated)
                power_rows.append(
                    {
                        "true_factor": float(factor),
                        "true_span_eV": float(factor * np.ptp(NIST_4S_EV)),
                        "nested_power_design_signals": float((d_truth > threshold).mean()),
                        "null_fp_rate": float((d_null > threshold).mean()),
                    }
                )
            nested_design = pd.DataFrame(power_rows)

        # ---------- A14b: response-definition cross-check ----------
        with timed("a14b_seconds", timings):
            raw_response = _analysis_arrays(pivot)[1]
            response_raw = (raw_response - raw_response.mean(axis=0)) / raw_response.std(axis=0, ddof=1)
            va_fr, frac, _ = _retarding_fractions(pivot, [0, 4, 6, 8, 10])
            response_frac = (frac - frac.mean(axis=0)) / frac.std(axis=0, ddof=1)

            def percentile_for(resp):
                q, _ = _template_basis(va, NIST_4S_EV)
                _, ops = _template_operators(va, q)
                err_nist, _ = _operator_cv(resp, ops)
                errs = []
                for energies in _same_span_templates(256, 20260831):
                    qf, _ = _template_basis(va, energies)
                    _, ops_f = _template_operators(va, qf)
                    e, _ = _operator_cv(resp, ops_f)
                    errs.append(e)
                return float((np.asarray(errs) <= err_nist).mean())

            pct_raw = percentile_for(response_raw)
            pct_frac = percentile_for(response_frac)

        # ---------- A14c: lag table with wide window ----------
        with timed("a14c_seconds", timings):
            from exp_v2.a12_robustness import _detrend, _lag_steps

            _, fractions5, _ = _retarding_fractions(pivot, [0, 4, 6, 8, 10])
            band_labels = ["Vr_0_4", "Vr_4_6", "Vr_6_8", "Vr_8_10", "Vr_gt_10"]
            detrended = {b: _detrend(va, fractions5[:, i]) for i, b in enumerate(band_labels)}
            wide_rows = []
            for left, right in ((0, 1), (1, 2), (2, 3), (3, 4), (0, 4)):
                x, y = detrended[band_labels[left]], detrended[band_labels[right]]
                lag, peak, zero = _lag_steps(x, y, max_steps=LAG_MAX_STEPS_WIDE)
                boundary = int(abs(lag) == LAG_MAX_STEPS_WIDE)
                wide_rows.append(
                    {
                        "pair": f"{band_labels[left]}-vs-{band_labels[right]}",
                        "lag_steps_wide": lag,
                        "lag_V": lag * 0.5,
                        "corr_gain": peak - zero,
                        "at_search_boundary": boundary,
                    }
                )
            wide = pd.DataFrame(wide_rows)

        with timed("gates_seconds", timings):
            nist_row = nested_design[np.isclose(nested_design["true_factor"], 1.0)].iloc[0]
            f12 = nested_design[nested_design["true_factor"] == 12.0].iloc[0]
            powers = nested_design["nested_power_design_signals"].to_numpy()
            gates = {
                "G_A14a_nested_works_on_design_signals": bool(float(nist_row["nested_power_design_signals"]) >= 0.30),
                "G_A14b_response_definition_consistent": bool(abs(pct_raw - pct_frac) <= 0.10),
                "a14a_nested_power_by_factor": {
                    float(r["true_factor"]): float(r["nested_power_design_signals"])
                    for _, r in nested_design.iterrows()
                },
                "a14a_power_min_max": [float(powers.min()), float(powers.max())],
                "a14a_signal_max_offdiag_correlation": max_offdiag,
                "a14b_percentile_raw_currents": pct_raw,
                "a14b_percentile_retarding_fractions": pct_frac,
                "a14c_wide_lags": {
                    row["pair"]: [int(row["lag_steps_wide"]), int(row["at_search_boundary"])]
                    for _, row in wide.iterrows()
                },
            }
            summary = {
                "experiment": "A14_instrument_completion",
                "dataset_sha256": audit["sha256"],
                "gates": gates,
                "gate_verdicts": {
                    "G_A14a": (
                        "PASS: the nested instrument detects design-constructed four-level signatures at the "
                        "archived amplitude - existence detection is real, and its span dependence below is quotable"
                        if gates["G_A14a_nested_works_on_design_signals"]
                        else "FAIL: even the nested instrument cannot see design-constructed signals at this amplitude - "
                        "the 58-61% figure must be retired as a degenerate-signal artifact"
                    ),
                    "G_A14b": (
                        "PASS: the two response definitions give consistent percentiles"
                        if gates["G_A14b_response_definition_consistent"]
                        else "FAIL: layer definitions change the percentile by >10 points"
                    ),
                },
                "claim_boundary": [
                    "design signals use equal-weight in-phase sums of the H1-orthogonal template components",
                    "wide lag search re-introduces period aliasing beyond half a period; adjacent-band lags remain the primary readout",
                ],
                "timings": timings,
            }

    out_dir = write_outputs(
        "a14_instrument_completion",
        {
            "a14a_signal_gram": signal_gram,
            "a14a_nested_power": nested_design,
            "a14b_response_definitions": pd.DataFrame(
                [{"response": "raw_currents", "percentile": pct_raw},
                 {"response": "retarding_fractions", "percentile": pct_frac}]
            ),
            "a14c_wide_lags": wide,
        },
        summary,
    )
    return {"summary": summary, "out_dir": out_dir}


if __name__ == "__main__":
    result = run()
    print(json.dumps(result["summary"]["gates"], indent=2, ensure_ascii=False))
    print(f"outputs -> {result['out_dir']}")
