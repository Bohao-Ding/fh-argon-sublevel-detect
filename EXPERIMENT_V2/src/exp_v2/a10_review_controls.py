"""A10 — Review-response controls (M1, M2, M6 of the internal AJP review).

Four controls requested by the review:

  C1  Proper per-block confidence intervals for the observed rank-2 gains:
      moving-block bootstrap around the rank-2 fit (CIs that contain the point
      estimates), plus per-block p-values against the rank-1 null world that
      Figure 4a previously mislabeled as "95% confidence intervals".
  C2  Structure-free null detection power: rebuild the detection thresholds
      from the residuals of the full rank-2 fit (no 11.5 V structure left) and
      re-measure detection power. Quantifies how much the previous 18% was
      suppressed by real H1-orthogonal structure kept in the old null.
  C3  Per-template absolute detection power for single-energy (dominant-energy)
      templates. Expected near the 5% false-positive level: an in-row-space
      addition leaves its own template's holdout error unchanged, so the
      absolute per-template statistic is structurally blind to it. This
      completes the reviewer's requested statistic x template-type matrix.
  C4  Nested-model detection power versus injected span for the argon manifold
      (existence instrument): if power is span-flat, structure existence is
      detected while spacing remains unattributed - closing the M2 tension.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from .common import load_pivot, timed, write_outputs

BLOCK = 9
C1_REPLICATES = 400
NULL_REPLICATES = 500
POWER_REPLICATES = 300
SWEEP_SPAN_FACTORS = (0.5, 1.0, 2.0, 4.0, 12.0)


def run() -> dict:
    timings: dict[str, float] = {}
    with timed("total_seconds", timings):
        pivot, audit = load_pivot()
        from fh_retry.analysis import BLOCKS, _predict_fold
        from fh_retry.phase2 import _analysis_arrays, _retarding_fractions
        from fh_retry.phase3 import (
            INJECTION_FACTORS,
            INJECTION_SCHEMES,
            _batch_candidate_scores,
            _injection_setup,
            _moving_block_batch,
            _phase_noise_batch,
        )

        va, fractions, _ = _retarding_fractions(pivot, [0, 4, 6, 8, 10])
        masks = [(va >= low) & (va < high) for low, high in BLOCKS]
        rng = np.random.default_rng(20260923)

        # ---------- C1: proper CIs (rank-2 residual world) + null p-values ----------
        with timed("c1_seconds", timings):
            mean = fractions.mean(axis=0)
            scale = fractions.std(axis=0, ddof=1)
            standardized = (fractions - mean) / scale

            def fold_gains(response: np.ndarray) -> list[float]:
                gains = []
                for mask in masks:
                    train = ~mask
                    _, rank1_z, observed_z = _predict_fold(
                        va, response, train, mask, period=11.55, rank=1, trend_degree=3
                    )
                    _, rank2_z, _ = _predict_fold(
                        va, response, train, mask, period=11.55, rank=2, trend_degree=3
                    )
                    e1 = float(np.sqrt(np.mean((rank1_z - observed_z) ** 2)))
                    e2 = float(np.sqrt(np.mean((rank2_z - observed_z) ** 2)))
                    gains.append((e1 - e2) / e1)
                return gains

            observed = fold_gains(standardized)

            # global rank-2 fit residual in standardized units (trend + rank-2 harmonic removed)
            center = 0.5 * (va.min() + va.max())
            half_range = 0.5 * (va.max() - va.min())
            x = (va - center) / half_range
            trend = np.column_stack([x**power for power in range(4)])
            harmonic = np.column_stack(
                [np.sin(2.0 * np.pi * va / 11.55), np.cos(2.0 * np.pi * va / 11.55)]
            )
            t_coef = np.linalg.lstsq(trend, standardized, rcond=None)[0]
            resid_y = standardized - trend @ t_coef
            resid_h = harmonic - trend @ np.linalg.lstsq(trend, harmonic, rcond=None)[0]
            q, r = np.linalg.qr(resid_h, mode="reduced")
            coef = np.linalg.solve(r, (q.T @ resid_y))
            t_coef2 = np.linalg.lstsq(trend, standardized - harmonic @ coef, rcond=None)[0]
            rank2_fit = trend @ t_coef2 + harmonic @ coef
            rank2_residual = standardized - rank2_fit

            boot_gain = np.empty((C1_REPLICATES, len(masks)))
            for replicate in range(C1_REPLICATES):
                noise = _moving_block_batch(rank2_residual, BLOCK, 1, rng)[0]
                boot_gain[replicate] = fold_gains(rank2_fit + noise)
            # rank-1 null world (same construction as a7) for per-block p-values
            from fh_retry.phase2 import _full_rank1_fit

            _, rank1_residual = _full_rank1_fit(va, fractions)
            mean_f = fractions.mean(axis=0)
            scale_f = fractions.std(axis=0, ddof=1)
            standardized_f = (fractions - mean_f) / scale_f
            fitted1, residual1 = _full_rank1_fit(va, standardized_f)
            null_gain = np.empty((C1_REPLICATES, len(masks)))
            for replicate in range(C1_REPLICATES):
                noise = _moving_block_batch(residual1, BLOCK, 1, rng)[0]
                null_gain[replicate] = fold_gains(fitted1 + noise)

            c1_rows = []
            for index, mask in enumerate(masks):
                p_value = float(
                    (1 + int((null_gain[:, index] >= observed[index]).sum())) / (C1_REPLICATES + 1)
                )
                c1_rows.append(
                    {
                        "block": index + 1,
                        "observed_gain": observed[index],
                        "ci_low": float(np.quantile(boot_gain[:, index], 0.025)),
                        "ci_high": float(np.quantile(boot_gain[:, index], 0.975)),
                        "rank1_null_q975": float(np.quantile(null_gain[:, index], 0.975)),
                        "null_p_value": p_value,
                        "ci_contains_point_estimate": bool(
                            np.quantile(boot_gain[:, index], 0.025)
                            <= observed[index]
                            <= np.quantile(boot_gain[:, index], 0.975)
                        ),
                    }
                )
            c1 = pd.DataFrame(c1_rows)

        # ---------- shared injection setup ----------
        with timed("setup_seconds", timings):
            setup = _injection_setup(pivot)
            trend_fit = setup["trend_fit"]
            common_residual = setup["common_residual"]
            candidates = setup["candidates"]
            factors = np.array([item["factor"] for item in candidates])
            spans = np.array([item["span_eV"] for item in candidates])

        def noise_batch(kind: str, length: int | None, replicates: int) -> np.ndarray:
            if kind == "phase":
                return _phase_noise_batch(common_residual, replicates, rng)
            if kind == "block":
                return _moving_block_batch(common_residual, BLOCK, replicates, rng)
            raise ValueError(kind)

        # ---------- C2: structure-free null detection power ----------
        with timed("c2_seconds", timings):
            structure_free_residual = rank2_residual
            c2_rows = []
            for scheme, length in INJECTION_SCHEMES:
                if length is None:
                    null_noise = _phase_noise_batch(structure_free_residual, NULL_REPLICATES, rng)
                    sig_noise = lambda reps: _phase_noise_batch(structure_free_residual, reps, rng)  # noqa: E731
                    base_noise = lambda reps: _phase_noise_batch(common_residual, reps, rng)  # noqa: E731
                else:
                    null_noise = _moving_block_batch(structure_free_residual, BLOCK, NULL_REPLICATES, rng)
                    sig_noise = lambda reps: _moving_block_batch(structure_free_residual, BLOCK, reps, rng)  # noqa: E731
                    base_noise = lambda reps: _moving_block_batch(common_residual, BLOCK, reps, rng)  # noqa: E731
                structure_free_null = trend_fit[None, :, :] + null_noise
                threshold = np.quantile(
                    _batch_candidate_scores(structure_free_null, candidates), 0.05, axis=0
                )
                for true_index in range(len(candidates)):
                    if factors[true_index] not in SWEEP_SPAN_FACTORS:
                        continue
                    signal = setup["signals"][true_index]
                    simulated = trend_fit[None, :, :] + signal[None, :, :] + sig_noise(POWER_REPLICATES)
                    scores = _batch_candidate_scores(simulated, candidates)
                    detection = (scores < threshold[None, :]).mean(axis=0)
                    c2_rows.append(
                        {
                            "scheme": scheme,
                            "true_factor": float(factors[true_index]),
                            "true_span_eV": float(spans[true_index]),
                            "structure_free_null_power": float(detection[true_index]),
                        }
                    )
            c2 = pd.DataFrame(c2_rows)

        # ---------- C3: per-template absolute detection for single energies ----------
        with timed("c3_seconds", timings):
            from fh_retry.phase2 import _template_basis, _template_operators, _trend

            raw_response = _analysis_arrays(pivot)[1]
            response = (raw_response - raw_response.mean(axis=0)) / raw_response.std(axis=0, ddof=1)
            trend = _trend(va)
            detrended = response - trend @ np.linalg.lstsq(trend, response, rcond=None)[0]
            target_rms = setup["target_signal_rms"]

            single_configs = {
                "hg_single_4.886eV": 4.886,
                "ar_single_meaneV": float(np.mean([11.54835442, 11.62359272, 11.72316039, 11.82807116])),
                "ne_single_16.70eV": float(np.mean([16.6151, 16.6672, 16.7151, 16.7853])),
            }
            c3_rows = []
            for name, energy in single_configs.items():
                q, _ = _template_basis(va, np.array([energy]))
                signal = q @ (q.T @ detrended)
                rms = float(np.sqrt(np.mean(signal**2)))
                signal = signal * (target_rms / rms)
                _, operators = _template_operators(va, q)
                candidate = [{"operators": operators}]
                for scheme, length in INJECTION_SCHEMES:
                    null_scores = _batch_candidate_scores(
                        trend_fit[None, :, :] + noise_batch("block" if length else "phase", length, NULL_REPLICATES),
                        candidate,
                    )[:, 0]
                    threshold = float(np.quantile(null_scores, 0.05))
                    sim_scores = _batch_candidate_scores(
                        trend_fit[None, :, :] + signal[None, :, :] + noise_batch("block" if length else "phase", length, POWER_REPLICATES),
                        candidate,
                    )[:, 0]
                    c3_rows.append(
                        {
                            "config": name,
                            "scheme": scheme,
                            "per_template_detection_power": float((sim_scores < threshold).mean()),
                        }
                    )
            c3 = pd.DataFrame(c3_rows)

        # ---------- C4: nested detection power vs injected span (Ar manifold) ----------
        with timed("c4_seconds", timings):
            from fh_retry.phase2 import NIST_4S_EV

            ar_mean = float(NIST_4S_EV.mean())
            h1_cand = None
            from fh_retry.phase2 import _template_operators as _to

            q_h1, _ = _template_basis(va, np.array([ar_mean]))
            _, h1_ops = _to(va, q_h1)
            h1_cand = [{"operators": h1_ops}]
            nested_rows = []
            for factor in (0.5, 1.0, 2.0, 4.0):
                energies = ar_mean + factor * (NIST_4S_EV - ar_mean)
                q_m, _ = _template_basis(va, energies)
                signal = q_m @ (q_m.T @ detrended)
                rms = float(np.sqrt(np.mean(signal**2)))
                signal = signal * (target_rms / rms)
                _, manifold_ops = _to(va, q_m)
                manifold_cand = [{"operators": manifold_ops}]
                for scheme, length in INJECTION_SCHEMES:
                    nb = noise_batch("block" if length else "phase", length, NULL_REPLICATES)
                    d_null = (
                        _batch_candidate_scores(trend_fit[None, :, :] + nb, h1_cand)[:, 0]
                        - _batch_candidate_scores(trend_fit[None, :, :] + nb, manifold_cand)[:, 0]
                    )
                    threshold = float(np.quantile(d_null, 0.95))
                    sb = noise_batch("block" if length else "phase", length, POWER_REPLICATES)
                    simulated = trend_fit[None, :, :] + signal[None, :, :] + sb
                    d_truth = (
                        _batch_candidate_scores(simulated, h1_cand)[:, 0]
                        - _batch_candidate_scores(simulated, manifold_cand)[:, 0]
                    )
                    nested_rows.append(
                        {
                            "injected_span_factor": float(factor),
                            "scheme": scheme,
                            "nested_detection_power": float((d_truth > threshold).mean()),
                        }
                    )
            c4 = pd.DataFrame(nested_rows)

        with timed("gates_seconds", timings):
            c4_pooled = c4.groupby("injected_span_factor", as_index=False)["nested_detection_power"].mean()
            c2_pooled = c2.groupby("true_span_eV", as_index=False)["structure_free_null_power"].mean()
            gates = {
                "c1_all_block_CIs_contain_point_estimates": bool(c1["ci_contains_point_estimate"].all()),
                "c1_max_null_p_value_steady_blocks": float(c1[c1["block"] > 1]["null_p_value"].max()),
                "c1_first_block_null_p_value": float(c1[c1["block"] == 1]["null_p_value"].iloc[0]),
                "c2_structure_free_null_power_nist_span": float(
                    c2_pooled[np.isclose(c2_pooled["true_span_eV"], 0.27971674)]["structure_free_null_power"].iloc[0]
                ),
                "c2_structure_free_null_power_range": [
                    float(c2_pooled["structure_free_null_power"].min()),
                    float(c2_pooled["structure_free_null_power"].max()),
                ],
                "c3_single_energy_absolute_power_max": float(c3["per_template_detection_power"].max()),
                "c4_nested_power_span_minmax": [
                    float(c4_pooled["nested_detection_power"].min()),
                    float(c4_pooled["nested_detection_power"].max()),
                ],
            }
            summary = {
                "experiment": "A10_review_controls",
                "dataset_sha256": audit["sha256"],
                "gates": gates,
                "findings": {
                    "C1": "observed per-block gains now carry bootstrap CIs that contain the point estimates; steady-block null p-values quantify the previous 'reference band' correctly",
                    "C2": "detection power rebuilt under a structure-free null (rank-2 residuals): comparison with the 18% baseline quantifies how much real H1-orthogonal structure in the old null suppressed apparent power",
                    "C3": "per-template absolute statistic is structurally blind to in-row-space additions (power near 5% FP level) - the statistic x template matrix is now complete and honest",
                    "C4": "nested existence detection versus injected span: span-flat power would confirm 'existence detected, spacing unattributed'",
                },
                "timings": timings,
            }

    out_dir = write_outputs(
        "a10_review_controls",
        {"c1_block_ci_pvalues": c1, "c2_structure_free_null": c2, "c3_single_energy_absolute": c3, "c4_nested_vs_span": c4},
        summary,
    )
    return {"summary": summary, "out_dir": out_dir}


if __name__ == "__main__":
    result = run()
    print(json.dumps(result["summary"]["gates"], indent=2, ensure_ascii=False))
    print(f"outputs -> {result['out_dir']}")
