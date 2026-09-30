"""A8 — Development-contamination cut confirmation.

Answers the two-round unresolved `dev-data-reuse` objection: re-derive the
core numbers under splits and period choices that were NOT part of the original
development loop.

  Part 1  time-forward / time-backward segment confirmation:
            forward  = fit on [34.0, 57.0) V, confirm on [57.0, 80.0) V
            backward = fit on [57.0, 80.0) V, confirm on [34.0, 57.0) V
  Part 2  period-prior variants for the standard 5-block CV:
            prior 11.55 V / fixed 11.45 V / fixed 11.50 V / per-fold
            train-selected period from a free 9.50-13.50 V scan (81 grid)
  Part 3  template-specificity percentile under response-trimming variants
            (Va>=15 baseline / Va>=22.5 / Va>=28.25) - the template machinery
            is period-free by construction, so the honest prior-free axis for
            C3 is the trimming choice.

Gates: G-A8a |median gain drift| <= 5 percentage points under every period
variant; G-A8b the NIST template stays non-unique (percentile > 5%) under all
trimmings.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from .common import load_pivot, timed, write_outputs

PERIOD_GRID = np.round(np.arange(9.50, 13.50 + 0.025, 0.05), 2)
PSEUDO_PERIODS = (9.0, 10.0, 13.5, 15.0)
TRIM_MIN_VA = (15.0, 22.5, 28.25)
PSEUDO_COUNT = 256
TEMPLATE_SEED = 20260831


def _segment_split(va: np.ndarray, fractions: np.ndarray) -> list[dict]:
    from fh_retry.analysis import _predict_fold

    segments = {
        "forward_fit_34_57_test_57_80": ((va >= 34.0) & (va < 57.0), (va >= 57.0) & (va < 80.0)),
        "backward_fit_57_80_test_34_57": ((va >= 57.0) & (va < 80.0), (va >= 34.0) & (va < 57.0)),
    }
    rows = []
    for name, (train, test) in segments.items():
        def error(period, rank, degree):
            _, predicted_z, observed_z = _predict_fold(
                va, fractions, train, test, period=period, rank=rank, trend_degree=degree
            )
            return float(np.sqrt(np.mean((predicted_z - observed_z) ** 2)))

        rank1 = error(11.55, 1, 3)
        rank2 = error(11.55, 2, 3)
        trend5 = error(None, None, 5)
        pseudo = {p: error(float(p), 2, 3) for p in PSEUDO_PERIODS}
        best_pseudo = min(pseudo, key=pseudo.get)
        rows.append(
            {
                "segment": name,
                "rank1_rmse": rank1,
                "rank2_rmse": rank2,
                "relative_gain": (rank1 - rank2) / rank1,
                "trend5_rmse": trend5,
                "best_pseudo_period_V": float(best_pseudo),
                "best_pseudo_rmse": pseudo[best_pseudo],
                "rank2_beats_both": bool(rank2 < trend5 and rank2 < pseudo[best_pseudo]),
            }
        )
    return rows


def _period_variant_cv(va: np.ndarray, fractions: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    from fh_retry.analysis import BLOCKS, _predict_fold

    variants = {
        "prior_11.55": 11.55,
        "fixed_11.45": 11.45,
        "fixed_11.50": 11.50,
        "free_train_selected": None,
    }
    detail_rows = []
    selected_rows = []
    for variant, period in variants.items():
        gains = []
        for block_index, (low, high) in enumerate(BLOCKS, start=1):
            test = (va >= low) & (va < high)
            train = ~test
            chosen = period
            if period is None:
                train_errors = []
                for candidate in PERIOD_GRID:
                    _, predicted_z, observed_z = _predict_fold(
                        va, fractions, train, train, period=float(candidate), rank=2, trend_degree=3
                    )
                    train_errors.append(float(np.sqrt(np.mean((predicted_z - observed_z) ** 2))))
                chosen = float(PERIOD_GRID[int(np.argmin(train_errors))])
                selected_rows.append(
                    {"variant": variant, "block": block_index, "selected_period_V": chosen}
                )
            _, rank1_z, observed_z = _predict_fold(
                va, fractions, train, test, period=chosen, rank=1, trend_degree=3
            )
            _, rank2_z, _ = _predict_fold(
                va, fractions, train, test, period=chosen, rank=2, trend_degree=3
            )
            rank1_error = float(np.sqrt(np.mean((rank1_z - observed_z) ** 2)))
            rank2_error = float(np.sqrt(np.mean((rank2_z - observed_z) ** 2)))
            gains.append((rank1_error - rank2_error) / rank1_error)
            detail_rows.append(
                {
                    "variant": variant,
                    "block": block_index,
                    "effective_period_V": chosen,
                    "relative_gain": gains[-1],
                }
            )
    detail = pd.DataFrame(detail_rows)
    summary = (
        detail.groupby("variant", as_index=False)
        .agg(median_gain=("relative_gain", "median"), worst_gain=("relative_gain", "min"))
    )
    return detail, summary.merge(
        pd.DataFrame(selected_rows).groupby("variant", as_index=False)["selected_period_V"].agg(
            lambda values: ";".join(f"{value:.2f}" for value in values)
        ).rename(columns={"selected_period_V": "selected_periods_V"}),
        on="variant",
        how="left",
    )


def _percentile_on_trim(pivot, min_va: float) -> dict:
    from fh_retry.phase2 import (
        NIST_4S_EV,
        _operator_cv,
        _same_span_templates,
        _template_basis,
        _template_operators,
    )

    va_all, currents = _analysis_arrays_trimmed(pivot, min_va)
    nist_q, _ = _template_basis(va_all, NIST_4S_EV)
    _, nist_operators = _template_operators(va_all, nist_q)
    nist_error, _ = _operator_cv(currents, nist_operators)
    errors = []
    for energies in _same_span_templates(PSEUDO_COUNT, TEMPLATE_SEED):
        q, _ = _template_basis(va_all, energies)
        _, operators = _template_operators(va_all, q)
        error, _ = _operator_cv(currents, operators)
        errors.append(error)
    errors = np.asarray(errors)
    percentile = float((errors <= nist_error).mean())
    return {"min_va_V": min_va, "nist_mean_balanced_rmse": nist_error, "nist_pseudo_percentile": percentile}


def _analysis_arrays_trimmed(pivot, min_va: float):
    from fh_retry.phase2 import _analysis_arrays

    va, response = _analysis_arrays(pivot)
    keep = va >= min_va
    return va[keep], response[keep]


def run() -> dict:
    timings: dict[str, float] = {}
    with timed("total_seconds", timings):
        pivot, audit = load_pivot()
        from fh_retry.phase2 import _retarding_fractions
        from fh_retry.phase3 import _rank_detail, _rank_summary

        va, fractions, _ = _retarding_fractions(pivot, [0, 4, 6, 8, 10])

        with timed("segments_seconds", timings):
            segments = pd.DataFrame(_segment_split(va, fractions))

        with timed("period_variants_seconds", timings):
            period_detail, period_summary = _period_variant_cv(va, fractions)

        with timed("template_trims_seconds", timings):
            trims = pd.DataFrame([_percentile_on_trim(pivot, value) for value in TRIM_MIN_VA])

        baseline = period_summary[period_summary["variant"] == "prior_11.55"]["median_gain"].iloc[0]
        drifts = (
            period_summary.assign(drift_pp=lambda frame: 100.0 * (frame["median_gain"] - baseline))
            .loc[:, ["variant", "median_gain", "drift_pp"]]
        )
        gates = {
            "G_A8a_period_variant_drift_within_5pp": bool(
                drifts["drift_pp"].abs().max() <= 5.0
            ),
            "G_A8b_nist_percentile_above_5pct_all_trims": bool(
                (trims["nist_pseudo_percentile"] > 0.05).all()
            ),
            "max_abs_drift_pp": float(drifts["drift_pp"].abs().max()),
            "percentile_range": [float(trims["nist_pseudo_percentile"].min()),
                                 float(trims["nist_pseudo_percentile"].max())],
            "forward_segment_gain": float(
                segments[segments["segment"].str.startswith("forward")]["relative_gain"].iloc[0]
            ),
            "backward_segment_gain": float(
                segments[segments["segment"].str.startswith("backward")]["relative_gain"].iloc[0]
            ),
            "forward_segment_rank2_beats_both": bool(
                segments[segments["segment"].str.startswith("forward")]["rank2_beats_both"].iloc[0]
            ),
            "backward_segment_rank2_beats_both": bool(
                segments[segments["segment"].str.startswith("backward")]["rank2_beats_both"].iloc[0]
            ),
        }
        summary = {
            "experiment": "A8_development_cut",
            "dataset_sha256": audit["sha256"],
            "gates": gates,
            "gate_verdicts": {
                "G_A8a": (
                    "PASS: rank-2 conclusion is insensitive to the 11.55 V literature prior (drift <= 5 pp)"
                    if gates["G_A8a_period_variant_drift_within_5pp"]
                    else "FAIL: conclusion depends on the literature prior; restrict wording"
                ),
                "G_A8b": (
                    "PASS: NIST template stays non-specific under all response trims (template step is prior-free by construction)"
                    if gates["G_A8b_nist_percentile_above_5pct_all_trims"]
                    else "FAIL: percentile crosses into uniqueness under some trim"
                ),
            },
            "note": (
                "the time-forward split is the closest local analogue to a discovery/confirmation "
                "division available on one archived dataset; it is not a substitute for new data"
            ),
            "timings": timings,
        }

    out_dir = write_outputs(
        "a8_development_cut",
        {
            "segment_split": segments,
            "period_variant_detail": period_detail,
            "period_variant_summary": drifts,
            "template_trim_percentiles": trims,
        },
        summary,
    )
    return {"summary": summary, "out_dir": out_dir}


if __name__ == "__main__":
    result = run()
    print(json.dumps(result["summary"]["gates"], indent=2, ensure_ascii=False))
    print(f"outputs -> {result['out_dir']}")
