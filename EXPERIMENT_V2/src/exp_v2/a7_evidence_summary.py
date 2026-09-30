"""A7 — Unified effect-size evidence summary (three-layer forest figure).

Fixes GAP_ANALYSIS_AJP 8.2D: the three evidence layers currently share a table
while using incomparable error metrics. This module computes one comparable
effect size per layer (leave-out improvement with a moving-block bootstrap CI
for C1; paired fold deltas for C2; specificity percentile and the debiased
recovery curve for C3) and renders a single three-panel summary figure.

Sources: fresh computation on the archived dataset (C1, C3 percentile),
EXPERIMENT_V2 A5 outputs (C3 recovery), and the frozen local artifacts of the
h4s comparison and the v2l collision-history model (C2).
"""

from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import RESULTS_ROOT, WORKSPACE_ROOT, load_pivot, timed, write_outputs

EVIDENCE_ROOT = WORKSPACE_ROOT / "source_data_package" / "research_evidence"
H4S_CONTRASTS_PATH = EVIDENCE_ROOT / "h4s_holdout" / "fold_median_contrasts.csv"
V2L_SUMMARY_PATH = EVIDENCE_ROOT / "collision_history" / "v2l_experiment_summary.json"
A5_POOLED_PATH = RESULTS_ROOT / "a5_injection_debias" / "power_at_archived_snr.csv"
A5_SUMMARY_PATH = RESULTS_ROOT / "a5_injection_debias" / "summary.json"
BOOTSTRAP_REPLICATES = 400
BOOTSTRAP_BLOCK = 9
PSEUDO_COUNT = 256
TEMPLATE_SEED = 20260831


def _c1_gains_with_bootstrap(va: np.ndarray, fractions: np.ndarray) -> tuple[pd.DataFrame, dict]:
    from fh_retry.analysis import BLOCKS, _predict_fold
    from fh_retry.phase2 import _full_rank1_fit
    from fh_retry.phase3 import _moving_block_batch

    fitted, residual = _full_rank1_fit(va, fractions)
    masks = [(va >= low) & (va < high) for low, high in BLOCKS]

    def fold_errors(response: np.ndarray) -> list[float]:
        errors = []
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
            errors.append((e1 - e2) / e1)
        return errors

    observed = fold_errors(fractions)
    rng = np.random.default_rng(20260921)
    boot = np.empty((BOOTSTRAP_REPLICATES, len(masks)))
    for replicate in range(BOOTSTRAP_REPLICATES):
        noise = _moving_block_batch(residual, BOOTSTRAP_BLOCK, 1, rng)[0]
        boot[replicate] = fold_errors(fitted + noise)
    rows = []
    for index, (low, high) in enumerate(BLOCKS):
        lo, hi = np.quantile(boot[:, index], [0.025, 0.975])
        rows.append(
            {
                "block": index + 1,
                "low_V": low,
                "high_V_exclusive": high,
                "relative_gain": observed[index],
                "ci_low": float(lo),
                "ci_high": float(hi),
            }
        )
    detail = pd.DataFrame(rows)
    summary = {
        "median_gain": float(np.median(observed)),
        "bootstrap_median_ci": [
            float(np.quantile(np.median(boot, axis=1), 0.025)),
            float(np.quantile(np.median(boot, axis=1), 0.975)),
        ],
    }
    return detail, summary


def _c1_adequacy(va: np.ndarray, fractions: np.ndarray) -> pd.DataFrame:
    from fh_retry.analysis import BLOCKS, _predict_fold

    rows = []
    for index, (low, high) in enumerate(BLOCKS, start=1):
        test = (va >= low) & (va < high)
        train = ~test

        def error(period, rank, degree):
            _, predicted_z, observed_z = _predict_fold(
                va, fractions, train, test, period=period, rank=rank, trend_degree=degree
            )
            return float(np.sqrt(np.mean((predicted_z - observed_z) ** 2)))

        rank2 = error(11.55, 2, 3)
        trend5 = error(None, None, 5)
        pseudo = {p: error(float(p), 2, 3) for p in (9.0, 10.0, 13.5, 15.0)}
        rows.append(
            {
                "block": index,
                "rank2_rmse": rank2,
                "trend5_rmse": trend5,
                "best_pseudo_rmse": min(pseudo.values()),
                "rank2_beats_both": bool(rank2 < trend5 and rank2 < min(pseudo.values())),
            }
        )
    return pd.DataFrame(rows)


def _c3_percentile(pivot) -> float:
    from fh_retry.phase2 import (
        NIST_4S_EV,
        _operator_cv,
        _same_span_templates,
        _template_basis,
        _template_operators,
    )
    from fh_retry.phase2 import _analysis_arrays

    va, response = _analysis_arrays(pivot)
    nist_q, _ = _template_basis(va, NIST_4S_EV)
    _, operators = _template_operators(va, nist_q)
    nist_error, _ = _operator_cv(response, operators)
    errors = []
    for energies in _same_span_templates(PSEUDO_COUNT, TEMPLATE_SEED):
        q, _ = _template_basis(va, energies)
        _, pseudo_operators = _template_operators(va, q)
        error, _ = _operator_cv(response, pseudo_operators)
        errors.append(error)
    return float((np.asarray(errors) <= nist_error).mean())


def _c2_artifacts() -> tuple[pd.DataFrame, dict]:
    # Read the published receipts, rather than substituting rounded prose values.
    frame = pd.read_csv(H4S_CONTRASTS_PATH)
    primary = frame[frame["contrast"] == "h4s_minus_h1"]
    fold_rows = primary[["heldout_vr", "delta_rmse", "delta_nrmse"]].to_dict(orient="records")
    source = "fresh_read"
    v2l = {"source_available": V2L_SUMMARY_PATH.exists()}
    payload = json.loads(V2L_SUMMARY_PATH.read_text(encoding="utf-8"))
    v2l.update(
        {
            "h1_gates": f"{payload['h1']['passed_checks']}/{payload['h1']['total_checks']}",
            "h4s_gates": f"{payload['h4s']['posthoc_passed_checks']}/{payload['h4s']['posthoc_total_checks']}",
            "h1_nrmse": payload["h1"]["mean_curve_nrmse"],
            "h4s_nrmse": payload["h4s"]["mean_curve_nrmse"],
            "top_channel_fraction": max(payload["h4s"]["channel_rate_fractions"]),
        }
    )
    return pd.DataFrame(fold_rows), {"v2l": v2l, "fold_source": source}


def _figure(c1: pd.DataFrame, c2: pd.DataFrame, recovery: pd.DataFrame | None, percentile: float, s50: float | None, out_dir) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    ax = axes[0]
    colors = ["#b2182b" if value < 0.05 else "#2166ac" for value in c1["relative_gain"]]
    ax.bar(c1["block"].astype(str), 100 * c1["relative_gain"], color=colors, alpha=0.85)
    lower = np.maximum(100 * (c1["relative_gain"] - c1["ci_low"]).to_numpy(), 0.0)
    upper = np.maximum(100 * (c1["ci_high"] - c1["relative_gain"]).to_numpy(), 0.0)
    ax.errorbar(
        c1["block"].astype(str),
        100 * c1["relative_gain"],
        yerr=np.vstack([lower, upper]),
        fmt="none",
        ecolor="black",
        capsize=3,
        lw=1,
    )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("holdout block (Va)")
    ax.set_ylabel("rank-2 vs rank-1 improvement (%)")
    ax.set_title("C1 retarding-response structure")

    ax = axes[1]
    x = c2["heldout_vr"].astype(float)
    ax.bar(x - 0.0, c2["delta_rmse"], width=0.6, color=np.where(c2["delta_rmse"] < 0, "#2166ac", "#b2182b"), alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("held-out $V_r$ (V)")
    ax.set_ylabel("$\\Delta$RMSE  (H4s $-$ H1)")
    ax.set_title("C2 model-class dependence of the 4s advantage")

    ax = axes[2]
    if recovery is not None and len(recovery):
        ax.plot(recovery["true_span_eV"], 100 * recovery["truth_detection_power"], "o-", color="#2166ac")
        ax.axhline(5.0, color="grey", ls=":", lw=1.0)
        ax.text(recovery["true_span_eV"].min(), 6.5, "5% false-positive level", fontsize=8, color="grey")
        ax.set_xlabel("injected template span (eV)")
        ax.set_ylabel("detection power at archived SNR (%)")
        ax.set_ylim(0, 100)
    nist_span = 0.27971674
    ax.axvline(nist_span, color="grey", ls=":", lw=1.0)
    ax.text(nist_span, 2, " NIST 4s span", rotation=90, va="bottom", fontsize=8, color="grey")
    ax.set_title(f"C3 identifiability boundary (template pct. {100 * percentile:.1f}%)")
    fig.suptitle("Evidence summary: what the archived curves do and do not determine", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "evidence_summary.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "evidence_summary.pdf", bbox_inches="tight")
    plt.close(fig)
    return fig


def run() -> dict:
    timings: dict[str, float] = {}
    with timed("total_seconds", timings):
        pivot, audit = load_pivot()
        from fh_retry.phase2 import _retarding_fractions

        va, fractions, _ = _retarding_fractions(pivot, [0, 4, 6, 8, 10])

        with timed("c1_bootstrap_seconds", timings):
            c1, c1_summary = _c1_gains_with_bootstrap(va, fractions)
            adequacy = _c1_adequacy(va, fractions)
        with timed("c3_percentile_seconds", timings):
            percentile = _c3_percentile(pivot)
        with timed("artifact_reads_seconds", timings):
            c2, c2_meta = _c2_artifacts()
            recovery = A5_POOLED_PATH if A5_POOLED_PATH.exists() else None
            recovery = pd.read_csv(recovery) if recovery is not None else None
            s50 = None
            if A5_SUMMARY_PATH.exists():
                payload = json.loads(A5_SUMMARY_PATH.read_text(encoding="utf-8"))
                gates_a5 = payload["summary"]["gates"]
                s50 = gates_a5.get("nist_span_detection_power_at_archived_snr")

        effect_rows = [
            {
                "claim": "C1",
                "layer": "retarding-response rank",
                "metric": "median leave-block improvement",
                "value": c1_summary["median_gain"],
                "ci_low": c1_summary["bootstrap_median_ci"][0],
                "ci_high": c1_summary["bootstrap_median_ci"][1],
                "source": "fresh",
            },
            {
                "claim": "C1",
                "layer": "retarding-response rank",
                "metric": "blocks beating trend5 and best pseudo",
                "value": float(adequacy["rank2_beats_both"].sum()),
                "ci_low": None,
                "ci_high": None,
                "source": "fresh",
            },
            {
                "claim": "C2",
                "layer": "phenomenological kernel",
                "metric": "H4s-H1 delta RMSE at Vr=4 (largest pro-4s fold)",
                "value": float(c2[c2["heldout_vr"] == 4.0]["delta_rmse"].iloc[0]) if len(c2) else None,
                "ci_low": None,
                "ci_high": None,
                "source": c2_meta["fold_source"],
            },
            {
                "claim": "C2",
                "layer": "phenomenological kernel",
                "metric": "H4s-H1 delta RMSE at Vr=10 (reversal)",
                "value": float(c2[c2["heldout_vr"] == 10.0]["delta_rmse"].iloc[0]) if len(c2) else None,
                "ci_low": None,
                "ci_high": None,
                "source": c2_meta["fold_source"],
            },
            {
                "claim": "C2",
                "layer": "collision-history kernel (v2l)",
                "metric": "top channel fraction of H4s rate",
                "value": c2_meta["v2l"].get("top_channel_fraction"),
                "ci_low": None,
                "ci_high": None,
                "source": "fresh_read" if c2_meta["v2l"]["source_available"] else "frozen_documented_values",
            },
            {
                "claim": "C3",
                "layer": "template specificity",
                "metric": "NIST percentile among 256 matched pseudo-templates",
                "value": percentile,
                "ci_low": None,
                "ci_high": None,
                "source": "fresh",
            },
        ]
        effects = pd.DataFrame(effect_rows)
        figure_dir = RESULTS_ROOT / "a7_evidence_summary" / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        with timed("figure_seconds", timings):
            _figure(c1, c2, recovery, percentile, s50, figure_dir)
        summary = {
            "experiment": "A7_evidence_summary",
            "dataset_sha256": audit["sha256"],
            "c1": c1_summary,
            "c2_meta": c2_meta,
            "c3_percentile_recomputed": percentile,
            "s50_from_a5": s50,
            "figure": str(figure_dir / "evidence_summary.png"),
            "note": "effect sizes are direction-aligned: larger C1 = more structure; C2 deltas < 0 favor H4s; C3 percentile near chance = non-specific",
            "timings": timings,
        }

    out_dir = write_outputs(
        "a7_evidence_summary",
        {"c1_block_gains_bootstrap": c1, "c1_adequacy": adequacy, "c2_fold_deltas": c2, "effect_table": effects},
        summary,
    )
    return {"summary": summary, "out_dir": out_dir}


if __name__ == "__main__":
    result = run()
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False, default=str))
    print(f"outputs -> {result['out_dir']}")
