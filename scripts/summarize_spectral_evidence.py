"""Summarize frozen comparisons without averaging over missing conditions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def summarize(evidence: Path) -> dict:
    scores = pd.read_csv(evidence / "outer_scores.csv")
    medians = scores.groupby(["heldout_vr", "family"]).nrmse.median().unstack()
    comparisons = []
    for family in medians:
        for baseline in ("H1", "d1", "g1", "equal4"):
            if family == baseline:
                continue
            paired = medians[[family, baseline]].dropna()
            comparisons.append({"family": family, "baseline": baseline,
                "matched_conditions": len(paired),
                "heldout_vr": json.dumps(paired.index.tolist()),
                "mean_candidate_nrmse": float(paired[family].mean()),
                "mean_baseline_nrmse": float(paired[baseline].mean()),
                "mean_difference": float((paired[family] - paired[baseline]).mean()),
                "conditions_better": int((paired[family] < paired[baseline]).sum())})
    pd.DataFrame(comparisons).to_csv(evidence / "matched_comparisons.csv", index=False)
    status = pd.read_csv(evidence / "fit_status.csv")
    final = status[status.phase.eq("final/coarse")]
    fields = ["mode_eV", "mean_eV", "median_eV", "sd_eV", "q05_eV", "q95_eV", "kernel_width_V"]

    def ranges(frame: pd.DataFrame) -> dict:
        return {column: {"median": float(frame[column].median()),
                         "min": float(frame[column].min()), "max": float(frame[column].max())}
                for column in fields}

    profile = status[status.phase.eq("final/kernel_profile")].copy()
    profile["fixed_width_V"] = profile.kernel_width_V.round(5)
    recovery = pd.read_csv(evidence / "synthetic_recovery_counts.csv")
    bootstrap = pd.read_csv(evidence / "bootstrap.csv")
    stress = pd.read_csv(evidence / "stress_scores.csv")
    quadrature = pd.read_csv(evidence / "quadrature.csv")
    return {"score_rule": "median over optimizer starts within each held-out condition; matched conditions only",
        "outer_available_conditions": medians.count().astype(int).to_dict(),
        "outer_four_condition_mean_nrmse": medians.loc[:, medians.count().eq(4)].mean().to_dict(),
        "final_coarse_start_ranges": {family: ranges(group) for family, group in final.groupby("family")},
        "final_width_profile_start_ranges": {str(width): ranges(group) for width, group in profile.groupby("fixed_width_V")},
        "bootstrap_training_objective_winners": bootstrap.training_objective_winner.fillna("no_eligible_candidate").value_counts().astype(int).to_dict(),
        "synthetic_best_local_counts": {truth: group.best_local_mean_holdout.fillna("no_eligible_candidate").value_counts().astype(int).to_dict()
                                         for truth, group in recovery.groupby("truth")},
        "synthetic_discrete_improvement_counts": recovery.groupby("truth").discrete_beats_delta_and_gaussian.sum().astype(int).to_dict(),
        "stress_seed_median_nrmse": stress.groupby("family").nrmse.median().to_dict(),
        "quadrature_max_difference_uA": float(quadrature.max_difference_uA.max()),
        "quadrature_units": len(quadrature), "quadrature_all_pass": bool(quadrature.passed.all()),
        "sources": ["outer_scores.csv", "fit_status.csv", "bootstrap.csv", "synthetic_recovery_counts.csv", "stress_scores.csv", "quadrature.csv"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.evidence), indent=2, allow_nan=False))
