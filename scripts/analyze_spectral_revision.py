"""Aggregate revised spectral evidence without treating starts as measurements."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sublevel_detect.validation_common import atomic_json_dump, sha256_file
from sublevel_detect.model import json_ready
from sublevel_detect.spectral_revision import concentration_status

QUANTITIES = ("mode_eV", "mean_eV", "median_eV", "sd_eV", "kernel_width_V", "phase_beta",
              "peak_fwhm_eV", "peak_halfheight_mass", "ar4s_mass", "below_ar4s_mass")


def ranges(frame, keys, columns):
    return frame.groupby(keys)[list(columns)].agg(["median", "min", "max"]).rename_axis(columns=["quantity", "statistic"])


def flatten(frame):
    frame = frame.copy()
    frame.columns = ["_".join(c) if isinstance(c, tuple) else c for c in frame.columns]
    return frame.reset_index()


def recovery_table(frame, keys):
    center = frame.groupby(keys)[["mode_eV", "mean_eV", "sd_eV", "truth_mode_eV", "truth_mean_eV", "truth_sd_eV"]].median().reset_index()
    for name in ("mode", "mean", "sd"):
        center[f"{name}_error_eV"] = center[f"{name}_eV"] - center[f"truth_{name}_eV"]
        center[f"{name}_absolute_error_eV"] = center[f"{name}_error_eV"].abs()
    count = frame.groupby(keys).peak_status.apply(lambda s: int((s == "concentrated").sum())).rename("concentrated_starts")
    return center.merge(count.reset_index(), on=keys)


def analyze(run, output, companion=None, phase=None):
    summary = json.loads((run / "summary.json").read_text())
    if not summary["ok"]:
        raise ValueError("completed_revision_required")
    output.mkdir(parents=True, exist_ok=True)
    read = lambda name: pd.read_csv(run / f"{name}.csv")
    status, outer, stress, training = [read(name) for name in ("fit_status", "outer_scores", "stress_scores", "training_scores")]
    distributions = read("distributions")
    scopes = distributions[["scope", "unit", "family", "seed"]].drop_duplicates()
    real_scopes = scopes[scopes.scope.eq("final") | scopes.scope.str.startswith("outer_") | scopes.scope.str.startswith("profile/")]
    concentrations = real_scopes.merge(status.drop(columns=["family", "seed"]), on="unit", validate="many_to_one")
    concentrations.to_csv(output / "concentration.csv", index=False)
    outer_medians = outer.groupby(["heldout_vr", "family"])[["nrmse", "rmse", "bias_uA"]].median().reset_index()
    outer_medians.to_csv(output / "outer_seed_medians.csv", index=False)
    flatten(ranges(outer, ["heldout_vr", "family"], ["nrmse", "rmse", "bias_uA"])).to_csv(output / "outer_start_ranges.csv", index=False)
    final = concentrations[concentrations.scope.eq("final")]
    usable = [c for c in QUANTITIES if c in final]
    flatten(ranges(final, ["family"], usable)).to_csv(output / "final_parameter_ranges.csv", index=False)
    boot = read("bootstrap")
    boot_medians = boot.groupby("replicate")[usable].median().reset_index()
    boot_medians.to_csv(output / "bootstrap_seed_medians.csv", index=False)
    boot_status = pd.DataFrame([dict(replicate=rep, **concentration_status(g.to_dict("records")))
                                for rep, g in boot.groupby("replicate")])
    boot_status.to_csv(output / "bootstrap_concentration_status.csv", index=False)
    boot_density = distributions[distributions.scope.str.startswith("bootstrap_")]
    boot_center = boot_density.groupby(["scope", "energy_eV"]).density_per_eV.median()
    band = boot_center.groupby("energy_eV").quantile([.025, .5, .975]).unstack()
    band.columns = ["lower_density", "median_density", "upper_density"]
    band.reset_index().to_csv(output / "bootstrap_density_band.csv", index=False)
    recovery = recovery_table(read("recovery"), ["truth", "replicate", "heldout_vr"])
    recovery.to_csv(output / "recovery_seed_medians.csv", index=False)
    threshold_rows = []
    continuous = concentrations[concentrations.family.eq("C")]
    for scope, group in continuous.groupby("scope"):
        for threshold in (1.25, 1.5, 2.0):
            # These descriptive thresholds do not feed any training decision.
            eligible = (~group.peak_status.eq("boundary_peak") & (group.peak_to_uniform >= threshold)
                        & (group.peak_fwhm_eV < (9 if scope == "profile/expanded_domain" else 7) - 1e-8))
            mode_range = float(group.mode_eV.max() - group.mode_eV.min())
            threshold_rows.append(dict(scope=scope, threshold=threshold, qualifying_starts=int(eligible.sum()),
                starts=len(group), mode_range_eV=mode_range, stable=bool(eligible.all() and mode_range <= .5)))
    pd.DataFrame(threshold_rows).to_csv(output / "peak_threshold_sensitivity.csv", index=False)
    matched = None
    if companion is not None:
        matched = recovery_table(pd.read_csv(companion / "matched_recovery.csv"), ["truth", "replicate", "kernel"])
        matched.to_csv(output / "matched_recovery_seed_medians.csv", index=False)
    phase_scores = None
    if phase is not None:
        phase_scores = pd.read_csv(phase / "phase_scores.csv").groupby(["heldout_vr", "family", "phase_response"]).nrmse.median().reset_index()
        phase_scores.to_csv(output / "phase_seed_medians.csv", index=False)
    profiles = concentrations[concentrations.scope.str.startswith("profile/")]
    flatten(ranges(profiles, ["scope"], usable)).to_csv(output / "profile_ranges.csv", index=False)
    claims = {
        "schema": summary["schema"], "scientific_run": summary["scientific_run"],
        "choices": summary["choices"],
        "outer_mean_seed_median_nrmse": outer_medians.groupby("family").nrmse.mean().to_dict(),
        "stress_seed_median_nrmse": stress.groupby("family").nrmse.median().to_dict(),
        "training_mean_seed_median_nrmse": training.groupby(["Vr", "family"]).nrmse.median().groupby("family").mean().to_dict(),
        "final": final.groupby("family")[usable].median().to_dict("index"),
        "final_start_ranges": flatten(ranges(final, ["family"], usable)).to_dict("records"),
        "concentration_status": summary["concentration_status"],
        "bootstrap_rep_median_intervals": {c: boot_medians[c].quantile([.025, .5, .975]).to_dict() for c in usable},
        "bootstrap_qualifying_start_counts": boot.peak_status.value_counts().to_dict(),
        "bootstrap_concentration_status_counts": boot_status.status.value_counts().to_dict(),
        "recovery": recovery.groupby("truth")[["mode_absolute_error_eV", "sd_eV", "sd_absolute_error_eV"]].median().to_dict("index"),
        "matched_recovery": None if matched is None else matched.groupby(["truth", "kernel"])[["mode_absolute_error_eV", "sd_eV", "sd_absolute_error_eV"]].median().reset_index().to_dict("records"),
        "matched_phase_diagnostic": None if phase_scores is None else phase_scores.groupby(["family", "phase_response"]).nrmse.mean().reset_index().to_dict("records"),
        "final_optimizer_status": final.status.value_counts().to_dict(),
        "fit_status_counts": summary["status_counts"], "unique_fits": summary["unique_fits"],
        "quadrature_max_difference_uA": summary["quadrature_max_difference_uA"],
        "scope": "One archive. Whole-condition holdouts; optimization starts are computational variation. Synthetic recovery is not new measurement.",
        "source_sha256": {p.name: sha256_file(p) for p in run.glob("*.csv")}
    }
    claims = json_ready(claims)
    atomic_json_dump(claims, output / "claim_summary.json")
    return claims


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "output/spectral_revision")
    parser.add_argument("--output", type=Path, default=ROOT / "output/spectral_revision/analysis")
    parser.add_argument("--companion", type=Path)
    parser.add_argument("--phase", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run.resolve(), args.output.resolve(), args.companion, args.phase), indent=2))
