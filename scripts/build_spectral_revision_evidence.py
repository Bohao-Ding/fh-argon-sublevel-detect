"""Publish revised aggregate evidence and real-data parameter receipts separately."""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import scipy
import torch
from analyze_spectral_revision import analyze

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sublevel_detect.validation_common import atomic_json_dump, sha256_file


def build(run, companion, phase, destination):
    result = json.loads((run / "summary.json").read_text())
    control = json.loads((companion / "summary.json").read_text())
    phase_control = json.loads((phase / "summary.json").read_text())
    if not (result["ok"] and result["scientific_run"] and result["outer_folds"] == 4 and result["bootstrap_count"] == 30
            and result["synthetic_datasets"] == 20 and control["ok"] and control["datasets"] == 20 and phase_control["ok"]):
        raise ValueError("complete_formal_and_matched_controls_required")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("refusing_to_replace_evidence_release")
    analysis = run / "analysis"
    analyze(run, analysis, companion, phase)
    destination.mkdir(parents=True, exist_ok=True)
    sources = {}

    def record(target, original, transform="exact byte copy"):
        sources[target.relative_to(destination).as_posix()] = {
            "source_path": original.relative_to(ROOT).as_posix(), "source_sha256": sha256_file(original),
            "transformation": transform}

    names = ("outer_scores.csv", "inner_scores.csv", "selection.csv", "concentration_status.csv", "fit_status.csv",
             "profiles.csv", "bootstrap.csv", "synthetic_scores.csv", "recovery.csv", "fixed_kernel_recovery.csv",
             "quadrature.csv", "stress_scores.csv", "training_scores.csv", "run_identity.json", "frozen_selection.json", "summary.json")
    for name in names:
        shutil.copyfile(run / name, destination / name)
        record(destination / name, run / name)
    for original in sorted(analysis.glob("*")):
        if original.is_file():
            shutil.copyfile(original, destination / original.name)
            record(destination / original.name, original, "start-aware aggregation; scripts/analyze_spectral_revision.py")
    for name in ("matched_recovery.csv", "noiseless_recovery.csv", "quadrature.csv", "run_identity.json", "summary.json"):
        target = destination / f"matched_{name}"
        shutil.copyfile(companion / name, target)
        record(target, companion / name)
    for name in ("phase_scores.csv", "fit_status.csv", "quadrature.csv", "run_identity.json", "summary.json"):
        target = destination / (name if name == "phase_scores.csv" else f"phase_{name}")
        shutil.copyfile(phase / name, target)
        record(target, phase / name)
    points = pd.read_csv(run / "prediction_points.csv")
    real = points[points.scope.isin(["outer_0V", "outer_4V", "outer_6V", "outer_8V", "final_training", "stress_10V"])]
    target = destination / "real_prediction_points.csv"
    real.to_csv(target, index=False)
    record(target, run / "prediction_points.csv", "retain actual outer, final training and frozen stress points")
    densities = pd.read_csv(run / "distributions.csv")
    keep = densities.scope.isin(["outer_0V", "outer_4V", "outer_6V", "outer_8V", "final"]) | densities.scope.str.startswith("profile/")
    densities.loc[keep].to_csv(destination / "distribution_reference.csv", index=False)
    record(destination / "distribution_reference.csv", run / "distributions.csv", "retain real-data outer/final/sensitivity measures")
    needed = set(real.unit) | set(densities.loc[keep, "unit"]) | set(pd.read_csv(run / "bootstrap.csv").unit) | set(pd.read_csv(phase / "fit_status.csv").unit)
    (destination / "units").mkdir()
    for unit in sorted(needed):
        original = run / "units" / unit[:20] / "fit.json"
        if not original.exists():
            original = phase / "units" / unit[:20] / "fit.json"
        info = json.loads(original.read_text())
        published = {k: v for k, v in info.items() if k not in ("history", "seconds")}
        target = destination / "units" / f"{unit[:20]}.json"
        atomic_json_dump(published, target)
        record(target, original, "parameter-only receipt; omit optimizer history and elapsed time")
    provenance = {"input": "data/argon/FHdata.xlsx", "input_sha256": sha256_file(ROOT / "data/argon/FHdata.xlsx"),
        "run": run.relative_to(ROOT).as_posix(), "matched_control_run": companion.relative_to(ROOT).as_posix(),
        "phase_control_run": phase.relative_to(ROOT).as_posix(),
        "sources": sources, "analysis_source_sha256": sha256_file(ROOT / "scripts/analyze_spectral_revision.py"),
        "excluded": ["checkpoints", "synthetic point predictions and densities", "manuscripts", "figures", "optimizer histories"]}
    atomic_json_dump(provenance, destination / "SOURCE_MAP.json")
    atomic_json_dump({"record_role": "publication and parameter replay environment",
        "python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
        "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__,
        "model_dtype": "float64", "science_device": "cpu", "numerical_threads_per_worker": 1},
        destination / "ENVIRONMENT.json")
    (destination / "README.md").write_text(
        "# Revised effective-spectrum evidence\n\n"
        "Independent release; historical spectrum and calibration evidence are unchanged. "
        "One 805-point archive; 644 points (0/4/6/8 V) train and select; unchanged 10 V is scored only after freezing.\n\n"
        "`claim_summary.json` and `outer_seed_medians.csv` summarize actual complete-condition prediction. "
        "Starts are optimizer variation, not new measurements. `selection.csv` retains nested one-SE decisions. "
        "`phase_scores.csv` compares fixed phase-on/off candidates at each training-selected C penalty; "
        "these are matched diagnostics, not reselection on outer outcomes. "
        "`concentration.csv` and `peak_threshold_sensitivity.csv` distinguish a peak, diffuse/boundary spectra, "
        "and agreement across starts. Atomic quantiles and integrated continuous CDF quantiles use different definitions.\n\n"
        "`bootstrap_seed_medians.csv` reports 30 conditional block resamples after median over three starts. "
        "`bootstrap_density_band.csv` uses those replicate-centered densities. Kernel, domain and smoothing are fixed; "
        "these intervals do not replace condition holdouts or sensitivity fits.\n\n"
        "`recovery_seed_medians.csv` uses three-curve synthetic recovery; `matched_recovery_seed_medians.csv` "
        "compares free/fixed kernels on the same four curves and same noise realization. Synthetic controls are not "
        "new measurements or a formal power calculation. Effective response masses are not atomic populations.\n\n"
        "Parameter-only receipts reproduce real predictions and distributions without checkpoints:\n\n"
        "```powershell\npython scripts/replay_spectral_revision.py\n```\n\n"
        "`fit_status.csv` retains all stationary/iteration-limit judgments; stationarity is local. "
        "`quadrature.csv` tests integration, not experimental resolution. `SOURCE_MAP.json` records original byte "
        "hashes and transformations. `FILE_INDEX.csv` omits itself/checksum list; `SHA256SUMS.txt` covers the index "
        "and every other release file except itself. Full outputs and plots remain local.\n", encoding="utf-8")
    paths = sorted(p for p in destination.rglob("*") if p.is_file())
    pd.DataFrame([{"path": p.relative_to(destination).as_posix(), "bytes": p.stat().st_size,
                   "sha256": sha256_file(p), **sources.get(p.relative_to(destination).as_posix(),
                   {"source_path": "generated", "source_sha256": "", "transformation": "release metadata"})}
                  for p in paths]).to_csv(destination / "FILE_INDEX.csv", index=False)
    paths = sorted(p for p in destination.rglob("*") if p.is_file())
    (destination / "SHA256SUMS.txt").write_text("".join(
        f"{sha256_file(p)}  {p.relative_to(destination).as_posix()}\n" for p in paths), encoding="utf-8")
    return {"files": len(paths) + 1, "parameter_receipts": len(needed), "real_points": len(real),
            "bytes": sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "output/spectral_revision")
    parser.add_argument("--companion", type=Path, default=ROOT / "output/spectral_revision_matched_recovery")
    parser.add_argument("--phase", type=Path, default=ROOT / "output/spectral_revision_phase_comparison")
    parser.add_argument("--destination", type=Path, default=ROOT / "source_data_package/spectral_revision_evidence")
    args = parser.parse_args()
    print(json.dumps(build(args.run.resolve(), args.companion.resolve(), args.phase.resolve(), args.destination.resolve()), indent=2))
