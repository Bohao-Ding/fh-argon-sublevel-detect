"""Publish a minimal, independently replayable receipt set from a completed run."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
from summarize_spectral_evidence import summarize
from summarize_spectral_concentration import summarize as summarize_concentration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sublevel_detect.validation_common import atomic_json_dump, sha256_file


def build(run: Path, destination: Path) -> dict:
    result = json.loads((run / "result.json").read_text())
    if not result["scientific_run"] or result["outer_folds"] != 4 or result["bootstrap_count"] != 30 or result["synthetic_datasets"] != 20:
        raise ValueError("A completed formal protocol is required; smoke evidence cannot be published")
    grid = run / f"grid_{result['grid_step_eV']:g}"
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Refusing to replace an existing evidence release")
    destination.mkdir(parents=True, exist_ok=True)
    sources = {}

    def source_record(target: Path, original: Path, transformation: str = "exact byte copy") -> None:
        sources[target.relative_to(destination).as_posix()] = {
            "source_path": original.relative_to(ROOT).as_posix(),
            "source_sha256": sha256_file(original), "transformation": transformation}

    names = ("outer_scores.csv", "outer_seed_medians.csv", "inner_scores.csv", "selection.csv",
             "fit_status.csv", "bootstrap.csv", "synthetic_scores.csv", "synthetic_recovery_counts.csv",
             "kernel_profile_scores.csv", "quadrature.csv", "final_training_scores.csv", "stress_scores.csv",
             "summary.json", "windows.json", "frozen_selection.json", "identity.json")
    for name in names:
        shutil.copyfile(grid / name, destination / name)
        source_record(destination / name, grid / name)
    shutil.copyfile(run / "run_identity.json", destination / "run_identity.json")
    source_record(destination / "run_identity.json", run / "run_identity.json")
    shutil.copyfile(run / "result.json", destination / "formal_result.json")
    source_record(destination / "formal_result.json", run / "result.json")
    points = pd.read_csv(grid / "prediction_points.csv")
    real = points[(points.scope.str.startswith("outer_") & points.heldout_vr.notna())
                  | points.scope.isin(["final_training", "stress_10V"])]
    real.to_csv(destination / "real_prediction_points.csv", index=False)
    source_record(destination / "real_prediction_points.csv", grid / "prediction_points.csv",
                  "retain outer whole-curve predictions, final training, and frozen 10 V stress only")
    density = pd.read_csv(grid / "distributions.csv")
    retained = (density.scope == "final") | (density.family.eq("C") & (
        density.scope.str.startswith("outer_") | density.scope.str.startswith("final/width_")
        | density.scope.str.startswith("bootstrap_")))
    density.loc[retained].to_csv(destination / "distribution_reference.csv", index=False)
    source_record(destination / "distribution_reference.csv", grid / "distributions.csv",
                  "retain final measures and continuous outer/profile/bootstrap densities")
    needed = set(real.unit) | set(density.loc[retained, "unit"])
    status = pd.read_csv(grid / "fit_status.csv").drop_duplicates("unit").set_index("unit")
    exported = set()

    def export_unit(unit: str) -> None:
        short = unit[:20]
        if short in exported:
            return
        original = grid / "units" / short / "fit.json"
        info = json.loads(original.read_text())
        spec = dict(info["spec"])
        if spec["coarse"]:
            parent = Path(spec["coarse"]).name
            parent_info = json.loads((grid / "units" / parent / "fit.json").read_text())
            export_unit(parent_info["identity"])
            spec["coarse"] = parent
        spec["training_vr"] = json.loads(status.loc[unit, "training_vr"])
        published = {key: value for key, value in info.items() if key not in ("history", "seconds", "spec")}
        published["spec"] = spec
        target = destination / "units" / f"{short}.json"
        atomic_json_dump(published, target)
        source_record(target, original, "omit optimizer history/time; use relative parent ID and explicit training voltages")
        exported.add(short)

    for unit in sorted(needed):
        export_unit(unit)
    provenance = {"run_relative_path": run.relative_to(ROOT).as_posix(),
        "formal_result_sha256": sha256_file(run / "result.json"),
        "shared_input_path": "data/argon/FHdata.xlsx",
        "shared_input_sha256": sha256_file(ROOT / "data/argon/FHdata.xlsx"),
        "sources": sources, "excluded": ["checkpoints", "optimizer histories", "full synthetic point predictions",
                                            "full synthetic densities", "manuscript PDFs", "figures"]}
    atomic_json_dump(summarize(destination), destination / "claim_summary.json")
    source_record(destination / "matched_comparisons.csv", grid / "outer_scores.csv",
                  "paired condition contrasts after median over optimizer starts")
    source_record(destination / "claim_summary.json", grid / "summary.json",
                  "scope-aware aggregation of the source tables listed in claim_summary.sources")
    summarize_concentration(destination).to_csv(destination / "concentration.csv", index=False)
    source_record(destination / "concentration.csv", grid / "distributions.csv",
                  "final and width-profile C/G1: normalized piecewise-linear density integrals and connected half-height lobe")
    provenance["concentration_script_sha256"] = sha256_file(ROOT / "scripts/summarize_spectral_concentration.py")
    atomic_json_dump(provenance, destination / "SOURCE_MAP.json")
    (destination / "README.md").write_text(
        "# Minimal spectral inference evidence\n\n"
        "One archived dataset; 0/4/6/8 V training (644 points), unchanged 10 V final known-anomaly stress. "
        "Effective excitation response density is conditional on the kernel; it is not an atomic continuum.\n\n"
        f"Completed: four nested outer holdouts, three outer/final starts, 30 conditional residual resamples, "
        f"20 synthetic datasets. Final grid: {result['grid_step_eV']} eV. "
        f"Fit statuses (unique units): {result['fit_status_counts']}. Epoch-limit fits are not convergence.\n\n"
        "`concentration.csv` locates the connected main half-height lobe and its mass, and compares the known "
        "Ar I 4s range (11.54835442–11.82807116 eV). Continuous inversion locates effective energy concentration; "
        "it does not naturally imply four separated states. Local discrete comparisons are supplementary exploration. "
        "Missing inner local choices remain missing; diagnostic candidate scores are not substituted for a nested selected-model score.\n\n"
        "`summary.json` and `outer_seed_medians.csv` give complete-curve prediction scores. "
        "`matched_comparisons.csv` restricts contrasts to conditions available for both candidates; "
        "`claim_summary.json` excludes incomplete candidates from four-condition means. "
        "`selection.csv` contains every one-SE decision; `windows.json` records eligible windows. "
        "`bootstrap.csv` uses conditional training-objective candidate ranking, not holdout selection. "
        "`synthetic_recovery_counts.csv` gives best held-out candidates and false-splitting controls; "
        "these are synthetic checks, not new observations or sufficient statistical power.\n\n"
        "Parameter-only unit receipts allow replay without checkpoint files. From the repository root:\n\n"
        "```powershell\npython scripts/replay_spectral_evidence.py\n```\n\n"
        "Old 805-point evidence and A1–A15 retain their original scope; A4 remains unexecuted. "
        "`SOURCE_MAP.json` maps original bytes and transformations. `FILE_INDEX.csv` indexes published files "
        "excluding itself and the checksum list; `SHA256SUMS.txt` covers all files except itself, including the index.\n",
        encoding="utf-8")
    rows = []
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            name = path.relative_to(destination).as_posix()
            rows.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path),
                         **sources.get(name, {"source_path": "generated", "source_sha256": "",
                                             "transformation": "release metadata"})})
    pd.DataFrame(rows).to_csv(destination / "FILE_INDEX.csv", index=False)
    checksum_paths = sorted(p for p in destination.rglob("*") if p.is_file())
    (destination / "SHA256SUMS.txt").write_text("".join(
        f"{sha256_file(path)}  {path.relative_to(destination).as_posix()}\n" for path in checksum_paths), encoding="utf-8")
    return {"published_files": len(checksum_paths) + 1, "parameter_receipts": len(exported),
            "real_prediction_rows": len(real), "distribution_rows": int(retained.sum()),
            "bytes": sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "output/spectral_v1")
    parser.add_argument("--destination", type=Path, default=ROOT / "source_data_package/spectral_evidence_v1")
    args = parser.parse_args()
    print(json.dumps(build(args.run.resolve(), args.destination.resolve()), indent=2))
