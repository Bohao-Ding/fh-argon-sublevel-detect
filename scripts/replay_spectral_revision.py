"""Verify revised evidence bytes, parameter predictions and continuous summaries."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sublevel_detect.spectral_revision import spectrum_summary
from sublevel_detect.spectral_revision_pipeline import SCHEMA, make_model, restore_parameters, revision_sources
from sublevel_detect.validation_common import sha256_file
from sublevel_detect.validation_holdout import prediction_metrics


def replay(evidence):
    torch.set_num_threads(1)
    files = 0
    for line in (evidence / "SHA256SUMS.txt").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if sha256_file(evidence / name) != expected:
            raise ValueError(f"evidence_byte_mismatch: {name}")
        files += 1
    identity = json.loads((evidence / "run_identity.json").read_text())
    if identity["schema"] != SCHEMA or identity["sources"] != revision_sources():
        raise ValueError("revision_source_schema_mismatch")
    if sha256_file(ROOT / "data/argon/FHdata.xlsx") != identity["input_sha256"]:
        raise ValueError("input_byte_mismatch")
    nets, receipts = {}, {}

    def load(unit):
        if unit not in nets:
            r = json.loads((evidence / "units" / f"{unit[:20]}.json").read_text())
            net = make_model(r["spec"])
            restore_parameters(net, r["raw_parameter_values"])
            nets[unit], receipts[unit] = net, r
        return nets[unit], receipts[unit]["spec"]

    points = pd.read_csv(evidence / "real_prediction_points.csv")
    largest, subset_error, count = 0., 0., 0
    recalculated = []
    for (unit, scope, vr), group in points.groupby(["unit", "scope", "Vr"]):
        group = group.sort_values("Va")
        net, spec = load(unit)
        va = torch.tensor(group.Va.to_numpy(), dtype=torch.float64)
        retarding = torch.full_like(va, vr)
        mode = "curve" if scope == "final_training" else "neutral"
        index = spec["training_vr"].index(vr) if mode == "curve" else 0
        with torch.no_grad():
            values = net(va, retarding, index, mode).numpy()
            selected = np.arange(0, len(va), 17)
            partial = net(va[selected], retarding[selected], index, mode).numpy()
        largest = max(largest, float(np.max(np.abs(values - group.predicted_uA.to_numpy()))))
        subset_error = max(subset_error, float(np.max(np.abs(partial - values[selected]))))
        recalculated.append(dict(unit=unit, scope=scope, Vr=vr, **prediction_metrics(group.observed_uA.to_numpy(), values,
                                                                                 denominator_observed=group.observed_uA.to_numpy())))
        count += len(group)
    if largest >= 1e-9 or subset_error >= 1e-9:
        raise ValueError(f"revision_prediction_replay_failed: {largest}, subset {subset_error}")
    score_error = 0.
    computed = pd.DataFrame(recalculated)
    for name in ("outer_scores", "stress_scores", "training_scores"):
        table = pd.read_csv(evidence / f"{name}.csv")
        joined = table.merge(computed, left_on=["unit", "heldout_vr" if name != "training_scores" else "Vr"],
                             right_on=["unit", "Vr"], suffixes=("_reported", "_replayed"), validate="one_to_one")
        if len(joined) != len(table):
            raise ValueError(f"incomplete_score_replay: {name}")
        for metric in ("rmse", "nrmse"):
            score_error = max(score_error, float((joined[f"{metric}_reported"] - joined[f"{metric}_replayed"]).abs().max()))
    if score_error >= 1e-9:
        raise ValueError(f"revision_score_replay_failed: {score_error}")
    density_error, summary_error = 0., 0.
    densities = pd.read_csv(evidence / "distribution_reference.csv")
    for unit, group in densities.groupby("unit"):
        net, _ = load(unit)
        if net.family != "H1":
            with torch.no_grad():
                value = net.density().numpy()
            density_error = max(density_error, float(np.max(np.abs(value - group.sort_values("energy_eV").density_per_eV.to_numpy()))))
        summary = spectrum_summary(net)
        for key, value in summary.items():
            stored = receipts[unit]["distribution"][key]
            if isinstance(value, (float, int)):
                summary_error = max(summary_error, abs(value - stored))
            elif value != stored:
                raise ValueError(f"distribution_status_mismatch: {unit}, {key}")
    if density_error >= 1e-9 or summary_error >= 1e-9:
        raise ValueError(f"revision_distribution_replay_failed: {density_error}, {summary_error}")
    return dict(ok=True, verified_files=files, parameter_models=len(nets), replayed_points=count,
                max_prediction_difference_uA=largest, max_subset_difference_uA=subset_error,
                max_score_difference=score_error, max_density_difference=density_error,
                max_summary_difference=summary_error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=ROOT / "source_data_package/spectral_revision_evidence")
    print(json.dumps(replay(parser.parse_args().evidence.resolve()), indent=2))
