"""Verify published bytes and replay real predictions from parameter-only receipts."""
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
from sublevel_detect.spectral_models import EffectiveSpectrum, LOCAL_FAMILIES, LocalSpectrum
from sublevel_detect.spectral_pipeline import SCHEMA, source_hashes
from sublevel_detect.validation_common import sha256_file


def replay(evidence: Path) -> dict:
    torch.set_num_threads(1)
    files = 0
    for line in (evidence / "SHA256SUMS.txt").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        if sha256_file(evidence / relative) != expected:
            raise ValueError(f"Evidence byte mismatch: {relative}")
        files += 1
    run_identity = json.loads((evidence / "run_identity.json").read_text())
    if run_identity["schema"] != SCHEMA or run_identity["sources"] != source_hashes():
        raise ValueError("Evidence source/schema mismatch")
    if sha256_file(ROOT / "data/argon/FHdata.xlsx") != run_identity["input_sha256"]:
        raise ValueError("Input byte mismatch")
    nets = {}

    def load(unit: str):
        unit = unit[:20]
        if unit in nets:
            return nets[unit]
        receipt = json.loads((evidence / "units" / f"{unit}.json").read_text())
        spec = receipt["spec"]
        if spec["family"] in LOCAL_FAMILIES:
            net = LocalSpectrum(load(spec["coarse"])[0], spec["family"], spec["window"])
        else:
            net = EffectiveSpectrum(spec["n_curves"], spec["family"], seed=spec["seed"],
                excitation=spec["excitation"], step=spec["step"], fixed_width=spec["fixed_width"])
        with torch.no_grad():
            for name, parameter in net.named_parameters():
                parameter.copy_(torch.tensor(receipt["raw_parameter_values"][name], dtype=parameter.dtype))
        nets[unit] = (net, spec)
        return nets[unit]

    points = pd.read_csv(evidence / "real_prediction_points.csv")
    largest, comparisons = 0.0, 0
    for (unit, scope, vr), group in points.groupby(["unit", "scope", "Vr"]):
        group = group.sort_values("Va")
        net, spec = load(unit)
        va = torch.tensor(group.Va.to_numpy(np.float32))
        retarding = torch.full_like(va, float(vr))
        mode = "curve" if scope == "final_training" else "neutral"
        index = spec["training_vr"].index(vr) if mode == "curve" else 0
        with torch.no_grad():
            values = net(va, retarding, curve_idx=index, nuisance_mode=mode).numpy()
        difference = float(np.max(np.abs(values - group.predicted_uA.to_numpy())))
        largest = max(largest, difference)
        comparisons += len(group)
    if largest >= 1e-6:
        raise ValueError(f"Prediction replay exceeds tolerance: {largest} uA")
    return {"ok": True, "verified_files": files, "replayed_points": comparisons,
            "max_prediction_difference_uA": largest, "parameter_models": len(nets)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=ROOT / "source_data_package/spectral_evidence_v1")
    print(json.dumps(replay(parser.parse_args().evidence.resolve()), indent=2))
