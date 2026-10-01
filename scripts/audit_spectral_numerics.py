"""Check quadrature at fitted parameters, including frozen-mass local replacements."""
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
from sublevel_detect.spectral_models import LocalSpectrum, refine_measure
from sublevel_detect.spectral_pipeline import load_fit, predict


def audit(grid: Path, output: Path) -> dict:
    torch.set_num_threads(1)
    va = np.arange(161, dtype=np.float32) * 0.5
    design = [dict(Va=va, Vr=vr) for vr in (0.0, 4.0, 6.0, 8.0)]
    rows = []
    for path in sorted((grid / "units").glob("*/fit.json")):
        info = json.loads(path.read_text())
        if info["status"] == "ineligible":
            continue
        net = load_fit(path.parent)
        fine = refine_measure(net, net.step / 2)
        maximum = max(float(np.max(np.abs(a - b))) for a, b in zip(predict(net, design), predict(fine, design)))
        rows.append({"unit": info["identity"], "family": info["spec"]["family"],
            "coarse_step_eV": net.step, "fine_step_eV": net.step / 2,
            "fixed_inside_mass": isinstance(net, LocalSpectrum),
            "max_difference_uA": maximum, "passed": maximum < 1e-4})
    pd.DataFrame(rows).to_csv(output, index=False)
    return {"units": len(rows), "max_difference_uA": max(r["max_difference_uA"] for r in rows),
            "failed": sum(not r["passed"] for r in rows), "tolerance_uA": 1e-4}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.grid, args.output), indent=2))
