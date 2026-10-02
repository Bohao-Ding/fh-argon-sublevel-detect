"""Small matched fit diagnostic; no held-out or stress observations are scored."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sublevel_detect.spectral_pipeline import load_curves, training_curves
from sublevel_detect.spectral_revision_pipeline import Experiment, settings
from sublevel_detect.validation_common import atomic_json_dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    curves = training_curves(load_curves(ROOT / "data/argon/FHdata.xlsx"))
    exp = Experiment(args.output.resolve(), settings("fullscan"))
    jobs = [exp.task(curves, family, 0, 1e-6 if family == "C" else 0, phase)
            for family in ("H1", "G1", "C") for phase in (False, True)]
    jobs += [exp.task(curves, "C", 1, 1e-6, phase) for phase in (False, True)]
    jobs += [exp.task(curves, "C", 0, 1e-6, False, knot_step=0.25),
             exp.task(curves, "C", 1, 1e-8, False)]
    exp.batch(jobs, "matched_pilot")
    rows = [{"unit": t["identity"], "family": t["spec"]["family"], "seed": t["spec"]["seed"],
        "phase_response": t["spec"]["phase_response"], "lambda": t["spec"]["lambda"],
        "knot_step": t["spec"]["knot_step"], "status": exp.receipts[t["identity"]]["status"],
        "data_mse": exp.receipts[t["identity"]]["data_mse"],
        **exp.receipts[t["identity"]]["distribution"]} for t in jobs]
    exp.write({"matched_pilot": rows})
    summary = {"role": "training_only_matched_diagnostic", "points": 644, "units": len(jobs),
               "stress_used": False, "rows": rows}
    atomic_json_dump(summary, exp.root / "pilot_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
