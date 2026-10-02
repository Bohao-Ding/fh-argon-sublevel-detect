"""Matched phase on/off holdouts; diagnostic scores do not reselect models."""
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

from sublevel_detect.spectral_pipeline import TRAIN_VR, load_curves, training_curves
from sublevel_detect.spectral_revision_pipeline import Experiment, quadrature_rows
from sublevel_detect.validation_common import atomic_json_dump, sha256_file
from sublevel_detect.validation_holdout import split_vr_fold
import pandas as pd


def run(main_run, output):
    identity = json.loads((main_run / "run_identity.json").read_text())
    selection = pd.read_csv(main_run / "selection.csv")
    selected = selection[selection.selected].set_index(["scope", "family"])
    if len(selected) != 15:
        raise ValueError("all_real_nested_choices_required")
    output.mkdir(parents=True, exist_ok=True)
    control = dict(main_identity_sha256=sha256_file(main_run / "run_identity.json"),
                   script_sha256=sha256_file(Path(__file__)), role="matched_phase_holdout_diagnostic", stress_used=False)
    if (output / "run_identity.json").exists() and json.loads((output / "run_identity.json").read_text()) != control:
        raise ValueError("incompatible_phase_comparison_identity")
    atomic_json_dump(control, output / "run_identity.json")
    curves = training_curves(load_curves(ROOT / "data/argon/FHdata.xlsx"))
    exp = Experiment(output, identity["config"])
    jobs, evaluation = [], []
    for heldout in (*TRAIN_VR, None):
        scope = "final" if heldout is None else f"outer_{heldout:g}V"
        train, test = (curves, []) if heldout is None else split_vr_fold(curves, heldout_vr=heldout)
        for family in ("H1", "G1", "C"):
            penalty = float(selected.loc[(scope, family), "lambda"])
            for phase in (False, True):
                for seed in identity["config"]["seeds"]:
                    task = exp.task(train, family, seed, penalty, phase)
                    existing = main_run / "units" / task["identity"][:20]
                    if (existing / "fit.json").exists():
                        task["directory"] = str(existing)
                    jobs.append(task)
                    if test:
                        evaluation.append((task, test[0], scope))
    exp.batch(jobs, "matched_phase")
    rows = [exp.evaluate(t, c, scope) for t, c, scope in evaluation]
    checks = quadrature_rows(exp, jobs, curves)
    exp.write({"phase_scores": rows, "quadrature": checks})
    if not all(r["passed"] for r in checks):
        raise ValueError("matched_phase_quadrature_gate_failed")
    atomic_json_dump(dict(ok=True, role="fixed-candidate diagnostic, not nested reselection",
                         units=len(jobs), stress_used=False,
                         quadrature_max_difference_uA=max(r["max_difference_uA"] for r in checks)), output / "summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "output/spectral_revision")
    parser.add_argument("--output", type=Path, default=ROOT / "output/spectral_revision_phase_comparison")
    args = parser.parse_args()
    run(args.run.resolve(), args.output.resolve())
