"""Match free/fixed-kernel recovery on the same four synthetic training curves."""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch
from sublevel_detect.spectral_pipeline import load_curves, training_curves, residual_sample
from sublevel_detect.spectral_revision_pipeline import Experiment, load_fit, predict
from sublevel_detect.validation_common import atomic_json_dump, sha256_file


def run(main_run, output):
    torch.set_num_threads(1)
    summary = json.loads((main_run / "summary.json").read_text())
    if not summary["scientific_run"] or not summary["ok"]:
        raise ValueError("completed_formal_revision_required")
    identity = json.loads((main_run / "run_identity.json").read_text())
    frozen = json.loads((main_run / "frozen_selection.json").read_text())
    receipts = [json.loads((main_run / "units" / f"{u[:20]}" / "fit.json").read_text())
                for u in frozen["final_units"]]
    candidates = sorted([r for r in receipts if r["spec"]["family"] == "C"], key=lambda r: r["best_objective"])
    reference = candidates[len(candidates) // 2]
    net = load_fit(main_run / "units" / reference["identity"][:20])
    main = training_curves(load_curves(ROOT / "data/argon/FHdata.xlsx"))
    residuals = [c["Ip"].astype(float) - p for c, p in zip(main, predict(net, main, neutral=False))]
    config = identity["config"]
    output.mkdir(parents=True, exist_ok=True)
    companion = {"main_identity_sha256": sha256_file(main_run / "run_identity.json"),
                 "script_sha256": sha256_file(Path(__file__)), "config": config,
                 "role": "matched_four_curve_synthetic_recovery", "stress_used": False}
    if (output / "run_identity.json").exists() and json.loads((output / "run_identity.json").read_text()) != companion:
        raise ValueError("incompatible_matched_recovery_identity")
    atomic_json_dump(companion, output / "run_identity.json")
    exp = Experiment(output, config)
    jobs, context, noiseless_jobs, noiseless_context = [], {}, [], {}
    for truth_index, truth in enumerate(("single", "gaussian", "broad", "asymmetric")):
        energy = net.energy_grid.clone()
        if truth == "single":
            energy, weights = torch.tensor([11.65], dtype=torch.float64), torch.ones(1, dtype=torch.float64)
        else:
            d = torch.exp(-0.5 * ((energy - 11.65) / (0.15 if truth == "gaussian" else 0.8)) ** 2)
            if truth == "asymmetric":
                d = .7/.25 * torch.exp(-.5*((energy-11.5)/.25)**2) + .3/.55 * torch.exp(-.5*((energy-12.3)/.55)**2)
            weights = d * net.quadrature
            weights /= weights.sum()
        generator = copy.deepcopy(net)
        generator.level_params = lambda spectrum_weights=None: {"energies": energy, "weights": weights, "gaps": energy.new_empty(0)}
        noiseless = predict(generator, main)
        truth_mean = float(energy @ weights)
        truth_sd = float(torch.sqrt(weights @ (energy - truth_mean).square()))
        fixed_parameters = {name: p.detach().numpy().tolist() for name, p in net.named_parameters() if name.startswith("raw_")}
        fixed_parameters.update(raw_curve_gain=[0.] * 4, raw_curve_bias=[0.] * 4)
        clean = [dict(c, Ip=p.astype(np.float32)) for c, p in zip(main, noiseless)]
        for penalty in config["lambdas"]:
            choice = frozen["choices"]["C"]
            task = exp.task(clean, "C", 0, penalty, choice["phase_response"], fixed_parameters=fixed_parameters)
            noiseless_jobs.append(task)
            noiseless_context[task["identity"]] = dict(truth=truth, **{"lambda": penalty},
                truth_mode_eV=float(energy[weights.argmax()]), truth_mean_eV=truth_mean, truth_sd_eV=truth_sd)
        for replicate in range(config["noise_replicates"]):
            rng = np.random.default_rng(20000 + truth_index * 5 + replicate)
            curves = [dict(c, Ip=(p + residual_sample(r, rng, config["block_length"])).astype(np.float32))
                      for c, p, r in zip(main, noiseless, residuals)]
            choice = frozen["choices"]["C"]
            for seed in config["seeds"]:
                task = exp.task(curves, "C", seed, choice["lambda"], choice["phase_response"])
                jobs.append(task)
                context[task["identity"]] = dict(truth=truth, replicate=replicate, seed=seed,
                    truth_mode_eV=float(energy[weights.argmax()]), truth_mean_eV=truth_mean, truth_sd_eV=truth_sd)
    exp.batch(jobs + noiseless_jobs, "matched_free_and_noiseless_controls")
    rows = [{**context[t["identity"]], "unit": t["identity"], "kernel": "free", "training_points": 644,
             "status": exp.receipts[t["identity"]]["status"], **exp.receipts[t["identity"]]["distribution"]} for t in jobs]
    fixed = pd.read_csv(main_run / "fixed_kernel_recovery.csv").assign(kernel="fixed", training_points=644)
    status = pd.read_csv(main_run / "fit_status.csv").set_index("unit").status
    fixed["status"] = fixed.unit.map(status)
    pd.concat([pd.DataFrame(rows), fixed], ignore_index=True).to_csv(output / "matched_recovery.csv", index=False)
    noiseless_rows = [{**noiseless_context[t["identity"]], "unit": t["identity"], "status": exp.receipts[t["identity"]]["status"],
                      "data_mse": exp.receipts[t["identity"]]["data_mse"], **exp.receipts[t["identity"]]["distribution"]}
                     for t in noiseless_jobs]
    exp.write({"free_recovery": rows, "noiseless_recovery": noiseless_rows})
    atomic_json_dump({"ok": True, "free_fits": len(jobs), "fixed_fits": len(fixed),
                      "datasets": 4 * config["noise_replicates"], "noiseless_controls": len(noiseless_jobs),
                      "stress_used": False}, output / "summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "output/spectral_revision")
    parser.add_argument("--output", type=Path, default=ROOT / "output/spectral_revision_matched_recovery")
    args = parser.parse_args()
    run(args.run.resolve(), args.output.resolve())
