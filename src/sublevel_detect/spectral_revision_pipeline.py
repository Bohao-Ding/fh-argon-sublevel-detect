"""Concentration-first refitting, nested prediction and conditional recovery."""
from __future__ import annotations

import copy
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

from . import model
from .spectral_models import simplex
from .spectral_pipeline import (TRAIN_VR, curve_vr, curves_hash, load_curves,
    morphology, residual_sample, source_hashes, training_curves, training_initialization)
from .spectral_revision import RevisedSpectrum, concentration_status, spectrum_summary
from .validation_common import atomic_json_dump, canonical_hash, sha256_file
from .validation_holdout import prediction_metrics, split_vr_fold

SCHEMA = "effective-spectrum-revision-schema1"


def settings(mode):
    smoke = mode == "smoke"
    return {"schema": SCHEMA, "mode": mode, "workers": 4,
            "seeds": [0] if smoke else [0, 1, 2], "inner_seed": 0,
            "lambdas": [1e-6] if smoke else [0.0, 1e-8, 1e-6, 1e-4],
            "phase_responses": [False, True], "knot_step": 0.1, "domain": [9.0, 16.0],
            "grid_step": 0.005, "lr": 0.002, "clip": 5.0, "common_regularization": 1e-4,
            "adam_epochs": 12 if smoke else 1500, "min_epochs": 3 if smoke else 300,
            "patience": 3 if smoke else 150, "relative_delta": 2e-5,
            "lbfgs_iterations": 8 if smoke else 400,
            "joint_iterations": 8 if smoke else 600, "gradient_tolerance": 1e-5,
            "simplex_tolerance": 1e-4, "bootstrap_count": 1 if smoke else 30,
            "noise_replicates": 1 if smoke else 5, "block_length": 9,
            "peak_to_uniform_threshold": 1.5, "mode_range_threshold_eV": 0.5}


def revision_sources():
    values = source_hashes()
    for name in ("spectral_revision.py", "spectral_revision_pipeline.py"):
        values[name] = sha256_file(Path(__file__).with_name(name))
    return values


def make_model(spec):
    keys = ("n_curves", "family", "seed", "excitation", "step", "knot_step",
            "domain", "phase_response", "normalization_voltage", "fixed_width")
    net = RevisedSpectrum(**{key: spec[key] for key in keys})
    if spec.get("fixed_parameters"):
        with torch.no_grad():
            for name, p in net.named_parameters():
                if name in spec["fixed_parameters"]:
                    p.copy_(torch.tensor(spec["fixed_parameters"][name], dtype=p.dtype))
                    p.requires_grad_(False)
    return net


def restore_parameters(net, values):
    with torch.no_grad():
        for name, p in net.named_parameters():
            p.copy_(torch.as_tensor(values[name], dtype=p.dtype))


def load_fit(directory):
    directory = Path(directory)
    receipt = json.loads((directory / "fit.json").read_text())
    if receipt["spec"]["schema"] != SCHEMA:
        raise ValueError("incompatible_revision_schema")
    if sha256_file(directory / "best.pt") != receipt["checkpoint_sha256"]:
        raise ValueError("checkpoint_byte_hash_mismatch")
    net = make_model(receipt["spec"])
    restore_parameters(net, torch.load(directory / "best.pt", map_location="cpu", weights_only=True))
    return net


def fit_unit(task):
    torch.set_num_threads(1)
    directory = Path(task["directory"])
    if (directory / "fit.json").exists():
        receipt = json.loads((directory / "fit.json").read_text())
        if receipt["identity"] != task["identity"]:
            raise ValueError("incompatible_revision_identity")
        load_fit(directory)
        return receipt
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    spec, cfg, curves = task["spec"], task["config"], task["curves"]
    net = make_model(spec)
    va = torch.tensor(np.stack([c["Va"] for c in curves]), dtype=torch.float64)
    vr = torch.tensor(np.stack([np.full_like(c["Va"], curve_vr(c)) for c in curves]), dtype=torch.float64)
    observed = torch.tensor(np.stack([c["Ip"] for c in curves]), dtype=torch.float64)
    ranges = (observed.amax(1) - observed.amin(1)).clamp_min(1e-8)[:, None]
    indices = torch.arange(len(curves))
    parameters = [p for p in net.parameters() if p.requires_grad]
    best = float("inf")
    best_parameters = {}
    history, simplex_runs, calls = [], [], 0

    def objective(weights=None):
        prediction = net(va, vr, indices, "curve", weights)
        data = ((prediction - observed) / ranges).square().mean()
        return data + cfg["common_regularization"] * net.common_penalty() + spec["lambda"] * net.curvature(weights), data

    def consider(loss, weights=None):
        nonlocal best, best_parameters, calls
        calls += 1
        value = float(loss.detach())
        if not np.isfinite(value):
            raise FloatingPointError("nonfinite_revision_objective")
        if value < best:
            best = value
            best_parameters = {name: p.detach().clone() for name, p in net.named_parameters()}
            if weights is not None:
                positive = weights.detach().clamp_min(1e-12)
                best_parameters["spectral_logits"] = torch.log(positive[:-1] / positive[-1])

    optimizer = torch.optim.AdamW(parameters, lr=cfg["lr"], weight_decay=0)
    stopper = model.EarlyStopper(warmup_epochs=0, min_epochs=cfg["min_epochs"],
        patience=cfg["patience"], min_delta_rel=cfg["relative_delta"], smoothing=7)
    for epoch in range(1, cfg["adam_epochs"] + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, data = objective()
        consider(loss)
        if epoch == 1 or epoch % 100 == 0:
            history.append({"stage": "adam", "iteration": epoch, "objective": float(loss.detach()), "data_mse": float(data.detach())})
        if stopper.update(epoch, float(loss.detach())).should_stop:
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, cfg["clip"])
        optimizer.step()
    restore_parameters(net, best_parameters)

    def gradient_diagnostics():
        net.zero_grad(set_to_none=True)
        loss, _ = objective()
        loss.backward()
        shared = [float(p.grad.abs().max()) for name, p in net.named_parameters()
                  if p.grad is not None and p.numel() and name != "spectral_logits"]
        kkt = 0.0
        if net.family == "C":
            w = simplex(net.spectral_logits).detach().requires_grad_()
            loss, _ = objective(w)
            g = torch.autograd.grad(loss, w)[0].detach().numpy()
            weights = w.detach().numpy()
            residual = g - weights @ g
            kkt = max(float(np.max(np.abs(residual[weights > 1e-7]))),
                      float(np.maximum(-residual[weights <= 1e-7], 0).max(initial=0)))
        return max(shared, default=0.0), kkt

    if parameters:
        optimizer = torch.optim.LBFGS(parameters, lr=1, max_iter=cfg["lbfgs_iterations"],
            tolerance_grad=1e-7, tolerance_change=1e-12, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss, _ = objective()
            consider(loss)
            loss.backward()
            return loss

        optimizer.step(closure)
        loss, _ = objective()
        consider(loss)
        restore_parameters(net, best_parameters)
    if net.family == "C":
        shared = [(name, p) for name, p in net.named_parameters()
                  if p.requires_grad and p.numel() and name != "spectral_logits"]
        counts = [p.numel() for _, p in shared]
        n_shared = sum(counts)
        weights0 = simplex(net.spectral_logits).detach().numpy()
        values0 = np.r_[np.concatenate([p.detach().numpy().ravel() for _, p in shared]) if shared else [], weights0]

        def joint_objective(values):
            with torch.no_grad():
                cursor = 0
                for (_, p), count in zip(shared, counts):
                    p.copy_(torch.tensor(values[cursor:cursor + count]).reshape_as(p))
                    cursor += count
            weights = torch.tensor(values[n_shared:], dtype=torch.float64, requires_grad=True)
            loss, _ = objective(weights)
            consider(loss, weights)
            gradients = torch.autograd.grad(loss, [p for _, p in shared] + [weights])
            gradient = np.concatenate([g.detach().numpy().ravel() for g in gradients])
            # Scale the objective, not the model, for the SQP initial Hessian.
            return 1000 * float(loss.detach()), 1000 * gradient

        result = minimize(joint_objective, values0, jac=True, method="SLSQP",
            bounds=[(None, None)] * n_shared + [(0, 1)] * len(weights0),
            constraints={"type": "eq", "fun": lambda v: v[n_shared:].sum() - 1,
                         "jac": lambda v: np.r_[np.zeros(n_shared), np.ones(len(weights0))]},
            options={"maxiter": cfg["joint_iterations"], "ftol": 1e-10})
        simplex_runs.append({"success": bool(result.success), "iterations": int(result.nit), "message": str(result.message)})
        restore_parameters(net, best_parameters)
    shared_gradient, simplex_kkt = gradient_diagnostics()
    loss, data = objective()
    history.append({"stage": "polish", "objective": float(loss.detach()),
        "data_mse": float(data.detach()), "shared_gradient_max": shared_gradient,
        "simplex_kkt": simplex_kkt})
    restore_parameters(net, best_parameters)
    shared_gradient, simplex_kkt = gradient_diagnostics()
    status = "stationary" if shared_gradient <= cfg["gradient_tolerance"] and simplex_kkt <= cfg["simplex_tolerance"] else "iteration_limit"
    torch.save(best_parameters, directory / "best.pt")
    loss, data = objective()
    receipt = {"identity": task["identity"], "spec": spec, "status": status,
        "adam_epochs": epoch, "objective_calls": calls, "best_objective": best,
        "data_mse": float(data.detach()), "common_penalty": float(net.common_penalty().detach()) * cfg["common_regularization"],
        "curvature_penalty": float(net.curvature().detach()) * spec["lambda"],
        "shared_gradient_max": shared_gradient, "simplex_kkt": simplex_kkt,
        "simplex_runs": simplex_runs, "history": history, "seconds": time.monotonic() - started,
        "distribution": spectrum_summary(net), "parameters": model.json_ready(net.phys_params()),
        "raw_parameter_values": model.json_ready(best_parameters),
        "checkpoint_sha256": sha256_file(directory / "best.pt")}
    atomic_json_dump(receipt, directory / "fit.json")
    return receipt


def choose(rows):
    frame = pd.DataFrame(rows)
    stats = []
    for (phase, penalty), group in frame.groupby(["phase_response", "lambda"]):
        scores = group.nrmse.to_numpy()
        stats.append({"phase_response": bool(phase), "lambda": float(penalty),
            "mean_nrmse": float(scores.mean()), "se_nrmse": float(scores.std(ddof=1) / np.sqrt(len(scores))) if len(scores) > 1 else 0})
    best = min(stats, key=lambda s: s["mean_nrmse"])
    threshold = best["mean_nrmse"] + best["se_nrmse"]
    eligible = [s for s in stats if s["mean_nrmse"] <= threshold]
    chosen = min(eligible, key=lambda s: (s["phase_response"], -s["lambda"]))
    for row in stats:
        row.update(selected=row is chosen, threshold=threshold)
    return {key: chosen[key] for key in ("phase_response", "lambda")}, stats


def predict(net, curves, neutral=True):
    va = torch.tensor(np.stack([c["Va"] for c in curves]), dtype=torch.float64)
    vr = torch.tensor(np.stack([np.full_like(c["Va"], curve_vr(c)) for c in curves]), dtype=torch.float64)
    with torch.no_grad():
        values = net(va, vr, torch.arange(len(curves)), "neutral" if neutral else "curve").numpy()
    return list(values)


class Experiment:
    def __init__(self, root, config):
        self.root, self.cfg = Path(root), config
        self.sources = revision_sources()
        self.receipts, self.points, self.densities = {}, [], []

    def task(self, curves, family, seed, penalty=0, phase=False, **overrides):
        spec = {"schema": SCHEMA, "family": family, "seed": seed, "lambda": penalty,
            "n_curves": len(curves), "training_vr": [curve_vr(c) for c in curves],
            "excitation": training_initialization(curves), "domain": self.cfg["domain"],
            "knot_step": self.cfg["knot_step"], "step": self.cfg["grid_step"],
            "phase_response": phase, "normalization_voltage": max(float(c["Va"].max()) for c in curves),
            "fixed_width": None, **overrides}
        identity = canonical_hash({"spec": spec, "config": self.cfg,
            "training_hash": curves_hash(curves), "sources": self.sources})
        return {"spec": spec, "config": self.cfg, "curves": curves, "identity": identity,
                "directory": str(self.root / "units" / identity[:20])}

    def batch(self, jobs, label):
        unique = {t["identity"]: t for t in jobs}
        with ProcessPoolExecutor(max_workers=self.cfg["workers"]) as pool:
            futures = {pool.submit(fit_unit, task): key for key, task in unique.items()}
            for future in as_completed(futures):
                self.receipts[futures[future]] = future.result()
                atomic_json_dump({"phase": label, "completed": sum(k in self.receipts for k in unique),
                                  "total": len(unique)}, self.root / "progress.json")
        print(f"{label}: {len(unique)} units", flush=True)

    def evaluate(self, task, curve, scope, **context):
        value = predict(load_fit(task["directory"]), [curve])[0]
        row = {"scope": scope, "family": task["spec"]["family"], "seed": task["spec"]["seed"],
            "phase_response": task["spec"]["phase_response"], "lambda": task["spec"]["lambda"],
            "unit": task["identity"], "heldout_vr": curve_vr(curve), **context,
            **prediction_metrics(curve["Ip"], value, denominator_observed=curve["Ip"]),
            "bias_uA": float(np.mean(value - curve["Ip"])), **morphology(curve["Va"], curve["Ip"], value)}
        for va, obs, pred in zip(curve["Va"], curve["Ip"], value):
            self.points.append({"scope": scope, "unit": task["identity"], "family": row["family"],
                "seed": row["seed"], "Vr": curve_vr(curve), "Va": float(va),
                "observed_uA": float(obs), "predicted_uA": float(pred), **context})
        return row

    def density(self, task, scope):
        net = load_fit(task["directory"])
        with torch.no_grad():
            levels = net.level_params()
            e, w = [levels[k].numpy() for k in ("energies", "weights")]
            d = net.density().numpy() if net.family != "H1" else np.full_like(e, np.nan)
        for energy, mass, density in zip(e, w, d):
            self.densities.append({"scope": scope, "family": net.family, "seed": task["spec"]["seed"],
                "unit": task["identity"], "energy_eV": float(energy), "mass": float(mass),
                "density_per_eV": float(density) if np.isfinite(density) else None})

    def tune(self, curves, scope):
        jobs, refs = [], []
        for curve in curves:
            train, test = split_vr_fold(curves, heldout_vr=curve_vr(curve))
            for family in ("H1", "G1", "C"):
                for phase in self.cfg["phase_responses"]:
                    for penalty in self.cfg["lambdas"] if family == "C" else [0]:
                        task = self.task(train, family, self.cfg["inner_seed"], penalty, phase)
                        jobs.append(task)
                        refs.append((task, test[0]))
        self.batch(jobs, f"{scope}/inner")
        rows = [self.evaluate(t, c, f"{scope}/inner") for t, c in refs]
        choices, decisions = {}, []
        for family in ("H1", "G1", "C"):
            choices[family], stats = choose([r for r in rows if r["family"] == family])
            decisions.extend({"scope": scope, "family": family, **s} for s in stats)
        return choices, decisions, rows

    def refit(self, curves, choices, scope, seeds=None):
        jobs = [self.task(curves, family, seed, c["lambda"], c["phase_response"])
                for family, c in choices.items() for seed in (self.cfg["seeds"] if seeds is None else seeds)]
        self.batch(jobs, scope)
        for task in jobs:
            self.density(task, scope)
        return jobs

    def write(self, tables):
        status = []
        for unit, r in self.receipts.items():
            status.append({"unit": unit, "family": r["spec"]["family"], "seed": r["spec"]["seed"],
                "training_vr": json.dumps(r["spec"]["training_vr"]), "phase_response": r["spec"]["phase_response"],
                "lambda": r["spec"]["lambda"], "status": r["status"], "seconds": r["seconds"],
                "best_objective": r["best_objective"], "data_mse": r["data_mse"],
                "shared_gradient_max": r["shared_gradient_max"], "simplex_kkt": r["simplex_kkt"],
                **r["distribution"]})
        tables.update(fit_status=status, prediction_points=self.points, distributions=self.densities)
        for name, rows in tables.items():
            pd.DataFrame(rows).to_csv(self.root / f"{name}.csv", index=False)


def quadrature_rows(exp, tasks, curves):
    rows = []
    for task in {t["identity"]: t for t in tasks}.values():
        original = load_fit(task["directory"])
        spec = dict(task["spec"], step=task["spec"]["step"] / 2)
        fine = make_model(spec)
        restore_parameters(fine, dict(original.named_parameters()))
        difference = max(float(np.max(np.abs(a - b))) for a, b in zip(predict(original, curves), predict(fine, curves)))
        rows.append({"unit": task["identity"], "step_eV": task["spec"]["step"],
                     "fine_step_eV": spec["step"], "max_difference_uA": difference, "passed": difference < 1e-4})
    return rows


def run(*, mode, input_path, output_root, device="cpu"):
    if device not in ("cpu", "auto"):
        raise ValueError("The revised reproducible protocol uses CPU workers")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    torch.set_num_threads(1)
    curves = load_curves(Path(input_path))
    if sorted(curve_vr(c) for c in curves) != [0, 4, 6, 8, 10] or any(
        len(c["Va"]) != 161 or not np.array_equal(c["Va"], np.arange(161) * 0.5) for c in curves):
        raise ValueError("requires_original_five_complete_curves")
    cfg, root = settings(mode), Path(output_root).resolve()
    identity = {"schema": SCHEMA, "input_sha256": sha256_file(Path(input_path)),
                "config": cfg, "sources": revision_sources(), "training_hash": curves_hash(training_curves(curves))}
    if (root / "run_identity.json").exists() and json.loads((root / "run_identity.json").read_text()) != identity:
        raise ValueError("incompatible_revision_run_identity")
    root.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(identity, root / "run_identity.json")
    exp = Experiment(root, cfg)
    main = training_curves(curves)
    folds = TRAIN_VR[:1] if mode == "smoke" else TRAIN_VR
    outer, selection, inner, concentration, all_tasks = [], [], [], [], []
    for heldout in folds:
        train, test = split_vr_fold(main, heldout_vr=heldout)
        scope = f"outer_{heldout:g}V"
        choices, decisions, rows = exp.tune(train, scope)
        selection += decisions
        inner += rows
        tasks = exp.refit(train, choices, scope)
        all_tasks += tasks
        outer += [exp.evaluate(t, test[0], scope) for t in tasks]
        summaries = [exp.receipts[t["identity"]]["distribution"] for t in tasks if t["spec"]["family"] == "C"]
        concentration.append({"scope": scope, **concentration_status(summaries)})
        exp.write({"outer_scores": outer, "selection": selection, "inner_scores": inner, "concentration_status": concentration})
    choices, decisions, rows = exp.tune(main, "final")
    selection += decisions
    inner += rows
    final = exp.refit(main, choices, "final")
    all_tasks += final
    final_c = [t for t in final if t["spec"]["family"] == "C"]
    concentration.append({"scope": "final", **concentration_status([exp.receipts[t["identity"]]["distribution"] for t in final_c])})
    reference = sorted(final_c, key=lambda t: exp.receipts[t["identity"]]["best_objective"])[len(final_c) // 2]
    net = load_fit(reference["directory"])
    profiles = []
    for label, overrides in [("width_1", {"fixed_width": 1.0}), ("width_2", {"fixed_width": 2.0}),
        ("width_3", {"fixed_width": 3.0}), ("coarse_knots", {"knot_step": 0.25}),
        ("fine_knots", {"knot_step": 0.05}), ("expanded_domain", {"domain": [8.0, 17.0]})]:
        tasks = [exp.task(main, "C", seed, choices["C"]["lambda"], choices["C"]["phase_response"], **overrides) for seed in cfg["seeds"]]
        exp.batch(tasks, f"profile/{label}")
        all_tasks += tasks
        for t in tasks:
            profiles.append({"scope": label, "unit": t["identity"], "seed": t["spec"]["seed"], **exp.receipts[t["identity"]]["distribution"]})
            exp.density(t, f"profile/{label}")
    fitted = predict(net, main, neutral=False)
    residuals = [c["Ip"].astype(float) - p for c, p in zip(main, fitted)]
    bootstrap = []
    for replicate in range(cfg["bootstrap_count"]):
        rng = np.random.default_rng(10000 + replicate)
        sampled = [dict(c, Ip=(p + residual_sample(r, rng, cfg["block_length"])).astype(np.float32))
                   for c, p, r in zip(main, fitted, residuals)]
        tasks = exp.refit(sampled, {"C": choices["C"]}, f"bootstrap_{replicate:02d}")
        all_tasks += tasks
        for t in tasks:
            bootstrap.append({"replicate": replicate, "unit": t["identity"], "seed": t["spec"]["seed"], **exp.receipts[t["identity"]]["distribution"]})
        exp.write({"outer_scores": outer, "selection": selection, "inner_scores": inner,
                   "concentration_status": concentration, "profiles": profiles, "bootstrap": bootstrap})
    synthetic, recovery, fixed_recovery = [], [], []
    for truth_index, truth in enumerate(("single", "gaussian", "broad", "asymmetric")):
        energy = net.energy_grid.clone()
        if truth == "single":
            energy, weights = torch.tensor([11.65], dtype=torch.float64), torch.ones(1, dtype=torch.float64)
        else:
            density = torch.exp(-0.5 * ((energy - 11.65) / (0.15 if truth == "gaussian" else 0.8)) ** 2)
            if truth == "asymmetric":
                density = 0.7 / 0.25 * torch.exp(-0.5 * ((energy - 11.5) / 0.25) ** 2) + 0.3 / 0.55 * torch.exp(-0.5 * ((energy - 12.3) / 0.55) ** 2)
            weights = density * net.quadrature
            weights /= weights.sum()
        truth_mean = float(energy @ weights)
        truth_sd = float(torch.sqrt(weights @ (energy - truth_mean).square()))
        truth_mode = float(energy[weights.argmax()])
        truth_net = copy.deepcopy(net)
        truth_net.level_params = lambda spectrum_weights=None: {"energies": energy, "weights": weights, "gaps": energy.new_empty(0)}
        noiseless = predict(truth_net, main)
        fixed_parameters = {name: p.detach().numpy().tolist() for name, p in net.named_parameters() if name.startswith("raw_")}
        fixed_parameters.update(raw_curve_gain=[0.0] * 4, raw_curve_bias=[0.0] * 4)
        for replicate in range(cfg["noise_replicates"]):
            rng = np.random.default_rng(20000 + truth_index * 5 + replicate)
            generated = [dict(c, Ip=(p + residual_sample(r, rng, cfg["block_length"])).astype(np.float32))
                         for c, p, r in zip(main, noiseless, residuals)]
            for heldout in folds:
                train, test = split_vr_fold(generated, heldout_vr=heldout)
                scope = f"synthetic_{truth}_{replicate}_{heldout:g}V"
                tasks = exp.refit(train, choices, scope)
                all_tasks += tasks
                for t in tasks:
                    synthetic.append(exp.evaluate(t, test[0], scope, truth=truth, replicate=replicate))
                    if t["spec"]["family"] == "C":
                        recovery.append({"truth": truth, "replicate": replicate, "heldout_vr": heldout,
                            "seed": t["spec"]["seed"], "unit": t["identity"], "truth_mode_eV": truth_mode,
                            "truth_mean_eV": truth_mean, "truth_sd_eV": truth_sd, **exp.receipts[t["identity"]]["distribution"]})
            jobs = [exp.task(generated, "C", seed, choices["C"]["lambda"], choices["C"]["phase_response"],
                            fixed_parameters=fixed_parameters) for seed in cfg["seeds"]]
            exp.batch(jobs, f"fixed_kernel/{truth}_{replicate}")
            all_tasks += jobs
            for t in jobs:
                fixed_recovery.append({"truth": truth, "replicate": replicate, "seed": t["spec"]["seed"],
                    "unit": t["identity"], "truth_mode_eV": truth_mode, "truth_mean_eV": truth_mean,
                    "truth_sd_eV": truth_sd, **exp.receipts[t["identity"]]["distribution"]})
            exp.write({"outer_scores": outer, "selection": selection, "inner_scores": inner,
                "concentration_status": concentration, "profiles": profiles, "bootstrap": bootstrap,
                "synthetic_scores": synthetic, "recovery": recovery, "fixed_kernel_recovery": fixed_recovery})
    checks = quadrature_rows(exp, all_tasks + [dict(directory=str(root / "units" / k[:20]), spec=r["spec"], identity=k)
        for k, r in exp.receipts.items()], main)
    if not all(r["passed"] for r in checks):
        exp.write({"outer_scores": outer, "selection": selection, "inner_scores": inner,
            "concentration_status": concentration, "profiles": profiles, "bootstrap": bootstrap,
            "synthetic_scores": synthetic, "recovery": recovery, "fixed_kernel_recovery": fixed_recovery,
            "quadrature": checks})
        raise ValueError("revision_quadrature_gate_failed; retain receipts and refine before scoring stress")
    freeze = {"choices": choices, "final_units": [t["identity"] for t in final], "stress_has_influenced_selection": False}
    atomic_json_dump(freeze, root / "frozen_selection.json")
    stress_curve = next(c for c in curves if curve_vr(c) == 10)
    stress = [exp.evaluate(t, stress_curve, "stress_10V") for t in final]
    training = []
    for t in final:
        for c, p in zip(main, predict(load_fit(t["directory"]), main, neutral=False)):
            training.append({"family": t["spec"]["family"], "seed": t["spec"]["seed"], "Vr": curve_vr(c),
                "unit": t["identity"], **prediction_metrics(c["Ip"], p, denominator_observed=c["Ip"])})
            for va, obs, pred in zip(c["Va"], c["Ip"], p):
                exp.points.append({"scope": "final_training", "unit": t["identity"], "family": t["spec"]["family"],
                    "seed": t["spec"]["seed"], "Vr": curve_vr(c), "Va": float(va), "observed_uA": float(obs), "predicted_uA": float(pred)})
    exp.write({"outer_scores": outer, "selection": selection, "inner_scores": inner,
        "concentration_status": concentration, "profiles": profiles, "bootstrap": bootstrap,
        "synthetic_scores": synthetic, "recovery": recovery, "fixed_kernel_recovery": fixed_recovery,
        "quadrature": checks, "stress_scores": stress, "training_scores": training})
    medians = pd.DataFrame(outer).groupby(["heldout_vr", "family"]).nrmse.median().unstack()
    summary = {"ok": True, "scientific_run": mode == "fullscan", "schema": SCHEMA,
        "train_points": sum(len(c["Va"]) for c in main), "outer_folds": len(folds),
        "outer_mean_seed_median_nrmse": medians.mean().to_dict(), "choices": choices,
        "concentration_status": concentration, "unique_fits": len(exp.receipts),
        "status_counts": pd.Series([r["status"] for r in exp.receipts.values()]).value_counts().to_dict(),
        "quadrature_max_difference_uA": max(r["max_difference_uA"] for r in checks),
        "bootstrap_count": cfg["bootstrap_count"], "synthetic_datasets": cfg["noise_replicates"] * 4,
        "local_discrete_stage": "not_run_in_concentration_protocol"}
    atomic_json_dump(model.json_ready(summary), root / "summary.json")
    return summary
