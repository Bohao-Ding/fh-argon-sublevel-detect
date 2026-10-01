"""Nested curve holdout, local replacement, and conditional recovery experiments."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks

from . import model, paths
from .spectral_models import (DOMAIN, LOCAL_DIMENSIONS, LOCAL_FAMILIES, NIST_4S,
    EffectiveSpectrum, LocalSpectrum, distribution_summary, main_peak_window, refine_measure)
from .validation_common import atomic_json_dump, canonical_hash, sha256_file
from .validation_holdout import prediction_metrics, split_vr_fold

TRAIN_VR = (0.0, 4.0, 6.0, 8.0)
LAMBDAS = (1e-6, 1e-4, 1e-2)
SCHEMA = "effective-spectrum-schema2"


def curve_vr(curve: dict) -> float:
    return float(np.asarray(curve["Vr"]).reshape(-1)[0])


def training_curves(curves: list[dict], heldout: float | None = None) -> list[dict]:
    chosen = [dict(c) for c in curves if curve_vr(c) in TRAIN_VR and curve_vr(c) != heldout]
    for i, curve in enumerate(chosen):
        curve["curve_idx"] = i
    return chosen


def load_curves(input_path: Path) -> list[dict]:
    # Reuse the input parser without invoking the legacy all-data initialization.
    frame = model.read_excel_or_csv(input_path)
    cid = model.pick_col(frame, ("curve_id", "curveID", "curve", "id"))
    vr = model.pick_col(frame, ("Vr", "vr", "V_r", "Ur"))
    va = model.pick_col(frame, ("Va", "va", "V_a", "Ua"))
    current = model.pick_col(frame, ("IuA", "Ip", "I", "current"))
    curves = []
    for index, (curve_id, group) in enumerate(frame.groupby(cid, sort=True)):
        group = group.sort_values(va)
        if group[vr].nunique() != 1:
            raise ValueError("retarding_voltage_must_be_constant_per_curve")
        curves.append({"curve_id": int(curve_id), "curve_idx": index,
            "Va": group[va].to_numpy(np.float32), "Vr": float(group[vr].iloc[0]),
            "Ip": group[current].to_numpy(np.float32)})
    return curves


def curves_hash(curves: list[dict]) -> str:
    digest = hashlib.sha256()
    for c in curves:
        digest.update(np.asarray([curve_vr(c)], dtype="<f8").tobytes())
        for key in ("Va", "Ip"):
            digest.update(np.asarray(c[key], dtype="<f8").tobytes())
    return digest.hexdigest().upper()


def training_initialization(curves: list[dict]) -> float:
    spacings = [s for c in curves if (s := model.estimate_peak_spacing(c["Va"], c["Ip"])) is not None and np.isfinite(s)]
    return float(np.clip(np.mean(spacings) if spacings else 11.5, 9.01, 15.99))


def settings(mode: str) -> dict:
    smoke = mode == "smoke"
    return {"schema": SCHEMA, "mode": mode, "max_epochs": 12 if smoke else 3500,
            "min_epochs": 3 if smoke else 300, "patience": 3 if smoke else 100,
            "relative_delta": 2e-4, "smoothing": 7, "lr": 0.002, "clip": 5.0,
            "common_regularization": 1e-4, "domain": DOMAIN, "knot_step": 0.25,
            "lambdas": LAMBDAS, "seeds": [0] if smoke else [0, 1, 2],
            "workers": 4, "bootstrap_replicates": 1 if smoke else 30,
            "synthetic_noise_replicates": 1 if smoke else 5, "block_length": 9,
            "train_vr": TRAIN_VR, "stress_vr": 10.0, "nist_4s": NIST_4S.tolist(),
            "synthetic_center": 11.65, "synthetic_sigma": 0.15,
            "synthetic_weights": [0.4, 0.3, 0.2, 0.1]}


def source_hashes() -> dict:
    files = ("spectral_models.py", "spectral_pipeline.py", "model.py",
             "validation_common.py", "validation_holdout.py")
    return {name: sha256_file(Path(__file__).with_name(name)) for name in files}


def load_fit(directory: str | Path) -> EffectiveSpectrum:
    directory = Path(directory)
    info = json.loads((directory / "fit.json").read_text())
    spec = info["spec"]
    if spec.get("schema") != SCHEMA:
        raise ValueError("incompatible_checkpoint_schema")
    if spec["family"] in LOCAL_FAMILIES:
        net = LocalSpectrum(load_fit(spec["coarse"]), spec["family"], spec["window"])
    else:
        net = EffectiveSpectrum(spec["n_curves"], spec["family"], seed=spec["seed"],
            excitation=spec["excitation"], step=spec["step"], fixed_width=spec["fixed_width"])
    checkpoint = directory / "best.pt"
    if sha256_file(checkpoint) != info["checkpoint_sha256"]:
        raise ValueError("checkpoint_byte_hash_mismatch")
    net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return net


def fit_unit(task: dict) -> dict:
    """The worker receives training observations only; evaluation happens later."""
    torch.set_num_threads(1)
    spec, cfg = task["spec"], task["config"]
    directory = Path(task["directory"])
    receipt = directory / "fit.json"
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old["identity"] != task["identity"]:
            raise ValueError("incompatible_unit_identity")
        if old["status"] == "ineligible":
            return old
        load_fit(directory)
        return old
    directory.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    try:
        if spec["family"] in LOCAL_FAMILIES:
            net = LocalSpectrum(load_fit(spec["coarse"]), spec["family"], spec["window"])
        else:
            net = EffectiveSpectrum(spec["n_curves"], spec["family"], seed=spec["seed"],
                excitation=spec["excitation"], step=spec["step"], fixed_width=spec["fixed_width"])
    except ValueError as error:
        if str(error) not in ("no_interior_main_peak", "outside_current_main_peak_candidates"):
            raise
        result = {"identity": task["identity"], "spec": spec, "status": "ineligible", "reason": str(error)}
        atomic_json_dump(result, receipt)
        return result
    device = torch.device(task["device"])
    net.to(device)
    curves = task["curves"]
    va = torch.tensor(np.stack([c["Va"] for c in curves]), device=device)
    vr = torch.tensor(np.stack([np.full_like(c["Va"], curve_vr(c)) for c in curves]), device=device)
    observed = torch.tensor(np.stack([c["Ip"] for c in curves]), device=device)
    ranges = (observed.amax(1) - observed.amin(1)).clamp_min(1e-8)[:, None]
    indices = torch.arange(len(curves), device=device)
    optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=cfg["lr"], weight_decay=0)
    stopper = model.EarlyStopper(warmup_epochs=0, min_epochs=cfg["min_epochs"],
        patience=cfg["patience"], min_delta_rel=cfg["relative_delta"], smoothing=cfg["smoothing"])
    best, best_epoch, history = float("inf"), 0, []
    status = "epoch_limit"
    for epoch in range(1, cfg["max_epochs"] + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = net(va, vr, curve_idx=indices, nuisance_mode="curve")
        data_loss = ((prediction - observed) / ranges).square().mean(1).mean()
        objective = data_loss + cfg["common_regularization"] * net.common_penalty() + spec["lambda"] * net.curvature()
        value = float(objective.detach())
        if not np.isfinite(value):
            raise FloatingPointError(f"Nonfinite training objective: {directory}")
        if value < best:
            best, best_epoch = value, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        if epoch == 1 or epoch % 50 == 0:
            history.append({"epoch": epoch, "objective": value, "data_mse": float(data_loss.detach())})
        if stopper.update(epoch, value).should_stop:
            status = "plateau"
            break
        objective.backward()
        torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], cfg["clip"])
        optimizer.step()
    net.cpu().load_state_dict(best_state)
    torch.save(best_state, directory / "best.pt")
    result = {"identity": task["identity"], "spec": spec, "status": status,
        "epochs": epoch, "best_epoch": best_epoch, "best_objective": best,
        "seconds": time.monotonic() - start, "distribution": distribution_summary(net),
        "parameters": model.json_ready(net.phys_params()),
        "raw_parameter_values": model.json_ready({name: p for name, p in net.named_parameters()}),
        "n_trainable": sum(p.numel() for p in net.parameters() if p.requires_grad),
        "checkpoint_sha256": sha256_file(directory / "best.pt"), "history": history}
    atomic_json_dump(result, receipt)
    return result


def one_se_selection(rows: list[dict], *, kind: str, expected_folds: int | None = None) -> tuple[list, list[dict]]:
    frame = pd.DataFrame(rows)
    key = "lambda" if kind == "lambda" else "family"
    grouped = []
    expected = frame["fold"].nunique() if expected_folds is None else expected_folds
    for candidate, sub in frame.groupby(key):
        if sub["fold"].nunique() != expected:
            continue
        scores = sub["nrmse"].to_numpy()
        grouped.append({"candidate": candidate, "mean_nrmse": float(scores.mean()),
            "se_nrmse": float(scores.std(ddof=1) / np.sqrt(len(scores))) if len(scores) > 1 else 0,
            "folds": len(scores)})
    if not grouped:
        return [], []
    best = min(grouped, key=lambda row: row["mean_nrmse"])
    threshold = best["mean_nrmse"] + best["se_nrmse"]
    eligible = [r["candidate"] for r in grouped if r["mean_nrmse"] <= threshold]
    if kind == "lambda":
        selected = [max(eligible)]
    else:
        dimension = min(LOCAL_DIMENSIONS[name] for name in eligible)
        selected = sorted(name for name in eligible if LOCAL_DIMENSIONS[name] == dimension)
    for row in grouped:
        row.update(threshold=threshold, selected=row["candidate"] in selected)
    return selected, grouped


def predict(net: EffectiveSpectrum, curves: list[dict], *, neutral: bool = True) -> list[np.ndarray]:
    with torch.no_grad():
        va = torch.tensor(np.stack([c["Va"] for c in curves]))
        vr = torch.tensor(np.stack([np.full_like(c["Va"], curve_vr(c)) for c in curves]))
        values = net(va, vr, curve_idx=torch.arange(len(curves)), nuisance_mode="neutral" if neutral else "curve")
    return list(values.numpy().astype(float))


def morphology(va: np.ndarray, observed: np.ndarray, predicted: np.ndarray) -> dict:
    result = {}
    prominence = max(float(np.ptp(observed)) * 0.025, 0.001)
    for label, sign in (("peak", 1), ("trough", -1)):
        actual = va[find_peaks(sign * model.moving_average_np(observed, 7), prominence=prominence, distance=8)[0]]
        fitted = va[find_peaks(sign * model.moving_average_np(predicted, 7), prominence=prominence, distance=8)[0]]
        distances = []
        if len(actual) and len(fitted):
            matrix = np.abs(actual[:, None] - fitted[None, :])
            i, j = linear_sum_assignment(matrix)
            distances = matrix[i, j][matrix[i, j] <= 3.0].tolist()
        result.update({f"observed_{label}_V": json.dumps(actual.tolist()),
            f"predicted_{label}_V": json.dumps(fitted.tolist()),
            f"missing_{label}": len(actual) - len(distances),
            f"extra_{label}": len(fitted) - len(distances),
            f"matched_{label}_mean_abs_V": float(np.mean(distances)) if distances else None})
    return result


def score(directory: str, curve: dict) -> tuple[dict, np.ndarray]:
    values = predict(load_fit(directory), [curve])[0]
    metrics = prediction_metrics(curve["Ip"], values, denominator_observed=curve["Ip"])
    metrics["bias_uA"] = float(np.mean(values - curve["Ip"]))
    metrics.update(morphology(curve["Va"], curve["Ip"], values))
    return metrics, values


def residual_sample(residual: np.ndarray, rng: np.random.Generator, block: int = 9) -> np.ndarray:
    residual = np.asarray(residual, dtype=float) - np.mean(residual)
    starts = rng.integers(0, len(residual), int(np.ceil(len(residual) / block)))
    indices = (starts[:, None] + np.arange(block)) % len(residual)
    return residual[indices.ravel()[:len(residual)]]


class Experiment:
    def __init__(self, root: Path, config: dict, step: float, device: str) -> None:
        self.root, self.cfg, self.step, self.device = root, config, step, device
        self.sources = source_hashes()
        self.fit_records, self.points, self.densities = [], [], []

    def task(self, curves: list[dict], family: str, seed: int, penalty: float = 0,
             *, coarse: str | None = None, window: dict | None = None,
             width: float | None = None) -> dict:
        spec = {"schema": SCHEMA, "family": family, "seed": seed,
            "lambda": float(penalty), "step": self.step, "n_curves": len(curves),
            "excitation": training_initialization(curves), "fixed_width": width,
            "window": window, "coarse": coarse}
        identity = canonical_hash({"spec": spec, "config": self.cfg,
            "training_hash": curves_hash(curves), "sources": self.sources,
            "coarse_sha256": sha256_file(Path(coarse) / "best.pt") if coarse else None})
        return {"spec": spec, "identity": identity, "directory": str(self.root / "units" / identity[:20]),
                "config": self.cfg, "curves": curves, "device": self.device}

    def batch(self, tasks: list[dict], phase: str) -> list[dict]:
        unique = {t["identity"]: t for t in tasks}
        completed = {}
        with ProcessPoolExecutor(max_workers=self.cfg["workers"]) as pool:
            futures = {pool.submit(fit_unit, task): key for key, task in unique.items()}
            for future in as_completed(futures):
                key = futures[future]
                result = future.result()
                completed[key] = result
                atomic_json_dump({"phase": phase, "completed": len(completed), "total": len(unique)}, self.root / "progress.json")
        for task in tasks:
            result = completed[task["identity"]]
            self.fit_records.append({"phase": phase, "unit": task["identity"],
                "family": task["spec"]["family"], "seed": task["spec"]["seed"],
                "lambda": task["spec"]["lambda"], "training_vr": json.dumps([curve_vr(c) for c in task["curves"]]),
                "status": result["status"], "epochs": result.get("epochs"),
                "best_epoch": result.get("best_epoch"), "objective": result.get("best_objective"),
                "reason": result.get("reason"), "seconds": result.get("seconds"),
                **result.get("distribution", {})})
        print(f"{phase}: {len(tasks)} fits ({len(unique)} unique)", flush=True)
        return [completed[t["identity"]] for t in tasks]

    def evaluate(self, task: dict, curve: dict, **context) -> dict | None:
        receipt = json.loads((Path(task["directory"]) / "fit.json").read_text())
        if receipt["status"] == "ineligible":
            return None
        metrics, values = score(task["directory"], curve)
        row = {**context, "family": task["spec"]["family"], "seed": task["spec"]["seed"],
               "lambda": task["spec"]["lambda"], "unit": task["identity"], **metrics}
        for va, obs, pred in zip(curve["Va"], curve["Ip"], values):
            self.points.append({**context, "family": row["family"], "seed": row["seed"],
                "unit": row["unit"], "Vr": curve_vr(curve), "Va": float(va),
                "observed_uA": float(obs), "predicted_uA": float(pred)})
        return row

    def density(self, task: dict, **context) -> None:
        info = json.loads((Path(task["directory"]) / "fit.json").read_text())
        if info["status"] == "ineligible":
            return
        net = load_fit(task["directory"])
        with torch.no_grad():
            levels = net.level_params()
            e, w = [levels[key].numpy() for key in ("energies", "weights")]
            q = net.density().numpy() if not isinstance(net, LocalSpectrum) and net.family != "H1" else np.full_like(e, np.nan)
        for energy, weight, density in zip(e, w, q):
            self.densities.append({**context, "family": task["spec"]["family"], "seed": task["spec"]["seed"],
                "unit": task["identity"], "energy_eV": float(energy), "mass": float(weight),
                "density_per_eV": float(density) if np.isfinite(density) else None})

    def tune(self, curves: list[dict], label: str) -> tuple[float, list[str], list[dict]]:
        jobs, refs = [], []
        for c in curves:
            train, held = split_vr_fold(curves, heldout_vr=curve_vr(c))
            for penalty in LAMBDAS:
                task = self.task(train, "C", 0, penalty)
                jobs.append(task)
                refs.append((task, held[0]))
        self.batch(jobs, f"{label}/inner_C")
        rows = [self.evaluate(task, held, scope=label, stage="continuous", fold=curve_vr(held)) for task, held in refs]
        chosen, stats = one_se_selection(rows, kind="lambda")
        penalty = float(chosen[0])
        local_refs, local_jobs = [], []
        for task, held in refs:
            if task["spec"]["lambda"] != penalty:
                continue
            window = main_peak_window([load_fit(task["directory"])])
            for family in LOCAL_FAMILIES:
                job = self.task(task["curves"], family, 0, coarse=task["directory"], window=window)
                local_jobs.append(job)
                local_refs.append((job, held))
        self.batch(local_jobs, f"{label}/inner_local")
        local_rows = [row for task, held in local_refs if (row := self.evaluate(task, held,
                      scope=label, stage="local", fold=curve_vr(held))) is not None]
        selected, local_stats = one_se_selection(local_rows, kind="local", expected_folds=len(curves)) if local_rows else ([], [])
        decisions = [{"scope": label, "stage": "lambda", **s} for s in stats]
        decisions += [{"scope": label, "stage": "local", **s} for s in local_stats]
        self.inner_rows = getattr(self, "inner_rows", []) + rows + local_rows
        return penalty, selected, decisions

    def refit(self, curves: list[dict], penalty: float, label: str,
              seeds: list[int] | None = None, comparators: bool = True) -> tuple[list[dict], dict]:
        seeds = self.cfg["seeds"] if seeds is None else seeds
        jobs = [self.task(curves, family, seed, penalty if family == "C" else 0)
                for family in (("H1", "G1", "C") if comparators else ("C",)) for seed in seeds]
        self.batch(jobs, f"{label}/coarse")
        coarse = [job for job in jobs if job["spec"]["family"] == "C"]
        window = main_peak_window([load_fit(task["directory"]) for task in coarse])
        local = [self.task(curves, family, task["spec"]["seed"], coarse=task["directory"], window=window)
                 for task in coarse for family in LOCAL_FAMILIES]
        self.batch(local, f"{label}/local")
        for task in jobs + local:
            self.density(task, scope=label)
        return jobs + local, window

    def write_tables(self, tables: dict) -> None:
        tables.update(fit_status=self.fit_records, prediction_points=self.points,
                      distributions=self.densities, inner_scores=getattr(self, "inner_rows", []))
        for name, rows in tables.items():
            pd.DataFrame(rows).to_csv(self.root / f"{name}.csv", index=False)


def quadrature_check(tasks: list[dict], curves: list[dict], step: float) -> list[dict]:
    results = []
    for task in tasks:
        coarse = load_fit(task["directory"])
        fine = refine_measure(coarse, step / 2)
        difference = max(float(np.max(np.abs(a - b))) for a, b in zip(predict(coarse, curves), predict(fine, curves)))
        results.append({"unit": Path(task["directory"]).name, "family": task["spec"]["family"],
                        "seed": task["spec"]["seed"], "coarse_step_eV": step,
                        "fine_step_eV": step / 2, "max_difference_uA": difference,
                        "passed": difference < 1e-4})
    return results


def completed_quadrature_tasks(root: Path) -> list[dict]:
    tasks = []
    for path in sorted((root / "units").glob("*/fit.json")):
        info = json.loads(path.read_text())
        if info["status"] != "ineligible":
            tasks.append({"directory": str(path.parent), "spec": info["spec"]})
    return tasks


def _run_grid(root: Path, curves: list[dict], cfg: dict, step: float, device: str) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    exp = Experiment(root, cfg, step, device)
    identity = {"schema": SCHEMA, "config": cfg, "step": step, "sources": exp.sources,
                "training_hash": curves_hash(training_curves(curves)), "raw_hash": curves_hash(curves)}
    receipt = root / "identity.json"
    if receipt.exists() and json.loads(receipt.read_text()) != model.json_ready(identity):
        raise ValueError(f"incompatible_run_identity: {root}")
    atomic_json_dump(identity, receipt)
    main = training_curves(curves)
    outer_rows, decisions, windows, profiles = [], [], [], []
    folds = TRAIN_VR[:1] if cfg["mode"] == "smoke" else TRAIN_VR
    for heldout in folds:
        train, test = split_vr_fold(main, heldout_vr=heldout)
        label = f"outer_{heldout:g}V"
        penalty, selected, stats = exp.tune(train, label)
        decisions += stats
        jobs, window = exp.refit(train, penalty, label)
        windows.append({"scope": label, "lambda": penalty, "selected_local": selected, **window})
        for task in jobs:
            row = exp.evaluate(task, test[0], scope=label, heldout_vr=heldout,
                               selected=task["spec"]["family"] in selected)
            if row is not None:
                outer_rows.append(row)
        profile_jobs = [exp.task(train, "C", seed, penalty, width=width)
                        for width in (1.0, 2.0, 3.0) for seed in cfg["seeds"]]
        exp.batch(profile_jobs, f"{label}/kernel_profile")
        for task in profile_jobs:
            profiles.append(exp.evaluate(task, test[0], scope=label, width_V=task["spec"]["fixed_width"]))
            exp.density(task, scope=f"{label}/width_{task['spec']['fixed_width']:g}")
    penalty, selected, stats = exp.tune(main, "final")
    decisions += stats
    final, window = exp.refit(main, penalty, "final")
    windows.append({"scope": "final", "lambda": penalty, "selected_local": selected, **window})
    final_profiles = [exp.task(main, "C", seed, penalty, width=width)
                     for width in (1.0, 2.0, 3.0) for seed in cfg["seeds"]]
    exp.batch(final_profiles, "final/kernel_profile")
    for task in final_profiles:
        exp.density(task, scope=f"final/width_{task['spec']['fixed_width']:g}")
    checks = quadrature_check(completed_quadrature_tasks(root), main, step)
    base_tables = {"outer_scores": outer_rows, "selection": decisions,
                   "kernel_profile_scores": profiles, "quadrature": checks}
    atomic_json_dump(windows, root / "windows.json")
    if not all(row["passed"] for row in checks):
        exp.write_tables(base_tables)
        return {"refine": True, "step": step, "checks": checks}
    continuous = [t for t in final if t["spec"]["family"] == "C"]
    reference = sorted(continuous, key=lambda t: json.loads((Path(t["directory"]) / "fit.json").read_text())["best_objective"])[len(continuous) // 2]
    net = load_fit(reference["directory"])
    fitted = predict(net, main, neutral=False)
    residuals = [c["Ip"].astype(float) - value for c, value in zip(main, fitted)]
    bootstrap, synthetic = [], []
    for index in range(cfg["bootstrap_replicates"]):
        rng = np.random.default_rng(10000 + index)
        sampled = [dict(c, Ip=(value + residual_sample(r, rng)).astype(np.float32))
                   for c, value, r in zip(main, fitted, residuals)]
        jobs, bw = exp.refit(sampled, penalty, f"bootstrap_{index:02d}", seeds=[0], comparators=False)
        valid = [t for t in jobs if t["spec"]["family"] in LOCAL_FAMILIES
                 and json.loads((Path(t["directory"]) / "fit.json").read_text())["status"] != "ineligible"]
        winner = min(valid, key=lambda t: json.loads((Path(t["directory"]) / "fit.json").read_text())["best_objective"])["spec"]["family"] if valid else None
        coarse = next(t for t in jobs if t["spec"]["family"] == "C")
        bootstrap.append({"replicate": index, "seed": 10000 + index,
            "training_objective_winner": winner, "window_center_eV": bw["center"],
            **distribution_summary(load_fit(coarse["directory"]))})
    for truth_index, truth in enumerate(("single", "gaussian", "nist4", "equal4")):
        truth_net = copy.deepcopy(net)
        with torch.no_grad():
            if truth == "gaussian":
                energy = truth_net.energy_grid.clone()
                weights = torch.exp(-0.5 * ((energy - cfg["synthetic_center"]) / cfg["synthetic_sigma"]) ** 2) * truth_net.quadrature
                weights /= weights.sum()
            elif truth == "single":
                energy, weights = torch.tensor([cfg["synthetic_center"]]), torch.ones(1)
            else:
                energy = torch.tensor(NIST_4S if truth == "nist4" else np.linspace(NIST_4S[0], NIST_4S[-1], 4), dtype=torch.float32)
                weights = torch.tensor(cfg["synthetic_weights"])
        # A small bound method avoids altering the response kernel for known truth.
        truth_net.level_params = lambda: {"energies": energy, "weights": weights, "gaps": energy.new_empty(0)}
        noiseless = predict(truth_net, main)
        for replicate in range(cfg["synthetic_noise_replicates"]):
            seed = 20000 + truth_index * 5 + replicate
            rng = np.random.default_rng(seed)
            generated = [dict(c, Ip=(value + residual_sample(r, rng)).astype(np.float32))
                         for c, value, r in zip(main, noiseless, residuals)]
            for heldout in folds:
                train, test = split_vr_fold(generated, heldout_vr=heldout)
                label = f"synthetic_{truth}_{replicate}_{heldout:g}V"
                jobs, sw = exp.refit(train, penalty, label, seeds=[0])
                for task in jobs:
                    row = exp.evaluate(task, test[0], scope=label, truth=truth, replicate=replicate,
                        noise_seed=seed, heldout_vr=heldout, window_center_eV=sw["center"])
                    if row is not None:
                        synthetic.append(row)
    # Recovery units must also pass the numerical gate before publication/stress.
    checks = quadrature_check(completed_quadrature_tasks(root), main, step)
    base_tables.update(quadrature=checks, bootstrap=bootstrap, synthetic_scores=synthetic)
    if not all(row["passed"] for row in checks):
        exp.write_tables(base_tables)
        return {"refine": True, "step": step, "checks": checks}
    # Every decision is written before any stress observation is scored.
    freeze = {"lambda": penalty, "selected_local": selected, "window": window,
              "stress_has_influenced_selection": False,
              "final_units": [t["identity"] for t in final]}
    atomic_json_dump(freeze, root / "frozen_selection.json")
    final_scores, stress = [], []
    stress_curve = next(c for c in curves if curve_vr(c) == 10.0)
    for task in final:
        info = json.loads((Path(task["directory"]) / "fit.json").read_text())
        if info["status"] == "ineligible":
            continue
        trained = load_fit(task["directory"])
        for curve, value in zip(main, predict(trained, main, neutral=False)):
            final_scores.append({"family": task["spec"]["family"], "seed": task["spec"]["seed"],
                "Vr": curve_vr(curve), "unit": task["identity"], "role": "training_diagnostic",
                **prediction_metrics(curve["Ip"], value, denominator_observed=curve["Ip"])})
            for va, obs, pred in zip(curve["Va"], curve["Ip"], value):
                exp.points.append({"scope": "final_training", "family": task["spec"]["family"],
                    "seed": task["spec"]["seed"], "unit": task["identity"], "Vr": curve_vr(curve),
                    "Va": float(va), "observed_uA": float(obs), "predicted_uA": float(pred)})
        stress.append(exp.evaluate(task, stress_curve, scope="stress_10V", selected=task["spec"]["family"] in selected))
    base_tables.update(bootstrap=bootstrap, synthetic_scores=synthetic,
                       final_training_scores=final_scores, stress_scores=stress)
    exp.write_tables(base_tables)
    summary = summarize(root, cfg, step, freeze)
    atomic_json_dump(summary, root / "summary.json")
    return summary


def summarize(root: Path, cfg: dict, step: float, freeze: dict) -> dict:
    outer = pd.read_csv(root / "outer_scores.csv")
    medians = outer.groupby(["heldout_vr", "family"])["nrmse"].median().unstack()
    medians.to_csv(root / "outer_seed_medians.csv")
    synthetic = pd.read_csv(root / "synthetic_scores.csv")
    counts = []
    for (truth, rep), group in synthetic.groupby(["truth", "replicate"]):
        local = group[group.family.isin(LOCAL_FAMILIES)]
        means = local.groupby("family").nrmse.mean()
        eligible = local.groupby("family").heldout_vr.nunique() == len(TRAIN_VR if cfg["mode"] == "fullscan" else TRAIN_VR[:1])
        means = means[eligible]
        winner = str(means.idxmin()) if len(means) else None
        discrete = means.reindex(["d2", "d3", "d4", "h4s", "equal4"]).dropna()
        split = bool(len(discrete) and "d1" in means and "g1" in means
                     and discrete.min() < min(means["d1"], means["g1"]))
        counts.append({"truth": truth, "replicate": rep, "best_local_mean_holdout": winner,
                       "discrete_beats_delta_and_gaussian": split})
    pd.DataFrame(counts).to_csv(root / "synthetic_recovery_counts.csv", index=False)
    statuses = pd.read_csv(root / "fit_status.csv").drop_duplicates("unit")
    boot = pd.read_csv(root / "bootstrap.csv")
    return {"ok": True, "scientific_run": cfg["mode"] == "fullscan", "grid_step_eV": step,
        "train_points": 644, "outer_folds": len(medians), "optimization_starts": cfg["seeds"],
        "outer_mean_seed_median_nrmse": model.json_ready(medians.mean().to_dict()),
        "outer_fold_seed_median_nrmse": model.json_ready(medians.reset_index().to_dict("records")),
        "final_selection": freeze,
        "fit_status_counts": {str(k): int(v) for k, v in statuses.status.value_counts().items()},
        "unique_fits": len(statuses), "bootstrap_count": len(boot),
        "bootstrap_quantiles": {key: {str(q): float(value) for q, value in boot[key].quantile([0.025, 0.5, 0.975]).items()}
            for key in ("mode_eV", "mean_eV", "sd_eV", "q05_eV", "q95_eV")},
        "synthetic_datasets": len(counts), "synthetic_discrete_improvement_counts":
            pd.DataFrame(counts).groupby("truth").discrete_beats_delta_and_gaussian.sum().astype(int).to_dict()}


def run(*, mode: str, input_path: str | Path, output_root: str | Path, device: str = "cpu") -> dict:
    if mode not in ("smoke", "fullscan"):
        raise ValueError(f"Unsupported spectrum mode: {mode}")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    torch.set_num_threads(1)
    device = str(model.resolve_device(device))
    cfg = settings(mode)
    input_path = paths.resolve_project_path(input_path)
    curves = load_curves(input_path)
    if sorted(curve_vr(c) for c in curves) != list(TRAIN_VR) + [10.0]:
        raise ValueError("missing_or_unexpected_retarding_curve")
    if any(len(c["Va"]) != 161 or not np.isfinite(c["Ip"]).all()
           or not np.array_equal(c["Va"], np.arange(161, dtype=np.float32) * 0.5) for c in curves):
        raise ValueError("invalid_curve_points")
    root = paths.resolve_project_path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    input_receipt = {"input_sha256": sha256_file(input_path), "schema": SCHEMA,
                     "config": cfg, "sources": source_hashes()}
    existing = root / "run_identity.json"
    if existing.exists() and json.loads(existing.read_text()) != model.json_ready(input_receipt):
        raise ValueError("incompatible_run_identity")
    atomic_json_dump(input_receipt, existing)
    refinements = []
    for step in (0.02, 0.01, 0.005):
        grid_root = root / f"grid_{step:g}"
        result = _run_grid(grid_root, curves, cfg, step, device)
        refinements.append({"grid_step_eV": step, "refinement_required": result.get("refine", False)})
        if not result.get("refine", False):
            result.update(output=str(grid_root), refinements=refinements)
            atomic_json_dump(result, root / "result.json")
            return result
    raise RuntimeError("quadrature_tolerance_not_met_at_0.005_eV")
