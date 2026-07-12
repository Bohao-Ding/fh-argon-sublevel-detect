from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import main_pipeline, model, paths
from .validation_common import (
    Baseline,
    atomic_json_dump,
    canonical_hash,
    write_progress,
    write_stage_status,
)


HELDOUT_VR_VALUES = (0.0, 4.0, 6.0, 8.0, 10.0)


@dataclass(frozen=True)
class HoldoutUnit:
    heldout_vr: float
    n_levels: int
    seed: int


def holdout_units(mode: str) -> list[HoldoutUnit]:
    if str(mode) == "smoke":
        return [HoldoutUnit(0.0, n_levels, 0) for n_levels in (1, 2)]
    return [
        HoldoutUnit(heldout_vr, n_levels, seed)
        for heldout_vr in HELDOUT_VR_VALUES
        for n_levels in range(1, 9)
        for seed in (0, 1, 2)
    ]


def _curve_vr(curve: Mapping[str, Any]) -> float:
    values = np.asarray(curve["Vr"], dtype=np.float64)
    return float(values.reshape(-1)[0])


def split_vr_fold(
    curves: Sequence[Mapping[str, Any]],
    *,
    heldout_vr: float,
    tolerance: float = 1e-6,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    training: list[dict[str, Any]] = []
    heldout: list[dict[str, Any]] = []
    for curve in curves:
        copied = dict(curve)
        if abs(_curve_vr(copied) - float(heldout_vr)) <= float(tolerance):
            heldout.append(copied)
        else:
            training.append(copied)
    if len(heldout) != 1:
        raise ValueError(f"Expected exactly one held-out curve for Vr={heldout_vr}, found {len(heldout)}")
    for curve_idx, curve in enumerate(training):
        curve["curve_idx"] = int(curve_idx)
    heldout[0]["curve_idx"] = 0
    return training, heldout


def calibration_split(n_points: int = 161) -> tuple[np.ndarray, np.ndarray]:
    if int(n_points) != 161:
        raise ValueError(f"Preregistered calibration split requires 161 points, found {n_points}")
    calibration = np.arange(0, 161, 8, dtype=int)
    evaluation = np.setdiff1d(np.arange(161, dtype=int), calibration)
    return calibration, evaluation


def prediction_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    denominator_observed: np.ndarray,
) -> dict[str, float]:
    observed_values = np.asarray(observed, dtype=np.float64)
    predicted_values = np.asarray(predicted, dtype=np.float64)
    if observed_values.shape != predicted_values.shape or observed_values.size == 0:
        raise ValueError("Observed and predicted arrays must have the same non-empty shape")
    denominator = np.asarray(denominator_observed, dtype=np.float64)
    observed_range = float(np.max(denominator) - np.min(denominator))
    if not np.isfinite(observed_range) or observed_range <= 0.0:
        raise ValueError("Held-out observed range must be finite and positive")
    error = predicted_values - observed_values
    rmse = float(np.sqrt(np.mean(np.square(error))))
    return {
        "rmse": rmse,
        "mae": float(np.mean(np.abs(error))),
        "nrmse": float(rmse / observed_range),
        "normalization_range": observed_range,
        "n_points": int(observed_values.size),
    }


class SparseNuisance(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.raw_gain = nn.Parameter(torch.zeros(()))
        self.raw_bias = nn.Parameter(torch.zeros(()))
        self.raw_delta_va = nn.Parameter(torch.zeros(()))

    def values(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gain = 1.0 + 0.12 * torch.tanh(self.raw_gain)
        bias = 0.12 * torch.tanh(self.raw_bias)
        delta_va = 1.5 * torch.tanh(self.raw_delta_va)
        return gain, bias, delta_va

    def forward(
        self,
        va: torch.Tensor,
        core_prediction: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        gain, bias, delta_va = self.values()
        prediction = gain * core_prediction(va + delta_va) + bias
        return torch.clamp(prediction, min=0.0)


def fit_sparse_nuisance(
    *,
    core_prediction: Callable[[torch.Tensor], torch.Tensor],
    va: torch.Tensor,
    target: torch.Tensor,
    epochs: int = 500,
    learning_rate: float = 0.02,
) -> dict[str, Any]:
    if va.shape != target.shape or va.numel() == 0:
        raise ValueError("Calibration Va and target must have the same non-empty shape")
    nuisance = SparseNuisance().to(device=va.device, dtype=va.dtype)
    optimizer = torch.optim.Adam(nuisance.parameters(), lr=float(learning_rate))
    loss = torch.zeros((), device=va.device, dtype=va.dtype)
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        prediction = nuisance(va, core_prediction)
        loss = torch.mean(torch.square(prediction - target))
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        gain, bias, delta_va = nuisance.values()
        final_loss = torch.mean(torch.square(nuisance(va, core_prediction) - target))
    return {
        "module": nuisance,
        "values": {
            "gain": float(gain.detach().cpu()),
            "bias": float(bias.detach().cpu()),
            "delta_va": float(delta_va.detach().cpu()),
        },
        "loss": float(final_loss.detach().cpu()),
        "epochs": int(epochs),
    }


def _baseline_config(baseline: Baseline) -> model.Config:
    payload = json.loads(baseline.files["config"].read_text(encoding="utf-8-sig"))
    allowed = {field.name for field in fields(model.Config)}
    return model.Config(**{key: value for key, value in payload.items() if key in allowed})


def _fold_config(
    *,
    mode: str,
    input_path: str | Path,
    baseline: Baseline,
    out_dir: Path,
    device: str,
    heldout_vr: float,
) -> model.Config:
    cfg = _baseline_config(baseline)
    if str(mode) == "smoke":
        for key, value in main_pipeline.profile_defaults("smoke").items():
            setattr(cfg, key, value)
    cfg.data_path = str(paths.resolve_project_path(input_path))
    cfg.out_dir = str(out_dir)
    cfg.profile = f"validation_holdout_{mode}"
    cfg.exclude_vr_values = str(float(heldout_vr))
    cfg.hyperopt_enabled = False
    cfg.scan_seeds = "0" if str(mode) == "smoke" else "0,1,2"
    cfg.level_scan_min = 1
    cfg.level_scan_max = 2 if str(mode) == "smoke" else 8
    cfg.device = str(device)
    cfg.selected_device = "cuda" if str(device) == "cuda" and model.torch.cuda.is_available() else "cpu"
    if cfg.selected_device == "cuda":
        cfg.cuda_workers = 4
        cfg.dispatch_strategy = "cuda_4"
    else:
        cfg.dispatch_strategy = "cpu_4" if cfg.cpu_workers > 1 else "single"
    cfg.resume_mode = "auto" if str(mode) == "fullscan" else "off"
    cfg.launch_monitor = False
    cfg.forward_evidence_path = str(out_dir / "forward_evidence.json")
    return cfg


def _full_curves(cfg: model.Config) -> list[dict[str, Any]]:
    payload = asdict(cfg)
    payload["exclude_vr_values"] = ""
    full_cfg = model.Config(**payload)
    curves, _ = model.load_curves_and_init(full_cfg)
    return curves


def load_candidate(
    *,
    cfg: model.Config,
    scan_dir: str | Path,
    n_levels: int,
    seed: int,
    device: str,
) -> model.PoissonRateFHCoreMultiLevel:
    training_curves, init = model.load_curves_and_init(cfg)
    torch_device = model.resolve_device(device)
    trained = model.PoissonRateFHCoreMultiLevel(
        n_curves=len(training_curves),
        n_max=int(cfg.n_max),
        n_levels=int(n_levels),
        min_level_gap=float(cfg.min_level_gap),
        device=torch_device,
        V_exc_init=float(init["V_exc_init"]),
        init_jitter_scale=float(cfg.init_jitter_scale),
        init_seed=int(seed),
    ).to(torch_device)
    checkpoint = (
        Path(scan_dir)
        / "levels"
        / f"L{int(n_levels):02d}"
        / "seeds"
        / f"seed_{int(seed):03d}"
        / "full"
        / "checkpoint_best.pt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing holdout candidate checkpoint: {checkpoint}")
    payload = torch.load(checkpoint, map_location=torch_device, weights_only=False)
    state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
    trained.load_state_dict(state, strict=True)
    trained.eval()
    for parameter in trained.parameters():
        parameter.requires_grad_(False)
    return trained


@torch.no_grad()
def neutral_predictions(
    trained: model.PoissonRateFHCoreMultiLevel,
    curve: Mapping[str, Any],
    *,
    device: str,
) -> np.ndarray:
    torch_device = model.resolve_device(device)
    va = torch.as_tensor(curve["Va"], dtype=torch.float32, device=torch_device)
    if bool(curve["Vr_is_vector"]):
        vr = torch.as_tensor(curve["Vr"], dtype=torch.float32, device=torch_device)
    else:
        vr = torch.full_like(va, float(curve["Vr"]))
    prediction = trained.forward_core(va, vr, nuisance_mode="neutral")
    return prediction.detach().cpu().numpy().astype(np.float64)


def evaluate_candidate(
    *,
    unit: HoldoutUnit,
    cfg: model.Config,
    scan_dir: str | Path,
    heldout_curve: Mapping[str, Any],
    device: str,
) -> dict[str, Any]:
    trained = load_candidate(
        cfg=cfg,
        scan_dir=scan_dir,
        n_levels=unit.n_levels,
        seed=unit.seed,
        device=device,
    )
    torch_device = model.resolve_device(device)
    observed = np.asarray(heldout_curve["Ip"], dtype=np.float64)
    va_values = np.asarray(heldout_curve["Va"], dtype=np.float64)
    zero_prediction = neutral_predictions(trained, heldout_curve, device=device)
    zero_metrics = prediction_metrics(observed, zero_prediction, denominator_observed=observed)
    calibration_indices, evaluation_indices = calibration_split(len(observed))
    va = torch.as_tensor(va_values, dtype=torch.float32, device=torch_device)
    target = torch.as_tensor(observed, dtype=torch.float32, device=torch_device)
    if bool(heldout_curve["Vr_is_vector"]):
        vr = torch.as_tensor(heldout_curve["Vr"], dtype=torch.float32, device=torch_device)
    else:
        vr = torch.full_like(va, float(heldout_curve["Vr"]))
    calibration_tensor = torch.as_tensor(calibration_indices, dtype=torch.long, device=torch_device)

    def calibration_core(shifted_va: torch.Tensor) -> torch.Tensor:
        return trained.forward_core(
            shifted_va,
            vr.index_select(0, calibration_tensor),
            nuisance_mode="neutral",
        )

    calibration = fit_sparse_nuisance(
        core_prediction=calibration_core,
        va=va.index_select(0, calibration_tensor),
        target=target.index_select(0, calibration_tensor),
    )
    nuisance = calibration["module"]
    with torch.no_grad():
        calibrated_prediction = nuisance(
            va,
            lambda shifted_va: trained.forward_core(shifted_va, vr, nuisance_mode="neutral"),
        ).detach().cpu().numpy().astype(np.float64)
    calibrated_metrics = prediction_metrics(
        observed[evaluation_indices],
        calibrated_prediction[evaluation_indices],
        denominator_observed=observed,
    )
    zero_rows = [
        {
            "index": int(index),
            "Va": float(va_values[index]),
            "observed": float(observed[index]),
            "predicted": float(zero_prediction[index]),
        }
        for index in range(len(observed))
    ]
    calibrated_rows = [
        {
            "index": int(index),
            "Va": float(va_values[index]),
            "observed": float(observed[index]),
            "predicted": float(calibrated_prediction[index]),
        }
        for index in evaluation_indices
    ]
    return {
        "unit": unit,
        "zero_metrics": zero_metrics,
        "calibrated_metrics": calibrated_metrics,
        "zero_predictions": zero_rows,
        "calibrated_predictions": calibrated_rows,
        "nuisance": calibration["values"],
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(row) for row in rows]).to_csv(path, index=False)


def _vr_label(value: float) -> str:
    return str(float(value)).replace("-", "m").replace(".", "p")


def run(
    *,
    mode: str,
    input_path: str | Path,
    baseline: Baseline,
    output_dir: str | Path,
    validation_root: str | Path,
    device: str,
) -> dict[str, Any]:
    target = Path(output_dir)
    validation = Path(validation_root)
    units = holdout_units(mode)
    fold_values = (0.0,) if str(mode) == "smoke" else HELDOUT_VR_VALUES
    base_cfg = _baseline_config(baseline)
    base_cfg.data_path = str(paths.resolve_project_path(input_path))
    all_curves = _full_curves(base_cfg)
    forward_evidence = json.loads(baseline.files["forward_evidence"].read_text(encoding="utf-8-sig"))
    fold_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    zero_metrics_rows: list[dict[str, Any]] = []
    calibrated_metrics_rows: list[dict[str, Any]] = []
    zero_prediction_rows: list[dict[str, Any]] = []
    calibrated_prediction_rows: list[dict[str, Any]] = []
    completed = 0
    failed = 0
    write_progress(validation / "progress.json", completed=0, failed=0, total=len(units), current_stage="holdout")
    for heldout_vr in fold_values:
        fold_units = [unit for unit in units if unit.heldout_vr == heldout_vr]
        fold_dir = target / "folds" / f"vr_{_vr_label(heldout_vr)}"
        scan_dir = fold_dir / "fullscan"
        cfg = _fold_config(
            mode=mode,
            input_path=input_path,
            baseline=baseline,
            out_dir=scan_dir,
            device=device,
            heldout_vr=heldout_vr,
        )
        _, heldout_curves = split_vr_fold(all_curves, heldout_vr=heldout_vr)
        heldout_curve = heldout_curves[0]
        scan_dir.mkdir(parents=True, exist_ok=True)
        atomic_json_dump(forward_evidence, scan_dir / "forward_evidence.json")
        fold_identity = {
            "baseline_sha256": baseline.identity_hash,
            "config_sha256": canonical_hash(asdict(cfg)),
            "heldout_vr": float(heldout_vr),
            "units": [asdict(unit) for unit in fold_units],
        }
        fold_manifest_path = fold_dir / "fold_manifest.json"
        if fold_manifest_path.exists():
            existing = json.loads(fold_manifest_path.read_text(encoding="utf-8-sig"))
            if existing.get("identity") != fold_identity:
                raise ValueError(f"Holdout fold identity mismatch; refusing to overwrite {fold_dir}")
        else:
            atomic_json_dump({"identity": fold_identity}, fold_manifest_path)
        try:
            scan_result = model.run_level_scan(cfg)
            selected_k = int(scan_result["decision"]["selected_k"])
            fold_rows.append(
                {
                    "heldout_vr": float(heldout_vr),
                    "selected_k": selected_k,
                    "status": "complete",
                    "config_sha256": fold_identity["config_sha256"],
                    "scan_dir": str(scan_dir),
                    "error": "",
                }
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            fold_rows.append(
                {
                    "heldout_vr": float(heldout_vr),
                    "selected_k": None,
                    "status": "failed",
                    "config_sha256": fold_identity["config_sha256"],
                    "scan_dir": str(scan_dir),
                    "error": error,
                }
            )
            for unit in fold_units:
                candidate_rows.append({**asdict(unit), "status": "failed", "selected_k": None, "error": error})
            failed += len(fold_units)
            write_progress(
                validation / "progress.json",
                completed=completed,
                failed=failed,
                total=len(units),
                current_stage="holdout",
                heldout_vr=float(heldout_vr),
            )
            continue

        for unit in fold_units:
            labels: list[str] = []
            if unit.n_levels in {1, 4, 8}:
                labels.append(f"k{unit.n_levels}")
            if unit.n_levels == selected_k:
                labels.append("selected_k")
            candidate = {
                **asdict(unit),
                "selected_k": selected_k,
                "is_selected_k": unit.n_levels == selected_k,
                "evaluated_labels": ";".join(labels),
                "status": "complete",
                "error": "",
            }
            if labels:
                try:
                    evaluation = evaluate_candidate(
                        unit=unit,
                        cfg=cfg,
                        scan_dir=scan_dir,
                        heldout_curve=heldout_curve,
                        device=device,
                    )
                    common = {**asdict(unit), "selected_k": selected_k}
                    for label in labels:
                        zero_metrics_rows.append(
                            {**common, "model_label": label, **evaluation["zero_metrics"]}
                        )
                        calibrated_metrics_rows.append(
                            {
                                **common,
                                "model_label": label,
                                **evaluation["calibrated_metrics"],
                                **evaluation["nuisance"],
                            }
                        )
                        zero_prediction_rows.extend(
                            {**common, "model_label": label, **row}
                            for row in evaluation["zero_predictions"]
                        )
                        calibrated_prediction_rows.extend(
                            {**common, "model_label": label, **row}
                            for row in evaluation["calibrated_predictions"]
                        )
                except Exception as exc:
                    candidate["status"] = "failed"
                    candidate["error"] = f"{type(exc).__name__}: {exc}"
            candidate_rows.append(candidate)
            if candidate["status"] == "complete":
                completed += 1
            else:
                failed += 1
            write_progress(
                validation / "progress.json",
                completed=completed,
                failed=failed,
                total=len(units),
                current_stage="holdout",
                heldout_vr=float(heldout_vr),
                n_levels=unit.n_levels,
                seed=unit.seed,
            )
        _write_csv(target / "candidate_status.csv", candidate_rows)

    def metric_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        frame = pd.DataFrame(rows)
        if frame.empty:
            return []
        return (
            frame.groupby("model_label", as_index=False)
            .agg(
                row_count=("rmse", "size"),
                rmse_mean=("rmse", "mean"),
                mae_mean=("mae", "mean"),
                nrmse_mean=("nrmse", "mean"),
            )
            .to_dict("records")
        )

    status = "incomplete" if failed else ("smoke_passed" if str(mode) == "smoke" else "complete")
    summary = {
        "ok": failed == 0,
        "status": status,
        "unit_total": len(units),
        "completed": completed,
        "failed": failed,
        "fold_count": len(fold_values),
        "zero_shot_summary": metric_summary(zero_metrics_rows),
        "calibrated_summary": metric_summary(calibrated_metrics_rows),
    }
    _write_csv(target / "fold_manifest.csv", fold_rows)
    _write_csv(target / "candidate_status.csv", candidate_rows)
    _write_csv(target / "zero_shot_metrics.csv", zero_metrics_rows)
    _write_csv(target / "calibrated_metrics.csv", calibrated_metrics_rows)
    _write_csv(target / "zero_shot_predictions.csv", zero_prediction_rows)
    _write_csv(target / "calibrated_predictions.csv", calibrated_prediction_rows)
    atomic_json_dump(summary, target / "holdout_summary.json")
    write_stage_status(
        target / "stage_status.json",
        status,
        unit_total=len(units),
        completed=completed,
        failed=failed,
        fold_count=len(fold_values),
    )
    return summary
