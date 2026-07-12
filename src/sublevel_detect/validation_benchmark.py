from __future__ import annotations

import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import model, paths
from .validation_common import atomic_json_dump, write_progress, write_stage_status
from .validation_holdout import HELDOUT_VR_VALUES, prediction_metrics, split_vr_fold


@dataclass(frozen=True)
class BenchmarkUnit:
    heldout_vr: float
    seed: int


@dataclass(frozen=True)
class Standardizer:
    mean: tuple[float, float]
    scale: tuple[float, float]


class BudgetMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 11),
            nn.Tanh(),
            nn.Linear(11, 1),
            nn.Softplus(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def benchmark_units(mode: str) -> list[BenchmarkUnit]:
    if str(mode) == "smoke":
        return [BenchmarkUnit(heldout_vr=0.0, seed=0)]
    return [
        BenchmarkUnit(heldout_vr=heldout_vr, seed=seed)
        for heldout_vr in HELDOUT_VR_VALUES
        for seed in (0, 1, 2)
    ]


def design_matrix(curves: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for curve in curves:
        va = np.asarray(curve["Va"], dtype=np.float64)
        if bool(curve["Vr_is_vector"]):
            vr = np.asarray(curve["Vr"], dtype=np.float64)
        else:
            vr = np.full_like(va, float(curve["Vr"]), dtype=np.float64)
        current = np.asarray(curve["Ip"], dtype=np.float64)
        if va.shape != vr.shape or va.shape != current.shape:
            raise ValueError("Va, Vr, and current must have identical shapes")
        features.append(np.column_stack([va, vr]))
        targets.append(current)
    if not features:
        raise ValueError("At least one curve is required")
    return np.concatenate(features, axis=0), np.concatenate(targets, axis=0)


def fit_standardizer(training_features: np.ndarray) -> Standardizer:
    values = np.asarray(training_features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
        raise ValueError("Training features must have shape (n, 2)")
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    return Standardizer(tuple(float(value) for value in mean), tuple(float(value) for value in scale))


def transform(features: np.ndarray, standardizer: Standardizer) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Features must have shape (n, 2)")
    return (values - np.asarray(standardizer.mean)) / np.asarray(standardizer.scale)


def _load_curves(input_path: str | Path) -> list[dict[str, Any]]:
    cfg = model.Config(data_path=str(paths.resolve_project_path(input_path)), exclude_vr_values="")
    curves, _ = model.load_curves_and_init(cfg)
    return curves


def _train_unit(
    *,
    unit: BenchmarkUnit,
    curves: Sequence[Mapping[str, Any]],
    mode: str,
    device: str,
) -> dict[str, Any]:
    training, heldout = split_vr_fold(curves, heldout_vr=unit.heldout_vr)
    train_features, train_target = design_matrix(training)
    test_features, test_target = design_matrix(heldout)
    standardizer = fit_standardizer(train_features)
    train_values = transform(train_features, standardizer)
    test_values = transform(test_features, standardizer)
    torch.manual_seed(int(unit.seed))
    np.random.seed(int(unit.seed))
    torch_device = model.resolve_device(device)
    network = BudgetMLP().to(torch_device)
    if sum(parameter.numel() for parameter in network.parameters()) != 45:
        raise AssertionError("BudgetMLP parameter count changed from 45")
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.0025, weight_decay=1e-4)
    x_train = torch.as_tensor(train_values, dtype=torch.float32, device=torch_device)
    y_train = torch.as_tensor(train_target, dtype=torch.float32, device=torch_device).view(-1, 1)
    x_test = torch.as_tensor(test_values, dtype=torch.float32, device=torch_device)
    max_epochs = 2 if str(mode) == "smoke" else 3500
    stopper = model.EarlyStopper(
        warmup_epochs=1 if str(mode) == "smoke" else 300,
        min_epochs=1 if str(mode) == "smoke" else 300,
        patience=10 if str(mode) == "smoke" else 45,
        min_delta_rel=2e-4,
        smoothing=7,
    )
    started = time.perf_counter()
    epochs_completed = 0
    final_train_loss = float("nan")
    for epoch in range(1, max_epochs + 1):
        network.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = network(x_train)
        loss = torch.mean(torch.square(prediction - y_train))
        loss.backward()
        optimizer.step()
        final_train_loss = float(loss.detach().cpu())
        epochs_completed = epoch
        if stopper.update(epoch, final_train_loss).should_stop:
            break
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    training_seconds = float(time.perf_counter() - started)
    network.eval()
    with torch.no_grad():
        test_prediction = network(x_test).view(-1).detach().cpu().numpy().astype(np.float64)
    metrics = prediction_metrics(
        test_target,
        test_prediction,
        denominator_observed=np.asarray(heldout[0]["Ip"], dtype=np.float64),
    )
    va = np.asarray(heldout[0]["Va"], dtype=np.float64)
    predictions = [
        {
            "index": int(index),
            "Va": float(va[index]),
            "Vr": float(unit.heldout_vr),
            "observed": float(test_target[index]),
            "predicted": float(test_prediction[index]),
        }
        for index in range(len(test_target))
    ]
    return {
        "metrics": metrics,
        "predictions": predictions,
        "standardizer": asdict(standardizer),
        "training_seconds": training_seconds,
        "epochs_completed": epochs_completed,
        "final_train_loss": final_train_loss,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(row) for row in rows]).to_csv(path, index=False)


def run(
    *,
    mode: str,
    input_path: str | Path,
    holdout_dir: str | Path,
    output_dir: str | Path,
    validation_root: str | Path,
    device: str,
) -> dict[str, Any]:
    target = Path(output_dir)
    validation = Path(validation_root)
    units = benchmark_units(mode)
    curves = _load_curves(input_path)
    metrics_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    completed = 0
    failed = 0
    write_progress(validation / "progress.json", completed=0, failed=0, total=len(units), current_stage="benchmark")
    for unit in units:
        row = {**asdict(unit), "status": "failed", "error": ""}
        try:
            trained = _train_unit(unit=unit, curves=curves, mode=mode, device=device)
            row.update(
                status="complete",
                **trained["metrics"],
                parameter_count=45,
                training_seconds=trained["training_seconds"],
                epochs_completed=trained["epochs_completed"],
                final_train_loss=trained["final_train_loss"],
                train_feature_mean=str(trained["standardizer"]["mean"]),
                train_feature_scale=str(trained["standardizer"]["scale"]),
            )
            prediction_rows.extend({**asdict(unit), **point} for point in trained["predictions"])
            completed += 1
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            failed += 1
        metrics_rows.append(row)
        _write_csv(target / "fold_metrics.csv", metrics_rows)
        write_progress(
            validation / "progress.json",
            completed=completed,
            failed=failed,
            total=len(units),
            current_stage="benchmark",
            heldout_vr=unit.heldout_vr,
            seed=unit.seed,
        )
    completed_metrics = pd.DataFrame([row for row in metrics_rows if row["status"] == "complete"])
    physical_path = Path(holdout_dir) / "zero_shot_metrics.csv"
    paired_rows: list[dict[str, Any]] = []
    if physical_path.is_file() and not completed_metrics.empty:
        physical = pd.read_csv(physical_path)
        physical = physical[physical["model_label"].astype(str) == "selected_k"].copy()
        merged = completed_metrics.merge(
            physical,
            on=["heldout_vr", "seed"],
            how="left",
            suffixes=("_mlp", "_physical"),
        )
        for _, row in merged.iterrows():
            paired_rows.append(
                {
                    "heldout_vr": float(row["heldout_vr"]),
                    "seed": int(row["seed"]),
                    "mlp_nrmse": float(row["nrmse_mlp"]),
                    "physical_nrmse": float(row["nrmse_physical"]) if pd.notna(row.get("nrmse_physical")) else None,
                    "nrmse_mlp_minus_physical": (
                        float(row["nrmse_mlp"] - row["nrmse_physical"])
                        if pd.notna(row.get("nrmse_physical"))
                        else None
                    ),
                    "mlp_mae": float(row["mae_mlp"]),
                    "physical_mae": float(row["mae_physical"]) if pd.notna(row.get("mae_physical")) else None,
                }
            )
    status = "incomplete" if failed else ("smoke_passed" if str(mode) == "smoke" else "complete")
    summary = {
        "ok": failed == 0,
        "status": status,
        "unit_total": len(units),
        "completed": completed,
        "failed": failed,
        "parameter_count": 45,
        "rmse_mean": float(completed_metrics["rmse"].mean()) if not completed_metrics.empty else None,
        "mae_mean": float(completed_metrics["mae"].mean()) if not completed_metrics.empty else None,
        "nrmse_mean": float(completed_metrics["nrmse"].mean()) if not completed_metrics.empty else None,
        "training_seconds_total": (
            float(completed_metrics["training_seconds"].sum()) if not completed_metrics.empty else 0.0
        ),
        "scope": "fixed 2-to-11-to-1 Tanh Softplus MLP only",
    }
    atomic_json_dump(
        {
            "inputs": ["Va", "Vr"],
            "architecture": "2->11->1",
            "hidden_activation": "Tanh",
            "output_activation": "Softplus",
            "parameter_count": 45,
            "optimizer": "AdamW",
            "learning_rate": 0.0025,
            "weight_decay": 1e-4,
            "max_epochs": 2 if str(mode) == "smoke" else 3500,
            "early_stop_min_epochs": 1 if str(mode) == "smoke" else 300,
            "early_stop_warmup": 1 if str(mode) == "smoke" else 300,
            "early_stop_patience": 10 if str(mode) == "smoke" else 45,
        },
        target / "mlp_config.json",
    )
    _write_csv(target / "fold_metrics.csv", metrics_rows)
    _write_csv(target / "predictions.csv", prediction_rows)
    _write_csv(target / "paired_comparison.csv", paired_rows)
    atomic_json_dump(summary, target / "benchmark_summary.json")
    write_stage_status(
        target / "stage_status.json",
        status,
        unit_total=len(units),
        completed=completed,
        failed=failed,
        parameter_count=45,
    )
    return summary
