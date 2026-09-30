from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import CurveData, load_curve_data
from .metrics import apply_single_level_gate, evaluate_predictions
from .model import ThinLayerFranckHertz


@dataclass(frozen=True)
class TrainConfig:
    data_path: str
    output_dir: str
    n_levels: int = 1
    seed: int = 0
    device: str = "auto"
    max_epochs: int = 1800
    min_epochs: int = 300
    patience: int = 120
    min_delta_rel: float = 1.0e-4
    smoothing: int = 15
    learning_rate: float = 0.020
    n_layers: int = 32
    max_collisions: int = 8
    quadrature_order: int = 10
    log_interval: int = 10


class PlateauStopper:
    def __init__(self, min_epochs: int, patience: int, min_delta_rel: float, smoothing: int) -> None:
        self.min_epochs = int(min_epochs)
        self.patience = int(patience)
        self.min_delta_rel = float(min_delta_rel)
        self.smoothing = max(1, int(smoothing))
        self.values: list[float] = []
        self.best = float("inf")
        self.wait = 0

    def update(self, epoch: int, value: float) -> tuple[bool, float]:
        self.values.append(float(value))
        smoothed = float(np.mean(self.values[-self.smoothing :]))
        improvement = self.best - smoothed
        required = self.min_delta_rel * max(abs(self.best), 1.0e-12)
        if not math.isfinite(self.best) or improvement > required:
            self.best = smoothed
            self.wait = 0
        else:
            self.wait += 1
        return bool(epoch >= self.min_epochs and self.wait >= self.patience), smoothed


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def _seed_everything(seed: int, device: torch.device) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


def _loss_terms(model: nn.Module, data: CurveData) -> dict[str, torch.Tensor]:
    predicted = model(data.va, data.vr)
    point_scale = data.curve_ranges[data.curve_index]
    normalized = (predicted - data.current) / point_scale
    raw = torch.mean(torch.square(normalized))
    derivative_terms = []
    for curve_index in range(len(data.curve_ids)):
        mask = data.curve_index == curve_index
        pred_curve = predicted[mask]
        target_curve = data.current[mask]
        if pred_curve.numel() >= 2:
            derivative_terms.append(
                torch.mean(
                    torch.square(
                        (torch.diff(pred_curve) - torch.diff(target_curve))
                        / data.curve_ranges[curve_index]
                    )
                )
            )
    derivative = torch.mean(torch.stack(derivative_terms))
    if bool(torch.any(data.zero_mask)):
        zero = torch.mean(torch.square(predicted[data.zero_mask] / point_scale[data.zero_mask]))
    else:
        zero = torch.zeros((), device=predicted.device)
    early_mask = data.va <= 24.0
    early = torch.mean(torch.square(normalized[early_mask]))
    energies, _ = model.excitation_parameters()
    energy_ceiling = torch.square(torch.relu(energies[-1] - 16.0))
    total = raw + 0.20 * derivative + 0.75 * early + 1.25 * zero + 0.002 * energy_ceiling
    return {
        "total": total,
        "raw": raw,
        "derivative": derivative,
        "zero": zero,
        "early": early,
        "energy_ceiling": energy_ceiling,
        "predicted": predicted,
    }


def _json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def _config_hash(config: TrainConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _plot_fit(frame: pd.DataFrame, output_path: Path) -> None:
    groups = list(frame.groupby("curve_id", sort=True))
    figure, axes = plt.subplots(3, 2, figsize=(11, 11), sharex=True)
    axes_flat = axes.ravel()
    for axis, (_, group) in zip(axes_flat, groups):
        group = group.sort_values("Va")
        vr = float(group["Vr"].iloc[0])
        axis.plot(group["Va"], group["IuA"], color="black", linewidth=1.4, label="observed")
        axis.plot(group["Va"], group["predicted"], color="#1f77b4", linewidth=1.4, label="model")
        axis.set_title(f"$V_r={vr:g}$ V")
        axis.set_ylabel("Plate current ($\\mu$A)")
        axis.grid(alpha=0.22)
    for axis in axes_flat[len(groups) :]:
        axis.axis("off")
    axes_flat[0].legend(frameon=False)
    for axis in axes[-1, :]:
        axis.set_xlabel("Accelerating voltage $V_a$ (V)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def fit_model(
    config: TrainConfig,
    baseline_path: str | Path | None = None,
    model_factory: Callable[[TrainConfig], nn.Module] | None = None,
    single_level_gate_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    material_passport: dict[str, Any] | None = None,
    initial_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    device = select_device(config.device)
    _seed_everything(config.seed, device)
    data = load_curve_data(config.data_path, device)
    if model_factory is None:
        model = ThinLayerFranckHertz(
            n_levels=config.n_levels,
            n_layers=config.n_layers,
            max_collisions=config.max_collisions,
            quadrature_order=config.quadrature_order,
            seed=config.seed,
        ).to(device)
    else:
        model = model_factory(config).to(device)
    initialization = None
    if initial_checkpoint is not None:
        checkpoint_path = Path(initial_checkpoint).resolve()
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
        source_state = payload["model_state"]
        target_state = model.state_dict()
        compatible = {
            name: value
            for name, value in source_state.items()
            if name in target_state and target_state[name].shape == value.shape
        }
        model.load_state_dict(compatible, strict=False)
        initialization = {
            "source_checkpoint": str(checkpoint_path),
            "loaded_tensor_count": len(compatible),
            "skipped_source_tensors": sorted(set(source_state) - set(compatible)),
        }
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config.learning_rate), weight_decay=1.0e-6
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(10, config.patience // 4), min_lr=2.0e-4
    )
    stopper = PlateauStopper(
        min_epochs=config.min_epochs,
        patience=config.patience,
        min_delta_rel=config.min_delta_rel,
        smoothing=config.smoothing,
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    best_epoch = 0
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    stop_reason = "max_epochs"

    for epoch in range(1, int(config.max_epochs) + 1):
        optimizer.zero_grad(set_to_none=True)
        terms = _loss_terms(model, data)
        terms["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        value = float(terms["total"].detach().cpu())
        scheduler.step(value)
        should_stop, smoothed = stopper.update(epoch, value)
        if value < best_loss:
            best_loss = value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % int(config.log_interval) == 0 or should_stop or epoch == config.max_epochs:
            rows.append(
                {
                    "epoch": epoch,
                    "loss_total": value,
                    "loss_smoothed": smoothed,
                    "loss_raw": float(terms["raw"].detach().cpu()),
                    "loss_derivative": float(terms["derivative"].detach().cpu()),
                    "loss_zero": float(terms["zero"].detach().cpu()),
                    "loss_early": float(terms["early"].detach().cpu()),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "early_stop_wait": stopper.wait,
                }
            )
        if should_stop:
            stop_reason = "plateau_early_stop"
            break

    if best_state is None:
        raise RuntimeError("Training produced no best state")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = model(data.va, data.vr).detach().cpu().numpy()
    elapsed = float(time.perf_counter() - started)
    prediction_frame = data.frame.copy()
    prediction_frame["observed"] = prediction_frame["IuA"]
    prediction_frame["predicted"] = prediction
    prediction_frame["residual"] = prediction_frame["observed"] - prediction_frame["predicted"]
    prediction_frame.to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(output_dir / "train_log.csv", index=False)
    torch.save(
        {
            "model_state": best_state,
            "config": asdict(config),
            "config_sha256": _config_hash(config),
            "best_epoch": best_epoch,
            "best_loss": best_loss,
            "initialization": initialization,
        },
        output_dir / "checkpoint_best.pt",
    )
    metrics = evaluate_predictions(data.frame, prediction)
    gate_function = single_level_gate_fn or apply_single_level_gate
    gate = gate_function(metrics) if config.n_levels == 1 else None
    baseline_metrics = None
    if baseline_path is not None and Path(baseline_path).is_file():
        baseline = pd.read_csv(baseline_path)
        if len(baseline) == len(data.frame) and "predicted" in baseline:
            baseline_metrics = evaluate_predictions(data.frame, baseline["predicted"].to_numpy())
    n_points = int(metrics["summary"]["n_points"])
    n_params = int(model.parameter_report()["trainable_parameter_count"])
    sse = max(float(metrics["summary"]["sse_uA2"]), 1.0e-12)
    metrics["summary"].update(
        {
            "n_params": n_params,
            "bic": float(n_points * math.log(sse / n_points) + n_params * math.log(n_points)),
            "aic": float(n_points * math.log(sse / n_points) + 2 * n_params),
        }
    )
    result = {
        "status": "completed",
        "config": asdict(config),
        "config_sha256": _config_hash(config),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "elapsed_seconds": elapsed,
        "epochs_completed": int(rows[-1]["epoch"]),
        "best_epoch": int(best_epoch),
        "best_loss": float(best_loss),
        "stop_reason": stop_reason,
        "parameters": model.parameter_report(),
        "metrics": metrics,
        "single_level_gate": gate,
        "old_k1_baseline_metrics": baseline_metrics,
        "initialization": initialization,
    }
    if material_passport is not None:
        result["material_passport"] = material_passport
    _json_dump(result, output_dir / "result.json")
    _plot_fit(prediction_frame, output_dir / "fit.png")
    return result
