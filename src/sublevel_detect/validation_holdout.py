from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn


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
