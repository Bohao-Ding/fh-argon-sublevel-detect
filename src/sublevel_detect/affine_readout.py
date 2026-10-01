"""Training-only readout calibration of a frozen physical response.

These coefficients describe residual response, not independently measured
instrument errors. The Vr-dependent option is an exploratory diagnostic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .spectral_pipeline import curve_vr, curves_hash


def fit_readout(curves: list[dict], predictions: list[np.ndarray], mode: str) -> dict:
    if mode not in ("none", "global", "linear_vr_diagnostic"):
        raise ValueError("unknown_readout_mode")
    matrices, targets = [], []
    for curve, values in zip(curves, predictions, strict=True):
        z = (curve_vr(curve) - 4.0) / 4.0
        columns = [values, np.ones_like(values)]
        if mode == "linear_vr_diagnostic":
            columns += [z * values, np.full_like(values, z)]
        scale = max(float(np.ptp(curve["Ip"])), 1e-8)
        matrices.append(np.column_stack(columns) / scale)
        targets.append(np.asarray(curve["Ip"], float) / scale)
    design = np.vstack(matrices)
    coefficients, _, rank, _ = np.linalg.lstsq(design, np.concatenate(targets), rcond=None)
    if mode == "none":
        coefficients = np.array([1.0, 0.0])
    return {"mode": mode, "coefficients": coefficients.tolist(),
            "training_vr": [curve_vr(c) for c in curves],
            "training_hash": curves_hash(curves), "design_rank": int(rank),
            "design_condition": float(np.linalg.cond(design))}


def readout_values(calibration: dict, vr: float) -> tuple[float, float]:
    coefficients = calibration["coefficients"]
    gain, bias = coefficients[:2]
    if calibration["mode"] == "linear_vr_diagnostic":
        z = (vr - 4.0) / 4.0
        gain += coefficients[2] * z
        bias += coefficients[3] * z
    return float(gain), float(bias)


def apply_readout(values: np.ndarray, vr: float, calibration: dict) -> np.ndarray:
    gain, bias = readout_values(calibration, vr)
    return np.maximum(gain * np.asarray(values) + bias, 0.0)


def select_readout(rows: list[dict]) -> tuple[str, list[dict]]:
    """One-SE comparison of identity and a common two-parameter affine readout."""
    frame = pd.DataFrame(rows)
    summaries = []
    for mode in ("none", "global"):
        values = frame.loc[frame["mode"] == mode, "nrmse"].to_numpy()
        summaries.append(dict(mode=mode, mean_nrmse=float(values.mean()),
            se_nrmse=float(values.std(ddof=1) / np.sqrt(len(values))), folds=len(values)))
    best = min(summaries, key=lambda row: row["mean_nrmse"])
    threshold = best["mean_nrmse"] + best["se_nrmse"]
    selected = next(row["mode"] for row in summaries if row["mean_nrmse"] <= threshold)
    for row in summaries:
        row.update(threshold=threshold, selected=row["mode"] == selected)
    return selected, summaries
