from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks, savgol_filter


SINGLE_LEVEL_GATE_THRESHOLDS: dict[str, tuple[str, float]] = {
    "mean_curve_nrmse": ("max", 0.060),
    "cutoff_mae_V": ("max", 1.50),
    "zero_region_mae_uA": ("max", 0.015),
    "peak_position_mae_V": ("max", 1.00),
    "trough_position_mae_V": ("max", 1.25),
    "peak_recall": ("min", 0.80),
    "trough_recall": ("min", 0.75),
    "peak_drift_rmse_V": ("max", 1.00),
    "vr0_spurious_feature_count": ("max", 0.0),
}

V2_SINGLE_LEVEL_GATE_THRESHOLDS: dict[str, tuple[str, float]] = {
    "mean_curve_nrmse": ("max", 0.055),
    "cutoff_mae_V": ("max", 1.50),
    "max_cutoff_error_V": ("max", 1.50),
    "zero_region_mae_uA": ("max", 0.015),
    "peak_position_mae_V": ("max", 1.00),
    "trough_position_mae_V": ("max", 1.25),
    "peak_recall": ("min", 0.80),
    "trough_recall": ("min", 0.75),
    "peak_drift_rmse_V": ("max", 1.00),
    "vr0_spurious_feature_count": ("max", 0.0),
}

V2B_SINGLE_LEVEL_GATE_THRESHOLDS: dict[str, tuple[str, float]] = {
    "mean_curve_nrmse": ("max", 0.050),
    "cutoff_mae_V": ("max", 1.00),
    "max_cutoff_error_V": ("max", 1.50),
    "zero_region_mae_uA": ("max", 0.015),
    "peak_position_mae_V": ("max", 1.00),
    "trough_position_mae_V": ("max", 1.25),
    "peak_recall": ("min", 0.80),
    "trough_recall": ("min", 0.75),
    "peak_drift_rmse_V": ("max", 1.00),
    "vr0_spurious_feature_count": ("max", 0.0),
    "vr0_low_voltage_mae_uA": ("max", 0.070),
    "late_trough_current_mae_uA": ("max", 0.205),
    "late_voltage_mae_uA": ("max", 0.110),
}

V2J_SINGLE_LEVEL_GATE_THRESHOLDS: dict[str, tuple[str, float]] = {
    **V2B_SINGLE_LEVEL_GATE_THRESHOLDS,
    "trough_precision": ("min", 1.0),
    "early_high_retarding_mae_uA": ("max", 0.075),
}


def _odd_window(length: int, preferred: int = 9) -> int:
    window = min(int(preferred), int(length) if int(length) % 2 else int(length) - 1)
    return max(window, 3)


def feature_positions(va: Iterable[float], current: Iterable[float]) -> dict[str, np.ndarray]:
    x = np.asarray(tuple(va), dtype=float)
    y = np.asarray(tuple(current), dtype=float)
    if x.size < 5:
        return {"peaks": np.array([], dtype=float), "troughs": np.array([], dtype=float)}
    smooth = savgol_filter(y, _odd_window(y.size), 3 if y.size >= 5 else 2)
    spacing = float(np.median(np.diff(x)))
    distance = max(2, int(round(7.0 / max(spacing, 1.0e-6))))
    prominence = max(0.035 * float(np.ptp(smooth)), 0.010)
    peak_indices, _ = find_peaks(smooth, prominence=prominence, distance=distance)
    trough_indices, _ = find_peaks(-smooth, prominence=prominence, distance=distance)
    peak_indices = peak_indices[x[peak_indices] >= 24.0]
    trough_indices = trough_indices[x[trough_indices] >= 24.0]
    return {"peaks": x[peak_indices], "troughs": x[trough_indices]}


def _match_features(
    observed: np.ndarray,
    predicted: np.ndarray,
    max_distance: float = 2.5,
) -> dict[str, Any]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if observed.size == 0 or predicted.size == 0:
        return {
            "matched": 0,
            "observed": int(observed.size),
            "predicted": int(predicted.size),
            "absolute_errors": [],
            "pairs": [],
        }
    cost = np.abs(observed[:, None] - predicted[None, :])
    observed_idx, predicted_idx = linear_sum_assignment(cost)
    pairs = []
    errors = []
    for left, right in zip(observed_idx, predicted_idx):
        error = float(cost[left, right])
        if error <= float(max_distance):
            pairs.append((float(observed[left]), float(predicted[right])))
            errors.append(error)
    return {
        "matched": len(errors),
        "observed": int(observed.size),
        "predicted": int(predicted.size),
        "absolute_errors": errors,
        "pairs": pairs,
    }


def _first_crossing(va: np.ndarray, current: np.ndarray, threshold: float = 0.01) -> float:
    indices = np.flatnonzero(np.asarray(current, dtype=float) >= float(threshold))
    return float(va[indices[0]]) if indices.size else float("nan")


def evaluate_predictions(frame: pd.DataFrame, predicted: Iterable[float]) -> dict[str, Any]:
    if len(frame) != len(tuple(predicted)):
        raise ValueError("Prediction length does not match the data frame")
    work = frame.copy()
    work["predicted"] = np.asarray(tuple(predicted), dtype=float)
    per_curve: list[dict[str, Any]] = []
    peak_errors: list[float] = []
    trough_errors: list[float] = []
    cutoff_errors: list[float] = []
    peak_current_errors: list[float] = []
    trough_current_errors: list[float] = []
    late_trough_current_errors: list[float] = []
    contrast_errors: list[float] = []
    low_voltage_mae: list[float] = []
    late_voltage_errors: list[float] = []
    early_high_retarding_errors: list[float] = []
    peak_totals = {"matched": 0, "observed": 0, "predicted": 0}
    trough_totals = {"matched": 0, "observed": 0, "predicted": 0}
    feature_map: dict[float, dict[str, dict[str, np.ndarray]]] = {}
    vr0_spurious = 0
    vr0_low_voltage_mae = float("nan")

    for curve_id, group in work.groupby("curve_id", sort=True):
        group = group.sort_values("Va")
        va = group["Va"].to_numpy(dtype=float)
        observed = group["IuA"].to_numpy(dtype=float)
        fit = group["predicted"].to_numpy(dtype=float)
        scale = max(float(np.ptp(observed)), 1.0e-8)
        nrmse = float(np.sqrt(np.mean(np.square(fit - observed))) / scale)
        observed_features = feature_positions(va, observed)
        predicted_features = feature_positions(va, fit)
        peaks = _match_features(observed_features["peaks"], predicted_features["peaks"])
        troughs = _match_features(observed_features["troughs"], predicted_features["troughs"])
        peak_errors.extend(peaks["absolute_errors"])
        trough_errors.extend(troughs["absolute_errors"])
        for key in peak_totals:
            peak_totals[key] += int(peaks[key])
            trough_totals[key] += int(troughs[key])
        observed_cutoff = _first_crossing(va, observed)
        predicted_cutoff = _first_crossing(va, fit)
        cutoff_error = abs(observed_cutoff - predicted_cutoff)
        if math.isfinite(cutoff_error):
            cutoff_errors.append(float(cutoff_error))
        vr = float(group["Vr"].iloc[0])
        observed_peak_indices = np.flatnonzero(np.isin(va, observed_features["peaks"]))
        observed_trough_indices = np.flatnonzero(np.isin(va, observed_features["troughs"]))
        peak_current_errors.extend((fit[observed_peak_indices] - observed[observed_peak_indices]).tolist())
        trough_current_errors.extend(
            (fit[observed_trough_indices] - observed[observed_trough_indices]).tolist()
        )
        late_trough_indices = observed_trough_indices[va[observed_trough_indices] >= 55.0]
        late_trough_current_errors.extend(
            (fit[late_trough_indices] - observed[late_trough_indices]).tolist()
        )
        for peak_index in observed_peak_indices:
            following = observed_trough_indices[observed_trough_indices > peak_index]
            if following.size:
                trough_index = int(following[0])
                observed_contrast = observed[peak_index] - observed[trough_index]
                predicted_contrast = fit[peak_index] - fit[trough_index]
                contrast_errors.append(float(predicted_contrast - observed_contrast))
        low_mask = va <= 15.0
        low_voltage_mae.append(float(np.mean(np.abs(fit[low_mask] - observed[low_mask]))))
        late_voltage_errors.extend((fit[va >= 60.0] - observed[va >= 60.0]).tolist())
        if vr >= 8.0:
            early_high_mask = (va >= 15.0) & (va <= 30.0)
            early_high_retarding_errors.extend(
                (fit[early_high_mask] - observed[early_high_mask]).tolist()
            )
        if abs(vr) < 1.0e-9:
            vr0_low_voltage_mae = float(np.mean(np.abs(fit[low_mask] - observed[low_mask])))
        if abs(vr) < 1.0e-9 and not (
            observed_features["peaks"].size or observed_features["troughs"].size
        ):
            vr0_spurious = int(
                predicted_features["peaks"].size + predicted_features["troughs"].size
            )
        feature_map[vr] = {"observed": observed_features, "predicted": predicted_features}
        per_curve.append(
            {
                "curve_id": int(curve_id),
                "Vr": vr,
                "nrmse": nrmse,
                "observed_cutoff_V": observed_cutoff,
                "predicted_cutoff_V": predicted_cutoff,
                "cutoff_error_V": float(cutoff_error),
                "observed_peaks_V": observed_features["peaks"].tolist(),
                "predicted_peaks_V": predicted_features["peaks"].tolist(),
                "observed_troughs_V": observed_features["troughs"].tolist(),
                "predicted_troughs_V": predicted_features["troughs"].tolist(),
            }
        )

    drift_errors: list[float] = []
    positive_vr = sorted(value for value in feature_map if value >= 4.0)
    if positive_vr:
        reference_vr = positive_vr[0]
        observed_reference = feature_map[reference_vr]["observed"]["peaks"]
        predicted_reference = feature_map[reference_vr]["predicted"]["peaks"]
        for vr in positive_vr[1:]:
            observed_peaks = feature_map[vr]["observed"]["peaks"]
            predicted_peaks = feature_map[vr]["predicted"]["peaks"]
            count = min(
                observed_reference.size,
                predicted_reference.size,
                observed_peaks.size,
                predicted_peaks.size,
            )
            for index in range(int(count)):
                observed_shift = observed_peaks[index] - observed_reference[index]
                predicted_shift = predicted_peaks[index] - predicted_reference[index]
                drift_errors.append(float(predicted_shift - observed_shift))

    zero_values = work.loc[work["IuA"] <= 1.0e-3, "predicted"].to_numpy(dtype=float)
    residual = work["predicted"].to_numpy(dtype=float) - work["IuA"].to_numpy(dtype=float)
    overall_range = max(float(np.ptp(work["IuA"].to_numpy(dtype=float))), 1.0e-8)
    summary = {
        "overall_nrmse": float(np.sqrt(np.mean(np.square(residual))) / overall_range),
        "mean_curve_nrmse": float(np.mean([row["nrmse"] for row in per_curve])),
        "cutoff_mae_V": float(np.mean(cutoff_errors)) if cutoff_errors else float("nan"),
        "max_cutoff_error_V": float(np.max(cutoff_errors)) if cutoff_errors else float("nan"),
        "zero_region_mae_uA": float(np.mean(np.abs(zero_values))) if zero_values.size else 0.0,
        "peak_position_mae_V": float(np.mean(peak_errors)) if peak_errors else float("inf"),
        "trough_position_mae_V": float(np.mean(trough_errors)) if trough_errors else float("inf"),
        "peak_recall": peak_totals["matched"] / max(peak_totals["observed"], 1),
        "peak_precision": peak_totals["matched"] / max(peak_totals["predicted"], 1),
        "trough_recall": trough_totals["matched"] / max(trough_totals["observed"], 1),
        "trough_precision": trough_totals["matched"] / max(trough_totals["predicted"], 1),
        "peak_drift_rmse_V": float(np.sqrt(np.mean(np.square(drift_errors)))) if drift_errors else float("inf"),
        "vr0_spurious_feature_count": int(vr0_spurious),
        "low_voltage_mae_uA": float(np.mean(low_voltage_mae)),
        "vr0_low_voltage_mae_uA": float(vr0_low_voltage_mae),
        "late_voltage_mae_uA": float(np.mean(np.abs(late_voltage_errors))),
        "early_high_retarding_mae_uA": float(
            np.mean(np.abs(early_high_retarding_errors))
        ),
        "peak_current_bias_uA": float(np.mean(peak_current_errors)),
        "peak_current_mae_uA": float(np.mean(np.abs(peak_current_errors))),
        "trough_current_bias_uA": float(np.mean(trough_current_errors)),
        "trough_current_mae_uA": float(np.mean(np.abs(trough_current_errors))),
        "late_trough_current_bias_uA": float(np.mean(late_trough_current_errors)),
        "late_trough_current_mae_uA": float(np.mean(np.abs(late_trough_current_errors))),
        "oscillation_contrast_bias_uA": float(np.mean(contrast_errors)),
        "oscillation_contrast_mae_uA": float(np.mean(np.abs(contrast_errors))),
        "sse_uA2": float(np.sum(np.square(residual))),
        "n_points": int(len(work)),
    }
    return {"summary": summary, "per_curve": per_curve}


def _apply_gate(
    metrics: dict[str, Any], thresholds: dict[str, tuple[str, float]]
) -> dict[str, Any]:
    summary = metrics["summary"]
    checks = []
    for name, (direction, threshold) in thresholds.items():
        value = float(summary.get(name, float("nan")))
        passed = math.isfinite(value) and (
            value <= threshold if direction == "max" else value >= threshold
        )
        checks.append(
            {
                "metric": name,
                "direction": direction,
                "threshold": threshold,
                "value": value,
                "passed": bool(passed),
            }
        )
    return {"passed": all(check["passed"] for check in checks), "checks": checks}


def apply_single_level_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    return _apply_gate(metrics, SINGLE_LEVEL_GATE_THRESHOLDS)


def apply_v2_single_level_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    return _apply_gate(metrics, V2_SINGLE_LEVEL_GATE_THRESHOLDS)


def apply_v2b_single_level_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    return _apply_gate(metrics, V2B_SINGLE_LEVEL_GATE_THRESHOLDS)


def apply_v2j_single_level_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    return _apply_gate(metrics, V2J_SINGLE_LEVEL_GATE_THRESHOLDS)
