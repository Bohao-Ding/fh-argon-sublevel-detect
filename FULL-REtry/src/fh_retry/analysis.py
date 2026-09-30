from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


EXPECTED_SHA256 = "FF4B316B7D11C84B99E07B9AE0E32AFD03EEFD33698CF9DE3D02062D7F1902F3"
EXPECTED_COLUMNS = ["curve_id", "Vr", "Va", "IuA"]
VR_VALUES = [0, 4, 6, 8, 10]
BAND_LABELS = ["Vr_0_4", "Vr_4_6", "Vr_6_8", "Vr_8_10", "Vr_gt_10"]
PERIOD_PRIOR = 11.55
PSEUDO_PERIODS = [9.0, 10.0, 13.5, 15.0]
BLOCKS = [(22.5 + 11.5 * i, 34.0 + 11.5 * i) for i in range(5)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_and_audit(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    actual_hash = sha256_file(path)
    if actual_hash != EXPECTED_SHA256:
        raise ValueError(f"dataset SHA-256 mismatch: {actual_hash}")

    data = pd.read_excel(path, sheet_name="Sheet1")
    if list(data.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"unexpected columns: {list(data.columns)}")
    if data.isna().any().any():
        raise ValueError("dataset contains missing values")
    if not np.isfinite(data[EXPECTED_COLUMNS].to_numpy(float)).all():
        raise ValueError("dataset contains non-finite values")
    if data.duplicated(["curve_id", "Va"]).any() or data.duplicated(["Vr", "Va"]).any():
        raise ValueError("dataset contains duplicate curve/grid keys")

    mapping = data[["curve_id", "Vr"]].drop_duplicates().sort_values("Vr")
    if mapping.shape[0] != 5 or mapping["Vr"].tolist() != VR_VALUES:
        raise ValueError("expected one curve for each Vr={0,4,6,8,10}")

    pivot = data.pivot(index="Va", columns="Vr", values="IuA").sort_index()
    if pivot.shape != (161, 5) or pivot.columns.tolist() != VR_VALUES:
        raise ValueError(f"unexpected pivot shape/conditions: {pivot.shape}, {pivot.columns.tolist()}")
    va = pivot.index.to_numpy(float)
    if not np.allclose(va, np.arange(0.0, 80.0 + 0.5, 0.5), atol=1e-12):
        raise ValueError("unexpected Va grid")

    currents = pivot.to_numpy(float).T
    upward = currents[1:] - currents[:-1]
    violations = upward > 1e-12
    unique_current = np.unique(data["IuA"].to_numpy(float))
    positive_steps = np.diff(unique_current)
    quantization = float(np.min(positive_steps[positive_steps > 1e-12]))
    audit = {
        "sha256": actual_hash,
        "rows": int(data.shape[0]),
        "columns": EXPECTED_COLUMNS,
        "curve_count": 5,
        "points_per_curve": 161,
        "vr_values_V": VR_VALUES,
        "va_min_V": float(va.min()),
        "va_max_V": float(va.max()),
        "va_step_V": float(np.diff(va).min()),
        "missing_values": 0,
        "duplicate_keys": 0,
        "current_quantization_uA": quantization,
        "monotonic_comparisons": int(upward.size),
        "monotonic_violations": int(violations.sum()),
        "max_monotonic_violation_uA": float(upward[violations].max(initial=0.0)),
    }
    return data, pivot, audit


def make_retarding_bands(pivot: pd.DataFrame, min_va: float = 12.0) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    va_all = pivot.index.to_numpy(float)
    currents = pivot[VR_VALUES].to_numpy(float).T
    raw = np.vstack(
        [
            currents[0] - currents[1],
            currents[1] - currents[2],
            currents[2] - currents[3],
            currents[3] - currents[4],
            currents[4],
        ]
    )
    telescope_error = np.max(np.abs(raw.sum(axis=0) - currents[0]))
    negative = raw < 0.0
    clipped = np.maximum(raw, 0.0)
    valid = (va_all >= min_va) & (currents[0] > 0.1)
    va = va_all[valid]
    clipped = clipped[:, valid]
    fractions = (clipped / clipped.sum(axis=0, keepdims=True)).T
    frame = pd.DataFrame(fractions, columns=BAND_LABELS)
    frame.insert(0, "Va", va)
    band_audit = {
        "normalization_min_va_V": min_va,
        "normalization_min_I0_uA": 0.1,
        "retained_voltage_points": int(valid.sum()),
        "negative_raw_band_values": int(negative.sum()),
        "total_clipping_correction_uA": float(-raw[negative].sum()),
        "max_telescope_error_uA": float(telescope_error),
        "fraction_sum_max_abs_error": float(np.max(np.abs(fractions.sum(axis=1) - 1.0))),
    }
    return va, fractions, frame, band_audit


def _dominant_period(va: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    periods = np.linspace(8.0, 16.0, 321)
    total = np.sum((score - score.mean()) ** 2)
    r2 = []
    for period in periods:
        design = np.column_stack(
            [np.ones_like(va), np.sin(2.0 * np.pi * va / period), np.cos(2.0 * np.pi * va / period)]
        )
        fitted = design @ np.linalg.lstsq(design, score, rcond=None)[0]
        r2.append(1.0 - np.sum((score - fitted) ** 2) / total)
    best = int(np.argmax(r2))
    return float(periods[best]), float(r2[best])


def _modal_fit(va: np.ndarray, fractions: np.ndarray, start: float, window: int) -> dict:
    keep = va >= start
    local_va = va[keep]
    local = fractions[keep]
    smooth = np.column_stack(
        [savgol_filter(local[:, j], window, 3, mode="interp") for j in range(local.shape[1])]
    )
    residual = local - smooth
    residual /= residual.std(axis=0, ddof=1)
    u, singular, vt = np.linalg.svd(residual, full_matrices=False)
    scores = u * singular
    loadings = vt.T
    for mode in range(2):
        anchor = int(np.argmax(np.abs(loadings[:, mode])))
        if loadings[anchor, mode] < 0:
            loadings[:, mode] *= -1.0
            scores[:, mode] *= -1.0
    explained = singular**2 / np.sum(singular**2)
    periods, period_r2 = zip(*[_dominant_period(local_va, scores[:, j]) for j in range(2)])
    return {
        "va": local_va,
        "singular": singular,
        "explained": explained,
        "scores": scores,
        "loadings": loadings,
        "periods": periods,
        "period_r2": period_r2,
    }


def modal_analysis(va: np.ndarray, fractions: np.ndarray) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reference = _modal_fit(va, fractions, 15.0, 41)
    reference_space = reference["loadings"][:, :2]
    records = []
    for start in [12.0, 15.0, 18.0]:
        for window in [31, 41, 51]:
            fit = _modal_fit(va, fractions, start, window)
            cosine = np.linalg.svd(reference_space.T @ fit["loadings"][:, :2], compute_uv=False)
            angle = np.degrees(np.arccos(np.clip(cosine.min(), -1.0, 1.0)))
            records.append(
                {
                    "start_V": start,
                    "window_points": window,
                    "cumulative_explained_2": float(fit["explained"][:2].sum()),
                    "s2_over_s3": float(fit["singular"][1] / fit["singular"][2]),
                    "mode1_period_V": float(fit["periods"][0]),
                    "mode2_period_V": float(fit["periods"][1]),
                    "mode1_period_r2": float(fit["period_r2"][0]),
                    "mode2_period_r2": float(fit["period_r2"][1]),
                    "subspace_angle_deg": float(angle),
                }
            )
    sensitivity = pd.DataFrame(records)
    scores = pd.DataFrame(
        {"Va": reference["va"], "mode_1": reference["scores"][:, 0], "mode_2": reference["scores"][:, 1]}
    )
    loadings = pd.DataFrame(
        {"band": BAND_LABELS, "mode_1": reference["loadings"][:, 0], "mode_2": reference["loadings"][:, 1]}
    )
    summary = {
        "reference_start_V": 15.0,
        "reference_window_points": 41,
        "singular_values": reference["singular"].tolist(),
        "explained_fraction": reference["explained"].tolist(),
        "cumulative_explained_2": float(reference["explained"][:2].sum()),
        "s2_over_s3": float(reference["singular"][1] / reference["singular"][2]),
        "mode_periods_V": list(reference["periods"]),
        "mode_period_r2": list(reference["period_r2"]),
    }
    return summary, sensitivity, scores, loadings


def _predict_fold(
    va: np.ndarray,
    fractions: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    *,
    period: float | None,
    rank: int | None,
    trend_degree: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = fractions[train].mean(axis=0)
    scale = fractions[train].std(axis=0, ddof=1)
    scale = np.where(scale > 1e-10, scale, 1.0)
    standardized = (fractions - mean) / scale
    center = 0.5 * (va.min() + va.max())
    half_range = 0.5 * (va.max() - va.min())
    x = (va - center) / half_range
    trend = np.column_stack([x**degree for degree in range(trend_degree + 1)])
    y_train = standardized[train]

    if period is None:
        trend_coef = np.linalg.lstsq(trend[train], y_train, rcond=None)[0]
        predicted_z = trend[test] @ trend_coef
    else:
        harmonic = np.column_stack(
            [np.sin(2.0 * np.pi * va / period), np.cos(2.0 * np.pi * va / period)]
        )
        trend_train = trend[train]
        harmonic_train = harmonic[train]
        trend_coef = np.linalg.lstsq(trend_train, y_train, rcond=None)[0]
        residual_y = y_train - trend_train @ trend_coef
        residual_h = harmonic_train - trend_train @ np.linalg.lstsq(
            trend_train, harmonic_train, rcond=None
        )[0]
        q, r = np.linalg.qr(residual_h, mode="reduced")
        coefficient_orthogonal = q.T @ residual_y
        u, singular, vt = np.linalg.svd(coefficient_orthogonal, full_matrices=False)
        use_rank = int(rank or 2)
        truncated = (u[:, :use_rank] * singular[:use_rank]) @ vt[:use_rank]
        harmonic_coef = np.linalg.solve(r, truncated)
        trend_coef = np.linalg.lstsq(
            trend_train, y_train - harmonic_train @ harmonic_coef, rcond=None
        )[0]
        predicted_z = trend[test] @ trend_coef + harmonic[test] @ harmonic_coef

    predicted = predicted_z * scale + mean
    return predicted, predicted_z, standardized[test]


def blocked_cv(va: np.ndarray, fractions: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = [
        ("rank1_P11.55", PERIOD_PRIOR, 1, 3),
        ("rank2_P11.55", PERIOD_PRIOR, 2, 3),
        ("trend_degree5", None, None, 5),
    ] + [(f"rank2_P{period:g}", period, 2, 3) for period in PSEUDO_PERIODS]
    metrics = []
    predictions = []
    for block_index, (low, high) in enumerate(BLOCKS, start=1):
        test = (va >= low) & (va < high)
        train = ~test
        for model, period, rank, degree in models:
            predicted, predicted_z, observed_z = _predict_fold(
                va,
                fractions,
                train,
                test,
                period=period,
                rank=rank,
                trend_degree=degree,
            )
            metrics.append(
                {
                    "block": block_index,
                    "low_V": low,
                    "high_V_exclusive": high,
                    "model": model,
                    "period_V": period,
                    "rank": rank,
                    "balanced_rmse": float(np.sqrt(np.mean((predicted_z - observed_z) ** 2))),
                    "fraction_rmse": float(np.sqrt(np.mean((predicted - fractions[test]) ** 2))),
                    "test_voltage_points": int(test.sum()),
                }
            )
            if model in {"rank1_P11.55", "rank2_P11.55"}:
                for row, voltage in enumerate(va[test]):
                    for band_index, band in enumerate(BAND_LABELS):
                        predictions.append(
                            {
                                "block": block_index,
                                "Va": voltage,
                                "band": band,
                                "model": model,
                                "observed": fractions[test][row, band_index],
                                "predicted": predicted[row, band_index],
                            }
                        )
    return pd.DataFrame(metrics), pd.DataFrame(predictions)


def make_decision(data_audit: dict, band_audit: dict, modal: dict, sensitivity: pd.DataFrame, cv: pd.DataFrame) -> dict:
    fixed = cv[cv["model"].isin(["rank1_P11.55", "rank2_P11.55"])].pivot(
        index="block", columns="model", values="balanced_rmse"
    )
    relative = (fixed["rank1_P11.55"] - fixed["rank2_P11.55"]) / fixed["rank1_P11.55"]
    mean_error = cv.groupby("model")["balanced_rmse"].mean()
    pseudo_best = min(float(mean_error[f"rank2_P{period:g}"]) for period in PSEUDO_PERIODS)
    physical_mean = float(mean_error["rank2_P11.55"])

    gates = {
        "G0_data": bool(
            data_audit["sha256"] == EXPECTED_SHA256
            and data_audit["max_monotonic_violation_uA"] <= 0.0010000001
            and band_audit["total_clipping_correction_uA"] <= 0.0010000001
        ),
        "G1_two_mode_stability": bool(
            sensitivity["cumulative_explained_2"].min() >= 0.90
            and sensitivity["s2_over_s3"].min() >= 3.0
            and sensitivity[["mode1_period_V", "mode2_period_V"]].to_numpy().min() >= 10.0
            and sensitivity[["mode1_period_V", "mode2_period_V"]].to_numpy().max() <= 13.0
            and sensitivity["subspace_angle_deg"].max() <= 10.0
        ),
        "G2_block_prediction": bool(
            int((relative > 0.0).sum()) >= 4
            and float(relative.median()) >= 0.10
            and float(relative.min()) >= -0.05
        ),
        "G3_negative_controls": bool(
            physical_mean < float(mean_error["trend_degree5"]) and physical_mean < pseudo_best
        ),
    }
    passed = all(gates.values())
    return {
        "gates": gates,
        "phase_a_positive_screen": passed,
        "scale_decision": "ALLOW_CPU_CONFIRMATION_ONLY" if passed else "STOP_AND_RETAIN_NEGATIVE_RESULT",
        "data_metrics": {
            "monotonic_violations": data_audit["monotonic_violations"],
            "max_violation_uA": data_audit["max_monotonic_violation_uA"],
            "clipping_correction_uA": band_audit["total_clipping_correction_uA"],
        },
        "modal_metrics": {
            "reference_cumulative_explained_2": modal["cumulative_explained_2"],
            "reference_s2_over_s3": modal["s2_over_s3"],
            "reference_periods_V": modal["mode_periods_V"],
            "sensitivity_min_cumulative_explained_2": float(sensitivity["cumulative_explained_2"].min()),
            "sensitivity_min_s2_over_s3": float(sensitivity["s2_over_s3"].min()),
            "sensitivity_max_subspace_angle_deg": float(sensitivity["subspace_angle_deg"].max()),
        },
        "cv_metrics": {
            "rank2_better_blocks": int((relative > 0.0).sum()),
            "relative_improvement_by_block": {str(k): float(v) for k, v in relative.items()},
            "median_relative_improvement": float(relative.median()),
            "worst_relative_improvement": float(relative.min()),
            "mean_balanced_rmse_rank1": float(mean_error["rank1_P11.55"]),
            "mean_balanced_rmse_rank2": physical_mean,
            "mean_balanced_rmse_trend_degree5": float(mean_error["trend_degree5"]),
            "best_pseudo_period_mean_balanced_rmse": pseudo_best,
        },
        "claim_limit": "stable two-dimensional period-locked coarse retarding-response redistribution",
        "prohibited_interpretations": [
            "resolved atomic sublevels",
            "identified physical collision channels",
            "calibrated EEDF or energy spectrum",
            "independent confirmatory evidence",
        ],
    }


def run_numeric_analysis(data_path: Path) -> dict:
    _, pivot, data_audit = load_and_audit(data_path)
    va_all, fractions_all, bands, band_audit = make_retarding_bands(pivot, min_va=12.0)
    modal, sensitivity, scores, loadings = modal_analysis(va_all, fractions_all)
    cv_keep = va_all >= 15.0
    cv, predictions = blocked_cv(va_all[cv_keep], fractions_all[cv_keep])
    decision = make_decision(data_audit, band_audit, modal, sensitivity, cv)
    return {
        "data_audit": data_audit,
        "band_audit": band_audit,
        "bands": bands,
        "modal": modal,
        "sensitivity": sensitivity,
        "scores": scores,
        "loadings": loadings,
        "cv": cv,
        "predictions": predictions,
        "decision": decision,
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
