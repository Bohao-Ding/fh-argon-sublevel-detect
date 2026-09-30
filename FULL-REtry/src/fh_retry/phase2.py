from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import BLOCKS, PERIOD_PRIOR, _predict_fold, load_and_audit


NIST_4S_EV = np.array([11.54835442, 11.62359272, 11.72316039, 11.82807116])
SPAN_FACTORS = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0])


def _analysis_arrays(pivot: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    va_all = pivot.index.to_numpy(float)
    keep = va_all >= 15.0
    return va_all[keep], pivot[[0, 4, 6, 8, 10]].to_numpy(float)[keep]


def _trend(va: np.ndarray, degree: int = 3) -> np.ndarray:
    center = 0.5 * (va.min() + va.max())
    half_range = 0.5 * (va.max() - va.min())
    x = (va - center) / half_range
    return np.column_stack([x**power for power in range(degree + 1)])


def _block_masks(va: np.ndarray) -> list[np.ndarray]:
    return [(va >= low) & (va < high) for low, high in BLOCKS]


def retarding_order_audit(pivot: pd.DataFrame) -> pd.DataFrame:
    levels = [0, 4, 6, 8, 10]
    currents = pivot[levels].to_numpy(float).T
    rows = []
    for permutation in itertools.permutations(range(5)):
        ordered = currents[list(permutation)]
        violations = int(((ordered[1:] - ordered[:-1]) > 1e-12).sum())
        rows.append(
            {
                "order": "-".join(str(levels[index]) for index in permutation),
                "violations": violations,
                "is_physical_order": permutation == tuple(range(5)),
            }
        )
    return pd.DataFrame(rows).sort_values(["violations", "order"]).reset_index(drop=True)


def _retarding_fractions(pivot: pd.DataFrame, levels: list[int]) -> tuple[np.ndarray, np.ndarray, dict]:
    va_all = pivot.index.to_numpy(float)
    currents = pivot[levels].to_numpy(float).T
    raw = np.vstack([currents[index] - currents[index + 1] for index in range(len(levels) - 1)] + [currents[-1]])
    keep = (va_all >= 15.0) & (currents[0] > 0.1)
    raw = raw[:, keep]
    clipped = np.maximum(raw, 0.0)
    fractions = (clipped / clipped.sum(axis=0, keepdims=True)).T
    audit = {
        "levels_V": levels,
        "negative_values": int((raw < 0).sum()),
        "negative_mass_fraction": float(-np.minimum(raw, 0.0).sum() / currents[0, keep].sum()),
    }
    return va_all[keep], fractions, audit


def _rank_gain(va: np.ndarray, response: np.ndarray) -> tuple[pd.DataFrame, dict]:
    rows = []
    for block, (low, high) in enumerate(BLOCKS, start=1):
        test = (va >= low) & (va < high)
        train = ~test
        _, rank1_z, observed_z = _predict_fold(
            va, response, train, test, period=PERIOD_PRIOR, rank=1, trend_degree=3
        )
        _, rank2_z, _ = _predict_fold(
            va, response, train, test, period=PERIOD_PRIOR, rank=2, trend_degree=3
        )
        rank1_error = float(np.sqrt(np.mean((rank1_z - observed_z) ** 2)))
        rank2_error = float(np.sqrt(np.mean((rank2_z - observed_z) ** 2)))
        rows.append(
            {
                "block": block,
                "rank1_rmse": rank1_error,
                "rank2_rmse": rank2_error,
                "relative_gain": (rank1_error - rank2_error) / rank1_error,
            }
        )
    frame = pd.DataFrame(rows)
    summary = {
        "improved_blocks": int((frame["relative_gain"] > 0).sum()),
        "median_gain": float(frame["relative_gain"].median()),
        "worst_gain": float(frame["relative_gain"].min()),
    }
    return frame, summary


def endpoint_exclusion(pivot: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail = []
    summaries = []
    for levels in ([0, 4, 6], [0, 4, 6, 8], [0, 4, 6, 8, 10]):
        va, fractions, audit = _retarding_fractions(pivot, list(levels))
        rows, summary = _rank_gain(va, fractions)
        rows.insert(0, "levels", "-".join(map(str, levels)))
        detail.append(rows)
        summaries.append({"levels": "-".join(map(str, levels)), **audit, **summary})
    return pd.concat(detail, ignore_index=True), pd.DataFrame(summaries)


def _full_rank1_fit(va: np.ndarray, response: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = response.mean(axis=0)
    scale = response.std(axis=0, ddof=1)
    standardized = (response - mean) / scale
    trend = _trend(va)
    harmonic = np.column_stack(
        [np.sin(2.0 * np.pi * va / PERIOD_PRIOR), np.cos(2.0 * np.pi * va / PERIOD_PRIOR)]
    )
    trend_coef = np.linalg.lstsq(trend, standardized, rcond=None)[0]
    residual_y = standardized - trend @ trend_coef
    residual_h = harmonic - trend @ np.linalg.lstsq(trend, harmonic, rcond=None)[0]
    q, r = np.linalg.qr(residual_h, mode="reduced")
    coefficient = q.T @ residual_y
    u, singular, vt = np.linalg.svd(coefficient, full_matrices=False)
    rank1 = (u[:, :1] * singular[:1]) @ vt[:1]
    harmonic_coef = np.linalg.solve(r, rank1)
    trend_coef = np.linalg.lstsq(trend, standardized - harmonic @ harmonic_coef, rcond=None)[0]
    fitted = trend @ trend_coef + harmonic @ harmonic_coef
    return fitted, standardized - fitted


def _moving_block_sample(residual: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    count = residual.shape[0]
    pieces = []
    filled = 0
    while filled < count:
        start = int(rng.integers(0, count))
        index = np.arange(start, start + length) % count
        pieces.append(residual[index])
        filled += length
    return np.vstack(pieces)[:count]


def rank1_conditional_surrogates(
    pivot: pd.DataFrame,
    *,
    replicates: int = 2000,
    seed: int = 20260831,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    va, fractions, _ = _retarding_fractions(pivot, [0, 4, 6, 8, 10])
    fitted, residual = _full_rank1_fit(va, fractions)
    _, observed = _rank_gain(va, fractions)
    rng = np.random.default_rng(seed)
    rows = []
    summaries = []
    for length in [3, 5, 9, 23]:
        local = []
        for replicate in range(replicates):
            surrogate = fitted + _moving_block_sample(residual, length, rng)
            _, metrics = _rank_gain(va, surrogate)
            record = {"block_length": length, "replicate": replicate, **metrics}
            rows.append(record)
            local.append(record)
        local_frame = pd.DataFrame(local)
        exceed = (local_frame["median_gain"] >= observed["median_gain"]) & (
            local_frame["improved_blocks"] >= observed["improved_blocks"]
        )
        summaries.append(
            {
                "block_length": length,
                "replicates": replicates,
                "observed_median_gain": observed["median_gain"],
                "observed_improved_blocks": observed["improved_blocks"],
                "surrogate_median_gain_q95": float(local_frame["median_gain"].quantile(0.95)),
                "surrogate_median_gain_q99": float(local_frame["median_gain"].quantile(0.99)),
                "conditional_p": float((1 + exceed.sum()) / (replicates + 1)),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summaries), observed


def _harmonic_design(va: np.ndarray, energies: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [function(2.0 * np.pi * va / energy) for energy in energies for function in (np.sin, np.cos)]
    )


def _template_basis(va: np.ndarray, energies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    trend = _trend(va)
    harmonic = _harmonic_design(va, energies)
    residual = harmonic - trend @ np.linalg.lstsq(trend, harmonic, rcond=None)[0]
    q, _ = np.linalg.qr(residual, mode="reduced")
    singular = np.linalg.svd(residual, compute_uv=False)
    return q, singular


def _template_operators(va: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    design = np.column_stack([_trend(va), q])
    operators = []
    for test in _block_masks(va):
        train = ~test
        operator = design[test] @ np.linalg.pinv(design[train], rcond=1e-12)
        operators.append({"train": train, "test": test, "operator": operator})
    return design, operators


def _operator_cv(response: np.ndarray, operators: list[dict]) -> tuple[float, list[float]]:
    errors = []
    for item in operators:
        train = item["train"]
        test = item["test"]
        predicted = item["operator"] @ response[train]
        scale = response[train].std(axis=0, ddof=1)
        errors.append(float(np.sqrt(np.mean(((predicted - response[test]) / scale) ** 2))))
    return float(np.mean(errors)), errors


def _principal_angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    cosine = np.linalg.svd(left.T @ right, compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(cosine.min(), -1.0, 1.0))))


def _same_span_templates(count: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    center = float(NIST_4S_EV.mean())
    span = float(np.ptp(NIST_4S_EV))
    templates = []
    for _ in range(count):
        unit = np.r_[0.0, np.sort(rng.random(2)), 1.0]
        templates.append((unit - unit.mean()) * span + center)
    return templates


def _ridge_fold_prediction(
    va: np.ndarray,
    response: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    energies: np.ndarray,
    ridge: float,
) -> float:
    trend = _trend(va)
    harmonic = _harmonic_design(va, energies)
    mean = response[train].mean(axis=0)
    scale = response[train].std(axis=0, ddof=1)
    standardized = (response - mean) / scale
    trend_train = trend[train]
    harmonic_train = harmonic[train]
    first_trend = np.linalg.lstsq(trend_train, standardized[train], rcond=None)[0]
    residual_h = harmonic_train - trend_train @ np.linalg.lstsq(
        trend_train, harmonic_train, rcond=None
    )[0]
    residual_y = standardized[train] - trend_train @ first_trend
    coefficient = np.linalg.solve(
        residual_h.T @ residual_h + ridge * np.eye(harmonic.shape[1]),
        residual_h.T @ residual_y,
    )
    trend_coef = np.linalg.lstsq(
        trend_train, standardized[train] - harmonic_train @ coefficient, rcond=None
    )[0]
    predicted = trend[test] @ trend_coef + harmonic[test] @ coefficient
    return float(np.sqrt(np.mean((predicted - standardized[test]) ** 2)))


def _nested_ridge_cv(
    va: np.ndarray,
    response: np.ndarray,
    energies: np.ndarray,
) -> tuple[float, list[float], list[float]]:
    masks = _block_masks(va)
    ridge_grid = np.logspace(-6, 3, 13)
    outer_errors = []
    selected = []
    for outer_index, outer in enumerate(masks):
        validation_error = []
        for ridge in ridge_grid:
            inner_errors = []
            for inner_index, inner in enumerate(masks):
                if inner_index == outer_index:
                    continue
                train = ~(outer | inner)
                inner_errors.append(_ridge_fold_prediction(va, response, train, inner, energies, ridge))
            validation_error.append(float(np.mean(inner_errors)))
        ridge = float(ridge_grid[int(np.argmin(validation_error))])
        selected.append(ridge)
        outer_errors.append(_ridge_fold_prediction(va, response, ~outer, outer, energies, ridge))
    return float(np.mean(outer_errors)), outer_errors, selected


def template_specificity(
    pivot: pd.DataFrame,
    *,
    pseudo_count: int = 512,
    seed: int = 20260831,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    va, response = _analysis_arrays(pivot)
    mean_energy = float(NIST_4S_EV.mean())
    span = float(np.ptp(NIST_4S_EV))
    nist_q, nist_singular = _template_basis(va, NIST_4S_EV)

    controls = [("H1_mean", np.array([mean_energy])), ("NIST_4s", NIST_4S_EV)]
    equal = np.linspace(mean_energy - span / 2, mean_energy + span / 2, 4)
    controls.append(("equal_spacing", equal))
    for factor in SPAN_FACTORS:
        controls.append((f"span_x{factor:g}", mean_energy + factor * (NIST_4S_EV - mean_energy)))

    control_rows = []
    span_rows = []
    operator_cache = {}
    for name, energies in controls:
        q, singular = _template_basis(va, np.asarray(energies))
        _, operators = _template_operators(va, q)
        operator_cache[name] = (q, operators)
        mean_error, block_errors = _operator_cv(response, operators)
        for block, error in enumerate(block_errors, start=1):
            control_rows.append(
                {
                    "model": name,
                    "block": block,
                    "balanced_rmse": error,
                    "mean_balanced_rmse": mean_error,
                    "parameter_count_per_curve": 4 + 2 * len(energies),
                }
            )
        if name.startswith("span_x"):
            factor = float(name.split("x", 1)[1])
            span_rows.append(
                {
                    "factor": factor,
                    "span_eV": factor * span,
                    "mean_balanced_rmse": mean_error,
                    "subspace_angle_from_NIST_deg": _principal_angle_deg(nist_q, q),
                    "raw_design_condition_number": float(singular[0] / singular[-1]),
                }
            )

    pseudo_rows = []
    pseudo_templates = _same_span_templates(pseudo_count, seed)
    for index, energies in enumerate(pseudo_templates):
        q, singular = _template_basis(va, energies)
        _, operators = _template_operators(va, q)
        mean_error, _ = _operator_cv(response, operators)
        pseudo_rows.append(
            {
                "template": index,
                "energy_1_eV": energies[0],
                "energy_2_eV": energies[1],
                "energy_3_eV": energies[2],
                "energy_4_eV": energies[3],
                "mean_balanced_rmse": mean_error,
                "subspace_angle_from_NIST_deg": _principal_angle_deg(nist_q, q),
                "raw_design_condition_number": float(singular[0] / singular[-1]),
            }
        )
    pseudo = pd.DataFrame(pseudo_rows)
    controls_frame = pd.DataFrame(control_rows)
    nist_error = float(
        controls_frame.loc[controls_frame["model"] == "NIST_4s", "mean_balanced_rmse"].iloc[0]
    )

    ridge_templates = [("NIST_4s", NIST_4S_EV), ("equal_spacing", equal)] + [
        (f"pseudo_{index}", energies) for index, energies in enumerate(pseudo_templates[:64])
    ]
    ridge_rows = []
    for name, energies in ridge_templates:
        mean_error, block_errors, selected = _nested_ridge_cv(va, response, energies)
        ridge_rows.append(
            {
                "model": name,
                "mean_balanced_rmse": mean_error,
                "block_errors": ";".join(f"{value:.9g}" for value in block_errors),
                "selected_ridge": ";".join(f"{value:.9g}" for value in selected),
            }
        )
    ridge = pd.DataFrame(ridge_rows)
    ridge_nist = float(ridge.loc[ridge["model"] == "NIST_4s", "mean_balanced_rmse"].iloc[0])
    ridge_pseudo = ridge[ridge["model"].str.startswith("pseudo_")]["mean_balanced_rmse"]
    summary = {
        "nist_span_eV": span,
        "nist_raw_design_condition_number": float(nist_singular[0] / nist_singular[-1]),
        "nist_mean_balanced_rmse": nist_error,
        "nist_pseudo_percentile": float((pseudo["mean_balanced_rmse"] <= nist_error).mean()),
        "pseudo_error_min": float(pseudo["mean_balanced_rmse"].min()),
        "pseudo_error_median": float(pseudo["mean_balanced_rmse"].median()),
        "pseudo_error_max": float(pseudo["mean_balanced_rmse"].max()),
        "pseudo_angle_max_deg": float(pseudo["subspace_angle_from_NIST_deg"].max()),
        "ridge_nist_mean_balanced_rmse": ridge_nist,
        "ridge_nist_pseudo_percentile": float((ridge_pseudo <= ridge_nist).mean()),
    }
    cache = {"va": va, "response": response, "operator_cache": operator_cache}
    return controls_frame, pseudo, pd.DataFrame(span_rows), ridge, summary, cache


def _phase_randomized_noise(residual: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    spectrum = np.fft.rfft(residual, axis=0)
    phase = rng.uniform(0.0, 2.0 * np.pi, spectrum.shape[0])
    phase[0] = 0.0
    if residual.shape[0] % 2 == 0:
        phase[-1] = 0.0
    randomized = spectrum * np.exp(1j * phase)[:, None]
    return np.fft.irfft(randomized, n=residual.shape[0], axis=0)


def template_injection_recovery(
    cache: dict,
    *,
    replicates: int = 500,
    seed: int = 20260901,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    va = cache["va"]
    raw_response = cache["response"]
    response = (raw_response - raw_response.mean(axis=0)) / raw_response.std(axis=0, ddof=1)
    mean_energy = float(NIST_4S_EV.mean())
    span = float(np.ptp(NIST_4S_EV))
    candidates = []
    for factor in SPAN_FACTORS:
        name = f"span_x{factor:g}"
        q, operators = cache["operator_cache"][name]
        design = np.column_stack([_trend(va), q])
        fitted = design @ np.linalg.lstsq(design, response, rcond=None)[0]
        candidates.append(
            {"factor": float(factor), "span_eV": float(factor * span), "q": q, "operators": operators, "fitted": fitted}
        )

    common_residual = response - candidates[-1]["fitted"]
    rng = np.random.default_rng(seed)
    rows = []
    schemes = [("self_block_3", 3), ("self_block_5", 5), ("self_block_9", 9), ("common_phase", None)]
    for scheme, block_length in schemes:
        for true_index, true in enumerate(candidates):
            residual = response - true["fitted"]
            counts = np.zeros(len(candidates), dtype=int)
            for _ in range(replicates):
                if block_length is None:
                    noise = _phase_randomized_noise(common_residual, rng)
                else:
                    noise = _moving_block_sample(residual, block_length, rng)
                simulated = true["fitted"] + noise
                scores = [_operator_cv(simulated, candidate["operators"])[0] for candidate in candidates]
                counts[int(np.argmin(scores))] += 1
            for selected_index, count in enumerate(counts):
                selected = candidates[selected_index]
                rows.append(
                    {
                        "noise_scheme": scheme,
                        "true_factor": true["factor"],
                        "true_span_eV": true["span_eV"],
                        "selected_factor": selected["factor"],
                        "selected_span_eV": selected["span_eV"],
                        "count": int(count),
                        "rate": float(count / replicates),
                        "is_correct": selected_index == true_index,
                    }
                )
    detail = pd.DataFrame(rows)
    recovery = (
        detail[detail["is_correct"]]
        .loc[:, ["noise_scheme", "true_factor", "true_span_eV", "rate"]]
        .rename(columns={"rate": "recovery_rate"})
        .reset_index(drop=True)
    )
    return detail, recovery


def gain_offset_stress(
    pivot: pd.DataFrame,
    *,
    replicates: int = 512,
    seed: int = 20260902,
) -> pd.DataFrame:
    va_all = pivot.index.to_numpy(float)
    keep = va_all >= 15.0
    va = va_all[keep]
    base = pivot[[0, 4, 6, 8, 10]].to_numpy(float).T[:, keep]
    rng = np.random.default_rng(seed)
    rows = []
    for half_range in [0.0, 0.005, 0.01, 0.02, 0.05]:
        local = []
        count = 1 if half_range == 0 else replicates
        for _ in range(count):
            if half_range == 0:
                perturbed = base.copy()
            else:
                gain = 1.0 + rng.uniform(-half_range, half_range, 5)
                offset = rng.uniform(-0.001, 0.001, 5)
                perturbed = base * gain[:, None] + offset[:, None]
            raw = np.vstack([perturbed[index] - perturbed[index + 1] for index in range(4)] + [perturbed[-1]])
            negative_mass = float(-np.minimum(raw, 0.0).sum() / perturbed[0].sum())
            clipped = np.maximum(raw, 0.0)
            fractions = (clipped / clipped.sum(axis=0, keepdims=True)).T
            _, metrics = _rank_gain(va, fractions)
            violation_count = int(((perturbed[1:] - perturbed[:-1]) > 0.0010001).sum())
            passed = (
                metrics["improved_blocks"] >= 4
                and metrics["median_gain"] >= 0.10
                and metrics["worst_gain"] >= -0.05
            )
            local.append(
                {
                    "gain_half_range_pct": 100.0 * half_range,
                    "offset_half_range_uA": 0.0 if half_range == 0 else 0.001,
                    "median_gain": metrics["median_gain"],
                    "improved_blocks": metrics["improved_blocks"],
                    "worst_gain": metrics["worst_gain"],
                    "monotonic_violations_over_quantization": violation_count,
                    "negative_mass_fraction": negative_mass,
                    "phase_screen_passed": passed,
                }
            )
        frame = pd.DataFrame(local)
        rows.append(
            {
                "gain_half_range_pct": 100.0 * half_range,
                "offset_half_range_uA": 0.0 if half_range == 0 else 0.001,
                "replicates": count,
                "phase_screen_pass_rate": float(frame["phase_screen_passed"].mean()),
                "median_gain_q05": float(frame["median_gain"].quantile(0.05)),
                "median_gain_q50": float(frame["median_gain"].median()),
                "median_gain_q95": float(frame["median_gain"].quantile(0.95)),
                "monotonic_violations_q95": float(frame["monotonic_violations_over_quantization"].quantile(0.95)),
                "negative_mass_fraction_q95": float(frame["negative_mass_fraction"].quantile(0.95)),
            }
        )
    return pd.DataFrame(rows)


def make_phase2_decision(
    order: pd.DataFrame,
    surrogate_summary: pd.DataFrame,
    endpoint_summary: pd.DataFrame,
    template_summary: dict,
    recovery: pd.DataFrame,
    stress: pd.DataFrame,
) -> dict:
    physical = order[order["is_physical_order"]].iloc[0]
    next_best = int(order.loc[~order["is_physical_order"], "violations"].min())
    without_vr10 = endpoint_summary[endpoint_summary["levels"] == "0-4-6-8"].iloc[0]
    nist_recovery = recovery[recovery["true_factor"] == 1.0]["recovery_rate"]
    stress_2pct = stress[stress["gain_half_range_pct"] == 2.0].iloc[0]
    gates = {
        "G4_unique_retarding_order": bool(physical["violations"] <= 1 and next_best >= 100),
        "G5_rank2_exceeds_conditional_null": bool((surrogate_summary["conditional_p"] <= 0.01).all()),
        "G6_not_driven_by_Vr10": bool(
            without_vr10["improved_blocks"] >= 4 and without_vr10["median_gain"] >= 0.10
        ),
        "G7_NIST_spacing_not_specific": bool(
            template_summary["nist_pseudo_percentile"] > 0.05
            and template_summary["ridge_nist_pseudo_percentile"] > 0.05
        ),
        "G8_0p28eV_not_reliably_recovered": bool((nist_recovery < 0.80).all()),
        "G9_phase_result_survives_2pct_gain_stress": bool(stress_2pct["phase_screen_pass_rate"] >= 0.95),
    }
    return {
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "recommended_claim": "robust coarse multichannel retarding-response structure with an explicit sublevel-identifiability boundary",
        "physical_order_violations": int(physical["violations"]),
        "next_best_order_violations": next_best,
        "rank2_conditional_p_max": float(surrogate_summary["conditional_p"].max()),
        "without_Vr10_improved_blocks": int(without_vr10["improved_blocks"]),
        "without_Vr10_median_gain": float(without_vr10["median_gain"]),
        "nist_pseudo_percentile": template_summary["nist_pseudo_percentile"],
        "ridge_nist_pseudo_percentile": template_summary["ridge_nist_pseudo_percentile"],
        "nist_raw_design_condition_number": template_summary["nist_raw_design_condition_number"],
        "nist_recovery_rate_min": float(nist_recovery.min()),
        "nist_recovery_rate_max": float(nist_recovery.max()),
        "gain_2pct_phase_screen_pass_rate": float(stress_2pct["phase_screen_pass_rate"]),
        "claim_boundary": [
            "modes are mathematical response modes, not identified atomic channels",
            "NIST 4s internal spacing is not specifically selected by these data",
            "injection recovery is conditional on the linear harmonic template and empirical residuals",
            "gain stress is not a substitute for measured calibration uncertainty",
            "no independent apparatus or repeat-scan confirmation is available",
        ],
    }


def run_phase2(
    data_path: Path,
    *,
    surrogate_reps: int = 2000,
    pseudo_count: int = 512,
    recovery_reps: int = 500,
    stress_reps: int = 512,
) -> dict:
    _, pivot, _ = load_and_audit(data_path)
    timings = {}

    started = time.perf_counter()
    order = retarding_order_audit(pivot)
    timings["retarding_order_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    endpoint_detail, endpoint_summary = endpoint_exclusion(pivot)
    timings["endpoint_exclusion_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    surrogate, surrogate_summary, observed_rank = rank1_conditional_surrogates(
        pivot, replicates=surrogate_reps
    )
    timings["rank1_surrogate_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    template_cv, pseudo, span_profile, ridge, template_summary, cache = template_specificity(
        pivot, pseudo_count=pseudo_count
    )
    timings["template_specificity_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    recovery_detail, recovery = template_injection_recovery(cache, replicates=recovery_reps)
    timings["injection_recovery_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    stress = gain_offset_stress(pivot, replicates=stress_reps)
    timings["gain_stress_seconds"] = time.perf_counter() - started
    timings["total_seconds"] = float(sum(timings.values()))

    decision = make_phase2_decision(
        order, surrogate_summary, endpoint_summary, template_summary, recovery, stress
    )
    return {
        "order": order,
        "endpoint_detail": endpoint_detail,
        "endpoint_summary": endpoint_summary,
        "surrogate": surrogate,
        "surrogate_summary": surrogate_summary,
        "observed_rank": observed_rank,
        "template_cv": template_cv,
        "pseudo_templates": pseudo,
        "span_profile": span_profile,
        "ridge_sensitivity": ridge,
        "template_summary": template_summary,
        "recovery_detail": recovery_detail,
        "recovery": recovery,
        "gain_stress": stress,
        "timings": timings,
        "decision": decision,
    }
