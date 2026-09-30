from __future__ import annotations

import hashlib
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import helmert

from .analysis import BLOCKS, PERIOD_PRIOR, PSEUDO_PERIODS, _predict_fold, load_and_audit
from .phase2 import (
    NIST_4S_EV,
    _analysis_arrays,
    _full_rank1_fit,
    _retarding_fractions,
    _template_basis,
    _template_operators,
    _trend,
    endpoint_exclusion,
)


REPRESENTATIONS = ("raw_fraction", "hellinger", "clr", "ilr")
GEOMETRY_REPRESENTATIONS = ("hellinger", "clr", "ilr")
SURROGATE_BLOCK_LENGTHS = (3, 5, 9, 23)
BLOCK_SHIFTS = (-0.50, -0.25, 0.0)
PERIOD_GRID = np.round(np.arange(9.50, 13.50 + 0.025, 0.05), 2)
INJECTION_FACTORS = np.array([0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0])
INJECTION_SCHEMES = (("common_h1_block_3", 3), ("common_h1_block_5", 5), ("common_h1_block_9", 9), ("common_h1_phase", None))


def resolve_workers(requested: int | None) -> int:
    logical = os.cpu_count() or 1
    if requested is None or requested <= 0:
        return max(1, min(8, logical // 2))
    return max(1, min(int(requested), logical, 16))


def _stable_seed(base: int, key: str) -> int:
    digest = hashlib.sha256(f"{base}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def _multiplicative_zero_replacement(
    fractions: np.ndarray, row_delta: np.ndarray
) -> tuple[np.ndarray, dict]:
    replaced = np.asarray(fractions, float).copy()
    zero = replaced <= 0.0
    applied = np.zeros(replaced.shape[0], dtype=float)
    for row in np.flatnonzero(zero.any(axis=1)):
        mask = zero[row]
        count = int(mask.sum())
        delta = min(float(row_delta[row]), 0.25 / count)
        remaining = 1.0 - count * delta
        positive_sum = float(replaced[row, ~mask].sum())
        replaced[row, mask] = delta
        replaced[row, ~mask] *= remaining / positive_sum
        applied[row] = delta
    audit = {
        "zero_values_before": int(zero.sum()),
        "zero_rows_before": int(zero.any(axis=1).sum()),
        "max_applied_replacement_fraction": float(applied.max(initial=0.0)),
        "min_positive_after": float(replaced.min()),
        "closure_max_abs_error": float(np.max(np.abs(replaced.sum(axis=1) - 1.0))),
    }
    return replaced, audit


def composition_representations(
    pivot: pd.DataFrame, quantization_uA: float
) -> tuple[np.ndarray, dict[str, np.ndarray], pd.DataFrame]:
    va, fractions, band_audit = _retarding_fractions(pivot, [0, 4, 6, 8, 10])
    i0 = pivot.loc[va, 0].to_numpy(float)
    row_delta = 0.5 * quantization_uA / i0
    positive, replacement_audit = _multiplicative_zero_replacement(fractions, row_delta)
    log_positive = np.log(positive)
    transforms = {
        "raw_fraction": fractions,
        "hellinger": np.sqrt(fractions),
        "clr": log_positive - log_positive.mean(axis=1, keepdims=True),
        "ilr": log_positive @ helmert(fractions.shape[1], full=False).T,
    }
    rows = []
    for name, response in transforms.items():
        rows.append(
            {
                "representation": name,
                "rows": response.shape[0],
                "dimensions": response.shape[1],
                "finite": bool(np.isfinite(response).all()),
                "min_value": float(response.min()),
                "max_value": float(response.max()),
                "zero_values_before_replacement": replacement_audit["zero_values_before"],
                "zero_rows_before_replacement": replacement_audit["zero_rows_before"],
                "max_applied_replacement_fraction": replacement_audit[
                    "max_applied_replacement_fraction"
                ],
                "replacement_closure_max_abs_error": replacement_audit[
                    "closure_max_abs_error"
                ],
                "raw_negative_mass_fraction": band_audit["negative_mass_fraction"],
            }
        )
    return va, transforms, pd.DataFrame(rows)


def shifted_blocks(shift_fraction: float) -> list[tuple[float, float]]:
    shift = shift_fraction * PERIOD_PRIOR
    return [(low + shift, high + shift) for low, high in BLOCKS]


def _rank_detail(
    va: np.ndarray,
    response: np.ndarray,
    *,
    period: float = PERIOD_PRIOR,
    blocks: list[tuple[float, float]] | None = None,
) -> pd.DataFrame:
    rows = []
    for block, (low, high) in enumerate(blocks or BLOCKS, start=1):
        test = (va >= low) & (va < high)
        train = ~test
        _, rank1_z, observed_z = _predict_fold(
            va, response, train, test, period=period, rank=1, trend_degree=3
        )
        _, rank2_z, _ = _predict_fold(
            va, response, train, test, period=period, rank=2, trend_degree=3
        )
        rank1_error = float(np.sqrt(np.mean((rank1_z - observed_z) ** 2)))
        rank2_error = float(np.sqrt(np.mean((rank2_z - observed_z) ** 2)))
        rows.append(
            {
                "block": block,
                "low_V": low,
                "high_V_exclusive": high,
                "test_points": int(test.sum()),
                "rank1_rmse": rank1_error,
                "rank2_rmse": rank2_error,
                "relative_gain": (rank1_error - rank2_error) / rank1_error,
            }
        )
    return pd.DataFrame(rows)


def _rank_summary(detail: pd.DataFrame) -> dict:
    return {
        "improved_blocks": int((detail["relative_gain"] > 0.0).sum()),
        "median_gain": float(detail["relative_gain"].median()),
        "worst_gain": float(detail["relative_gain"].min()),
        "mean_rank1_rmse": float(detail["rank1_rmse"].mean()),
        "mean_rank2_rmse": float(detail["rank2_rmse"].mean()),
    }


def _moving_block_batch(
    residual: np.ndarray, length: int, replicates: int, rng: np.random.Generator
) -> np.ndarray:
    count = residual.shape[0]
    block_count = int(np.ceil(count / length))
    starts = rng.integers(0, count, size=(replicates, block_count))
    indices = (starts[:, :, None] + np.arange(length)[None, None, :]) % count
    indices = indices.reshape(replicates, -1)[:, :count]
    return residual[indices]


def _surrogate_worker(task: dict) -> dict:
    started = time.perf_counter()
    rng = np.random.default_rng(task["seed"])
    rows = []
    for replicate in range(task["replicates"]):
        count = task["residual"].shape[0]
        length = task["block_length"]
        block_count = int(np.ceil(count / length))
        starts = rng.integers(0, count, size=block_count)
        index = (starts[:, None] + np.arange(length)[None, :]) % count
        noise = task["residual"][index.reshape(-1)[:count]]
        summary = _rank_summary(_rank_detail(task["va"], task["fitted"] + noise))
        rows.append(
            {
                "representation": task["representation"],
                "block_length": length,
                "replicate": replicate,
                "improved_blocks": summary["improved_blocks"],
                "median_gain": summary["median_gain"],
                "worst_gain": summary["worst_gain"],
            }
        )
    frame = pd.DataFrame(rows)
    observed = task["observed"]
    exceed = (frame["median_gain"] >= observed["median_gain"]) & (
        frame["improved_blocks"] >= observed["improved_blocks"]
    )
    summary = {
        "representation": task["representation"],
        "block_length": task["block_length"],
        "replicates": task["replicates"],
        "observed_improved_blocks": observed["improved_blocks"],
        "observed_median_gain": observed["median_gain"],
        "observed_worst_gain": observed["worst_gain"],
        "surrogate_median_gain_q95": float(frame["median_gain"].quantile(0.95)),
        "surrogate_median_gain_q99": float(frame["median_gain"].quantile(0.99)),
        "conditional_p": float((1 + exceed.sum()) / (task["replicates"] + 1)),
        "task_seconds": time.perf_counter() - started,
    }
    return {"samples": frame, "summary": summary}


def _model_error(
    va: np.ndarray,
    response: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    *,
    period: float | None,
    rank: int | None,
    trend_degree: int,
) -> float:
    _, predicted, observed = _predict_fold(
        va,
        response,
        train,
        test,
        period=period,
        rank=rank,
        trend_degree=trend_degree,
    )
    return float(np.sqrt(np.mean((predicted - observed) ** 2)))


def period_and_block_sensitivity(
    va: np.ndarray, transforms: dict[str, np.ndarray]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    profile_rows = []
    shift_detail = []
    shift_summary = []
    adequacy_rows = []
    for representation, response in transforms.items():
        for period in PERIOD_GRID:
            detail = _rank_detail(va, response, period=float(period))
            profile_rows.append(
                {"representation": representation, "period_V": period, **_rank_summary(detail)}
            )
        for shift in BLOCK_SHIFTS:
            detail = _rank_detail(va, response, blocks=shifted_blocks(shift))
            detail.insert(0, "block_shift_fraction", shift)
            detail.insert(0, "representation", representation)
            shift_detail.append(detail)
            shift_summary.append(
                {
                    "representation": representation,
                    "block_shift_fraction": shift,
                    **_rank_summary(detail),
                }
            )
        for block, (low, high) in enumerate(BLOCKS, start=1):
            test = (va >= low) & (va < high)
            train = ~test
            rank1 = _model_error(
                va, response, train, test, period=PERIOD_PRIOR, rank=1, trend_degree=3
            )
            rank2 = _model_error(
                va, response, train, test, period=PERIOD_PRIOR, rank=2, trend_degree=3
            )
            trend5 = _model_error(
                va, response, train, test, period=None, rank=None, trend_degree=5
            )
            pseudo = [
                _model_error(va, response, train, test, period=p, rank=2, trend_degree=3)
                for p in PSEUDO_PERIODS
            ]
            adequacy_rows.append(
                {
                    "representation": representation,
                    "block": block,
                    "low_V": low,
                    "high_V_exclusive": high,
                    "rank1_rmse": rank1,
                    "rank2_rmse": rank2,
                    "trend_degree5_rmse": trend5,
                    "best_pseudo_period_rmse": min(pseudo),
                    "best_pseudo_period_V": PSEUDO_PERIODS[int(np.argmin(pseudo))],
                    "rank2_better_than_trend": rank2 < trend5,
                    "rank2_better_than_best_pseudo": rank2 < min(pseudo),
                    "rank2_better_than_both": rank2 < min(trend5, min(pseudo)),
                }
            )
    return (
        pd.DataFrame(profile_rows),
        pd.concat(shift_detail, ignore_index=True),
        pd.DataFrame(shift_summary),
        pd.DataFrame(adequacy_rows),
    )


def endpoint_gate_consistency(pivot: pd.DataFrame) -> pd.DataFrame:
    _, summary = endpoint_exclusion(pivot)
    frame = summary.copy()
    frame["passes_improved_blocks"] = frame["improved_blocks"] >= 4
    frame["passes_median_gain"] = frame["median_gain"] >= 0.10
    frame["passes_worst_gain"] = frame["worst_gain"] >= -0.05
    frame["passes_original_G2_full_rule"] = frame[
        ["passes_improved_blocks", "passes_median_gain", "passes_worst_gain"]
    ].all(axis=1)
    return frame


def _injection_setup(pivot: pd.DataFrame) -> dict:
    va, raw_response = _analysis_arrays(pivot)
    response = (raw_response - raw_response.mean(axis=0)) / raw_response.std(axis=0, ddof=1)
    trend = _trend(va)
    trend_fit = trend @ np.linalg.lstsq(trend, response, rcond=None)[0]
    detrended = response - trend_fit
    center = float(NIST_4S_EV.mean())
    span = float(np.ptp(NIST_4S_EV))

    candidates = []
    raw_signals = []
    for factor in INJECTION_FACTORS:
        energies = center + factor * (NIST_4S_EV - center)
        q, _ = _template_basis(va, energies)
        signal = q @ (q.T @ detrended)
        _, operators = _template_operators(va, q)
        candidates.append(
            {
                "factor": float(factor),
                "span_eV": float(factor * span),
                "operators": operators,
            }
        )
        raw_signals.append(signal)

    nist_index = int(np.flatnonzero(np.isclose(INJECTION_FACTORS, 1.0))[0])
    target_rms = float(np.sqrt(np.mean(raw_signals[nist_index] ** 2)))
    signals = []
    for signal in raw_signals:
        current_rms = float(np.sqrt(np.mean(signal**2)))
        signals.append(signal * (target_rms / current_rms))

    h1_q, _ = _template_basis(va, np.array([center]))
    h1_signal = h1_q @ (h1_q.T @ detrended)
    common_residual = response - trend_fit - h1_signal
    return {
        "va": va,
        "trend_fit": trend_fit,
        "signals": signals,
        "common_residual": common_residual,
        "candidates": candidates,
        "target_signal_rms": target_rms,
    }


def _phase_noise_batch(
    residual: np.ndarray, replicates: int, rng: np.random.Generator
) -> np.ndarray:
    spectrum = np.fft.rfft(residual, axis=0)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(replicates, spectrum.shape[0]))
    phases[:, 0] = 0.0
    if residual.shape[0] % 2 == 0:
        phases[:, -1] = 0.0
    randomized = spectrum[None, :, :] * np.exp(1j * phases[:, :, None])
    return np.fft.irfft(randomized, n=residual.shape[0], axis=1)


def _batch_candidate_scores(simulated: np.ndarray, candidates: list[dict]) -> np.ndarray:
    replicates = simulated.shape[0]
    scores = np.zeros((replicates, len(candidates)), dtype=float)
    first_operators = candidates[0]["operators"]
    cache = []
    for item in first_operators:
        train = item["train"]
        test = item["test"]
        scale = simulated[:, train, :].std(axis=1, ddof=1)
        scale = np.where(scale > 1e-10, scale, 1.0)
        cache.append((train, test, scale))
    for candidate_index, candidate in enumerate(candidates):
        block_errors = []
        for item, (train, test, scale) in zip(candidate["operators"], cache, strict=True):
            predicted = np.einsum(
                "ij,rjk->rik", item["operator"], simulated[:, train, :], optimize=True
            )
            standardized = (predicted - simulated[:, test, :]) / scale[:, None, :]
            block_errors.append(np.sqrt(np.mean(standardized**2, axis=(1, 2))))
        scores[:, candidate_index] = np.mean(np.column_stack(block_errors), axis=1)
    return scores


def _injection_worker(task: dict) -> dict:
    started = time.perf_counter()
    rng = np.random.default_rng(task["seed"])
    if task["block_length"] is None:
        noise = _phase_noise_batch(task["common_residual"], task["replicates"], rng)
    else:
        noise = _moving_block_batch(
            task["common_residual"], task["block_length"], task["replicates"], rng
        )
    simulated = task["trend_fit"][None, :, :] + task["signal"][None, :, :] + noise
    scores = _batch_candidate_scores(simulated, task["candidates"])
    selected = np.argmin(scores, axis=1)
    true_index = task["true_index"]
    factors = np.array([item["factor"] for item in task["candidates"]])
    selected_factors = factors[selected]
    rows = []
    for replicate, selected_index in enumerate(selected):
        rows.append(
            {
                "noise_scheme": task["scheme"],
                "true_factor": factors[true_index],
                "true_span_eV": task["candidates"][true_index]["span_eV"],
                "replicate": replicate,
                "selected_factor": factors[selected_index],
                "selected_span_eV": task["candidates"][selected_index]["span_eV"],
                "selected_score": scores[replicate, selected_index],
                "true_score": scores[replicate, true_index],
                "is_exact": selected_index == true_index,
                "within_one_grid_step": abs(int(selected_index) - true_index) <= 1,
            }
        )
    frame = pd.DataFrame(rows)
    return {
        "samples": frame,
        "summary": {
            "noise_scheme": task["scheme"],
            "true_factor": factors[true_index],
            "true_span_eV": task["candidates"][true_index]["span_eV"],
            "replicates": task["replicates"],
            "exact_recovery_rate": float(frame["is_exact"].mean()),
            "within_one_grid_step_rate": float(frame["within_one_grid_step"].mean()),
            "median_selected_factor": float(np.median(selected_factors)),
            "median_abs_log_factor_error": float(
                np.median(np.abs(np.log(selected_factors / factors[true_index])))
            ),
            "task_seconds": time.perf_counter() - started,
        },
    }


def _run_tasks(function, tasks: list[dict], workers: int) -> list[dict]:
    if workers == 1:
        return [function(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(function, tasks, chunksize=1))


def _make_decision(
    composition_summary: pd.DataFrame,
    shift_summary: pd.DataFrame,
    profile: pd.DataFrame,
    adequacy: pd.DataFrame,
    endpoint: pd.DataFrame,
    injection_summary: pd.DataFrame,
) -> dict:
    geometry = composition_summary[
        composition_summary["representation"].isin(GEOMETRY_REPRESENTATIONS)
    ]
    composition_pass = (
        (geometry["improved_blocks"] >= 4)
        & (geometry["median_gain"] >= 0.10)
        & (geometry["worst_gain"] >= -0.05)
    ).all()

    shifted = shift_summary[shift_summary["representation"].isin(GEOMETRY_REPRESENTATIONS)]
    shifted_pass = (
        (shifted["improved_blocks"] >= 4) & (shifted["median_gain"] >= 0.10)
    ).all()

    period_ratios = {}
    period_passes = []
    for representation in GEOMETRY_REPRESENTATIONS:
        local = profile[profile["representation"] == representation]
        prior = float(local.loc[np.isclose(local["period_V"], PERIOD_PRIOR), "mean_rank2_rmse"].iloc[0])
        optimum = float(local["mean_rank2_rmse"].min())
        period_ratios[representation] = prior / optimum
        period_passes.append(prior <= 1.10 * optimum)

    adequacy_counts = (
        adequacy[adequacy["representation"].isin(GEOMETRY_REPRESENTATIONS)]
        .groupby("representation")["rank2_better_than_both"]
        .sum()
        .astype(int)
        .to_dict()
    )
    adequacy_pass = all(count >= 4 for count in adequacy_counts.values())

    without_vr10 = endpoint[endpoint["levels"] == "0-4-6-8"].iloc[0]
    endpoint_pass = bool(without_vr10["passes_original_G2_full_rule"])

    nist = injection_summary[np.isclose(injection_summary["true_factor"], 1.0)]
    injection_pass = bool((nist["exact_recovery_rate"] < 0.80).all())

    gates = {
        "G10_composition_robustness": bool(composition_pass),
        "G11_shifted_block_robustness": bool(shifted_pass),
        "G12_prior_period_near_profile_optimum": bool(all(period_passes)),
        "G13_absolute_block_adequacy": bool(adequacy_pass),
        "G14_endpoint_full_rule": endpoint_pass,
        "G15_nist_span_not_recoverable": injection_pass,
    }
    return {
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "recommended_claim": "composition-audited nonseparable retarding-response structure with explicit local failures and conditional NIST-spacing non-recoverability",
        "composition_summary": geometry.to_dict(orient="records"),
        "prior_period_error_ratio": period_ratios,
        "absolute_adequacy_blocks": adequacy_counts,
        "without_Vr10": {
            "improved_blocks": int(without_vr10["improved_blocks"]),
            "median_gain": float(without_vr10["median_gain"]),
            "worst_gain": float(without_vr10["worst_gain"]),
            "passes_full_rule": endpoint_pass,
        },
        "nist_exact_recovery_by_scheme": {
            row["noise_scheme"]: float(row["exact_recovery_rate"])
            for _, row in nist.iterrows()
        },
        "claim_boundary": [
            "rank-2 means nonseparable sine/cosine response coefficients, not two physical channels",
            "all gates are internal development checks on one archived dataset",
            "endpoint robustness fails if the original worst-block rule fails",
            "injection recovery is conditional on the candidate grid and candidate-independent empirical residual",
            "no apparatus calibration, repeat scan, or independent confirmation is supplied",
        ],
    }


def run_phase3(
    data_path: Path,
    *,
    workers: int | None = None,
    surrogate_replicates: int = 2000,
    injection_replicates: int = 500,
) -> dict:
    started = time.perf_counter()
    worker_count = resolve_workers(workers)
    _, pivot, data_audit = load_and_audit(data_path)
    va, transforms, transform_audit = composition_representations(
        pivot, data_audit["current_quantization_uA"]
    )

    composition_detail = []
    composition_summary_rows = []
    surrogate_tasks = []
    for representation, response in transforms.items():
        detail = _rank_detail(va, response)
        detail.insert(0, "representation", representation)
        composition_detail.append(detail)
        observed = _rank_summary(detail)
        composition_summary_rows.append({"representation": representation, **observed})
        fitted, residual = _full_rank1_fit(va, response)
        for length in SURROGATE_BLOCK_LENGTHS:
            surrogate_tasks.append(
                {
                    "representation": representation,
                    "block_length": length,
                    "replicates": surrogate_replicates,
                    "seed": _stable_seed(20260903, f"surrogate:{representation}:{length}"),
                    "va": va,
                    "fitted": fitted,
                    "residual": residual,
                    "observed": observed,
                }
            )

    sensitivity_started = time.perf_counter()
    profile, shift_detail, shift_summary, adequacy = period_and_block_sensitivity(
        va, transforms
    )
    endpoint = endpoint_gate_consistency(pivot)
    sensitivity_seconds = time.perf_counter() - sensitivity_started

    injection = _injection_setup(pivot)
    injection_tasks = []
    for scheme, length in INJECTION_SCHEMES:
        for true_index, signal in enumerate(injection["signals"]):
            injection_tasks.append(
                {
                    "scheme": scheme,
                    "block_length": length,
                    "true_index": true_index,
                    "replicates": injection_replicates,
                    "seed": _stable_seed(20260904, f"injection:{scheme}:{true_index}"),
                    "trend_fit": injection["trend_fit"],
                    "signal": signal,
                    "common_residual": injection["common_residual"],
                    "candidates": injection["candidates"],
                }
            )

    surrogate_started = time.perf_counter()
    surrogate_results = _run_tasks(_surrogate_worker, surrogate_tasks, worker_count)
    surrogate_wall = time.perf_counter() - surrogate_started

    injection_started = time.perf_counter()
    injection_results = _run_tasks(_injection_worker, injection_tasks, worker_count)
    injection_wall = time.perf_counter() - injection_started

    surrogate_samples = pd.concat(
        [item["samples"] for item in surrogate_results], ignore_index=True
    )
    surrogate_summary = pd.DataFrame([item["summary"] for item in surrogate_results])
    injection_samples = pd.concat(
        [item["samples"] for item in injection_results], ignore_index=True
    )
    injection_summary = pd.DataFrame([item["summary"] for item in injection_results])
    confusion = (
        injection_samples.groupby(
            ["noise_scheme", "true_factor", "true_span_eV", "selected_factor", "selected_span_eV"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "count"})
    )
    confusion["rate"] = confusion["count"] / injection_replicates
    selection_bias = (
        injection_samples.groupby(
            ["noise_scheme", "selected_factor", "selected_span_eV"], as_index=False
        )
        .size()
        .rename(columns={"size": "count"})
    )
    selections_per_scheme = len(INJECTION_FACTORS) * injection_replicates
    selection_bias["marginal_selection_rate"] = (
        selection_bias["count"] / selections_per_scheme
    )
    selection_bias["selection_rank_within_scheme"] = (
        selection_bias.groupby("noise_scheme")["count"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    composition_summary = pd.DataFrame(composition_summary_rows)
    decision = _make_decision(
        composition_summary,
        shift_summary,
        profile,
        adequacy,
        endpoint,
        injection_summary,
    )
    timings = {
        "logical_cpus": os.cpu_count() or 1,
        "workers": worker_count,
        "surrogate_tasks": len(surrogate_tasks),
        "injection_tasks": len(injection_tasks),
        "surrogate_replicates_per_task": surrogate_replicates,
        "injection_replicates_per_task": injection_replicates,
        "sensitivity_seconds": sensitivity_seconds,
        "surrogate_parallel_wall_seconds": surrogate_wall,
        "surrogate_task_seconds_sum": float(surrogate_summary["task_seconds"].sum()),
        "surrogate_parallelism_ratio": float(
            surrogate_summary["task_seconds"].sum() / max(surrogate_wall, 1e-12)
        ),
        "injection_parallel_wall_seconds": injection_wall,
        "injection_task_seconds_sum": float(injection_summary["task_seconds"].sum()),
        "injection_parallelism_ratio": float(
            injection_summary["task_seconds"].sum() / max(injection_wall, 1e-12)
        ),
        "total_seconds": time.perf_counter() - started,
    }
    return {
        "data_audit": data_audit,
        "transform_audit": transform_audit,
        "composition_detail": pd.concat(composition_detail, ignore_index=True),
        "composition_summary": composition_summary,
        "surrogate_samples": surrogate_samples,
        "surrogate_summary": surrogate_summary,
        "period_profile": profile,
        "block_shift_detail": shift_detail,
        "block_shift_summary": shift_summary,
        "absolute_adequacy": adequacy,
        "endpoint_gate": endpoint,
        "injection_samples": injection_samples,
        "injection_confusion": confusion,
        "injection_summary": injection_summary,
        "injection_selection_bias": selection_bias,
        "injection_setup": {
            "candidate_factors": INJECTION_FACTORS.tolist(),
            "candidate_spans_eV": [item["span_eV"] for item in injection["candidates"]],
            "candidate_independent_noise_model": "cubic trend + single mean-energy harmonic residual",
            "equalized_signal_rms": injection["target_signal_rms"],
        },
        "decision": decision,
        "timings": timings,
    }
