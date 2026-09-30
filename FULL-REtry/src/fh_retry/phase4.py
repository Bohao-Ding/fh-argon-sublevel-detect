from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.linalg import helmert

from .analysis import BLOCKS, PERIOD_PRIOR, PSEUDO_PERIODS, load_and_audit
from .phase2 import _retarding_fractions
from .phase3 import (
    _model_error,
    _multiplicative_zero_replacement,
    _rank_detail,
    _rank_summary,
)


EXISTING_LEVELS = (0, 4, 6, 8, 10)
VIRTUAL_LEVELS = (0, 2, 4, 6, 8, 10)
LAG_GRID_V = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
REFERENCE_LAG_V = 0.5
REPRESENTATIONS = ("raw_fraction", "hellinger", "clr", "ilr")


def resolve_workers(requested: int | None, task_count: int) -> int:
    logical = os.cpu_count() or 1
    if requested is None or requested <= 0:
        return max(1, min(8, logical // 2, task_count))
    return max(1, min(int(requested), logical, task_count, 16))


def make_equilibrium_pivot(pivot: pd.DataFrame, design: str) -> tuple[pd.DataFrame, list[int]]:
    if design == "observed_5level":
        return pivot[list(EXISTING_LEVELS)].copy(), list(EXISTING_LEVELS)
    if design not in {"linear_vr2", "pchip_vr2"}:
        raise ValueError(f"unknown design: {design}")

    observed = pivot[list(EXISTING_LEVELS)].to_numpy(float)
    if design == "linear_vr2":
        vr2 = 0.5 * (observed[:, 0] + observed[:, 1])
    else:
        vr2 = PchipInterpolator(
            np.asarray(EXISTING_LEVELS, float), observed, axis=1
        )(2.0)

    equilibrium = pd.DataFrame(index=pivot.index)
    equilibrium[0] = pivot[0]
    equilibrium[2] = vr2
    for level in EXISTING_LEVELS[1:]:
        equilibrium[level] = pivot[level]
    return equilibrium[list(VIRTUAL_LEVELS)], list(VIRTUAL_LEVELS)


def apply_voltage_lag(
    equilibrium: pd.DataFrame, lag_v: float, direction: str
) -> pd.DataFrame:
    if lag_v < 0.0:
        raise ValueError("lag_v must be nonnegative")
    if direction not in {"forward", "reverse"}:
        raise ValueError("direction must be 'forward' or 'reverse'")
    if lag_v == 0.0:
        return equilibrium.copy()

    va = equilibrium.index.to_numpy(float)
    steps = np.diff(va)
    if not np.allclose(steps, steps[0], atol=1e-12):
        raise ValueError("Va grid must be uniform")
    alpha = 1.0 - np.exp(-float(steps[0]) / lag_v)
    values = equilibrium.to_numpy(float)
    filtered = np.empty_like(values)
    order = np.arange(len(va)) if direction == "forward" else np.arange(len(va) - 1, -1, -1)
    filtered[order[0]] = values[order[0]]
    for previous, current in zip(order[:-1], order[1:], strict=True):
        filtered[current] = filtered[previous] + alpha * (
            values[current] - filtered[previous]
        )
    return pd.DataFrame(filtered, index=equilibrium.index, columns=equilibrium.columns)


def _response_representations(
    pivot: pd.DataFrame, levels: list[int], quantization_ua: float
) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    va, fractions, band_audit = _retarding_fractions(pivot, levels)
    i0 = pivot.loc[va, levels[0]].to_numpy(float)
    row_delta = 0.5 * quantization_ua / i0
    positive, replacement_audit = _multiplicative_zero_replacement(fractions, row_delta)
    log_positive = np.log(positive)
    transforms = {
        "raw_fraction": fractions,
        "hellinger": np.sqrt(fractions),
        "clr": log_positive - log_positive.mean(axis=1, keepdims=True),
        "ilr": log_positive @ helmert(fractions.shape[1], full=False).T,
    }
    audit = {
        **band_audit,
        **replacement_audit,
        "fraction_dimensions": int(fractions.shape[1]),
        "all_finite": bool(all(np.isfinite(value).all() for value in transforms.values())),
    }
    return va, transforms, audit


def _order_audit(pivot: pd.DataFrame, levels: list[int]) -> dict:
    currents = pivot[levels].to_numpy(float).T
    upward = currents[1:] - currents[:-1]
    violations = upward > 1e-12
    return {
        "comparisons": int(upward.size),
        "violations": int(violations.sum()),
        "max_violation_uA": float(upward[violations].max(initial=0.0)),
        "total_violation_uA": float(upward[violations].sum()),
    }


def _absolute_adequacy(va: np.ndarray, response: np.ndarray) -> tuple[int, float]:
    passed = 0
    margins = []
    for low, high in BLOCKS:
        test = (va >= low) & (va < high)
        train = ~test
        rank2 = _model_error(
            va, response, train, test, period=PERIOD_PRIOR, rank=2, trend_degree=3
        )
        trend = _model_error(
            va, response, train, test, period=None, rank=None, trend_degree=5
        )
        pseudo = min(
            _model_error(
                va, response, train, test, period=period, rank=2, trend_degree=3
            )
            for period in PSEUDO_PERIODS
        )
        comparator = min(trend, pseudo)
        passed += int(rank2 < comparator)
        margins.append((comparator - rank2) / comparator)
    return passed, float(min(margins))


def _curve_frame(
    pivot: pd.DataFrame,
    *,
    design: str,
    lag_v: float,
    direction: str,
) -> pd.DataFrame:
    frame = pivot.rename_axis("Va").reset_index().melt(
        id_vars="Va", var_name="Vr", value_name="IuA_virtual"
    )
    frame.insert(0, "direction", direction)
    frame.insert(0, "lag_V", lag_v)
    frame.insert(0, "design", design)
    frame["source_kind"] = np.where(
        frame["Vr"].eq(2), "interpolated_Vr2", "derived_from_measured_curve"
    )
    return frame


def _analyze_task(task: dict) -> dict:
    started = time.perf_counter()
    scan = apply_voltage_lag(task["equilibrium"], task["lag_v"], task["direction"])
    levels = task["levels"]
    va, transforms, band_audit = _response_representations(
        scan, levels, task["quantization_uA"]
    )
    detail_frames = []
    summaries = []
    for representation in REPRESENTATIONS:
        detail = _rank_detail(va, transforms[representation])
        detail.insert(0, "representation", representation)
        detail_frames.append(detail)
        summary = _rank_summary(detail)
        adequate_blocks, worst_adequacy_margin = _absolute_adequacy(
            va, transforms[representation]
        )
        summaries.append(
            {
                "representation": representation,
                **summary,
                "strict_G2_pass": bool(
                    summary["improved_blocks"] >= 4
                    and summary["median_gain"] >= 0.10
                    and summary["worst_gain"] >= -0.05
                ),
                "majority_pass": bool(
                    summary["improved_blocks"] >= 4 and summary["median_gain"] >= 0.10
                ),
                "absolute_adequacy_blocks": adequate_blocks,
                "worst_adequacy_margin": worst_adequacy_margin,
            }
        )

    metadata = {
        "design": task["design"],
        "lag_V": task["lag_v"],
        "direction": task["direction"],
        "level_count": len(levels),
    }
    detail = pd.concat(detail_frames, ignore_index=True)
    summary = pd.DataFrame(summaries)
    for key, value in reversed(tuple(metadata.items())):
        detail.insert(0, key, value)
        summary.insert(0, key, value)
    return {
        "detail": detail,
        "summary": summary,
        "order": {**metadata, **_order_audit(scan, levels), **band_audit},
        "curves": _curve_frame(
            scan,
            design=task["design"],
            lag_v=task["lag_v"],
            direction=task["direction"],
        ),
        "task_seconds": time.perf_counter() - started,
    }


def _direction_differences(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    virtual = curves[curves["design"].isin(["linear_vr2", "pchip_vr2"])]
    for (design, lag_v, vr), local in virtual.groupby(["design", "lag_V", "Vr"]):
        pivot = local.pivot(index="Va", columns="direction", values="IuA_virtual")
        delta = pivot["forward"].to_numpy() - pivot["reverse"].to_numpy()
        span = float(local["IuA_virtual"].max() - local["IuA_virtual"].min())
        rows.append(
            {
                "design": design,
                "lag_V": lag_v,
                "Vr": vr,
                "direction_rmse_uA": float(np.sqrt(np.mean(delta**2))),
                "direction_max_abs_uA": float(np.max(np.abs(delta))),
                "direction_rmse_over_curve_span": float(
                    np.sqrt(np.mean(delta**2)) / max(span, 1e-12)
                ),
                "absolute_hysteresis_area_uA_V": float(
                    np.trapezoid(np.abs(delta), pivot.index.to_numpy(float))
                ),
            }
        )
    return pd.DataFrame(rows)


def _partition_diagnostic(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    virtual = curves[curves["design"].isin(["linear_vr2", "pchip_vr2"])]
    for (design, lag_v, direction), local in virtual.groupby(
        ["design", "lag_V", "direction"]
    ):
        pivot = local.pivot(index="Va", columns="Vr", values="IuA_virtual")
        keep = pivot.index.to_numpy(float) >= 15.0
        d02 = (pivot[0] - pivot[2]).to_numpy(float)[keep]
        d24 = (pivot[2] - pivot[4]).to_numpy(float)[keep]
        pair = np.column_stack([d02, d24])
        standardized = (pair - pair.mean(axis=0)) / pair.std(axis=0, ddof=1)
        singular = np.linalg.svd(standardized, compute_uv=False)
        rows.append(
            {
                "design": design,
                "lag_V": lag_v,
                "direction": direction,
                "D02_D24_correlation": float(np.corrcoef(d02, d24)[0, 1]),
                "D02_D24_rmse_uA": float(np.sqrt(np.mean((d02 - d24) ** 2))),
                "second_singular_variance_fraction": float(
                    singular[1] ** 2 / np.sum(singular**2)
                ),
            }
        )
    return pd.DataFrame(rows)


def measured_partition_audit(
    pivot: pd.DataFrame, quantization_ua: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_frames = []
    summaries = []
    for size in range(1, 4):
        for internal in combinations((4, 6, 8), size):
            levels = [0, *internal, 10]
            va, transforms, _ = _response_representations(
                pivot[levels], levels, quantization_ua
            )
            label = "-".join(map(str, levels))
            for representation in REPRESENTATIONS:
                detail = _rank_detail(va, transforms[representation])
                detail.insert(0, "representation", representation)
                detail.insert(0, "levels", label)
                detail_frames.append(detail)
                summary = _rank_summary(detail)
                adequate_blocks, worst_adequacy_margin = _absolute_adequacy(
                    va, transforms[representation]
                )
                summaries.append(
                    {
                        "levels": label,
                        "level_count": len(levels),
                        "representation": representation,
                        **summary,
                        "strict_G2_pass": bool(
                            summary["improved_blocks"] >= 4
                            and summary["median_gain"] >= 0.10
                            and summary["worst_gain"] >= -0.05
                        ),
                        "majority_pass": bool(
                            summary["improved_blocks"] >= 4
                            and summary["median_gain"] >= 0.10
                        ),
                        "absolute_adequacy_blocks": adequate_blocks,
                        "worst_adequacy_margin": worst_adequacy_margin,
                    }
                )
    return pd.concat(detail_frames, ignore_index=True), pd.DataFrame(summaries)


def _make_decision(
    summary: pd.DataFrame,
    direction: pd.DataFrame,
    partition: pd.DataFrame,
    measured_partitions: pd.DataFrame,
) -> dict:
    raw = summary[summary["representation"] == "raw_fraction"]
    virtual_designs = ("linear_vr2", "pchip_vr2")
    available_lags = sorted(float(value) for value in raw["lag_V"].unique())
    pass_rows = []
    for design in virtual_designs:
        for lag_v in available_lags:
            local = raw[(raw["design"] == design) & np.isclose(raw["lag_V"], lag_v)]
            pass_rows.append(
                {
                    "design": design,
                    "lag_V": lag_v,
                    "both_directions_strict_pass": bool(
                        len(local) == 2 and local["strict_G2_pass"].all()
                    ),
                    "minimum_median_gain": float(local["median_gain"].min()),
                    "minimum_worst_gain": float(local["worst_gain"].min()),
                }
            )
    lag_gate = pd.DataFrame(pass_rows)
    max_passed = {}
    for design in virtual_designs:
        local = lag_gate[
            (lag_gate["design"] == design) & lag_gate["both_directions_strict_pass"]
        ]
        max_passed[design] = None if local.empty else float(local["lag_V"].max())

    observed_pass = []
    for lag_v in available_lags:
        local = raw[
            (raw["design"] == "observed_5level") & np.isclose(raw["lag_V"], lag_v)
        ]
        if len(local) == 2 and local["strict_G2_pass"].all():
            observed_pass.append(lag_v)
    max_observed_pass = None if not observed_pass else float(max(observed_pass))

    observed_zero = raw[
        (raw["design"] == "observed_5level") & np.isclose(raw["lag_V"], 0.0)
    ]
    virtual_zero = raw[
        raw["design"].isin(virtual_designs) & np.isclose(raw["lag_V"], 0.0)
    ]

    reference = summary[
        (summary["design"] == "linear_vr2")
        & np.isclose(summary["lag_V"], REFERENCE_LAG_V)
    ]
    reference_raw = reference[reference["representation"] == "raw_fraction"]
    reference_geometry = reference[reference["representation"] != "raw_fraction"]
    reference_pass = bool(
        len(reference_raw) == 2
        and reference_raw["strict_G2_pass"].all()
        and reference_geometry["majority_pass"].all()
    )
    reference_direction = direction[np.isclose(direction["lag_V"], REFERENCE_LAG_V)]
    reference_partition = partition[
        (partition["design"] == "linear_vr2")
        & np.isclose(partition["lag_V"], REFERENCE_LAG_V)
    ]
    measured_raw = measured_partitions[
        measured_partitions["representation"] == "raw_fraction"
    ]
    measured_failures = measured_raw[~measured_raw["strict_G2_pass"]]
    return {
        "evidence_identity": "synthetic direction-lag stress test derived from one archived dataset",
        "reference_design": "linear_vr2",
        "reference_lag_V": REFERENCE_LAG_V,
        "reference_12_curve_gate_passed": reference_pass,
        "reference_raw_results": reference_raw.to_dict(orient="records"),
        "reference_max_direction_rmse_uA": float(
            reference_direction["direction_rmse_uA"].max()
        ),
        "reference_max_direction_rmse_over_curve_span": float(
            reference_direction["direction_rmse_over_curve_span"].max()
        ),
        "max_tested_lag_with_raw_strict_pass_both_directions_V": max_passed,
        "observed_5level_max_tested_lag_with_raw_strict_pass_both_directions_V": max_observed_pass,
        "observed_5level_zero_lag_raw": observed_zero.to_dict(orient="records"),
        "virtual_6level_zero_lag_raw": virtual_zero.to_dict(orient="records"),
        "reference_D02_D24_min_correlation": float(
            reference_partition["D02_D24_correlation"].min()
        ),
        "reference_D02_D24_max_second_singular_variance_fraction": float(
            reference_partition["second_singular_variance_fraction"].max()
        ),
        "measured_partition_raw_strict_pass_count": int(
            measured_raw["strict_G2_pass"].sum()
        ),
        "measured_partition_raw_total": int(len(measured_raw)),
        "measured_partition_raw_failures": measured_failures.to_dict(orient="records"),
        "lag_gate": lag_gate.to_dict(orient="records"),
        "expand_to_36_synthetic_cycles": False,
        "next_stage_decision": "DO_NOT_GENERATE_36_SYNTHETIC_THERMAL_CYCLES",
        "reason": (
            "The interpolated Vr=2 V curve adds no independent measurement and the six-band strict gate "
            "already fails at zero lag. Additional synthetic cycles would repeat assumptions rather than "
            "add independent physical information."
        ),
        "allowed_claim": (
            "The archived-data conclusion was stress-tested against a specified first-order "
            "scan-direction lag model and an interpolated Vr=2 V curve."
        ),
        "prohibited_claims": [
            "new experimental repeat scans were acquired",
            "scan hysteresis was measured",
            "thermal-cycle repeatability was established",
            "the virtual lag parameter is an instrument calibration",
        ],
    }


def run_phase4(
    data_path,
    *,
    workers: int | None = None,
    lag_grid_v: tuple[float, ...] = LAG_GRID_V,
) -> dict:
    started = time.perf_counter()
    _, pivot, data_audit = load_and_audit(data_path)
    designs = {}
    for design in ("observed_5level", "linear_vr2", "pchip_vr2"):
        designs[design] = make_equilibrium_pivot(pivot, design)

    tasks = []
    for design, (equilibrium, levels) in designs.items():
        for lag_v in lag_grid_v:
            for direction in ("forward", "reverse"):
                tasks.append(
                    {
                        "design": design,
                        "equilibrium": equilibrium,
                        "levels": levels,
                        "lag_v": float(lag_v),
                        "direction": direction,
                        "quantization_uA": data_audit["current_quantization_uA"],
                    }
                )

    worker_count = resolve_workers(workers, len(tasks))
    parallel_started = time.perf_counter()
    if worker_count == 1:
        results = [_analyze_task(task) for task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = list(executor.map(_analyze_task, tasks))
    parallel_wall = time.perf_counter() - parallel_started

    detail = pd.concat([item["detail"] for item in results], ignore_index=True)
    summary = pd.concat([item["summary"] for item in results], ignore_index=True)
    order = pd.DataFrame([item["order"] for item in results])
    curves = pd.concat([item["curves"] for item in results], ignore_index=True)
    direction = _direction_differences(curves)
    partition = _partition_diagnostic(curves)
    measured_partition_detail, measured_partition_summary = measured_partition_audit(
        pivot, data_audit["current_quantization_uA"]
    )
    reference_curves = curves[
        (curves["design"] == "linear_vr2")
        & np.isclose(curves["lag_V"], REFERENCE_LAG_V)
    ].reset_index(drop=True)
    decision = _make_decision(
        summary, direction, partition, measured_partition_summary
    )
    task_seconds = float(sum(item["task_seconds"] for item in results))
    timings = {
        "logical_cpus": os.cpu_count() or 1,
        "workers": worker_count,
        "tasks": len(tasks),
        "parallel_wall_seconds": parallel_wall,
        "task_seconds_sum": task_seconds,
        "parallelism_ratio": task_seconds / max(parallel_wall, 1e-12),
        "total_seconds": time.perf_counter() - started,
    }
    return {
        "data_audit": data_audit,
        "analysis_summary": summary,
        "block_detail": detail,
        "order_audit": order,
        "all_virtual_curves": curves,
        "reference_12_curves": reference_curves,
        "direction_differences": direction,
        "partition_diagnostic": partition,
        "measured_partition_detail": measured_partition_detail,
        "measured_partition_summary": measured_partition_summary,
        "decision": decision,
        "timings": timings,
        "config": {
            "existing_levels_V": list(EXISTING_LEVELS),
            "virtual_levels_V": list(VIRTUAL_LEVELS),
            "lag_grid_V": list(lag_grid_v),
            "reference_lag_V": REFERENCE_LAG_V,
            "interpolation_designs": ["linear_vr2", "pchip_vr2"],
            "scan_lag_model": "first-order causal response in the voltage domain",
        },
    }
