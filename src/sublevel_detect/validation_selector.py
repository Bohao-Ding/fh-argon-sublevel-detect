from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import model
from .validation_common import Baseline, atomic_json_dump, write_stage_status


COMPONENT_METRICS = {
    "rmse": "rmse_mean",
    "summary": "cv_rmse_mean",
    "d1": "d1_rmse_mean",
    "d2": "d2_rmse_mean",
    "structure": "structure_score",
    "physical": "vr_physical_response_score",
    "bic": "bic",
    "aic": "aic",
    "degeneracy": "degeneracy_penalty",
}
COMPONENTS = tuple(COMPONENT_METRICS)


@dataclass(frozen=True)
class SelectorScenario:
    name: str
    weights: dict[str, float]


def _weights_with_zeroes(*removed: str) -> dict[str, float]:
    weights = {key: float(value) for key, value in model.DEFAULT_SELECTOR_RANK_WEIGHTS.items()}
    for key in removed:
        weights[key] = 0.0
    return weights


def selector_scenarios() -> list[SelectorScenario]:
    return [
        SelectorScenario("production", _weights_with_zeroes()),
        SelectorScenario("leave_one_rmse_out", _weights_with_zeroes("rmse")),
        SelectorScenario("leave_one_legacy_summary_out", _weights_with_zeroes("summary")),
        SelectorScenario("leave_one_d1_out", _weights_with_zeroes("d1")),
        SelectorScenario("leave_one_d2_out", _weights_with_zeroes("d2")),
        SelectorScenario("leave_one_structure_out", _weights_with_zeroes("structure")),
        SelectorScenario("leave_one_physical_out", _weights_with_zeroes("physical")),
        SelectorScenario("leave_one_bic_out", _weights_with_zeroes("bic")),
        SelectorScenario("leave_one_aic_out", _weights_with_zeroes("aic")),
        SelectorScenario("leave_one_degeneracy_out", _weights_with_zeroes("degeneracy")),
        SelectorScenario("remove_fit_group", _weights_with_zeroes("rmse", "summary")),
        SelectorScenario("remove_shape_group", _weights_with_zeroes("d1", "d2", "structure")),
        SelectorScenario("remove_physical_group", _weights_with_zeroes("physical")),
        SelectorScenario("remove_complexity_group", _weights_with_zeroes("bic", "aic", "degeneracy")),
        SelectorScenario(
            "fit_complexity_only",
            _weights_with_zeroes("summary", "d1", "d2", "structure", "physical"),
        ),
    ]


def competition_ranks(values: Sequence[float], *, tolerance: float = 1e-12) -> list[float]:
    numeric = np.asarray(values, dtype=np.float64)
    finite_indices = [int(index) for index in np.flatnonzero(np.isfinite(numeric))]
    finite_indices.sort(key=lambda index: (float(numeric[index]), index))
    ranks = [float(len(finite_indices) + 1)] * len(numeric)
    previous_value: float | None = None
    previous_rank = 0
    for position, index in enumerate(finite_indices, start=1):
        value = float(numeric[index])
        scale = max(1.0, abs(value), abs(previous_value) if previous_value is not None else 0.0)
        if previous_value is None or abs(value - previous_value) > float(tolerance) * scale:
            previous_rank = position
            previous_value = value
        ranks[index] = float(previous_rank)
    return ranks


def component_values(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}
    for component, metric in COMPONENT_METRICS.items():
        raw = [float(row.get(metric, float("nan"))) for row in rows]
        values[component] = raw if component == "degeneracy" else competition_ranks(raw)
    return values


def scenario_scores(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("Selector audit requires at least one model-selection row")
    ordered = sorted((dict(row) for row in rows), key=lambda row: int(row["n_levels"]))
    component_by_row = component_values(ordered)
    score_rows: list[dict[str, Any]] = []
    for scenario in selector_scenarios():
        for index, row in enumerate(ordered):
            contributions = {
                component: float(scenario.weights[component] * component_by_row[component][index])
                for component in COMPONENTS
            }
            score_rows.append(
                {
                    "scenario": scenario.name,
                    "n_levels": int(row["n_levels"]),
                    "composite_rank_score": float(sum(contributions.values())),
                    **{f"{component}_component": float(component_by_row[component][index]) for component in COMPONENTS},
                    **{f"{component}_contribution": value for component, value in contributions.items()},
                }
            )
    return score_rows


def run_scenarios(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    scores = scenario_scores(rows)
    scenarios = {scenario.name: scenario for scenario in selector_scenarios()}
    decisions: list[dict[str, Any]] = []
    for name in [scenario.name for scenario in selector_scenarios()]:
        candidates = [row for row in scores if row["scenario"] == name]
        selected = min(candidates, key=lambda row: (float(row["composite_rank_score"]), int(row["n_levels"])))
        decisions.append(
            {
                "scenario": name,
                "selected_k": int(selected["n_levels"]),
                "selected_composite_rank_score": float(selected["composite_rank_score"]),
                **{f"weight_{key}": float(value) for key, value in scenarios[name].weights.items()},
            }
        )
    return decisions


def rank_correlation(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted((dict(row) for row in rows), key=lambda row: int(row["n_levels"]))
    values = component_values(ordered)
    result: list[dict[str, Any]] = []
    for left in COMPONENTS:
        for right in COMPONENTS:
            if left == right:
                rho = 1.0
            elif np.std(values[left]) == 0.0 or np.std(values[right]) == 0.0:
                rho = float("nan")
            else:
                rho = float(spearmanr(values[left], values[right]).statistic)
            result.append({"component_x": left, "component_y": right, "spearman_rho": rho})
    return result


def selected_k_distribution(decisions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    total = len(decisions)
    if total == 0:
        return []
    counts: dict[int, int] = {}
    for row in decisions:
        selected_k = int(row["selected_k"])
        counts[selected_k] = counts.get(selected_k, 0) + 1
    return [
        {
            "selected_k": selected_k,
            "scenario_count": count,
            "scenario_fraction": float(count / total),
        }
        for selected_k, count in sorted(counts.items())
    ]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(row) for row in rows]).to_csv(path, index=False)


def run(*, baseline: Baseline, output_dir: str | Path, mode: str) -> dict[str, Any]:
    target = Path(output_dir)
    source = pd.read_csv(baseline.files["model_selection"])
    rows = source.to_dict("records")
    scores = scenario_scores(rows)
    decisions = run_scenarios(rows)
    distribution = selected_k_distribution(decisions)
    correlations = rank_correlation(rows)
    production = next(row for row in decisions if row["scenario"] == "production")
    legacy_removed = next(row for row in decisions if row["scenario"] == "leave_one_legacy_summary_out")
    status = "smoke_passed" if str(mode) == "smoke" else "complete"
    summary = {
        "ok": True,
        "status": status,
        "baseline_kind": baseline.kind,
        "baseline_model_selection_sha256": baseline.hashes["model_selection"],
        "scenario_count": len(decisions),
        "production_selected_k": int(production["selected_k"]),
        "without_legacy_summary_selected_k": int(legacy_removed["selected_k"]),
        "legacy_summary_removal_changes_k": bool(production["selected_k"] != legacy_removed["selected_k"]),
        "selected_k_distribution": distribution,
    }
    _write_csv(target / "selector_scenarios.csv", decisions)
    _write_csv(target / "selector_scores.csv", scores)
    _write_csv(target / "rank_correlation.csv", correlations)
    _write_csv(target / "selected_k_distribution.csv", distribution)
    atomic_json_dump(summary, target / "selector_summary.json")
    write_stage_status(
        target / "stage_status.json",
        status,
        scenario_count=len(decisions),
        failed_count=0,
        baseline_model_selection_sha256=baseline.hashes["model_selection"],
    )
    return summary
