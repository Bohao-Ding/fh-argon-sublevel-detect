from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .train import TrainConfig, fit_model


def _dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def _scan_row(result: dict[str, Any]) -> dict[str, Any]:
    summary = result["metrics"]["summary"]
    params = result["parameters"]
    return {
        "K": int(params["n_levels"]),
        "epochs_completed": int(result["epochs_completed"]),
        "stop_reason": result["stop_reason"],
        "elapsed_seconds": float(result["elapsed_seconds"]),
        "n_params": int(summary["n_params"]),
        "mean_curve_nrmse": float(summary["mean_curve_nrmse"]),
        "overall_nrmse": float(summary["overall_nrmse"]),
        "cutoff_mae_V": float(summary["cutoff_mae_V"]),
        "zero_region_mae_uA": float(summary["zero_region_mae_uA"]),
        "peak_position_mae_V": float(summary["peak_position_mae_V"]),
        "trough_position_mae_V": float(summary["trough_position_mae_V"]),
        "peak_drift_rmse_V": float(summary["peak_drift_rmse_V"]),
        "aic": float(summary["aic"]),
        "bic": float(summary["bic"]),
        "energies_eV": json.dumps(params["energies_eV"]),
        "channel_weights": json.dumps(params["channel_weights"]),
    }


def _plot_scan(frame: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(frame["K"], frame["mean_curve_nrmse"], marker="o", color="#1f77b4")
    axes[0].set_xlabel("Number of excitation channels K")
    axes[0].set_ylabel("Mean curve NRMSE")
    axes[0].grid(alpha=0.25)
    axes[1].plot(frame["K"], frame["bic"], marker="o", color="#d62728", label="BIC")
    axes[1].plot(frame["K"], frame["aic"], marker="s", color="#2ca02c", label="AIC")
    axes[1].set_xlabel("Number of excitation channels K")
    axes[1].set_ylabel("Information criterion")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_single(config: TrainConfig, baseline_path: str | Path | None) -> dict[str, Any]:
    return fit_model(replace(config, n_levels=1), baseline_path=baseline_path)


def run_scan(
    base_config: TrainConfig,
    output_root: str | Path,
    baseline_path: str | Path | None,
    maximum_k: int = 8,
    existing_k1: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    for k in range(1, int(maximum_k) + 1):
        if k == 1 and existing_k1 is not None:
            result = existing_k1
        else:
            result = fit_model(
                replace(base_config, n_levels=k, output_dir=str(root / f"k{k}")),
                baseline_path=baseline_path if k == 1 else None,
            )
        results[str(k)] = {
            "result_path": str(Path(result["config"]["output_dir"]) / "result.json"),
            "metrics": result["metrics"]["summary"],
            "parameters": result["parameters"],
        }
        rows.append(_scan_row(result))
    table = pd.DataFrame(rows)
    table.to_csv(root / "level_scan.csv", index=False)
    _plot_scan(table, root / "level_scan.png")
    best_nrmse = int(table.loc[table["mean_curve_nrmse"].idxmin(), "K"])
    best_bic = int(table.loc[table["bic"].idxmin(), "K"])
    summary = {
        "status": "completed",
        "K_range": [1, int(maximum_k)],
        "best_mean_curve_nrmse_K": best_nrmse,
        "best_bic_K": best_bic,
        "selection_boundary": (
            "Fit and information criteria are reported separately. This scan does not, by itself, "
            "identify a unique physical level count."
        ),
        "results": results,
    }
    _dump(summary, root / "level_scan_summary.json")
    return summary


def run_auto(
    base_config: TrainConfig,
    output_root: str | Path,
    baseline_path: str | Path | None,
    maximum_k: int = 8,
    bypass_gate: bool = False,
) -> dict[str, Any]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    k1 = run_single(replace(base_config, output_dir=str(root / "k1")), baseline_path)
    gate_passed = bool(k1["single_level_gate"]["passed"])
    if not gate_passed and not bypass_gate:
        summary = {
            "status": "single_level_gate_failed",
            "single_level_result": str(root / "k1" / "result.json"),
            "single_level_gate": k1["single_level_gate"],
            "scan_started": False,
        }
        _dump(summary, root / "pipeline_summary.json")
        return summary
    scan = run_scan(
        base_config=base_config,
        output_root=root / "scan",
        baseline_path=baseline_path,
        maximum_k=maximum_k,
        existing_k1=k1,
    )
    summary = {
        "status": "completed",
        "single_level_result": str(root / "k1" / "result.json"),
        "single_level_gate": k1["single_level_gate"],
        "gate_bypassed": bool(bypass_gate and not gate_passed),
        "scan_summary": str(root / "scan" / "level_scan_summary.json"),
        "best_mean_curve_nrmse_K": scan["best_mean_curve_nrmse_K"],
        "best_bic_K": scan["best_bic_K"],
    }
    _dump(summary, root / "pipeline_summary.json")
    return summary

