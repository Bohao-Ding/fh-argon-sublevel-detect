from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .metrics import apply_v2_single_level_gate
from .model_v2 import CompetingChannelFranckHertz
from .train import TrainConfig, fit_model


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _passport(config: TrainConfig, hypothesis: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "input_path": str(Path(config.data_path).resolve()),
        "input_sha256": _sha256(config.data_path),
        "model_family": "v2a-competing-history",
        "hypothesis": hypothesis,
        "physical_scope": (
            "Saturating space-charge supply, uniform-field thin-layer competing collision "
            "master equation, and signed axial energy-angle collector gate."
        ),
        "constant_source": {
            "name": "NIST Atomic Spectra Database, Ar I levels",
            "url": "https://physics.nist.gov/PhysRefData/Handbook/Tables/argontable5.htm",
        },
        "evidence_boundary": (
            "In-sample structural fit only; not a full Boltzmann or trajectory Monte Carlo solver, "
            "and not evidence that the experiment uniquely resolves the fitted level count."
        ),
    }


def _model_factory(config: TrainConfig) -> CompetingChannelFranckHertz:
    hypothesis = "h1" if int(config.n_levels) == 1 else "h4s"
    if int(config.n_levels) not in {1, 4}:
        raise ValueError("v2a supports only the pre-registered H1 and H4s hypotheses")
    return CompetingChannelFranckHertz(
        hypothesis=hypothesis,
        n_layers=config.n_layers,
        max_collisions=config.max_collisions,
        quadrature_order=config.quadrature_order,
        seed=config.seed,
    )


def run_hypothesis(
    base_config: TrainConfig,
    hypothesis: str,
    output_dir: str | Path,
    baseline_path: str | Path | None = None,
    initial_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    if hypothesis not in {"h1", "h4s"}:
        raise ValueError("hypothesis must be 'h1' or 'h4s'")
    config = replace(
        base_config,
        n_levels=1 if hypothesis == "h1" else 4,
        output_dir=str(Path(output_dir).resolve()),
    )
    return fit_model(
        config,
        baseline_path=baseline_path if hypothesis == "h1" else None,
        model_factory=_model_factory,
        single_level_gate_fn=apply_v2_single_level_gate,
        material_passport=_passport(config, hypothesis),
        initial_checkpoint=initial_checkpoint,
    )


def compare_hypotheses(h1: dict[str, Any], h4s: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "mean_curve_nrmse",
        "overall_nrmse",
        "cutoff_mae_V",
        "max_cutoff_error_V",
        "zero_region_mae_uA",
        "peak_position_mae_V",
        "trough_position_mae_V",
        "peak_drift_rmse_V",
        "aic",
        "bic",
    )
    h1_summary = h1["metrics"]["summary"]
    h4s_summary = h4s["metrics"]["summary"]
    return {
        "status": "completed",
        "hypotheses": ["h1", "h4s"],
        "delta_h4s_minus_h1": {
            key: float(h4s_summary[key]) - float(h1_summary[key]) for key in keys
        },
        "h1_result": str(Path(h1["config"]["output_dir"]) / "result.json"),
        "h4s_result": str(Path(h4s["config"]["output_dir"]) / "result.json"),
        "interpretation_boundary": (
            "H4s is an in-sample fit comparison after an H1 structural gate. A better fit does not "
            "by itself establish unique level resolution; held-out validation is outside this run."
        ),
    }


def dump_json(payload: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
