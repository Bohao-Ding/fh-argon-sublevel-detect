from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from .metrics import apply_v2b_single_level_gate
from .model_v2c import IsotropicLastCollisionFranckHertz
from .pipeline_v2 import _passport
from .train import TrainConfig, fit_model


def _model_factory(config: TrainConfig) -> IsotropicLastCollisionFranckHertz:
    hypothesis = "h1" if int(config.n_levels) == 1 else "h4s"
    if int(config.n_levels) not in {1, 4}:
        raise ValueError("v2c supports only H1 and H4s")
    return IsotropicLastCollisionFranckHertz(
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
    passport = _passport(config, hypothesis)
    passport["model_family"] = "v2c-fixed-isotropic-scattering"
    passport["physical_scope"] = (
        "v2b last-collision position closure with fixed isotropic forward-flux "
        "post-collision angular distribution."
    )
    return fit_model(
        config,
        baseline_path=baseline_path if hypothesis == "h1" else None,
        model_factory=_model_factory,
        single_level_gate_fn=apply_v2b_single_level_gate,
        material_passport=passport,
        initial_checkpoint=initial_checkpoint,
    )
