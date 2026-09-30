from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from .metrics import apply_v2j_single_level_gate
from .model_v2m import (
    CrossSectionTable,
    TabulatedPartialCrossSectionFranckHertz,
    load_cross_section_table,
)
from .pipeline_v2 import _passport
from .train import TrainConfig, fit_model


def run_hypothesis(
    base_config: TrainConfig,
    hypothesis: str,
    output_dir: str | Path,
    baseline_path: str | Path | None = None,
    initial_checkpoint: str | Path | None = None,
    cross_section_table_path: str | Path | None = None,
) -> dict[str, Any]:
    if hypothesis not in {"h1", "h4s"}:
        raise ValueError("hypothesis must be 'h1' or 'h4s'")
    if cross_section_table_path is None:
        raise ValueError("v2m-H1 and v2m-H4s both require --cross-section-table")
    table: CrossSectionTable = load_cross_section_table(cross_section_table_path)

    config = replace(
        base_config,
        n_levels=1 if hypothesis == "h1" else 4,
        output_dir=str(Path(output_dir).resolve()),
    )

    def model_factory(model_config: TrainConfig) -> TabulatedPartialCrossSectionFranckHertz:
        return TabulatedPartialCrossSectionFranckHertz(
            hypothesis=hypothesis,
            n_layers=model_config.n_layers,
            max_collisions=model_config.max_collisions,
            quadrature_order=model_config.quadrature_order,
            seed=model_config.seed,
            cross_section_table=table,
        )

    passport = _passport(config, hypothesis)
    passport["model_family"] = "v2m-tabulated-partial-cross-sections"
    passport["physical_scope"] = (
        "v2l transport with matched tabulated collision rates: H1 uses the sum of "
        "four partial cross sections, while H4s fixes NIST loss energies and uses "
        "the same table as four competing hazards."
    )
    passport["cross_section_source"] = {
        "name": table.source_name,
        "url": table.source_url,
        "csv_sha256": table.csv_sha256,
        "csv_path": table.csv_path,
    }
    return fit_model(
        config,
        baseline_path=baseline_path if hypothesis == "h1" else None,
        model_factory=model_factory,
        single_level_gate_fn=apply_v2j_single_level_gate,
        material_passport=passport,
        initial_checkpoint=initial_checkpoint,
    )
