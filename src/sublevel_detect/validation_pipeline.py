from __future__ import annotations

from pathlib import Path
from typing import Any


VALIDATION_STAGES = ("selector", "bootstrap", "holdout", "synthetic", "benchmark")


def requested_stages(stage: str | None) -> list[str]:
    if stage is None:
        return list(VALIDATION_STAGES)
    if stage not in VALIDATION_STAGES:
        raise ValueError(f"Unsupported validation stage: {stage}")
    if stage == "benchmark":
        return ["holdout", "benchmark"]
    return [stage]


def run(
    *,
    mode: str,
    input_path: str | Path,
    output_root: str | Path,
    device: str,
    requested_stage: str | None,
) -> dict[str, Any]:
    del input_path, output_root, device
    return {
        "ok": True,
        "mode": str(mode),
        "requested_stages": requested_stages(requested_stage),
    }
