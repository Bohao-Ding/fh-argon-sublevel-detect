from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import (
    model,
    paths,
    validation_benchmark,
    validation_bootstrap,
    validation_holdout,
    validation_selector,
    validation_synthetic,
)
from .validation_common import atomic_json_dump, resolve_baseline, write_progress


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
    package_root: str | Path | None = None,
) -> dict[str, Any]:
    stages = requested_stages(requested_stage)
    validation_root = paths.validation_dir(output_root)
    package = Path(package_root) if package_root is not None else paths.PROJECT_ROOT / "source_data_package"
    baseline = resolve_baseline(paths.resolve_project_path(output_root), package_root=package)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=paths.PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=paths.PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    selected_device = "cuda" if str(device) == "cuda" and model.torch.cuda.is_available() else "cpu"
    manifest = {
        "schema_version": 1,
        "invocation": {
            "mode": str(mode),
            "input": str(paths.resolve_project_path(input_path)),
            "output": str(paths.resolve_project_path(output_root)),
            "device_requested": str(device),
            "requested_stage": requested_stage,
            "requested_stages": stages,
        },
        "git": {"commit": commit, "dirty": dirty},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": model.torch.__version__,
            "cuda_runtime": model.torch.version.cuda,
            "cuda_available": bool(model.torch.cuda.is_available()),
            "device_selected": selected_device,
        },
        "baseline": {
            "kind": baseline.kind,
            "root": str(baseline.root),
            "identity_sha256": baseline.identity_hash,
            "files": {key: str(path) for key, path in baseline.files.items()},
            "hashes": baseline.hashes,
        },
        "frozen_stage_order": list(VALIDATION_STAGES),
    }
    atomic_json_dump(manifest, validation_root / "experiment_manifest.json")
    write_progress(validation_root / "progress.json", completed=0, failed=0, total=len(stages))
    results: dict[str, Any] = {}
    completed = 0
    all_ok = True
    for stage in stages:
        if stage == "selector":
            result = validation_selector.run(
                baseline=baseline,
                output_dir=validation_root / "selector_audit",
                mode=str(mode),
            )
        elif stage == "bootstrap":
            result = validation_bootstrap.run(
                mode=str(mode),
                input_path=input_path,
                baseline=baseline,
                output_dir=validation_root / "block_bootstrap",
                validation_root=validation_root,
                device=str(device),
            )
        elif stage == "holdout":
            result = validation_holdout.run(
                mode=str(mode),
                input_path=input_path,
                baseline=baseline,
                output_dir=validation_root / "holdout",
                validation_root=validation_root,
                device=str(device),
            )
        elif stage == "synthetic":
            result = validation_synthetic.run(
                mode=str(mode),
                input_path=input_path,
                baseline=baseline,
                output_dir=validation_root / "synthetic_recovery",
                validation_root=validation_root,
                device=str(device),
            )
        elif stage == "benchmark":
            result = validation_benchmark.run(
                mode=str(mode),
                input_path=input_path,
                holdout_dir=validation_root / "holdout",
                output_dir=validation_root / "benchmark",
                validation_root=validation_root,
                device=str(device),
            )
        else:
            raise NotImplementedError(f"Validation stage is not implemented yet: {stage}")
        results[stage] = result
        completed += 1
        all_ok = all_ok and bool(result.get("ok", False))
        if "unit_total" in result:
            write_progress(
                validation_root / "progress.json",
                completed=int(result.get("completed", 0)),
                failed=int(result.get("failed", 0)),
                total=int(result["unit_total"]),
                current_stage=stage,
                completed_stages=completed,
                total_stages=len(stages),
            )
        else:
            write_progress(
                validation_root / "progress.json",
                completed=completed,
                failed=0 if bool(result.get("ok", False)) else 1,
                total=len(stages),
                current_stage=stage,
            )
    return {
        "ok": all_ok,
        "mode": str(mode),
        "baseline_kind": baseline.kind,
        "requested_stages": stages,
        "stage_results": results,
        "manifest": str(validation_root / "experiment_manifest.json"),
    }
