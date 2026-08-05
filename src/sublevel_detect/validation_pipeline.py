from __future__ import annotations

import platform
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import (
    model,
    paths,
    validation_bootstrap,
    validation_h4s,
    validation_holdout,
    validation_selector,
)
from .validation_common import (
    atomic_json_dump,
    canonical_hash,
    resolve_baseline,
    sha256_file,
    write_progress,
)


VALIDATION_STAGES = ("selector", "bootstrap", "holdout", "h4s")

STAGE_DIR_NAMES = {
    "selector": "selector_audit",
    "bootstrap": "block_bootstrap",
    "holdout": "holdout",
    "h4s": "h4s_comparison",
}

STAGE_MODULES = {
    "selector": (validation_selector, model),
    "bootstrap": (validation_bootstrap, model),
    "holdout": (validation_holdout, model),
    "h4s": (validation_h4s, validation_holdout, model),
}


def requested_stages(stage: str | None) -> list[str]:
    if stage is None:
        return list(VALIDATION_STAGES)
    if stage not in VALIDATION_STAGES:
        raise ValueError(f"Unsupported validation stage: {stage}")
    return [stage]


def _stage_matrix(stage: str, mode: str) -> list[dict[str, Any]]:
    if stage == "selector":
        return [asdict(item) for item in validation_selector.selector_scenarios()]
    if stage == "bootstrap":
        return [asdict(item) for item in validation_bootstrap.bootstrap_units(mode)]
    if stage == "holdout":
        return [asdict(item) for item in validation_holdout.holdout_units(mode)]
    if stage == "h4s":
        return [asdict(item) for item in validation_h4s.h4s_units(mode)]
    raise ValueError(f"Unsupported validation stage: {stage}")


def _stage_identity(
    *,
    stage: str,
    mode: str,
    input_path: str | Path,
    baseline_hash: str,
    selected_device: str,
    validation_root: Path,
) -> dict[str, Any]:
    matrix = _stage_matrix(stage, mode)
    seeds = sorted({int(row[key]) for row in matrix for key in ("seed", "train_seed") if key in row})
    source_hashes = {
        Path(module.__file__).name: sha256_file(Path(module.__file__))
        for module in STAGE_MODULES[stage]
    }
    resolved_input = paths.resolve_project_path(input_path)
    return {
        "schema_version": 1,
        "stage": stage,
        "mode": str(mode),
        "baseline_sha256": baseline_hash,
        "input_sha256": sha256_file(resolved_input) if resolved_input.is_file() else "MISSING",
        "selected_device": selected_device,
        "matrix_sha256": canonical_hash(matrix),
        "seed_sha256": canonical_hash(seeds),
        "source_hashes": source_hashes,
        "dependency_hashes": {},
    }


def _write_validation_summary(
    *,
    validation_root: Path,
    mode: str,
    baseline_kind: str,
    stages: list[str],
    results: dict[str, Any],
    reused_stages: list[str],
) -> dict[str, Any]:
    ok = all(bool(results[stage].get("ok", False)) for stage in stages)
    payload = {
        "ok": ok,
        "mode": str(mode),
        "baseline_kind": baseline_kind,
        "requested_stages": stages,
        "reused_stages": reused_stages,
        "stage_results": results,
        "evidence": "Each stage result is linked to a stage manifest with baseline, input, matrix, seed, device, and source hashes.",
        "inference": "Only completed preregistered endpoints may be interpreted; stage direction is not a completion criterion.",
        "limitations": "Validation results remain conditional on the packaged data, frozen model family, priors, and computation budget.",
    }
    atomic_json_dump(payload, validation_root / "validation_summary.json")
    lines = [
        "# Supplementary Validation Summary",
        "",
        f"- Overall status: {'complete' if ok else 'incomplete'}",
        f"- Mode: `{mode}`",
        f"- Baseline: `{baseline_kind}`",
        f"- Requested stages: {', '.join(stages)}",
        f"- Reused stages: {', '.join(reused_stages) if reused_stages else 'none'}",
        "",
        "## Evidence",
        "",
        str(payload["evidence"]),
        "",
        "## Inference",
        "",
        str(payload["inference"]),
        "",
        "## Limitations",
        "",
        str(payload["limitations"]),
        "",
    ]
    (validation_root / "validation_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


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
        "preregistered_unit_counts": {
            stage: {
                "smoke": len(_stage_matrix(stage, "smoke")),
                "fullscan": len(_stage_matrix(stage, "fullscan")),
            }
            for stage in VALIDATION_STAGES
        },
    }
    atomic_json_dump(manifest, validation_root / "experiment_manifest.json")
    write_progress(validation_root / "progress.json", completed=0, failed=0, total=len(stages))
    results: dict[str, Any] = {}
    completed = 0
    all_ok = True
    reused_stages: list[str] = []
    for stage in stages:
        stage_dir = validation_root / STAGE_DIR_NAMES[stage]
        stage_manifest_path = stage_dir / "stage_manifest.json"
        stage_result_path = stage_dir / "stage_result.json"
        identity = _stage_identity(
            stage=stage,
            mode=str(mode),
            input_path=input_path,
            baseline_hash=baseline.identity_hash,
            selected_device=selected_device,
            validation_root=validation_root,
        )
        result: dict[str, Any] | None = None
        if stage_manifest_path.exists():
            existing_manifest = json.loads(stage_manifest_path.read_text(encoding="utf-8-sig"))
            if existing_manifest.get("identity") != identity:
                raise ValueError(
                    f"Validation stage identity mismatch; refusing to reuse or overwrite {stage_dir}"
                )
            if stage_result_path.exists():
                existing_result = json.loads(stage_result_path.read_text(encoding="utf-8-sig"))
                if bool(existing_result.get("ok", False)) and str(existing_result.get("status")) in {
                    "smoke_passed",
                    "complete",
                }:
                    result = existing_result
                    result["reused"] = True
                    reused_stages.append(stage)
        else:
            atomic_json_dump({"identity": identity}, stage_manifest_path)

        if result is None and stage == "selector":
            result = validation_selector.run(
                baseline=baseline,
                output_dir=stage_dir,
                mode=str(mode),
            )
        elif result is None and stage == "bootstrap":
            result = validation_bootstrap.run(
                mode=str(mode),
                input_path=input_path,
                baseline=baseline,
                output_dir=stage_dir,
                validation_root=validation_root,
                device=str(device),
            )
        elif result is None and stage == "holdout":
            result = validation_holdout.run(
                mode=str(mode),
                input_path=input_path,
                baseline=baseline,
                output_dir=stage_dir,
                validation_root=validation_root,
                device=str(device),
            )
        elif result is None and stage == "h4s":
            result = validation_h4s.run(
                mode=str(mode),
                input_path=input_path,
                baseline=baseline,
                output_dir=stage_dir,
                validation_root=validation_root,
                device=str(device),
            )
        elif result is None:
            raise NotImplementedError(f"Validation stage is not implemented yet: {stage}")
        if not result.get("reused", False):
            atomic_json_dump(result, stage_result_path)
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
            stage_failed = 0 if bool(result.get("ok", False)) else 1
            write_progress(
                validation_root / "progress.json",
                completed=completed - stage_failed,
                failed=stage_failed,
                total=len(stages),
                current_stage=stage,
            )
    _write_validation_summary(
        validation_root=validation_root,
        mode=str(mode),
        baseline_kind=baseline.kind,
        stages=stages,
        results=results,
        reused_stages=reused_stages,
    )
    return {
        "ok": all_ok,
        "mode": str(mode),
        "baseline_kind": baseline.kind,
        "requested_stages": stages,
        "reused_stages": reused_stages,
        "stage_results": results,
        "manifest": str(validation_root / "experiment_manifest.json"),
    }
