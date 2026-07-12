from __future__ import annotations

import math
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import main_pipeline, model, paths
from .validation_common import (
    Baseline,
    atomic_json_dump,
    canonical_hash,
    sha256_file,
    wilson_interval,
    write_progress,
    write_stage_status,
)


@dataclass(frozen=True)
class BootstrapUnit:
    block_length: int
    replicate: int
    train_seed: int

    @property
    def data_id(self) -> str:
        return f"block_{self.block_length:02d}_replicate_{self.replicate:03d}"

    @property
    def unit_id(self) -> str:
        return f"{self.data_id}_seed_{self.train_seed}"


def bootstrap_units(mode: str) -> list[BootstrapUnit]:
    if str(mode) == "smoke":
        return [BootstrapUnit(block_length=7, replicate=0, train_seed=0)]
    units = [BootstrapUnit(7, replicate, 0) for replicate in range(30)]
    units.extend(BootstrapUnit(7, replicate, seed) for replicate in range(5) for seed in (1, 2))
    units.extend(BootstrapUnit(5, replicate, 0) for replicate in range(10))
    units.extend(BootstrapUnit(9, replicate, 0) for replicate in range(10))
    return units


def circular_moving_block_sample(
    pool: np.ndarray,
    *,
    size: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.asarray(pool, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Residual pool must be a non-empty one-dimensional array")
    if int(size) < 0 or int(block_length) <= 0:
        raise ValueError("Sample size must be non-negative and block length must be positive")
    if int(size) == 0:
        return np.asarray([], dtype=np.float64)
    block_count = math.ceil(int(size) / int(block_length))
    starts = rng.integers(0, values.size, size=block_count)
    offsets = np.arange(int(block_length), dtype=int)
    blocks = [values[(int(start) + offsets) % values.size] for start in starts]
    return np.concatenate(blocks)[: int(size)]


def centered_residual_pools(prediction_points: pd.DataFrame) -> dict[int, np.ndarray]:
    required = {"curve_id"}
    if not required.issubset(prediction_points.columns):
        raise ValueError("Prediction points must contain curve_id")
    if "residual" in prediction_points.columns:
        residual = prediction_points["residual"].astype(float)
    elif {"observed", "predicted"}.issubset(prediction_points.columns):
        residual = prediction_points["observed"].astype(float) - prediction_points["predicted"].astype(float)
    else:
        raise ValueError("Prediction points must contain residual or observed/predicted")
    frame = prediction_points.copy()
    frame["_residual"] = residual
    pools: dict[int, np.ndarray] = {}
    for curve_id, subset in frame.groupby("curve_id", sort=True):
        values = subset["_residual"].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError(f"Curve {curve_id} has no finite residuals")
        pools[int(curve_id)] = values - float(np.mean(values))
    return pools


def _input_columns(frame: pd.DataFrame) -> dict[str, str]:
    return {
        "curve": model.pick_col(frame, ("curve_id", "curveID", "curve", "id")),
        "vr": model.pick_col(frame, ("Vr", "vr", "V_r", "Ur")),
        "va": model.pick_col(frame, ("Va", "va", "V_a", "Ua")),
        "ip": model.pick_col(frame, ("IuA", "Ip", "ip", "I", "I_meas", "current", "Ip_uA", "I_uA")),
    }


def _validate_prediction_alignment(source: pd.DataFrame, points: pd.DataFrame, columns: Mapping[str, str]) -> None:
    if len(source) != len(points):
        raise ValueError(f"Source/prediction row count mismatch: {len(source)} != {len(points)}")
    source_curve = source[columns["curve"]].astype(int).to_numpy()
    point_curve = points["curve_id"].astype(int).to_numpy()
    if not np.array_equal(source_curve, point_curve):
        raise ValueError("Source and prediction curve order differ")
    if "Va" in points.columns and not np.allclose(
        source[columns["va"]].astype(float).to_numpy(), points["Va"].astype(float).to_numpy(), atol=1e-9
    ):
        raise ValueError("Source and prediction Va order differ")
    if "Vr" in points.columns and not np.allclose(
        source[columns["vr"]].astype(float).to_numpy(), points["Vr"].astype(float).to_numpy(), atol=1e-9
    ):
        raise ValueError("Source and prediction Vr order differ")


def write_bootstrap_input(
    *,
    source: pd.DataFrame,
    prediction_points: pd.DataFrame,
    unit: BootstrapUnit,
    output_path: str | Path,
    noise_scale: float = 1.0,
) -> dict[str, Any]:
    frame = source.copy()
    points = prediction_points.reset_index(drop=True).copy()
    columns = _input_columns(frame)
    _validate_prediction_alignment(frame.reset_index(drop=True), points, columns)
    if "predicted" not in points.columns:
        raise ValueError("Prediction points must contain predicted")
    pools = centered_residual_pools(points)
    curve_ids = frame[columns["curve"]].astype(int).to_numpy()
    generated = points["predicted"].astype(float).to_numpy(dtype=np.float64).copy()
    rng = np.random.default_rng(int(unit.block_length) * 100_000 + int(unit.replicate))
    for curve_id in sorted(set(curve_ids.tolist())):
        mask = curve_ids == curve_id
        if int(curve_id) not in pools:
            raise ValueError(f"Missing centered residual pool for curve_id={curve_id}")
        noise = circular_moving_block_sample(
            pools[int(curve_id)],
            size=int(np.sum(mask)),
            block_length=int(unit.block_length),
            rng=rng,
        )
        generated[mask] += float(noise_scale) * noise
    frame[columns["ip"]] = generated
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return {
        "data_id": unit.data_id,
        "block_length": int(unit.block_length),
        "replicate": int(unit.replicate),
        "rng_seed": int(unit.block_length) * 100_000 + int(unit.replicate),
        "noise_scale": float(noise_scale),
        "row_count": int(len(frame)),
        "curve_count": int(len(set(curve_ids.tolist()))),
        "sha256": sha256_file(target),
        "path": str(target),
    }


def summarize_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_primary_total: int = 30,
) -> dict[str, Any]:
    primary = [
        row for row in rows if int(row["block_length"]) == 7 and int(row["train_seed"]) == 0
    ]
    completed = [row for row in primary if str(row.get("status")) == "complete"]
    k4_count = sum(int(row.get("selected_k", -1) or -1) == 4 for row in completed)
    explicit_failures = sum(str(row.get("status")) != "complete" for row in primary)
    missing = max(0, int(expected_primary_total) - len(primary))
    total = int(expected_primary_total)
    low, high = wilson_interval(k4_count, total)
    distribution: dict[str, int] = {}
    for row in completed:
        key = str(int(row["selected_k"]))
        distribution[key] = distribution.get(key, 0) + 1
    return {
        "primary": {
            "block_length": 7,
            "train_seed": 0,
            "total": total,
            "completed": len(completed),
            "failed": explicit_failures + missing,
            "k4_count": int(k4_count),
            "k4_fraction": float(k4_count / total),
            "wilson_low": float(low),
            "wilson_high": float(high),
            "selected_k_counts": distribution,
        }
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(row) for row in rows]).to_csv(path, index=False)


def _baseline_config(baseline: Baseline) -> model.Config:
    payload = json.loads(baseline.files["config"].read_text(encoding="utf-8-sig"))
    allowed = {field.name for field in fields(model.Config)}
    return model.Config(**{key: value for key, value in payload.items() if key in allowed})


def _unit_config(
    *,
    mode: str,
    baseline: Baseline,
    input_path: Path,
    out_dir: Path,
    device: str,
    train_seed: int,
) -> model.Config:
    cfg = _baseline_config(baseline)
    if str(mode) == "smoke":
        for key, value in main_pipeline.profile_defaults("smoke").items():
            setattr(cfg, key, value)
    cfg.data_path = str(input_path)
    cfg.out_dir = str(out_dir)
    cfg.profile = f"validation_bootstrap_{mode}"
    cfg.device = str(device)
    cfg.selected_device = "cuda" if str(device) == "cuda" and model.torch.cuda.is_available() else "cpu"
    if cfg.selected_device == "cuda":
        cfg.cuda_workers = 4
        cfg.dispatch_strategy = "cuda_4"
    else:
        cfg.dispatch_strategy = "cpu_4" if cfg.cpu_workers > 1 else "single"
    cfg.seed = int(train_seed)
    cfg.scan_seeds = str(int(train_seed))
    cfg.hyperopt_enabled = False
    cfg.resume_mode = "auto" if str(mode) == "fullscan" else "off"
    cfg.launch_monitor = False
    cfg.forward_evidence_path = str(out_dir / "forward_evidence.json")
    return cfg


def _condition_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    block_length: int,
    expected_total: int,
) -> dict[str, Any]:
    subset = [
        row for row in rows if int(row["block_length"]) == int(block_length) and int(row["train_seed"]) == 0
    ]
    completed = [row for row in subset if str(row.get("status")) == "complete"]
    failed = sum(str(row.get("status")) != "complete" for row in subset) + max(0, expected_total - len(subset))
    counts = {k: sum(int(row.get("selected_k", -1) or -1) == k for row in completed) for k in range(1, 9)}
    return {
        "block_length": int(block_length),
        "total": int(expected_total),
        "completed": len(completed),
        "failed": int(failed),
        **{f"k{k}_count": int(counts[k]) for k in range(1, 9)},
    }


def run(
    *,
    mode: str,
    input_path: str | Path,
    baseline: Baseline,
    output_dir: str | Path,
    validation_root: str | Path,
    device: str,
) -> dict[str, Any]:
    target = Path(output_dir)
    validation = Path(validation_root)
    units = bootstrap_units(mode)
    source = model.read_excel_or_csv(paths.resolve_project_path(input_path))
    prediction_points = pd.read_csv(baseline.files["prediction_points"])
    unique_data = {(unit.block_length, unit.replicate): unit for unit in units}
    data_manifest: list[dict[str, Any]] = []
    data_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for key, unit in sorted(unique_data.items()):
        path = target / "perturbed_inputs" / f"{unit.data_id}.csv"
        metadata = write_bootstrap_input(
            source=source,
            prediction_points=prediction_points,
            unit=unit,
            output_path=path,
            noise_scale=1.0,
        )
        data_manifest.append(metadata)
        data_by_key[key] = metadata
    _write_csv(target / "replicate_manifest.csv", data_manifest)

    selection_rows: list[dict[str, Any]] = []
    write_progress(validation / "progress.json", completed=0, failed=0, total=len(units), current_stage="bootstrap")
    completed_count = 0
    failed_count = 0
    forward_evidence = json.loads(baseline.files["forward_evidence"].read_text(encoding="utf-8-sig"))
    for unit in units:
        metadata = data_by_key[(unit.block_length, unit.replicate)]
        unit_root = target / "fits" / unit.unit_id
        scan_dir = unit_root / "fullscan"
        scan_dir.mkdir(parents=True, exist_ok=True)
        cfg = _unit_config(
            mode=mode,
            baseline=baseline,
            input_path=Path(metadata["path"]),
            out_dir=scan_dir,
            device=device,
            train_seed=unit.train_seed,
        )
        atomic_json_dump(forward_evidence, scan_dir / "forward_evidence.json")
        identity = {
            "baseline_sha256": baseline.identity_hash,
            "input_sha256": metadata["sha256"],
            "unit": asdict(unit),
            "config_sha256": canonical_hash(asdict(cfg)),
        }
        unit_manifest_path = unit_root / "unit_manifest.json"
        result_path = unit_root / "result.json"
        if unit_manifest_path.exists():
            existing_manifest = json.loads(unit_manifest_path.read_text(encoding="utf-8-sig"))
            if existing_manifest.get("identity") != identity:
                raise ValueError(
                    f"Bootstrap unit identity mismatch; refusing to overwrite {unit_root}"
                )
        elif result_path.exists():
            raise ValueError(f"Bootstrap result exists without unit manifest: {result_path}")
        else:
            atomic_json_dump({"identity": identity}, unit_manifest_path)
        row: dict[str, Any] = {
            **asdict(unit),
            "unit_id": unit.unit_id,
            "data_sha256": metadata["sha256"],
            "config_sha256": identity["config_sha256"],
            "scan_dir": str(scan_dir),
        }
        if result_path.exists():
            existing_result = json.loads(result_path.read_text(encoding="utf-8-sig"))
            if existing_result.get("identity") != identity:
                raise ValueError(
                    f"Bootstrap result identity mismatch; refusing to reuse {result_path}"
                )
            if isinstance(existing_result.get("decision"), dict):
                row.update(
                    status="complete",
                    selected_k=int(existing_result["decision"]["selected_k"]),
                    error="",
                    reused=True,
                )
                completed_count += 1
                selection_rows.append(row)
                _write_csv(target / "selection_rows.csv", selection_rows)
                write_progress(
                    validation / "progress.json",
                    completed=completed_count,
                    failed=failed_count,
                    total=len(units),
                    current_stage="bootstrap",
                    last_unit=unit.unit_id,
                )
                continue
        try:
            result = model.run_level_scan(cfg)
            decision = result["decision"]
            row.update(status="complete", selected_k=int(decision["selected_k"]), error="", reused=False)
            atomic_json_dump({"identity": identity, "decision": decision}, result_path)
            completed_count += 1
        except Exception as exc:
            row.update(status="failed", selected_k=None, error=f"{type(exc).__name__}: {exc}")
            atomic_json_dump({"identity": identity, "error": row["error"]}, result_path)
            failed_count += 1
        selection_rows.append(row)
        _write_csv(target / "selection_rows.csv", selection_rows)
        write_progress(
            validation / "progress.json",
            completed=completed_count,
            failed=failed_count,
            total=len(units),
            current_stage="bootstrap",
            last_unit=unit.unit_id,
        )

    expected = {7: 1} if str(mode) == "smoke" else {5: 10, 7: 30, 9: 10}
    block_summary = [
        _condition_summary(selection_rows, block_length=block, expected_total=total)
        for block, total in sorted(expected.items())
    ]
    seed_sensitivity = [
        row
        for row in selection_rows
        if int(row["block_length"]) == 7 and int(row["replicate"]) < 5
    ]
    summary = summarize_bootstrap(
        selection_rows,
        expected_primary_total=1 if str(mode) == "smoke" else 30,
    )
    status = (
        "incomplete"
        if failed_count
        else ("smoke_passed" if str(mode) == "smoke" else "complete")
    )
    summary.update(
        {
            "ok": failed_count == 0,
            "status": status,
            "unit_total": len(units),
            "completed": completed_count,
            "failed": failed_count,
            "baseline_prediction_sha256": baseline.hashes["prediction_points"],
            "block_length_summary": block_summary,
        }
    )
    _write_csv(target / "selection_rows.csv", selection_rows)
    _write_csv(target / "block_length_summary.csv", block_summary)
    _write_csv(target / "seed_sensitivity.csv", seed_sensitivity)
    atomic_json_dump(summary, target / "block_bootstrap_summary.json")
    write_stage_status(
        target / "stage_status.json",
        status,
        unit_total=len(units),
        completed=completed_count,
        failed=failed_count,
    )
    return summary
