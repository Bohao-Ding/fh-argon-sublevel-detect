from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

from . import main_pipeline, model, paths
from .validation_bootstrap import centered_residual_pools, circular_moving_block_sample
from .validation_common import (
    Baseline,
    atomic_json_dump,
    canonical_hash,
    sha256_file,
    wilson_interval,
    write_progress,
    write_stage_status,
)


K4_WEIGHTS = (0.359396, 0.396105, 0.086224, 0.158275)


@dataclass(frozen=True)
class ChannelTruth:
    n_levels: int
    energies: tuple[float, ...]
    weights: tuple[float, ...]


@dataclass(frozen=True)
class SyntheticUnit:
    truth_k: int
    delta: float | None
    noise_scale: float
    replicate: int
    train_seed: int
    prior_enabled: bool

    @property
    def cell_id(self) -> str:
        delta = "none" if self.delta is None else str(self.delta).replace(".", "p")
        noise = str(self.noise_scale).replace(".", "p")
        prior = "prior_on" if self.prior_enabled else "prior_off"
        return f"k{self.truth_k}_delta_{delta}_noise_{noise}_{prior}"

    @property
    def unit_id(self) -> str:
        return f"{self.cell_id}_replicate_{self.replicate:03d}_seed_{self.train_seed}"


def truth_channels(
    n_levels: int,
    *,
    delta: float | None,
    channel_table: pd.DataFrame,
) -> ChannelTruth:
    n_levels = int(n_levels)
    if n_levels == 1:
        return ChannelTruth(1, (11.5,), (1.0,))
    if n_levels == 2:
        if delta not in {0.25, 0.5, 1.0}:
            raise ValueError("K=2 requires delta in {0.25, 0.5, 1.0}")
        return ChannelTruth(2, (11.5, 11.5 + float(delta)), (0.5, 0.5))
    if n_levels == 4:
        if delta not in {0.25, 0.5, 1.0}:
            raise ValueError("K=4 requires delta in {0.25, 0.5, 1.0}")
        weights = np.asarray(K4_WEIGHTS, dtype=np.float64)
        weights = weights / np.sum(weights)
        return ChannelTruth(
            4,
            (11.5, 11.5 + float(delta), 12.594, 13.965),
            tuple(float(value) for value in weights),
        )
    if n_levels == 8:
        required = {"k", "channel", "energy_v", "weight"}
        if not required.issubset(channel_table.columns):
            raise ValueError(f"K=8 channel table lacks columns: {sorted(required - set(channel_table.columns))}")
        rows = channel_table[channel_table["k"].astype(int) == 8].sort_values("channel")
        if len(rows) != 8:
            raise ValueError(f"Expected 8 packaged K=8 channel rows, found {len(rows)}")
        energies = tuple(float(value) for value in rows["energy_v"])
        weights_array = rows["weight"].to_numpy(dtype=np.float64)
        if np.any(weights_array < 0.0) or float(np.sum(weights_array)) <= 0.0:
            raise ValueError("K=8 channel weights must be non-negative with positive sum")
        weights_array = weights_array / np.sum(weights_array)
        return ChannelTruth(8, energies, tuple(float(value) for value in weights_array))
    raise ValueError(f"Unsupported synthetic truth K={n_levels}")


def apply_channel_truth(
    network: model.PoissonRateFHCoreMultiLevel,
    truth: ChannelTruth,
) -> None:
    if int(network.n_levels) != int(truth.n_levels):
        raise ValueError(
            f"Network/truth level mismatch: {network.n_levels} != {truth.n_levels}"
        )
    energies = np.asarray(truth.energies, dtype=np.float64)
    weights = np.asarray(truth.weights, dtype=np.float64)
    if energies.size != truth.n_levels or weights.size != truth.n_levels:
        raise ValueError("Synthetic truth channel lengths do not match n_levels")
    if np.any(np.diff(energies) <= float(network.min_level_gap)):
        raise ValueError("Synthetic truth gaps must exceed the model minimum level gap")
    if np.any(weights <= 0.0) or not np.isclose(np.sum(weights), 1.0, atol=1e-8):
        raise ValueError("Synthetic truth weights must be positive and normalized")
    raw_e1 = model.inv_bounded(float(energies[0]), 9.0, 14.5)
    gaps = np.diff(energies) - float(network.min_level_gap)
    raw_gaps = np.log(np.expm1(gaps)) if gaps.size else np.asarray([], dtype=np.float64)
    logits = np.log(weights)
    with torch.no_grad():
        network.raw_E1.copy_(
            torch.as_tensor(raw_e1, dtype=network.raw_E1.dtype, device=network.raw_E1.device)
        )
        if gaps.size:
            network.raw_dE.copy_(
                torch.as_tensor(raw_gaps, dtype=network.raw_dE.dtype, device=network.raw_dE.device)
            )
        network.raw_level_logits.copy_(
            torch.as_tensor(logits, dtype=network.raw_level_logits.dtype, device=network.raw_level_logits.device)
        )


def _screening_cells() -> list[tuple[int, float | None, float]]:
    cells: list[tuple[int, float | None, float]] = []
    for truth_k in (1, 8):
        for noise in (0.5, 1.0, 1.5):
            cells.append((truth_k, None, noise))
    for truth_k in (2, 4):
        for delta in (0.25, 0.5, 1.0):
            for noise in (0.5, 1.0, 1.5):
                cells.append((truth_k, delta, noise))
    return cells


def _confirmatory_cells() -> tuple[tuple[int, float | None, float], ...]:
    return (
        (1, None, 1.0),
        (2, 0.25, 1.0),
        (4, 0.25, 1.0),
        (4, 0.5, 1.0),
        (8, None, 1.0),
    )


def synthetic_units(mode: str) -> list[SyntheticUnit]:
    if str(mode) == "smoke":
        return [SyntheticUnit(4, 0.25, 1.0, replicate=0, train_seed=0, prior_enabled=True)]
    units: list[SyntheticUnit] = []
    for truth_k, delta, noise in _screening_cells():
        units.extend(
            SyntheticUnit(truth_k, delta, noise, replicate, 0, True)
            for replicate in range(3)
        )
    for truth_k, delta, noise in _confirmatory_cells():
        units.extend(
            SyntheticUnit(truth_k, delta, noise, replicate, 0, True)
            for replicate in range(3, 30)
        )
        units.extend(
            SyntheticUnit(truth_k, delta, noise, replicate, seed, True)
            for replicate in range(5)
            for seed in (1, 2)
        )
    units.extend(
        SyntheticUnit(4, 0.25, 1.0, replicate, 0, False)
        for replicate in range(30)
    )
    units.extend(
        SyntheticUnit(4, 0.25, 1.0, replicate, seed, False)
        for replicate in range(5)
        for seed in (1, 2)
    )
    return units


def choose_epochs(rows: Sequence[Mapping[str, Any]]) -> int:
    by_k = {int(row["n_levels"]): row for row in rows}
    if set(by_k) != set(range(1, 9)):
        return 3500
    selected_consistent = all(
        int(row["selected_k_1600"]) == int(row["selected_k_3500"])
        for row in by_k.values()
    )
    rmse_consistent = all(
        abs(float(row["rmse_1600"]) - float(row["rmse_3500"]))
        / max(abs(float(row["rmse_3500"])), 1e-12)
        < 0.01
        for row in by_k.values()
    )
    return 1600 if selected_consistent and rmse_consistent else 3500


def match_channels(truth: ChannelTruth, estimate: ChannelTruth) -> dict[str, Any]:
    truth_energy = np.asarray(truth.energies, dtype=np.float64)
    estimate_energy = np.asarray(estimate.energies, dtype=np.float64)
    truth_weight = np.asarray(truth.weights, dtype=np.float64)
    estimate_weight = np.asarray(estimate.weights, dtype=np.float64)
    cost = np.abs(truth_energy[:, None] - estimate_energy[None, :])
    truth_indices, estimate_indices = linear_sum_assignment(cost)
    energy_error = estimate_energy[estimate_indices] - truth_energy[truth_indices]
    weight_error = estimate_weight[estimate_indices] - truth_weight[truth_indices]
    pairs = [
        {
            "truth_channel": int(truth_index + 1),
            "estimated_channel": int(estimate_index + 1),
            "truth_energy_v": float(truth_energy[truth_index]),
            "estimated_energy_v": float(estimate_energy[estimate_index]),
            "truth_weight": float(truth_weight[truth_index]),
            "estimated_weight": float(estimate_weight[estimate_index]),
        }
        for truth_index, estimate_index in zip(truth_indices, estimate_indices)
    ]
    return {
        "matched_count": len(pairs),
        "energy_rmse_v": float(np.sqrt(np.mean(np.square(energy_error)))) if len(pairs) else float("nan"),
        "weight_mae": float(np.mean(np.abs(weight_error))) if len(pairs) else float("nan"),
        "pairs": pairs,
    }


def summarize_recovery(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_total: int,
) -> dict[str, Any]:
    total = int(expected_total)
    completed = [row for row in rows if str(row.get("status")) == "complete"]
    exact = sum(int(row.get("selected_k", -1) or -1) == int(row["truth_k"]) for row in completed)
    over = sum(int(row.get("selected_k", -1) or -1) > int(row["truth_k"]) for row in completed)
    under = sum(0 < int(row.get("selected_k", -1) or -1) < int(row["truth_k"]) for row in completed)
    explicit_failures = sum(str(row.get("status")) != "complete" for row in rows)
    failed = explicit_failures + max(0, total - len(rows))
    low, high = wilson_interval(exact, total)
    return {
        "total": total,
        "completed": len(completed),
        "failed": failed,
        "exact_k_count": exact,
        "exact_k_fraction": float(exact / total),
        "over_selected_count": over,
        "under_selected_count": under,
        "wilson_low": float(low),
        "wilson_high": float(high),
        "reliable_recovery": bool(low >= 0.80),
    }


def _baseline_config(baseline: Baseline) -> model.Config:
    payload = json.loads(baseline.files["config"].read_text(encoding="utf-8-sig"))
    allowed = {field.name for field in fields(model.Config)}
    return model.Config(**{key: value for key, value in payload.items() if key in allowed})


def _channel_table(baseline: Baseline) -> pd.DataFrame:
    if "channel_parameters" in baseline.files:
        return pd.read_csv(baseline.files["channel_parameters"])
    candidates = list(
        (baseline.root / "main" / "fullscan" / "levels" / "L08" / "seeds").glob(
            "seed_*/full/scorecard.json"
        )
    )
    scorecards = []
    for path in candidates:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        rmse = float((payload.get("metrics_curve") or {}).get("rmse_mean", float("inf")))
        scorecards.append((rmse, payload))
    if not scorecards:
        raise FileNotFoundError("Local baseline has no packaged channel table or K=8 scorecard")
    payload = min(scorecards, key=lambda item: item[0])[1]
    params = payload.get("params") or {}
    return pd.DataFrame(
        [
            {
                "k": 8,
                "channel": channel,
                "energy_v": params[f"level_energy_{channel:02d}"],
                "weight": params[f"level_weight_{channel:02d}"],
            }
            for channel in range(1, 9)
        ]
    )


def _load_production_state(
    *,
    baseline: Baseline,
    input_path: str | Path,
    device: str,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], model.Config, torch.device]:
    cfg = _baseline_config(baseline)
    cfg.data_path = str(paths.resolve_project_path(input_path))
    cfg.exclude_vr_values = ""
    curves, init = model.load_curves_and_init(cfg)
    torch_device = model.resolve_device(device)
    production = model.PoissonRateFHCoreMultiLevel(
        n_curves=len(curves),
        n_max=int(cfg.n_max),
        n_levels=4,
        min_level_gap=float(cfg.min_level_gap),
        device=torch_device,
        V_exc_init=float(init["V_exc_init"]),
        init_jitter_scale=float(cfg.init_jitter_scale),
        init_seed=0,
    ).to(torch_device)
    payload = torch.load(
        baseline.files["selected_checkpoint"],
        map_location=torch_device,
        weights_only=False,
    )
    state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
    production.load_state_dict(state, strict=True)
    return production.state_dict(), curves, cfg, torch_device


def _generator_for_truth(
    *,
    production_state: Mapping[str, torch.Tensor],
    curves: Sequence[Mapping[str, Any]],
    cfg: model.Config,
    truth: ChannelTruth,
    device: torch.device,
) -> model.PoissonRateFHCoreMultiLevel:
    generator = model.PoissonRateFHCoreMultiLevel(
        n_curves=len(curves),
        n_max=int(cfg.n_max),
        n_levels=truth.n_levels,
        min_level_gap=float(cfg.min_level_gap),
        device=device,
        V_exc_init=float(truth.energies[0]),
    ).to(device)
    shared = {
        key: value
        for key, value in production_state.items()
        if key not in {"raw_E1", "raw_dE", "raw_level_logits"}
    }
    missing, unexpected = generator.load_state_dict(shared, strict=False)
    if set(missing) != {"raw_E1", "raw_dE", "raw_level_logits"} or unexpected:
        raise ValueError(f"Unexpected shared-state mismatch: missing={missing}, unexpected={unexpected}")
    apply_channel_truth(generator, truth)
    generator.eval()
    return generator


def _input_columns(frame: pd.DataFrame) -> dict[str, str]:
    return {
        "curve": model.pick_col(frame, ("curve_id", "curveID", "curve", "id")),
        "vr": model.pick_col(frame, ("Vr", "vr", "V_r", "Ur")),
        "va": model.pick_col(frame, ("Va", "va", "V_a", "Ua")),
        "ip": model.pick_col(frame, ("IuA", "Ip", "ip", "I", "I_meas", "current", "Ip_uA", "I_uA")),
    }


def prepare_synthetic_inputs(
    *,
    mode: str,
    units: Sequence[SyntheticUnit],
    input_path: str | Path,
    baseline: Baseline,
    output_dir: str | Path,
    device: str,
) -> dict[tuple[int, float | None, float, int], dict[str, Any]]:
    source = model.read_excel_or_csv(paths.resolve_project_path(input_path))
    columns = _input_columns(source)
    production_points = pd.read_csv(baseline.files["prediction_points"])
    residual_pools = centered_residual_pools(production_points)
    production_state, curves, cfg, torch_device = _load_production_state(
        baseline=baseline,
        input_path=input_path,
        device=device,
    )
    channel_table = _channel_table(baseline)
    keys = sorted(
        {(unit.truth_k, unit.delta, unit.noise_scale, unit.replicate) for unit in units},
        key=lambda key: (key[0], -1.0 if key[1] is None else key[1], key[2], key[3]),
    )
    prepared: dict[tuple[int, float | None, float, int], dict[str, Any]] = {}
    generator_cache: dict[tuple[int, float | None], tuple[ChannelTruth, model.PoissonRateFHCoreMultiLevel]] = {}
    target_root = Path(output_dir) / "generated_inputs"
    source_curve_ids = source[columns["curve"]].astype(int).to_numpy()
    for truth_k, delta, noise_scale, replicate in keys:
        truth_key = (truth_k, delta)
        if truth_key not in generator_cache:
            truth = truth_channels(truth_k, delta=delta, channel_table=channel_table)
            generator_cache[truth_key] = (
                truth,
                _generator_for_truth(
                    production_state=production_state,
                    curves=curves,
                    cfg=cfg,
                    truth=truth,
                    device=torch_device,
                ),
            )
        truth, generator = generator_cache[truth_key]
        generated_points = pd.DataFrame(
            model.prediction_point_rows(generator, curves, torch_device, nuisance_mode="curve")
        )
        if len(generated_points) != len(source):
            raise ValueError("Synthetic generator/source row count mismatch")
        values = generated_points["predicted"].to_numpy(dtype=np.float64)
        rng_seed = int(
            canonical_hash(
                {"truth_k": truth_k, "delta": delta, "noise": noise_scale, "replicate": replicate}
            )[:16],
            16,
        )
        rng = np.random.default_rng(rng_seed)
        for curve_id in sorted(set(source_curve_ids.tolist())):
            mask = source_curve_ids == curve_id
            values[mask] += float(noise_scale) * circular_moving_block_sample(
                residual_pools[int(curve_id)],
                size=int(np.sum(mask)),
                block_length=7,
                rng=rng,
            )
        frame = source.copy()
        frame[columns["ip"]] = values
        delta_label = "none" if delta is None else str(delta).replace(".", "p")
        noise_label = str(noise_scale).replace(".", "p")
        path = target_root / f"k{truth_k}_delta_{delta_label}_noise_{noise_label}_replicate_{replicate:03d}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        prepared[(truth_k, delta, noise_scale, replicate)] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "row_count": len(frame),
            "rng_seed": rng_seed,
            "truth": truth,
        }
    return prepared


def _fit_config(
    *,
    mode: str,
    baseline: Baseline,
    input_path: str | Path,
    out_dir: Path,
    device: str,
    seed: int,
    epochs: int,
    prior_enabled: bool,
) -> model.Config:
    cfg = _baseline_config(baseline)
    if str(mode) == "smoke":
        for key, value in main_pipeline.profile_defaults("smoke").items():
            setattr(cfg, key, value)
    cfg.data_path = str(input_path)
    cfg.out_dir = str(out_dir)
    cfg.profile = f"validation_synthetic_{mode}"
    cfg.epochs = int(epochs)
    cfg.scan_seeds = str(int(seed))
    cfg.level_scan_min = 1
    cfg.level_scan_max = 2 if str(mode) == "smoke" else 8
    cfg.hyperopt_enabled = False
    cfg.device = str(device)
    cfg.selected_device = "cuda" if str(device) == "cuda" and model.torch.cuda.is_available() else "cpu"
    if cfg.selected_device == "cuda":
        cfg.cuda_workers = 4
        cfg.dispatch_strategy = "cuda_4"
    else:
        cfg.dispatch_strategy = "cpu_4" if cfg.cpu_workers > 1 else "single"
    cfg.resume_mode = "auto" if str(mode) == "fullscan" else "off"
    cfg.launch_monitor = False
    cfg.forward_evidence_path = str(out_dir / "forward_evidence.json")
    if not prior_enabled:
        cfg.forward_prior_mode = "off"
        cfg.w_prior_anchor = 0.0
        cfg.w_prior_gap = 0.0
    return cfg


def estimated_channels(*, scan_dir: str | Path, selected_k: int, seed: int) -> ChannelTruth:
    scorecard = (
        Path(scan_dir)
        / "levels"
        / f"L{int(selected_k):02d}"
        / "seeds"
        / f"seed_{int(seed):03d}"
        / "full"
        / "scorecard.json"
    )
    if not scorecard.is_file():
        raise FileNotFoundError(f"Missing selected synthetic scorecard: {scorecard}")
    payload = json.loads(scorecard.read_text(encoding="utf-8-sig"))
    params = payload.get("params") or {}
    return ChannelTruth(
        int(selected_k),
        tuple(float(params[f"level_energy_{channel:02d}"]) for channel in range(1, int(selected_k) + 1)),
        tuple(float(params[f"level_weight_{channel:02d}"]) for channel in range(1, int(selected_k) + 1)),
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(row) for row in rows]).to_csv(path, index=False)


def _run_epoch_gate(
    *,
    baseline: Baseline,
    prepared: Mapping[tuple[int, float | None, float, int], Mapping[str, Any]],
    output_dir: Path,
    device: str,
) -> tuple[int, list[dict[str, Any]]]:
    representative = prepared[(4, 0.25, 1.0, 0)]
    decisions: dict[int, dict[str, Any]] = {}
    forward = json.loads(baseline.files["forward_evidence"].read_text(encoding="utf-8-sig"))
    for epochs in (800, 1600, 3500):
        scan_dir = output_dir / "epoch_gate" / f"epochs_{epochs}" / "fullscan"
        cfg = _fit_config(
            mode="fullscan",
            baseline=baseline,
            input_path=representative["path"],
            out_dir=scan_dir,
            device=device,
            seed=0,
            epochs=epochs,
            prior_enabled=True,
        )
        atomic_json_dump(forward, scan_dir / "forward_evidence.json")
        decisions[epochs] = model.run_level_scan(cfg)["decision"]
    rows: list[dict[str, Any]] = []
    scored_1600 = {int(row["n_levels"]): row for row in decisions[1600]["scored_levels"]}
    scored_3500 = {int(row["n_levels"]): row for row in decisions[3500]["scored_levels"]}
    for n_levels in range(1, 9):
        rows.append(
            {
                "n_levels": n_levels,
                "rmse_1600": float(scored_1600[n_levels]["rmse_mean"]),
                "rmse_3500": float(scored_3500[n_levels]["rmse_mean"]),
                "selected_k_1600": int(decisions[1600]["selected_k"]),
                "selected_k_3500": int(decisions[3500]["selected_k"]),
                "rmse_relative_difference": abs(
                    float(scored_1600[n_levels]["rmse_mean"])
                    - float(scored_3500[n_levels]["rmse_mean"])
                )
                / max(abs(float(scored_3500[n_levels]["rmse_mean"])), 1e-12),
            }
        )
    return choose_epochs(rows), rows


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
    units = synthetic_units(mode)
    prepared = prepare_synthetic_inputs(
        mode=mode,
        units=units,
        input_path=input_path,
        baseline=baseline,
        output_dir=target,
        device=device,
    )
    truth_manifest = [
        {
            "truth_k": key[0],
            "delta": key[1],
            "noise_scale": key[2],
            "replicate": key[3],
            "path": value["path"],
            "sha256": value["sha256"],
            "energies": json.dumps(value["truth"].energies),
            "weights": json.dumps(value["truth"].weights),
        }
        for key, value in prepared.items()
    ]
    _write_csv(target / "truth_manifest.csv", truth_manifest)
    if str(mode) == "smoke":
        selected_epochs = 2
        gate_rows: list[dict[str, Any]] = []
    else:
        selected_epochs, gate_rows = _run_epoch_gate(
            baseline=baseline,
            prepared=prepared,
            output_dir=target,
            device=device,
        )
    _write_csv(target / "epoch_gate.csv", gate_rows)
    atomic_json_dump({"selected_epochs": selected_epochs, "rows": gate_rows}, target / "epoch_gate.json")
    forward = json.loads(baseline.files["forward_evidence"].read_text(encoding="utf-8-sig"))
    selection_rows: list[dict[str, Any]] = []
    matching_rows: list[dict[str, Any]] = []
    completed = 0
    failed = 0
    write_progress(validation / "progress.json", completed=0, failed=0, total=len(units), current_stage="synthetic")
    for unit in units:
        prepared_row = prepared[(unit.truth_k, unit.delta, unit.noise_scale, unit.replicate)]
        truth = prepared_row["truth"]
        scan_dir = target / "fits" / unit.unit_id / "fullscan"
        cfg = _fit_config(
            mode=mode,
            baseline=baseline,
            input_path=prepared_row["path"],
            out_dir=scan_dir,
            device=device,
            seed=unit.train_seed,
            epochs=selected_epochs,
            prior_enabled=unit.prior_enabled,
        )
        scan_dir.mkdir(parents=True, exist_ok=True)
        atomic_json_dump(forward, scan_dir / "forward_evidence.json")
        row: dict[str, Any] = {
            **asdict(unit),
            "unit_id": unit.unit_id,
            "input_sha256": prepared_row["sha256"],
            "epochs": selected_epochs,
            "status": "failed",
            "selected_k": None,
            "error": "",
        }
        try:
            decision = model.run_level_scan(cfg)["decision"]
            selected_k = int(decision["selected_k"])
            estimate = estimated_channels(scan_dir=scan_dir, selected_k=selected_k, seed=unit.train_seed)
            matching = match_channels(truth, estimate)
            pair_by_truth = {int(pair["truth_channel"]): pair for pair in matching["pairs"]}
            near_pair_recovered = False
            if truth.n_levels in {2, 4} and {1, 2}.issubset(pair_by_truth):
                near_pair_recovered = all(
                    abs(pair_by_truth[channel]["estimated_energy_v"] - pair_by_truth[channel]["truth_energy_v"])
                    <= 0.125
                    for channel in (1, 2)
                )
            row.update(
                status="complete",
                selected_k=selected_k,
                exact_k=selected_k == truth.n_levels,
                energy_rmse_v=matching["energy_rmse_v"],
                weight_mae=matching["weight_mae"],
                near_pair_recovered=near_pair_recovered,
            )
            for pair in matching["pairs"]:
                matching_rows.append({**asdict(unit), "selected_k": selected_k, **pair})
            completed += 1
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            failed += 1
        selection_rows.append(row)
        _write_csv(target / "selection_rows.csv", selection_rows)
        write_progress(
            validation / "progress.json",
            completed=completed,
            failed=failed,
            total=len(units),
            current_stage="synthetic",
            last_unit=unit.unit_id,
        )
    grouped: list[dict[str, Any]] = []
    frame = pd.DataFrame(selection_rows)
    group_columns = ["truth_k", "delta", "noise_scale", "prior_enabled"]
    for keys, subset in frame.groupby(group_columns, dropna=False):
        expected_total = len(subset)
        recovery = summarize_recovery(subset.to_dict("records"), expected_total=expected_total)
        truth_k, delta, noise_scale, prior_enabled = keys
        grouped.append(
            {
                "truth_k": int(truth_k),
                "delta": None if pd.isna(delta) else float(delta),
                "noise_scale": float(noise_scale),
                "prior_enabled": bool(prior_enabled),
                **recovery,
                "energy_rmse_mean_v": float(subset["energy_rmse_v"].dropna().mean()) if "energy_rmse_v" in subset else float("nan"),
                "weight_mae_mean": float(subset["weight_mae"].dropna().mean()) if "weight_mae" in subset else float("nan"),
                "near_pair_recovery_fraction": float(subset["near_pair_recovered"].fillna(False).mean()) if "near_pair_recovered" in subset else float("nan"),
            }
        )
    status = "incomplete" if failed else ("smoke_passed" if str(mode) == "smoke" else "complete")
    summary = {
        "ok": failed == 0,
        "status": status,
        "unit_total": len(units),
        "completed": completed,
        "failed": failed,
        "selected_epochs": selected_epochs,
        "near_pair_energy_tolerance_v": 0.125,
        "recovery_cells": grouped,
    }
    _write_csv(target / "selection_rows.csv", selection_rows)
    _write_csv(target / "channel_matching.csv", matching_rows)
    _write_csv(target / "recovery_summary.csv", grouped)
    atomic_json_dump(summary, target / "synthetic_summary.json")
    write_stage_status(
        target / "stage_status.json",
        status,
        unit_total=len(units),
        completed=completed,
        failed=failed,
        selected_epochs=selected_epochs,
    )
    return summary
