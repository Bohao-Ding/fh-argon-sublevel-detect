from __future__ import annotations

import concurrent.futures
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from . import main_pipeline, model, paths, validation_holdout
from .validation_common import (
    Baseline,
    atomic_json_dump,
    canonical_hash,
    sha256_file,
    write_progress,
    write_stage_status,
)


NIST_AR_I_4S_ENERGIES_EV = (
    11.54835442,
    11.62359272,
    11.72316039,
    11.82807116,
)
NIST_SOURCE_URL = (
    "https://physics.nist.gov/cgi-bin/ASD/energy1.pl?"
    "spectrum=Ar+I&units=1&level_out=on&conf_out=on&term_out=on&j_out=on"
)
COMMON_ENERGY_SHIFT_LIMIT_EV = 0.25
INITIALIZATION_JITTER_SCALE = 0.05
BACKGROUND_PARAMETER_NAMES = (
    "raw_high_energy_loss_strength",
    "raw_high_energy_loss_onset",
    "raw_high_energy_loss_width",
)


@dataclass(frozen=True)
class H4sHypothesis:
    key: str
    label: str
    nist_constrained: bool
    background_enabled: bool


HYPOTHESIS_SPECS = (
    H4sHypothesis("h1", "H1", False, False),
    H4sHypothesis("h1_background", "H1+B", False, True),
    H4sHypothesis("h4s", "H4s", True, False),
    H4sHypothesis("h4s_background", "H4s+B", True, True),
)
HYPOTHESIS_BY_KEY = {spec.key: spec for spec in HYPOTHESIS_SPECS}
H4S_HYPOTHESES = tuple(spec.key for spec in HYPOTHESIS_SPECS)

CONTRASTS = (
    ("h4s_minus_h1", "h4s", "h1"),
    (
        "h4s_background_minus_h1_background",
        "h4s_background",
        "h1_background",
    ),
    ("h1_background_minus_h1", "h1_background", "h1"),
    ("h4s_background_minus_h4s", "h4s_background", "h4s"),
)


@dataclass(frozen=True)
class H4sUnit:
    heldout_vr: float
    hypothesis: str
    seed: int


def h4s_units(mode: str) -> list[H4sUnit]:
    if str(mode) == "smoke":
        return [H4sUnit(0.0, hypothesis, 0) for hypothesis in H4S_HYPOTHESES]
    if str(mode) != "fullscan":
        raise ValueError(f"Unsupported H4s validation mode: {mode}")
    return [
        H4sUnit(heldout_vr, hypothesis, seed)
        for heldout_vr in validation_holdout.HELDOUT_VR_VALUES
        for hypothesis in H4S_HYPOTHESES
        for seed in (0, 1, 2)
    ]


class H4sHypothesisModel(model.PoissonRateFHCoreMultiLevel):
    def __init__(
        self,
        *,
        hypothesis: str,
        n_curves: int,
        n_max: int,
        min_level_gap: float,
        V_exc_init: float,
        init_jitter_scale: float,
        init_seed: int,
        device: torch.device | None = None,
    ) -> None:
        if hypothesis not in HYPOTHESIS_BY_KEY:
            raise ValueError(f"Unsupported H4s hypothesis: {hypothesis}")
        self.hypothesis_spec = HYPOTHESIS_BY_KEY[hypothesis]
        super().__init__(
            n_curves=n_curves,
            n_max=n_max,
            n_levels=4 if self.hypothesis_spec.nist_constrained else 1,
            min_level_gap=min_level_gap,
            device=device,
            V_exc_init=V_exc_init,
            init_jitter_scale=init_jitter_scale,
            init_seed=init_seed,
        )
        self.register_buffer(
            "nist_4s_energies_eV",
            torch.tensor(NIST_AR_I_4S_ENERGIES_EV, dtype=torch.float32),
        )
        if self.hypothesis_spec.nist_constrained:
            self.raw_E1.requires_grad_(False)
            self.raw_dE.requires_grad_(False)
            self.raw_common_energy_shift = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        else:
            self.register_parameter("raw_common_energy_shift", None)
        if not self.hypothesis_spec.background_enabled:
            for name in BACKGROUND_PARAMETER_NAMES:
                getattr(self, name).requires_grad_(False)

    def common_energy_shift(self) -> torch.Tensor:
        if self.raw_common_energy_shift is None:
            return torch.zeros((), device=self.raw_E1.device, dtype=self.raw_E1.dtype)
        return COMMON_ENERGY_SHIFT_LIMIT_EV * torch.tanh(self.raw_common_energy_shift)

    def level_params(self) -> dict[str, torch.Tensor]:
        if not self.hypothesis_spec.nist_constrained:
            return super().level_params()
        energies = self.nist_4s_energies_eV + self.common_energy_shift()
        weights = F.softmax(self.raw_level_logits, dim=0)
        gaps = torch.diff(energies)
        return {"energies": energies, "weights": weights, "gaps": gaps}

    def phys_params(self) -> dict[str, torch.Tensor]:
        params = super().phys_params()
        if not self.hypothesis_spec.background_enabled:
            params["high_energy_loss_strength"] = torch.zeros_like(
                params["high_energy_loss_strength"]
            )
        return params

    def regularization_loss(self, cfg: model.Config) -> torch.Tensor:
        reference = next(parameter for parameter in self.parameters())
        loss = torch.zeros((), device=reference.device, dtype=reference.dtype)
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad or parameter.numel() == 0:
                continue
            scale = 0.25 if name.startswith("raw_curve") else 1.0
            loss = loss + scale * torch.mean(torch.square(parameter))
        weights = self.level_params()["weights"]
        entropy = -torch.sum(weights * torch.log(weights + 1e-12))
        max_entropy = math.log(max(int(self.n_levels), 2))
        entropy_penalty = (max_entropy - entropy) / max_entropy
        return float(cfg.w_reg) * loss + float(cfg.level_weight_entropy) * entropy_penalty

    def trainable_parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad))


def load_all_curves(*, baseline: Baseline, input_path: str | Path) -> list[dict[str, Any]]:
    cfg = validation_holdout._baseline_config(baseline)
    cfg.data_path = str(paths.resolve_project_path(input_path))
    return validation_holdout._full_curves(cfg)


def _unit_config(
    *,
    unit: H4sUnit,
    mode: str,
    input_path: str | Path,
    baseline: Baseline,
    unit_dir: Path,
    device: str,
) -> model.Config:
    cfg = validation_holdout._baseline_config(baseline)
    if str(mode) == "smoke":
        for key, value in main_pipeline.profile_defaults("smoke").items():
            setattr(cfg, key, value)
    cfg.data_path = str(paths.resolve_project_path(input_path))
    cfg.out_dir = str(unit_dir / "fit")
    cfg.profile = f"validation_h4s_{mode}_{unit.hypothesis}"
    cfg.seed = int(unit.seed)
    cfg.exclude_vr_values = str(float(unit.heldout_vr))
    cfg.hyperopt_enabled = False
    cfg.launch_monitor = False
    cfg.forward_prior_mode = "off"
    cfg.forward_evidence_path = ""
    cfg.w_prior_anchor = 0.0
    cfg.w_prior_gap = 0.0
    cfg.init_jitter_scale = INITIALIZATION_JITTER_SCALE
    cfg.device = str(device)
    cfg.selected_device = (
        "cuda" if str(device) == "cuda" and torch.cuda.is_available() else "cpu"
    )
    cfg.resume_mode = "auto" if str(mode) == "fullscan" else "off"
    return cfg


def _vr_label(value: float) -> str:
    return str(float(value)).replace("-", "m").replace(".", "p")


def _unit_dir(root: Path, unit: H4sUnit) -> Path:
    return (
        root
        / "folds"
        / f"vr_{_vr_label(unit.heldout_vr)}"
        / unit.hypothesis
        / f"seed_{unit.seed:03d}"
    )


def _unit_identity(
    *,
    unit: H4sUnit,
    cfg: model.Config,
    baseline: Baseline,
    input_path: str | Path,
) -> dict[str, Any]:
    resolved_input = paths.resolve_project_path(input_path)
    return {
        "schema_version": 1,
        "unit": asdict(unit),
        "hypothesis": asdict(HYPOTHESIS_BY_KEY[unit.hypothesis]),
        "baseline_sha256": baseline.identity_hash,
        "input_sha256": sha256_file(resolved_input) if resolved_input.is_file() else "MISSING",
        "config_sha256": canonical_hash(asdict(cfg)),
        "nist_4s_energies_eV": list(NIST_AR_I_4S_ENERGIES_EV),
        "common_energy_shift_limit_eV": COMMON_ENERGY_SHIFT_LIMIT_EV,
        "initialization_jitter_scale": INITIALIZATION_JITTER_SCALE,
    }


def execute_unit(
    *,
    unit: H4sUnit,
    mode: str,
    input_path: str | Path,
    baseline: Baseline,
    output_dir: str | Path,
    device: str,
    all_curves: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    target = _unit_dir(Path(output_dir), unit)
    target.mkdir(parents=True, exist_ok=True)
    cfg = _unit_config(
        unit=unit,
        mode=mode,
        input_path=input_path,
        baseline=baseline,
        unit_dir=target,
        device=device,
    )
    identity = _unit_identity(unit=unit, cfg=cfg, baseline=baseline, input_path=input_path)
    manifest_path = target / "unit_manifest.json"
    result_path = target / "unit_result.json"
    predictions_path = target / "zero_shot_predictions.csv"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if existing.get("identity") != identity:
            raise ValueError(f"H4s unit identity mismatch; refusing to overwrite {target}")
    else:
        atomic_json_dump({"identity": identity}, manifest_path)
    if result_path.exists() and predictions_path.exists():
        existing_result = json.loads(result_path.read_text(encoding="utf-8-sig"))
        if bool(existing_result.get("ok", False)):
            return {
                **existing_result,
                "predictions": pd.read_csv(predictions_path).to_dict("records"),
                "reused": True,
            }

    training_curves, init = model.load_curves_and_init(cfg)
    heldout_values = {
        float(np.asarray(curve["Vr"], dtype=np.float64).reshape(-1)[0])
        for curve in training_curves
    }
    if any(
        abs(value - float(unit.heldout_vr)) <= 1e-6 for value in heldout_values
    ):
        raise AssertionError(f"Held-out Vr={unit.heldout_vr} leaked into training curves")
    _, heldout_curves = validation_holdout.split_vr_fold(
        all_curves,
        heldout_vr=unit.heldout_vr,
    )
    heldout_curve = heldout_curves[0]
    torch_device = model.resolve_device(cfg.selected_device)
    model.set_seed(unit.seed)
    trained = H4sHypothesisModel(
        hypothesis=unit.hypothesis,
        n_curves=len(training_curves),
        n_max=int(cfg.n_max),
        min_level_gap=float(cfg.min_level_gap),
        device=torch_device,
        V_exc_init=float(init["V_exc_init"]),
        init_jitter_scale=float(cfg.init_jitter_scale),
        init_seed=unit.seed,
    ).to(torch_device)
    resume_path = target / "fit" / "checkpoint_last.pt"
    scorecard = model.train_multilevel(
        cfg,
        trained,
        training_curves,
        torch_device,
        target / "fit",
        int(cfg.epochs),
        resume_path if cfg.resume_mode == "auto" and resume_path.is_file() else None,
    )
    observed = np.asarray(heldout_curve["Ip"], dtype=np.float64)
    va_values = np.asarray(heldout_curve["Va"], dtype=np.float64)
    va = torch.as_tensor(va_values, dtype=torch.float32, device=torch_device)
    if bool(heldout_curve["Vr_is_vector"]):
        vr = torch.as_tensor(heldout_curve["Vr"], dtype=torch.float32, device=torch_device)
    else:
        vr = torch.full_like(va, float(heldout_curve["Vr"]))
    trained.eval()
    with torch.no_grad():
        predicted = (
            trained.forward_core(va, vr, nuisance_mode="neutral")
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
    metrics = validation_holdout.prediction_metrics(
        observed,
        predicted,
        denominator_observed=observed,
    )
    extracted = model.extract_params(trained)
    extracted["common_energy_shift_eV"] = (
        float(trained.common_energy_shift().detach().cpu())
        if trained.hypothesis_spec.nist_constrained
        else None
    )
    predictions = [
        {
            "index": int(index),
            "Va": float(va_values[index]),
            "observed": float(observed[index]),
            "predicted": float(predicted[index]),
        }
        for index in range(len(observed))
    ]
    pd.DataFrame(predictions).to_csv(predictions_path, index=False)
    result = {
        "ok": True,
        "status": "complete",
        "unit": asdict(unit),
        "metrics": metrics,
        "elapsed_seconds": float(scorecard.get("elapsed_seconds", 0.0)),
        "trainable_parameter_count": trained.trainable_parameter_count(),
        "params": extracted,
        "reused": False,
    }
    atomic_json_dump(result, result_path)
    return {**result, "predictions": predictions}


def _worker(payload: dict[str, Any]) -> dict[str, Any]:
    return execute_unit(**payload)


def build_paired_contrasts(
    metric_rows: Sequence[Mapping[str, Any]],
    *,
    expected_units: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    observed = {
        (float(row["heldout_vr"]), int(row["seed"]), str(row["hypothesis"])): dict(row)
        for row in metric_rows
    }
    fold_seeds = sorted(
        {
            (float(unit["heldout_vr"]), int(unit["seed"]))
            for unit in expected_units
        }
    )
    rows: list[dict[str, Any]] = []
    for heldout_vr, seed in fold_seeds:
        for contrast, candidate_name, reference_name in CONTRASTS:
            candidate = observed.get((heldout_vr, seed, candidate_name))
            reference = observed.get((heldout_vr, seed, reference_name))
            row: dict[str, Any] = {
                "contrast": contrast,
                "candidate": candidate_name,
                "reference": reference_name,
                "heldout_vr": heldout_vr,
                "seed": seed,
                "status": "complete" if candidate is not None and reference is not None else "unavailable",
            }
            for metric_name in ("rmse", "mae", "nrmse"):
                if candidate is not None and reference is not None:
                    candidate_value = float(candidate[metric_name])
                    reference_value = float(reference[metric_name])
                    row[f"candidate_{metric_name}"] = candidate_value
                    row[f"reference_{metric_name}"] = reference_value
                    row[f"delta_{metric_name}"] = candidate_value - reference_value
                else:
                    row[f"candidate_{metric_name}"] = None
                    row[f"reference_{metric_name}"] = None
                    row[f"delta_{metric_name}"] = None
            rows.append(row)
    return rows


def _metric_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    return (
        frame.groupby("hypothesis", as_index=False)
        .agg(
            row_count=("rmse", "size"),
            rmse_mean=("rmse", "mean"),
            mae_mean=("mae", "mean"),
            nrmse_mean=("nrmse", "mean"),
            elapsed_seconds_mean=("elapsed_seconds", "mean"),
            trainable_parameter_count=("trainable_parameter_count", "max"),
        )
        .to_dict("records")
    )


def _contrast_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame([row for row in rows if row.get("status") == "complete"])
    if frame.empty:
        return []
    return (
        frame.groupby("contrast", as_index=False)
        .agg(
            pair_count=("delta_nrmse", "size"),
            delta_rmse_mean=("delta_rmse", "mean"),
            delta_mae_mean=("delta_mae", "mean"),
            delta_nrmse_mean=("delta_nrmse", "mean"),
            nrmse_improvement_count=("delta_nrmse", lambda values: int((values < 0.0).sum())),
        )
        .to_dict("records")
    )


def _fold_median_contrasts(metric_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(metric_rows)
    if frame.empty:
        return []
    medians = (
        frame.groupby(["heldout_vr", "hypothesis"], as_index=False)
        .agg(rmse=("rmse", "median"), mae=("mae", "median"), nrmse=("nrmse", "median"))
    )
    lookup = {
        (float(row.heldout_vr), str(row.hypothesis)): row
        for row in medians.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for heldout_vr in sorted(frame["heldout_vr"].unique()):
        for contrast, candidate_name, reference_name in CONTRASTS:
            candidate = lookup.get((float(heldout_vr), candidate_name))
            reference = lookup.get((float(heldout_vr), reference_name))
            if candidate is None or reference is None:
                continue
            rows.append(
                {
                    "contrast": contrast,
                    "heldout_vr": float(heldout_vr),
                    "delta_rmse": float(candidate.rmse - reference.rmse),
                    "delta_mae": float(candidate.mae - reference.mae),
                    "delta_nrmse": float(candidate.nrmse - reference.nrmse),
                }
            )
    return rows


def _direction_consistency(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    summaries: list[dict[str, Any]] = []
    for contrast, group in frame.groupby("contrast", sort=False):
        deltas = group["delta_nrmse"].astype(float)
        lower_count = int((deltas < 0.0).sum())
        higher_count = int((deltas > 0.0).sum())
        tied_count = int((deltas == 0.0).sum())
        fold_count = int(len(deltas))
        summaries.append(
            {
                "contrast": str(contrast),
                "fold_count": fold_count,
                "candidate_lower_nrmse_fold_count": lower_count,
                "candidate_higher_nrmse_fold_count": higher_count,
                "tied_fold_count": tied_count,
                "candidate_lower_nrmse_fold_fraction": lower_count / fold_count,
            }
        )
    return summaries


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(row) for row in rows], columns=list(columns)).to_csv(path, index=False)


def _worker_count(mode: str, device: str, baseline: Baseline, unit_count: int) -> int:
    if str(mode) == "smoke" or unit_count <= 1:
        return 1
    cfg = validation_holdout._baseline_config(baseline)
    if str(device) == "cuda" and torch.cuda.is_available():
        return min(unit_count, max(1, int(cfg.cuda_workers)))
    return min(unit_count, max(1, int(cfg.cpu_workers)))


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
    target.mkdir(parents=True, exist_ok=True)
    units = h4s_units(mode)
    all_curves = load_all_curves(baseline=baseline, input_path=input_path)
    hypothesis_manifest = {
        "schema_version": 1,
        "nist_source": NIST_SOURCE_URL,
        "nist_4s_energies_eV": list(NIST_AR_I_4S_ENERGIES_EV),
        "common_energy_shift_limit_eV": COMMON_ENERGY_SHIFT_LIMIT_EV,
        "background_definition": "Existing smooth high_energy_loss term; it is not a fifth excitation channel.",
        "hypotheses": [asdict(spec) for spec in HYPOTHESIS_SPECS],
        "contrasts": [
            {"name": name, "candidate": candidate, "reference": reference}
            for name, candidate, reference in CONTRASTS
        ],
        "claim_decision": "not_automated",
    }
    atomic_json_dump(hypothesis_manifest, target / "hypothesis_manifest.json")

    status_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    completed = 0
    failed = 0
    workers = _worker_count(mode, device, baseline, len(units))
    write_progress(
        validation / "progress.json",
        completed=0,
        failed=0,
        total=len(units),
        current_stage="h4s",
    )

    def record(unit: H4sUnit, result: dict[str, Any] | None, error: str = "") -> None:
        nonlocal completed, failed
        if result is None:
            failed += 1
            status_rows.append({**asdict(unit), "status": "failed", "reused": False, "error": error})
        else:
            completed += 1
            common = asdict(unit)
            metric_rows.append(
                {
                    **common,
                    **result["metrics"],
                    "elapsed_seconds": float(result["elapsed_seconds"]),
                    "trainable_parameter_count": int(result["trainable_parameter_count"]),
                    "common_energy_shift_eV": result["params"].get("common_energy_shift_eV"),
                }
            )
            prediction_rows.extend({**common, **row} for row in result["predictions"])
            status_rows.append(
                {
                    **common,
                    "status": "complete",
                    "reused": bool(result.get("reused", False)),
                    "error": "",
                }
            )
        write_progress(
            validation / "progress.json",
            completed=completed,
            failed=failed,
            total=len(units),
            current_stage="h4s",
            heldout_vr=float(unit.heldout_vr),
            hypothesis=unit.hypothesis,
            seed=int(unit.seed),
        )

    payloads = [
        {
            "unit": unit,
            "mode": str(mode),
            "input_path": input_path,
            "baseline": baseline,
            "output_dir": target,
            "device": str(device),
            "all_curves": all_curves,
        }
        for unit in units
    ]
    if workers == 1:
        for unit, payload in zip(units, payloads):
            try:
                record(unit, execute_unit(**payload))
            except Exception as exc:
                record(unit, None, f"{type(exc).__name__}: {exc}")
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            future_map = {pool.submit(_worker, payload): unit for payload, unit in zip(payloads, units)}
            for future in concurrent.futures.as_completed(future_map):
                unit = future_map[future]
                try:
                    record(unit, future.result())
                except Exception as exc:
                    record(unit, None, f"{type(exc).__name__}: {exc}")

    status_rows.sort(key=lambda row: (row["heldout_vr"], H4S_HYPOTHESES.index(row["hypothesis"]), row["seed"]))
    metric_rows.sort(key=lambda row: (row["heldout_vr"], H4S_HYPOTHESES.index(row["hypothesis"]), row["seed"]))
    prediction_rows.sort(
        key=lambda row: (
            row["heldout_vr"],
            H4S_HYPOTHESES.index(row["hypothesis"]),
            row["seed"],
            row["index"],
        )
    )
    contrasts = build_paired_contrasts(
        metric_rows,
        expected_units=[asdict(unit) for unit in units],
    )
    fold_medians = _fold_median_contrasts(metric_rows)
    status = "incomplete" if failed else ("smoke_passed" if str(mode) == "smoke" else "complete")
    claim_evaluable = str(mode) == "fullscan" and failed == 0 and completed == len(units)
    summary = {
        "ok": failed == 0 and completed == len(units),
        "status": status,
        "mode": str(mode),
        "unit_total": len(units),
        "completed": completed,
        "failed": failed,
        "fold_count": 1 if str(mode) == "smoke" else len(validation_holdout.HELDOUT_VR_VALUES),
        "hypothesis_count": len(H4S_HYPOTHESES),
        "worker_count": workers,
        "claim_evaluable": claim_evaluable,
        "claim_decision": "not_automated",
        "seed_interpretation": "Optimization restarts only; seeds are not independent samples.",
        "zero_shot_summary": _metric_summary(metric_rows),
        "paired_contrast_summary": _contrast_summary(contrasts),
        "fold_median_contrasts": fold_medians,
        "fold_median_direction_consistency": _direction_consistency(fold_medians),
    }
    _write_csv(
        target / "unit_status.csv",
        status_rows,
        ("heldout_vr", "hypothesis", "seed", "status", "reused", "error"),
    )
    _write_csv(
        target / "zero_shot_metrics.csv",
        metric_rows,
        (
            "heldout_vr",
            "hypothesis",
            "seed",
            "rmse",
            "mae",
            "nrmse",
            "normalization_range",
            "n_points",
            "elapsed_seconds",
            "trainable_parameter_count",
            "common_energy_shift_eV",
        ),
    )
    _write_csv(
        target / "zero_shot_predictions.csv",
        prediction_rows,
        ("heldout_vr", "hypothesis", "seed", "index", "Va", "observed", "predicted"),
    )
    _write_csv(
        target / "paired_contrasts.csv",
        contrasts,
        (
            "contrast",
            "candidate",
            "reference",
            "heldout_vr",
            "seed",
            "status",
            "candidate_rmse",
            "reference_rmse",
            "delta_rmse",
            "candidate_mae",
            "reference_mae",
            "delta_mae",
            "candidate_nrmse",
            "reference_nrmse",
            "delta_nrmse",
        ),
    )
    atomic_json_dump(summary, target / "h4s_comparison_summary.json")
    write_stage_status(
        target / "stage_status.json",
        status,
        unit_total=len(units),
        completed=completed,
        failed=failed,
        claim_evaluable=claim_evaluable,
    )
    return summary
