from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fh_transport.data import load_curve_data  # noqa: E402
from fh_transport.metrics import apply_v2j_single_level_gate, evaluate_predictions  # noqa: E402
from fh_transport.model_v2 import _inverse_bounded  # noqa: E402
from fh_transport.model_v2j import CollisionFieldEnsembleFranckHertz  # noqa: E402
from fh_transport.model_v2l import CalibratedCollisionFieldFranckHertz  # noqa: E402
from fh_transport.train import (  # noqa: E402
    TrainConfig,
    _config_hash,
    _json_dump,
    _loss_terms,
    _plot_fit,
    select_device,
)


SPREAD_GRID_EV = (10.5, 10.8, 11.1, 11.4, 11.6, 11.75, 11.85, 11.92, 11.97)
DECAY_GRID_V = (24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate the v2l apparatus closure.")
    parser.add_argument(
        "--checkpoint",
        default=str(PROJECT_ROOT / "output" / "v2j_iteration_01" / "h1" / "checkpoint_best.pt"),
    )
    parser.add_argument(
        "--baseline", default=str(PROJECT_ROOT / "data" / "old_k1_prediction_points.csv")
    )
    parser.add_argument(
        "--output", default=str(PROJECT_ROOT / "output" / "v2l_iteration_01")
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--layers", type=int)
    parser.add_argument("--max-collisions", type=int)
    parser.add_argument("--quadrature-order", type=int)
    return parser


def _load_compatible(model: torch.nn.Module, source: dict[str, torch.Tensor]) -> dict[str, object]:
    target = model.state_dict()
    compatible = {
        name: value
        for name, value in source.items()
        if name in target and target[name].shape == value.shape
    }
    model.load_state_dict(compatible, strict=False)
    return {
        "loaded_tensor_count": len(compatible),
        "skipped_source_tensors": sorted(set(source) - set(compatible)),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = copy.deepcopy(payload["config"])
    if args.layers is not None:
        config["n_layers"] = int(args.layers)
    if args.max_collisions is not None:
        config["max_collisions"] = int(args.max_collisions)
    if args.quadrature_order is not None:
        config["quadrature_order"] = int(args.quadrature_order)
    config["output_dir"] = str((output_dir / "h1").resolve())
    config_sha256 = _config_hash(TrainConfig(**config))
    h1_dir = Path(config["output_dir"])
    h1_dir.mkdir(parents=True, exist_ok=True)
    data = load_curve_data(config["data_path"], device)

    scan_model = CollisionFieldEnsembleFranckHertz(
        hypothesis="h1",
        n_layers=config["n_layers"],
        max_collisions=config["max_collisions"],
        quadrature_order=config["quadrature_order"],
        seed=config["seed"],
    ).to(device)
    _load_compatible(scan_model, payload["model_state"])
    scan_model.eval()
    scan_rows: list[dict[str, object]] = []
    for spread in SPREAD_GRID_EV:
        for decay in DECAY_GRID_V:
            with torch.no_grad():
                scan_model.raw_collision_field_spread_v2j.fill_(
                    _inverse_bounded(spread, 0.0, 12.0)
                )
                scan_model.raw_collision_spread_decay_v2j.fill_(
                    _inverse_bounded(decay, 8.0, 80.0)
                )
                prediction = scan_model(data.va, data.vr).detach().cpu().numpy()
            metrics = evaluate_predictions(data.frame, prediction)
            gate = apply_v2j_single_level_gate(metrics)
            summary = metrics["summary"]
            scan_rows.append(
                {
                    "collision_field_spread_eV": spread,
                    "collision_spread_decay_V": decay,
                    "gate_passed": bool(gate["passed"]),
                    "failed_checks": ";".join(
                        check["metric"] for check in gate["checks"] if not check["passed"]
                    ),
                    "mean_curve_nrmse": summary["mean_curve_nrmse"],
                    "max_cutoff_error_V": summary["max_cutoff_error_V"],
                    "trough_precision": summary["trough_precision"],
                    "early_high_retarding_mae_uA": summary[
                        "early_high_retarding_mae_uA"
                    ],
                    "late_trough_current_mae_uA": summary[
                        "late_trough_current_mae_uA"
                    ],
                }
            )
    scan_frame = pd.DataFrame(scan_rows)
    feasible = scan_frame.loc[scan_frame["gate_passed"]].sort_values(
        ["mean_curve_nrmse", "collision_field_spread_eV", "collision_spread_decay_V"]
    )
    if feasible.empty:
        raise RuntimeError("The frozen v2l calibration grid contains no feasible point")
    selected = feasible.iloc[0]
    if not math.isclose(
        float(selected["collision_field_spread_eV"]),
        CalibratedCollisionFieldFranckHertz.calibrated_spread_eV,
    ) or not math.isclose(
        float(selected["collision_spread_decay_V"]),
        CalibratedCollisionFieldFranckHertz.calibrated_decay_V,
    ):
        raise RuntimeError("The frozen v2l model constants do not match the grid selection")
    scan_frame.to_csv(h1_dir / "calibration_scan.csv", index=False)

    model = CalibratedCollisionFieldFranckHertz(
        hypothesis="h1",
        n_layers=config["n_layers"],
        max_collisions=config["max_collisions"],
        quadrature_order=config["quadrature_order"],
        seed=config["seed"],
    ).to(device)
    initialization = _load_compatible(model, payload["model_state"])
    initialization["source_checkpoint"] = str(checkpoint_path)
    model.eval()
    with torch.no_grad():
        prediction = model(data.va, data.vr).detach().cpu().numpy()
        selected_loss = float(_loss_terms(model, data)["total"].detach().cpu())

    prediction_frame = data.frame.copy()
    prediction_frame["observed"] = prediction_frame["IuA"]
    prediction_frame["predicted"] = prediction
    prediction_frame["residual"] = prediction_frame["observed"] - prediction_frame["predicted"]
    prediction_frame.to_csv(h1_dir / "predictions.csv", index=False)
    _plot_fit(prediction_frame, h1_dir / "fit.png")

    metrics = evaluate_predictions(data.frame, prediction)
    gate = apply_v2j_single_level_gate(metrics)
    n_points = int(metrics["summary"]["n_points"])
    n_params = int(model.parameter_report()["trainable_parameter_count"])
    sse = max(float(metrics["summary"]["sse_uA2"]), 1.0e-12)
    metrics["summary"].update(
        {
            "n_params": n_params,
            "bic": float(n_points * math.log(sse / n_points) + n_params * math.log(n_points)),
            "aic": float(n_points * math.log(sse / n_points) + 2 * n_params),
        }
    )
    baseline_metrics = None
    baseline_path = Path(args.baseline).resolve()
    if baseline_path.is_file():
        baseline = pd.read_csv(baseline_path)
        if len(baseline) == len(data.frame) and "predicted" in baseline:
            baseline_metrics = evaluate_predictions(data.frame, baseline["predicted"].to_numpy())

    source_result_path = checkpoint_path.parent / "result.json"
    source_training = None
    if source_result_path.is_file():
        source_result = json.loads(source_result_path.read_text(encoding="utf-8"))
        source_training = {
            "result": str(source_result_path),
            "epochs_completed": source_result.get("epochs_completed"),
            "best_epoch": source_result.get("best_epoch"),
            "stop_reason": source_result.get("stop_reason"),
        }
    result = {
        "status": "completed",
        "config": config,
        "config_sha256": config_sha256,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "elapsed_seconds": float(time.perf_counter() - started),
        "epochs_completed": 0,
        "best_epoch": 0,
        "best_loss": selected_loss,
        "stop_reason": "post_plateau_coordinate_calibration",
        "parameters": model.parameter_report(),
        "metrics": metrics,
        "single_level_gate": gate,
        "old_k1_baseline_metrics": baseline_metrics,
        "initialization": initialization,
        "source_training": source_training,
        "calibration": {
            "selection_rule": "minimum mean_curve_nrmse among frozen-grid 15/15-feasible points",
            "candidate_count": int(len(scan_frame)),
            "feasible_count": int(len(feasible)),
            "selected_collision_field_spread_eV": float(
                selected["collision_field_spread_eV"]
            ),
            "selected_collision_spread_decay_V": float(
                selected["collision_spread_decay_V"]
            ),
            "scan_artifact": str(h1_dir / "calibration_scan.csv"),
        },
        "material_passport": {
            "model_family": "v2l-calibrated-collision-field",
            "hypothesis": "h1",
            "physical_scope": (
                "Plateau-stopped v2j continuous fit followed by a frozen-grid "
                "apparatus collision-dispersion calibration."
            ),
        },
    }
    if not gate["passed"]:
        raise RuntimeError("Selected v2l calibration did not pass the frozen H1 gate")
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "config_sha256": config_sha256,
            "best_epoch": 0,
            "best_loss": selected_loss,
            "initialization": initialization,
            "calibration": result["calibration"],
        },
        h1_dir / "checkpoint_best.pt",
    )
    _json_dump(result, h1_dir / "result.json")
    summary = {
        "status": "completed",
        "h1_result": str(h1_dir / "result.json"),
        "single_level_gate": gate,
    }
    _json_dump(summary, output_dir / "pipeline_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
