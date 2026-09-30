from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fh_transport.data import load_curve_data  # noqa: E402
from fh_transport.model_v2 import NIST_ARGON_4S_EV  # noqa: E402
from fh_transport.model_v2l import CalibratedCollisionFieldFranckHertz  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CPU-only local identifiability diagnostic for a v2l-H4s checkpoint."
    )
    parser.add_argument("--data", default=str(PROJECT_ROOT / "data" / "FHdata.xlsx"))
    parser.add_argument(
        "--checkpoint",
        default=str(
            PROJECT_ROOT
            / "output"
            / "v2l_q4_h4s_iteration_01"
            / "h4s"
            / "checkpoint_best.pt"
        ),
    )
    parser.add_argument("--epsilon", type=float, default=1.0e-2)
    parser.add_argument("--threads", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if float(args.epsilon) <= 0.0:
        raise ValueError("--epsilon must be positive")
    torch.set_num_threads(max(1, int(args.threads)))
    device = torch.device("cpu")
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint["config"]
    if int(config.get("n_levels", 0)) != 4:
        raise ValueError("The checkpoint is not an H4s fit")
    model = CalibratedCollisionFieldFranckHertz(
        hypothesis="h4s",
        n_layers=int(config["n_layers"]),
        max_collisions=int(config["max_collisions"]),
        quadrature_order=int(config["quadrature_order"]),
        seed=int(config["seed"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    data = load_curve_data(Path(args.data).resolve(), device)

    contrasts = torch.tensor(
        [
            [1.0, -1.0, 0.0, 0.0],
            [1.0, 1.0, -2.0, 0.0],
            [1.0, 1.0, 1.0, -3.0],
        ],
        dtype=torch.float32,
    )
    contrasts = contrasts / torch.linalg.vector_norm(contrasts, dim=1, keepdim=True)
    base_logits = model.raw_rate_logits.detach().clone()
    epsilon = float(args.epsilon)
    columns = []
    with torch.no_grad():
        base = model(data.va, data.vr).detach().numpy()
        for direction in contrasts:
            model.raw_rate_logits.copy_(base_logits + epsilon * direction)
            plus = model(data.va, data.vr).detach().numpy()
            model.raw_rate_logits.copy_(base_logits - epsilon * direction)
            minus = model(data.va, data.vr).detach().numpy()
            columns.append((plus - minus) / (2.0 * epsilon))
        model.raw_rate_logits.copy_(base_logits)

    jacobian = np.column_stack(columns)
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    residual = data.current.detach().numpy() - base
    step, *_ = np.linalg.lstsq(jacobian, residual, rcond=None)
    linearized_residual = residual - jacobian @ step
    level_span = float(max(NIST_ARGON_4S_EV) - min(NIST_ARGON_4S_EV))
    probe_voltages = (0.0, 10.0, 20.0, 40.0, 60.0, 80.0)
    spread = {
        str(int(voltage)): {
            "sigma_eV": 10.5 * math.exp(-voltage / 32.0),
            "sigma_over_4s_span": 10.5 * math.exp(-voltage / 32.0) / level_span,
        }
        for voltage in probe_voltages
    }
    payload = {
        "status": "completed",
        "device": "cpu",
        "checkpoint": str(checkpoint_path),
        "finite_difference_epsilon": epsilon,
        "learned_channel_fractions": torch.softmax(base_logits, dim=0).tolist(),
        "nist_4s_span_eV": level_span,
        "collector_gate_sigma_over_4s_span": 3.0 / level_span,
        "collision_field_spread": spread,
        "rate_contrast_jacobian": {
            "singular_values_uA_per_logit": singular_values.tolist(),
            "condition_number": float(singular_values[0] / singular_values[-1]),
            "column_correlation": np.corrcoef(jacobian, rowvar=False).tolist(),
            "current_residual_rmse_uA": float(np.sqrt(np.mean(residual**2))),
            "best_linearized_rate_only_rmse_uA": float(
                np.sqrt(np.mean(linearized_residual**2))
            ),
            "linearized_contrast_step": step.tolist(),
        },
        "interpretation_boundary": (
            "This is a local, fixed-common-parameter diagnostic of the fitted checkpoint; "
            "it is not a profile likelihood or held-out model comparison."
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
