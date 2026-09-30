from __future__ import annotations

import argparse
from functools import partial
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fh_transport.pipeline_v2 import (  # noqa: E402
    compare_hypotheses,
    dump_json,
    run_hypothesis as run_v2a_hypothesis,
)
from fh_transport.pipeline_v2b import run_hypothesis as run_v2b_hypothesis  # noqa: E402
from fh_transport.pipeline_v2c import run_hypothesis as run_v2c_hypothesis  # noqa: E402
from fh_transport.pipeline_v2d import run_hypothesis as run_v2d_hypothesis  # noqa: E402
from fh_transport.pipeline_v2e import run_hypothesis as run_v2e_hypothesis  # noqa: E402
from fh_transport.pipeline_v2f import run_hypothesis as run_v2f_hypothesis  # noqa: E402
from fh_transport.pipeline_v2g import run_hypothesis as run_v2g_hypothesis  # noqa: E402
from fh_transport.pipeline_v2h import run_hypothesis as run_v2h_hypothesis  # noqa: E402
from fh_transport.pipeline_v2i import run_hypothesis as run_v2i_hypothesis  # noqa: E402
from fh_transport.pipeline_v2j import run_hypothesis as run_v2j_hypothesis  # noqa: E402
from fh_transport.pipeline_v2k import run_hypothesis as run_v2k_hypothesis  # noqa: E402
from fh_transport.pipeline_v2l import run_hypothesis as run_v2l_hypothesis  # noqa: E402
from fh_transport.pipeline_v2m import run_hypothesis as run_v2m_hypothesis  # noqa: E402
from fh_transport.train import TrainConfig  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit a competing-history Franck-Hertz kernel.")
    parser.add_argument(
        "--kernel",
        choices=("v2a", "v2b", "v2c", "v2d", "v2e", "v2f", "v2g", "v2h", "v2i", "v2j", "v2k", "v2l", "v2m"),
        default="v2a",
    )
    parser.add_argument("--stage", choices=("h1", "h4s", "auto"), default="auto")
    parser.add_argument("--data", default=str(PROJECT_ROOT / "data" / "FHdata.xlsx"))
    parser.add_argument(
        "--baseline", default=str(PROJECT_ROOT / "data" / "old_k1_prediction_points.csv")
    )
    parser.add_argument("--output", default=str(PROJECT_ROOT / "output" / "v2_formal"))
    parser.add_argument("--gate-result", help="Passed v2 H1 result.json required by --stage h4s")
    parser.add_argument(
        "--cross-section-table",
        help="Provenance-bearing four-channel partial cross-section CSV required by v2m H1/H4s",
    )
    parser.add_argument(
        "--initial-checkpoint",
        help="Optional compatible checkpoint used to initialize H1 common parameters",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=1800)
    parser.add_argument("--min-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.020)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--max-collisions", type=int, default=8)
    parser.add_argument("--quadrature-order", type=int, default=6)
    parser.add_argument("--smoke", action="store_true", help="Engineering check only; delete output after use")
    return parser


def _passed_gate(path: str | Path, expected_model_version: str) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not bool(payload.get("single_level_gate", {}).get("passed")):
        raise ValueError("The supplied v2 H1 result did not pass the pre-registered gate")
    if payload.get("parameters", {}).get("model_version") != expected_model_version:
        raise ValueError(f"The supplied gate result is not a {expected_model_version} result")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runners = {
        "v2a": (run_v2a_hypothesis, "v2a-competing-history"),
        "v2b": (run_v2b_hypothesis, "v2b-last-collision-moment"),
        "v2c": (run_v2c_hypothesis, "v2c-fixed-isotropic-scattering"),
        "v2d": (run_v2d_hypothesis, "v2d-energy-dependent-scattering"),
        "v2e": (run_v2e_hypothesis, "v2e-exact-last-collision"),
        "v2f": (run_v2f_hypothesis, "v2f-threshold-gaussian"),
        "v2g": (run_v2g_hypothesis, "v2g-generalized-supply"),
        "v2h": (run_v2h_hypothesis, "v2h-reachability-gated"),
        "v2i": (run_v2i_hypothesis, "v2i-finite-angular-aperture"),
        "v2j": (run_v2j_hypothesis, "v2j-collision-field-ensemble"),
        "v2k": (run_v2k_hypothesis, "v2k-truncated-gaussian-gate"),
        "v2l": (run_v2l_hypothesis, "v2l-calibrated-collision-field"),
        "v2m": (run_v2m_hypothesis, "v2m-tabulated-partial-cross-sections"),
    }
    run_hypothesis, expected_model_version = runners[args.kernel]
    if args.kernel == "v2m":
        if args.cross_section_table is None:
            raise ValueError("v2m H1/H4s requires --cross-section-table")
        run_hypothesis = partial(
            run_hypothesis, cross_section_table_path=args.cross_section_table
        )
    elif args.cross_section_table is not None:
        raise ValueError("--cross-section-table is only valid with --kernel v2m")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        max_epochs, min_epochs, patience = 20, 8, 5
        layers, max_collisions, quadrature_order = 6, 3, 4
    else:
        max_epochs, min_epochs, patience = args.max_epochs, args.min_epochs, args.patience
        layers = args.layers
        max_collisions = args.max_collisions
        quadrature_order = args.quadrature_order
    base_config = TrainConfig(
        data_path=str(Path(args.data).resolve()),
        output_dir=str(output),
        seed=int(args.seed),
        device=args.device,
        max_epochs=int(max_epochs),
        min_epochs=int(min_epochs),
        patience=int(patience),
        learning_rate=float(args.learning_rate),
        n_layers=int(layers),
        max_collisions=int(max_collisions),
        quadrature_order=int(quadrature_order),
    )
    baseline = Path(args.baseline).resolve() if args.baseline else None

    if args.stage == "h1":
        h1 = run_hypothesis(
            base_config,
            "h1",
            output / "h1",
            baseline,
            initial_checkpoint=args.initial_checkpoint,
        )
        summary = {
            "status": "completed" if h1["single_level_gate"]["passed"] else "h1_gate_failed",
            "h1_result": str(output / "h1" / "result.json"),
            "single_level_gate": h1["single_level_gate"],
        }
    elif args.stage == "h4s":
        if not args.gate_result:
            raise ValueError("--stage h4s requires --gate-result from a passed v2 H1 run")
        h1 = _passed_gate(args.gate_result, expected_model_version)
        checkpoint = Path(args.gate_result).resolve().parent / "checkpoint_best.pt"
        h4s = run_hypothesis(
            base_config, "h4s", output / "h4s", initial_checkpoint=checkpoint
        )
        comparison = compare_hypotheses(h1, h4s)
        dump_json(comparison, output / "comparison.json")
        summary = {
            "status": "completed",
            "h4s_result": str(output / "h4s" / "result.json"),
            "comparison": str(output / "comparison.json"),
        }
    else:
        h1 = run_hypothesis(
            base_config,
            "h1",
            output / "h1",
            baseline,
            initial_checkpoint=args.initial_checkpoint,
        )
        if not bool(h1["single_level_gate"]["passed"]):
            summary = {
                "status": "h1_gate_failed",
                "h1_result": str(output / "h1" / "result.json"),
                "single_level_gate": h1["single_level_gate"],
                "h4s_started": False,
            }
        else:
            checkpoint = Path(h1["config"]["output_dir"]) / "checkpoint_best.pt"
            h4s = run_hypothesis(
                base_config, "h4s", output / "h4s", initial_checkpoint=checkpoint
            )
            comparison = compare_hypotheses(h1, h4s)
            dump_json(comparison, output / "comparison.json")
            summary = {
                "status": "completed",
                "h1_result": str(output / "h1" / "result.json"),
                "h4s_result": str(output / "h4s" / "result.json"),
                "comparison": str(output / "comparison.json"),
                "single_level_gate": h1["single_level_gate"],
            }
    dump_json(summary, output / "pipeline_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("status") == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
