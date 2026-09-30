from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fh_transport.pipeline import run_auto, run_scan, run_single  # noqa: E402
from fh_transport.train import TrainConfig  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit the thin-layer Franck-Hertz transport model.")
    parser.add_argument("--stage", choices=("single", "scan", "auto"), default="auto")
    parser.add_argument("--data", default=str(PROJECT_ROOT / "data" / "FHdata.xlsx"))
    parser.add_argument(
        "--baseline", default=str(PROJECT_ROOT / "data" / "old_k1_prediction_points.csv")
    )
    parser.add_argument("--output", default=str(PROJECT_ROOT / "output" / "formal"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=1800)
    parser.add_argument("--min-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.020)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--max-collisions", type=int, default=8)
    parser.add_argument("--scan-max", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--gate-result", help="Passed K=1 result.json required by --stage scan.")
    parser.add_argument("--smoke", action="store_true", help="Fast engineering check; not scientific evidence.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    if args.smoke:
        max_epochs = 30
        min_epochs = 10
        patience = 6
        layers = 8
        max_collisions = 4
        scan_max = min(int(args.scan_max), 2)
    else:
        max_epochs = int(args.max_epochs)
        min_epochs = int(args.min_epochs)
        patience = int(args.patience)
        layers = int(args.layers)
        max_collisions = int(args.max_collisions)
        scan_max = int(args.scan_max)
    config = TrainConfig(
        data_path=str(Path(args.data).resolve()),
        output_dir=str(output / "k1"),
        seed=int(args.seed),
        device=args.device,
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        patience=patience,
        learning_rate=float(args.learning_rate),
        n_layers=layers,
        max_collisions=max_collisions,
    )
    baseline = Path(args.baseline).resolve() if args.baseline else None
    if args.stage == "single":
        result = run_single(config, baseline)
        summary = {
            "status": "completed",
            "result": str(Path(result["config"]["output_dir"]) / "result.json"),
            "single_level_gate": result["single_level_gate"],
        }
    elif args.stage == "scan":
        if not args.gate_result:
            raise ValueError("--stage scan requires --gate-result from a passed single-level run")
        gate_payload = json.loads(Path(args.gate_result).read_text(encoding="utf-8"))
        if not bool(gate_payload.get("single_level_gate", {}).get("passed")):
            raise ValueError("The supplied single-level gate did not pass")
        summary = run_scan(
            config,
            output,
            baseline,
            maximum_k=scan_max,
            existing_k1=gate_payload,
        )
    else:
        summary = run_auto(
            config,
            output,
            baseline,
            maximum_k=scan_max,
            bypass_gate=bool(args.smoke),
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("status") == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
