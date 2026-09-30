"""Run all EXPERIMENT_V2 experiments in dependency order and log gate verdicts.

Order: the registered diagnostic modules A1--A15, followed by A7 (summary
figure, consumes earlier outputs). Each module writes CSV tables + summary.json with SHA-256
receipts and per-stage timings under EXPERIMENT_V2/results/<experiment>/.
"""

from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE / "src") not in sys.path:
    sys.path.insert(0, str(HERE / "src"))

EXPERIMENTS = [
    "a1_block1_diagnosis",
    "a2_endpoint_mechanism",
    "a3_composition_truncation",
    "a5_injection_debias",
    "a6_gas_controls",
    "a8_development_cut",
    "a9_band_periods",
    "a10_review_controls",
    "a11_direct_evidence",
    "a12_robustness",
    "a13_method_hardening",
    "a14_instrument_completion",
    "a15_shift_equivalence",
    "a7_evidence_summary",
]


def main() -> int:
    log = {}
    failures = []
    for name in EXPERIMENTS:
        module = importlib.import_module(f"exp_v2.{name}")
        started = time.perf_counter()
        print(f"[run_all] >>> {name}", flush=True)
        try:
            result = module.run()
            wall = time.perf_counter() - started
            gates = result["summary"].get("gates", {})
            log[name] = {"wall_seconds": wall, "gates": gates, "out_dir": str(result["out_dir"])}
            print(f"[run_all] <<< {name} done in {wall:.1f}s", flush=True)
            for key, value in gates.items():
                if key.startswith("G_"):
                    print(f"          {key} = {value}", flush=True)
        except Exception as error:  # noqa: BLE001 - orchestrator keeps going
            wall = time.perf_counter() - started
            failures.append(name)
            log[name] = {"wall_seconds": wall, "error": repr(error)}
            print(f"[run_all] !!! {name} FAILED in {wall:.1f}s: {error!r}", flush=True)
    log_path = HERE / "results" / "run_all_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps(log, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"[run_all] log -> {log_path}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
