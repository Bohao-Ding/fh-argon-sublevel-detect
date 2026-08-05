from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(
    os.environ.get("FH_H4S_PROJECT_ROOT", Path(__file__).resolve().parents[1])
).resolve()
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sublevel_detect import model, paths, validation_h4s, validation_holdout  # noqa: E402
from sublevel_detect.validation_common import (  # noqa: E402
    Baseline,
    atomic_json_dump,
    resolve_baseline,
    sha256_file,
)


BENCHMARK_HYPOTHESES = validation_h4s.H4S_HYPOTHESES
BENCHMARK_SEEDS = (0, 1)
BENCHMARK_HELDOUT_VR = 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the fixed H4s runtime audit matrix.")
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input", default=paths.default_input_text())
    parser.add_argument("--label", required=True)
    return parser


def _git_value(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "UNKNOWN"


def _configure_worker(threads_per_worker: int) -> None:
    if hasattr(model, "configure_worker_runtime"):
        model.configure_worker_runtime(threads_per_worker)
        return
    threads = max(1, int(threads_per_worker))
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _benchmark_unit(payload: dict[str, Any]) -> dict[str, Any]:
    _configure_worker(int(payload["threads_per_worker"]))
    baseline: Baseline = payload["baseline"]
    unit = validation_h4s.H4sUnit(**payload["unit"])
    target = Path(payload["target"])
    target.mkdir(parents=True, exist_ok=False)
    cfg = validation_h4s._unit_config(
        unit=unit,
        mode="fullscan",
        input_path=payload["input_path"],
        baseline=baseline,
        unit_dir=target,
        device=str(payload["device"]),
    )
    epochs = int(payload["epochs"])
    cfg.epochs = epochs
    cfg.early_stop_min_epochs = epochs + 1
    cfg.early_stop_warmup = epochs + 1
    cfg.resume_mode = "off"
    all_curves = validation_h4s.load_all_curves(
        baseline=baseline,
        input_path=payload["input_path"],
    )
    if hasattr(validation_h4s, "prepare_h4s_fold"):
        training_curves, heldout, init = validation_h4s.prepare_h4s_fold(
            all_curves,
            heldout_vr=unit.heldout_vr,
            cfg=cfg,
        )
    else:
        training_curves, init = model.load_curves_and_init(cfg)
        _, heldout_curves = validation_holdout.split_vr_fold(
            all_curves,
            heldout_vr=unit.heldout_vr,
        )
        heldout = heldout_curves[0]
    torch_device = model.resolve_device(cfg.selected_device)
    if torch_device.type == "cuda":
        torch.cuda.set_device(torch_device.index if torch_device.index is not None else 0)
        torch.cuda.reset_peak_memory_stats(torch_device)
    model.set_seed(unit.seed)
    trained = validation_h4s.H4sHypothesisModel(
        hypothesis=unit.hypothesis,
        n_curves=len(training_curves),
        n_max=int(cfg.n_max),
        min_level_gap=float(cfg.min_level_gap),
        device=torch_device,
        V_exc_init=float(init["V_exc_init"]),
        init_jitter_scale=float(cfg.init_jitter_scale),
        init_seed=unit.seed,
    ).to(torch_device)
    started = time.perf_counter()
    scorecard = model.train_multilevel(
        cfg,
        trained,
        training_curves,
        torch_device,
        target / "fit",
        epochs,
    )
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    wall_seconds = time.perf_counter() - started

    va_values = np.asarray(heldout["Va"], dtype=np.float64)
    observed = np.asarray(heldout["Ip"], dtype=np.float64)
    va = torch.as_tensor(va_values, dtype=torch.float32, device=torch_device)
    if bool(heldout["Vr_is_vector"]):
        vr = torch.as_tensor(heldout["Vr"], dtype=torch.float32, device=torch_device)
    else:
        vr = torch.full_like(va, float(heldout["Vr"]))
    trained.eval()
    with torch.no_grad():
        predicted = trained.forward_core(va, vr, nuisance_mode="neutral").cpu().numpy()
    metrics = validation_holdout.prediction_metrics(
        observed,
        predicted,
        denominator_observed=observed,
    )
    params = model.extract_params(trained)
    params["common_energy_shift_eV"] = (
        float(trained.common_energy_shift().detach().cpu())
        if trained.hypothesis_spec.nist_constrained
        else None
    )
    result = {
        "unit": asdict(unit),
        "epochs_completed": int(scorecard["epochs_completed"]),
        "best_epoch": int(scorecard["best_epoch"]),
        "best_loss": float(scorecard["best_loss"]),
        "unit_wall_seconds": float(wall_seconds),
        "train_elapsed_seconds": float(scorecard["elapsed_seconds"]),
        "metrics": metrics,
        "params": params,
        "predictions": [float(value) for value in predicted],
        "checkpoint_last_sha256": sha256_file(target / "fit" / "checkpoint_last.pt"),
        "checkpoint_best_sha256": sha256_file(target / "fit" / "checkpoint_best.pt"),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(torch_device))
            if torch_device.type == "cuda"
            else 0
        ),
    }
    atomic_json_dump(result, target / "benchmark_unit.json")
    return result


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Benchmark output already exists: {output}")
    if int(args.workers) < 1 or int(args.threads_per_worker) < 1:
        raise ValueError("workers and threads-per-worker must be positive")
    if int(args.epochs) < 1 or int(args.repeat) < 1:
        raise ValueError("epochs and repeat must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but CUDA is unavailable")
    output.mkdir(parents=True)
    baseline = resolve_baseline(output, package_root=PROJECT_ROOT / "source_data_package")
    source_paths = {
        "model.py": PROJECT_ROOT / "src" / "sublevel_detect" / "model.py",
        "validation_h4s.py": PROJECT_ROOT / "src" / "sublevel_detect" / "validation_h4s.py",
        "benchmark_h4s_runtime.py": Path(__file__).resolve(),
    }
    manifest = {
        "schema_version": 1,
        "label": str(args.label),
        "claim_evaluable": False,
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device": str(args.device),
        "workers": int(args.workers),
        "worker_backend": "process",
        "threads_per_worker": int(args.threads_per_worker),
        "epochs": int(args.epochs),
        "repeat": int(args.repeat),
        "heldout_vr": BENCHMARK_HELDOUT_VR,
        "hypotheses": list(BENCHMARK_HYPOTHESES),
        "seeds": list(BENCHMARK_SEEDS),
        "baseline_sha256": baseline.identity_hash,
        "source_hashes": {name: sha256_file(path) for name, path in source_paths.items()},
        "command": [str(value) for value in sys.argv],
    }
    atomic_json_dump(manifest, output / "benchmark_manifest.json")

    jobs_by_repeat: list[list[dict[str, Any]]] = []
    for repeat_index in range(int(args.repeat)):
        jobs: list[dict[str, Any]] = []
        for hypothesis in BENCHMARK_HYPOTHESES:
            for seed in BENCHMARK_SEEDS:
                jobs.append(
                    {
                        "unit": asdict(
                            validation_h4s.H4sUnit(
                                BENCHMARK_HELDOUT_VR,
                                hypothesis,
                                seed,
                            )
                        ),
                        "target": str(
                            output
                            / f"repeat_{repeat_index:02d}"
                            / hypothesis
                            / f"seed_{seed:03d}"
                        ),
                        "input_path": str(args.input),
                        "baseline": baseline,
                        "device": str(args.device),
                        "epochs": int(args.epochs),
                        "threads_per_worker": int(args.threads_per_worker),
                    }
                )
        jobs_by_repeat.append(jobs)
    results: list[dict[str, Any]] = []
    repeat_wall_seconds: list[float] = []
    total_started = time.perf_counter()
    for jobs in jobs_by_repeat:
        started = time.perf_counter()
        if int(args.workers) == 1:
            repeat_results = [_benchmark_unit(payload) for payload in jobs]
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
                repeat_results = list(pool.map(_benchmark_unit, jobs))
        repeat_wall_seconds.append(float(time.perf_counter() - started))
        results.extend(repeat_results)
    total_wall_seconds = float(time.perf_counter() - total_started)
    wall_seconds = float(np.median(repeat_wall_seconds))
    unit_total = sum(len(jobs) for jobs in jobs_by_repeat)
    summary = {
        **manifest,
        "ok": len(results) == unit_total,
        "unit_total": unit_total,
        "completed": len(results),
        "failed": 0,
        "wall_seconds": float(wall_seconds),
        "repeat_wall_seconds": repeat_wall_seconds,
        "total_wall_seconds": total_wall_seconds,
        "unit_wall_seconds_median": float(
            np.median([float(row["unit_wall_seconds"]) for row in results])
        ),
        "peak_cuda_memory_bytes": max(
            int(row["peak_cuda_memory_bytes"]) for row in results
        ),
        "results": results,
    }
    atomic_json_dump(summary, output / "benchmark_summary.json")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_benchmark(args)
    print(
        json.dumps(
            {
                "ok": result["ok"],
                "output": str(Path(args.output).expanduser().resolve()),
                "unit_total": result["unit_total"],
                "wall_seconds": result["wall_seconds"],
                "unit_wall_seconds_median": result["unit_wall_seconds_median"],
                "peak_cuda_memory_bytes": result["peak_cuda_memory_bytes"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
