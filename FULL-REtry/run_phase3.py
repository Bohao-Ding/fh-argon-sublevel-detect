from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import multiprocessing
import os
import platform
import sys
from pathlib import Path


for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fh_retry.analysis import sha256_file  # noqa: E402
from fh_retry.phase3 import run_phase3  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_workers(value: str) -> int | None:
    if value.lower() == "auto":
        return None
    workers = int(value)
    if workers < 1:
        raise argparse.ArgumentTypeError("workers must be 'auto' or a positive integer")
    return workers


def _environment(workers: int) -> dict:
    packages = {}
    for name in ("numpy", "pandas", "scipy", "openpyxl", "pytest"):
        packages[name] = importlib.metadata.version(name)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count() or 1,
        "workers": workers,
        "packages": packages,
        "thread_limits": {
            name: os.environ[name]
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
    }


def _decision_report(result: dict) -> str:
    decision = result["decision"]
    timings = result["timings"]
    gates = "\n".join(
        f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in decision["gates"].items()
    )
    endpoint = decision["without_Vr10"]
    nist = "；".join(
        f"{name}={rate:.1%}" for name, rate in decision["nist_exact_recovery_by_scheme"].items()
    )
    dominant = result["injection_selection_bias"].loc[
        result["injection_selection_bias"]["selection_rank_within_scheme"] == 1
    ]
    selection_bias = "；".join(
        f"{row['noise_scheme']} 选择 factor={row['selected_factor']:g} 的边际比例 "
        f"{row['marginal_selection_rate']:.1%}"
        for _, row in dominant.iterrows()
    )
    return f"""# Phase 3 v1 决策记录

## 门控

{gates}

全部门控：`{decision['all_gates_passed']}`。

## 结果

- 推荐主张：`{decision['recommended_claim']}`。
- 去掉 Vr=10 V：{endpoint['improved_blocks']}/5 块改善，中位 {endpoint['median_gain']:.2%}，最差 {endpoint['worst_gain']:.2%}，按原 G2 完整规则：{'PASS' if endpoint['passes_full_rule'] else 'FAIL'}。
- NIST 跨度精确恢复率：{nist}。
- 全候选边际选择诊断：{selection_bias}。因此完整回收曲线不能解释成能量分辨率阈值。
- 使用 {timings['workers']} 个工作进程；总墙钟时间 {timings['total_seconds']:.2f} s。
- 条件代理任务时间和/并行墙钟比：{timings['surrogate_parallelism_ratio']:.2f}；注入任务时间和/并行墙钟比：{timings['injection_parallelism_ratio']:.2f}。

## 解释边界

- Phase 3 检验的是二维、非可分的正余弦响应系数，不是两个物理碰撞通道。
- 组成表示、周期和块边界检查仍使用同一归档数据，属于内部开发性稳健性分析。
- 注入回收使用候选无关的经验残差和等信号功率，但仍不是仪器能量分辨率标定。
- 任何失败门和失败块均保留在 CSV 中，不以平均结果覆盖。
"""


def _write_hash_manifest(output: Path) -> Path:
    fixed_inputs = [
        ROOT / "data" / "FHdata.xlsx",
        ROOT / "PHASE3_PLAN.md",
        ROOT / "requirements-lock.txt",
        ROOT / "run_phase3.py",
        ROOT / "src" / "fh_retry" / "analysis.py",
        ROOT / "src" / "fh_retry" / "phase2.py",
        ROOT / "src" / "fh_retry" / "phase3.py",
    ]
    generated = sorted(
        path
        for path in output.iterdir()
        if path.is_file() and path.name not in {"artifacts_sha256.csv", "RUN_RECEIPT.md"}
    )
    rows = []
    for path in fixed_inputs + generated:
        rows.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = output / "artifacts_sha256.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the parallel Phase 3 robustness analysis")
    parser.add_argument("--workers", type=_parse_workers, default=None, metavar="N|auto")
    parser.add_argument("--surrogate-replicates", type=int, default=2000)
    parser.add_argument("--injection-replicates", type=int, default=500)
    args = parser.parse_args()
    if args.surrogate_replicates < 1 or args.injection_replicates < 1:
        parser.error("replicate counts must be positive")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    output = ROOT / "results" / "phase3_v1"
    output.mkdir(parents=True, exist_ok=True)

    result = run_phase3(
        ROOT / "data" / "FHdata.xlsx",
        workers=args.workers,
        surrogate_replicates=args.surrogate_replicates,
        injection_replicates=args.injection_replicates,
    )
    tables = {
        "composition_transform_audit.csv": result["transform_audit"],
        "composition_block_cv.csv": result["composition_detail"],
        "composition_summary.csv": result["composition_summary"],
        "composition_surrogate_samples.csv": result["surrogate_samples"],
        "composition_surrogate_summary.csv": result["surrogate_summary"],
        "period_profile.csv": result["period_profile"],
        "block_shift_detail.csv": result["block_shift_detail"],
        "block_shift_summary.csv": result["block_shift_summary"],
        "absolute_block_adequacy.csv": result["absolute_adequacy"],
        "endpoint_gate_consistency.csv": result["endpoint_gate"],
        "injection_selected_samples.csv": result["injection_samples"],
        "injection_confusion_calibrated.csv": result["injection_confusion"],
        "injection_recovery_calibrated.csv": result["injection_summary"],
        "injection_selection_bias.csv": result["injection_selection_bias"],
    }
    for filename, frame in tables.items():
        frame.to_csv(output / filename, index=False)

    _write_json(output / "data_audit.json", result["data_audit"])
    _write_json(output / "injection_setup.json", result["injection_setup"])
    _write_json(output / "performance.json", result["timings"])
    _write_json(output / "decision.json", result["decision"])
    _write_json(
        output / "analysis_config.json",
        {
            "workers_requested": "auto" if args.workers is None else args.workers,
            "workers_used": result["timings"]["workers"],
            "surrogate_replicates_per_task": args.surrogate_replicates,
            "injection_replicates_per_task": args.injection_replicates,
            "phase3_plan": "PHASE3_PLAN.md",
        },
    )
    _write_json(output / "environment.json", _environment(result["timings"]["workers"]))
    report = _decision_report(result)
    (output / "PHASE3_DECISION.md").write_text(report, encoding="utf-8")
    manifest = _write_hash_manifest(output)
    receipt = f"""# Phase 3 v1 运行收据

## 命令

```powershell
python run_phase3.py --workers {'auto' if args.workers is None else args.workers} --surrogate-replicates {args.surrogate_replicates} --injection-replicates {args.injection_replicates}
```

## 状态

- 退出状态：0
- 工作进程：{result['timings']['workers']}
- 总墙钟时间：{result['timings']['total_seconds']:.3f} s
- 条件代理任务时间和/墙钟比：{result['timings']['surrogate_parallelism_ratio']:.3f}
- 注入任务时间和/墙钟比：{result['timings']['injection_parallelism_ratio']:.3f}
- 输入数据 SHA-256：`{result['data_audit']['sha256']}`
- 哈希清单：`{manifest.relative_to(ROOT).as_posix()}`
- 哈希清单自身 SHA-256：`{sha256_file(manifest)}`

## 证据身份

内部开发性稳健性分析；不是独立实验确认、物理通道识别或仪器分辨率标定。
"""
    (output / "RUN_RECEIPT.md").write_text(receipt, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
