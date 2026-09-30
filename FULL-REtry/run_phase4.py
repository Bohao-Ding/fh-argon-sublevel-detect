from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path


for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fh_retry.analysis import sha256_file  # noqa: E402
from fh_retry.phase4 import run_phase4  # noqa: E402


def _parse_workers(value: str) -> int | None:
    if value.lower() == "auto":
        return None
    workers = int(value)
    if workers < 1:
        raise argparse.ArgumentTypeError("workers must be 'auto' or a positive integer")
    return workers


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _environment(workers: int) -> dict:
    packages = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "pandas", "scipy", "openpyxl", "pytest")
    }
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


def _interpretation(result: dict) -> str:
    decision = result["decision"]
    rows = decision["reference_raw_results"]
    reference = "\n".join(
        f"- {row['direction']}：{row['improved_blocks']}/5 块改善，"
        f"中位 {row['median_gain']:.2%}，最差 {row['worst_gain']:.2%}，"
        f"严格规则 {'PASS' if row['strict_G2_pass'] else 'FAIL'}。"
        for row in rows
    )
    limits = "；".join(
        f"{name}={value if value is not None else '未通过'} V"
        for name, value in decision[
            "max_tested_lag_with_raw_strict_pass_both_directions_V"
        ].items()
    )
    observed_zero = decision["observed_5level_zero_lag_raw"][0]
    virtual_zero = decision["virtual_6level_zero_lag_raw"]
    zero_lines = "\n".join(
        f"- {row['design']} / {row['direction']}：{row['improved_blocks']}/5，"
        f"中位 {row['median_gain']:.2%}，最差 {row['worst_gain']:.2%}。"
        for row in virtual_zero
    )
    partition_failures = decision["measured_partition_raw_failures"]
    measured_partition_text = (
        "无严格规则失败。"
        if not partition_failures
        else "；".join(
            f"{row['levels']}：{row['improved_blocks']}/5，中位 {row['median_gain']:.2%}，"
            f"最差 {row['worst_gain']:.2%}"
            for row in partition_failures
        )
    )
    return f"""# Phase 4：12 条虚拟正反向扫描结果解释

## 证据身份

本阶段没有获得新的实验曲线。主数据集中的 `Vr=2 V` 曲线由 `Vr=0` 与 `4 V` 的逐点中值构造，正、反向差异由冻结的一阶电压域滞后模型产生。结果只能用于分析敏感性，不能作为扫描迟滞、重复性或热循环稳定性的实测证据。

## 主设计

主设计使用六个阻滞条件、两个扫描方向和 `tau_V=0.5 V`，共 12 条虚拟曲线。

{reference}

- 12 曲线联合门：`{'PASS' if decision['reference_12_curve_gate_passed'] else 'FAIL'}`。
- 最大方向 RMSE：`{decision['reference_max_direction_rmse_uA']:.6f} uA`。
- 最大方向 RMSE/曲线幅度：`{decision['reference_max_direction_rmse_over_curve_span']:.2%}`。

## 滞后与插值敏感性

- 原五个实测阻滞条件在零滞后时复现 {observed_zero['improved_blocks']}/5 块改善、中位 {observed_zero['median_gain']:.2%}、最差 {observed_zero['worst_gain']:.2%}。
- 原五条件在正、反向同时通过严格规则的最大已测试滞后为 `{decision['observed_5level_max_tested_lag_with_raw_strict_pass_both_directions_V']} V`。
- 正、反向同时通过原始分数严格规则的最大已测试滞后：{limits}。
- `linear_vr2` 是主插值，`pchip_vr2` 是对 `Vr=2 V` 构造方式的敏感性检查。

六带设计在零滞后时已经失败：

{zero_lines}

线性 `Vr=2 V` 插值把原 `0–4 V` 响应带精确分成两个重复子带。主设计中两子带的最小相关系数为 `{decision['reference_D02_D24_min_correlation']:.12f}`，第二独立成分的最大方差份额为 `{decision['reference_D02_D24_max_second_singular_variance_fraction']:.3e}`。因此，六带失败首先是人为带划分对评价权重的影响，而不是新的方向测量推翻了原结论。

## 实测阻滞边界合并检查

在只使用实测 `Vr={0,4,6,8,10} V` 且固定保留 0 和 10 V 端点时，共枚举 7 种内部边界组合。原始分数严格规则通过 `{decision['measured_partition_raw_strict_pass_count']}/{decision['measured_partition_raw_total']}` 种。失败方案：{measured_partition_text}。

这项检查使用的仍是同一数据，但没有插值生成新条件。它直接限定“非共同波形”结论对实测阻滞带合并的敏感程度。

## 阶段决策

`{decision['next_stage_decision']}`。

不生成 36 条虚拟热循环。增加合成循环只会重复相同假设，不能补充独立物理信息。可纳入正文的最强表述是：现有结论已经在一个明确的一阶扫描滞后模型和两种 `Vr=2 V` 插值下接受方向敏感性检查。是否以及如何写入正文，必须同时报告失败滞后和合成证据身份。

## 禁止表述

- 获得了新的正反向实验扫描；
- 测得了扫描迟滞；
- 证明了热循环重复性；
- `tau_V` 是经过装置校准的时间常数或电压误差。
"""


def _write_manifest(output: Path) -> Path:
    fixed = [
        ROOT / "data" / "FHdata.xlsx",
        ROOT / "PHASE4_PLAN.md",
        ROOT / "requirements-lock.txt",
        ROOT / "run_phase4.py",
        ROOT / "src" / "fh_retry" / "analysis.py",
        ROOT / "src" / "fh_retry" / "phase2.py",
        ROOT / "src" / "fh_retry" / "phase3.py",
        ROOT / "src" / "fh_retry" / "phase4.py",
        ROOT / "tests" / "test_phase4.py",
    ]
    generated = sorted(
        path
        for path in output.iterdir()
        if path.is_file() and path.name not in {"artifacts_sha256.csv", "RUN_RECEIPT.md"}
    )
    rows = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in fixed + generated
    ]
    manifest = output / "artifacts_sha256.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 4 virtual scan-direction stress test")
    parser.add_argument("--workers", type=_parse_workers, default=None, metavar="N|auto")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    output = ROOT / "results" / "phase4_virtual_scans_v1"
    output.mkdir(parents=True, exist_ok=True)
    result = run_phase4(ROOT / "data" / "FHdata.xlsx", workers=args.workers)

    tables = {
        "virtual_scans_reference_12.csv": result["reference_12_curves"],
        "virtual_scans_all_sensitivity.csv": result["all_virtual_curves"],
        "lag_direction_summary.csv": result["analysis_summary"],
        "lag_direction_block_detail.csv": result["block_detail"],
        "retarding_order_audit.csv": result["order_audit"],
        "forward_reverse_difference.csv": result["direction_differences"],
        "vr2_partition_diagnostic.csv": result["partition_diagnostic"],
        "measured_partition_summary.csv": result["measured_partition_summary"],
        "measured_partition_block_detail.csv": result["measured_partition_detail"],
    }
    for filename, frame in tables.items():
        frame.to_csv(output / filename, index=False)

    _write_json(output / "data_audit.json", result["data_audit"])
    _write_json(output / "analysis_config.json", result["config"])
    _write_json(output / "decision.json", result["decision"])
    _write_json(output / "performance.json", result["timings"])
    _write_json(output / "environment.json", _environment(result["timings"]["workers"]))
    report = _interpretation(result)
    (output / "RESULTS_INTERPRETATION.md").write_text(report, encoding="utf-8")
    manifest = _write_manifest(output)
    receipt = f"""# Phase 4 虚拟扫描运行收据

## 命令

```powershell
python run_phase4.py --workers {'auto' if args.workers is None else args.workers}
```

## 状态

- 退出状态：0
- 分析任务：{result['timings']['tasks']}
- 工作线程：{result['timings']['workers']}
- 总墙钟时间：{result['timings']['total_seconds']:.3f} s
- 任务时间和/并行墙钟比：{result['timings']['parallelism_ratio']:.3f}
- 主虚拟曲线：12 条，共 {len(result['reference_12_curves'])} 个点
- 输入数据 SHA-256：`{result['data_audit']['sha256']}`
- 哈希清单：`{manifest.relative_to(ROOT).as_posix()}`
- 哈希清单自身 SHA-256：`{sha256_file(manifest)}`

## 证据身份

合成方向—滞后敏感性分析；不是新的实验扫描、热循环重复或仪器标定。
"""
    (output / "RUN_RECEIPT.md").write_text(receipt, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
