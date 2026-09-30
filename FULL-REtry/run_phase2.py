from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fh_retry.phase2 import run_phase2  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    output = ROOT / "results" / "phase2_v1"
    output.mkdir(parents=True, exist_ok=True)
    result = run_phase2(ROOT / "data" / "FHdata.xlsx")

    tables = {
        "retarding_order_permutations.csv": result["order"],
        "endpoint_exclusion_detail.csv": result["endpoint_detail"],
        "endpoint_exclusion_summary.csv": result["endpoint_summary"],
        "rank1_surrogate_samples.csv": result["surrogate"],
        "rank1_surrogate_summary.csv": result["surrogate_summary"],
        "template_cv.csv": result["template_cv"],
        "pseudo_templates.csv": result["pseudo_templates"],
        "span_profile.csv": result["span_profile"],
        "ridge_sensitivity.csv": result["ridge_sensitivity"],
        "injection_confusion.csv": result["recovery_detail"],
        "injection_recovery.csv": result["recovery"],
        "gain_offset_stress.csv": result["gain_stress"],
    }
    for filename, frame in tables.items():
        frame.to_csv(output / filename, index=False)

    _write_json(output / "template_summary.json", result["template_summary"])
    _write_json(output / "performance.json", result["timings"])
    _write_json(output / "decision.json", result["decision"])

    decision = result["decision"]
    gates = "\n".join(
        f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in decision["gates"].items()
    )
    report = f"""# Phase 2 v1 决策记录

## 结果

{gates}

全部门控：`{decision['all_gates_passed']}`。

推荐主张：`{decision['recommended_claim']}`。

## 关键数值

- 正确阻滞电压顺序仅有 {decision['physical_order_violations']} 个单调违例；其他排列的最佳值为 {decision['next_best_order_violations']}。
- rank-2 条件代理检验的最大 p 值：{decision['rank2_conditional_p_max']:.6f}。
- 去掉 Vr=10 V 后：{decision['without_Vr10_improved_blocks']}/5 分块改善，中位改善 {decision['without_Vr10_median_gain']:.2%}。
- NIST 模板在同均值、同跨度伪模板中的误差百分位：{decision['nist_pseudo_percentile']:.3f}；嵌套 ridge 敏感性百分位：{decision['ridge_nist_pseudo_percentile']:.3f}。
- NIST 原始四频设计条件数：{decision['nist_raw_design_condition_number']:.1f}。
- 0.280 eV 注入模板的正确恢复率范围：{decision['nist_recovery_rate_min']:.1%}–{decision['nist_recovery_rate_max']:.1%}。
- ±2% 曲线级增益压力下，phase screen 通过率：{decision['gain_2pct_phase_screen_pass_rate']:.1%}。

## 物理解释

数据稳定支持粗粒度、周期锁定且跨阻滞条件非可分的响应结构；但同参数量模板不能选择出 NIST 4s 的内部间隔，0.280 eV 注入也不能稳定恢复。因此，多通道宏观响应与亚能级分辨必须作为两个不同层级的结论。

以上结果仍是同一归档数据上的内部方法证据，不替代装置元数据、重复扫描或独立实验。
"""
    (output / "PHASE2_DECISION.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"总计算时间：{result['timings']['total_seconds']:.2f} s")


if __name__ == "__main__":
    main()
