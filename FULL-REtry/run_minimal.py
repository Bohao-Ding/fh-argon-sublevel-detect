from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fh_retry.analysis import run_numeric_analysis, write_json  # noqa: E402


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    data_path = ROOT / "data" / "FHdata.xlsx"
    output = ROOT / "results" / "minimal_v1"
    output.mkdir(parents=True, exist_ok=True)
    result = run_numeric_analysis(data_path)

    write_json(output / "data_audit.json", {**result["data_audit"], **result["band_audit"]})
    write_json(output / "modal_reference.json", result["modal"])
    write_json(output / "decision.json", result["decision"])
    result["bands"].to_csv(output / "coarse_bands.csv", index=False)
    result["sensitivity"].to_csv(output / "modal_sensitivity.csv", index=False)
    result["scores"].to_csv(output / "modal_scores.csv", index=False)
    result["loadings"].to_csv(output / "modal_loadings.csv", index=False)
    result["cv"].to_csv(output / "blocked_cv.csv", index=False)
    result["predictions"].to_csv(output / "blocked_predictions.csv", index=False)

    decision = result["decision"]
    cv = decision["cv_metrics"]
    modal = decision["modal_metrics"]
    gate_lines = "\n".join(
        f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in decision["gates"].items()
    )
    report = f"""# Minimal v1 决策记录

## 结论

`{decision['scale_decision']}`

Phase A 正向筛查：`{decision['phase_a_positive_screen']}`。这只放行下一轮 CPU 确认，不构成独立确认，也不允许把数学模态解释为具体亚能级。

## 门控

{gate_lines}

## 关键数值

- 单调违例：{decision['data_metrics']['monotonic_violations']} 个；最大 {decision['data_metrics']['max_violation_uA']:.6f} µA。
- 参考设置前两模态累计解释率：{modal['reference_cumulative_explained_2']:.4f}。
- 参考设置 `s2/s3`：{modal['reference_s2_over_s3']:.3f}。
- 参考设置主周期：{modal['reference_periods_V'][0]:.3f} V、{modal['reference_periods_V'][1]:.3f} V。
- 敏感性设置最低前两模态累计解释率：{modal['sensitivity_min_cumulative_explained_2']:.4f}。
- rank-2 改善分块：{cv['rank2_better_blocks']}/5；相对改善中位数 {cv['median_relative_improvement']:.2%}。
- 平均 balanced RMSE：rank-1={cv['mean_balanced_rmse_rank1']:.4f}，rank-2={cv['mean_balanced_rmse_rank2']:.4f}，等复杂度五次趋势={cv['mean_balanced_rmse_trend_degree5']:.4f}。
- 最佳错位周期对照平均 balanced RMSE：{cv['best_pseudo_period_mean_balanced_rmse']:.4f}。

## 主张上限

可报告为：`{decision['claim_limit']}`。

不得报告为：已分辨原子亚能级、已识别具体碰撞通道、已恢复校准 EEDF/能谱，或独立确认性证据。
"""
    (output / "DECISION.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
