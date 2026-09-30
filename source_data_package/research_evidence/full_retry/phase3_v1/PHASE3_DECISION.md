# Phase 3 v1 决策记录

## 门控

- G10_composition_robustness: FAIL
- G11_shifted_block_robustness: PASS
- G12_prior_period_near_profile_optimum: PASS
- G13_absolute_block_adequacy: PASS
- G14_endpoint_full_rule: FAIL
- G15_nist_span_not_recoverable: PASS

全部门控：`False`。

## 结果

- 推荐主张：`composition-audited nonseparable retarding-response structure with explicit local failures and conditional NIST-spacing non-recoverability`。
- 去掉 Vr=10 V：4/5 块改善，中位 31.40%，最差 -24.83%，按原 G2 完整规则：FAIL。
- NIST 跨度精确恢复率：common_h1_block_3=0.2%；common_h1_block_5=0.0%；common_h1_block_9=0.0%；common_h1_phase=0.2%。
- 全候选边际选择诊断：common_h1_block_3 选择 factor=12 的边际比例 85.5%；common_h1_block_5 选择 factor=12 的边际比例 86.8%；common_h1_block_9 选择 factor=12 的边际比例 87.2%；common_h1_phase 选择 factor=12 的边际比例 81.0%。因此完整回收曲线不能解释成能量分辨率阈值。
- 使用 8 个工作进程；总墙钟时间 18.02 s。
- 条件代理任务时间和/并行墙钟比：7.10；注入任务时间和/并行墙钟比：4.43。

## 解释边界

- Phase 3 检验的是二维、非可分的正余弦响应系数，不是两个物理碰撞通道。
- 组成表示、周期和块边界检查仍使用同一归档数据，属于内部开发性稳健性分析。
- 注入回收使用候选无关的经验残差和等信号功率，但仍不是仪器能量分辨率标定。
- 任何失败门和失败块均保留在 CSV 中，不以平均结果覆盖。
