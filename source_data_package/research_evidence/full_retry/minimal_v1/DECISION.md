# Minimal v1 决策记录

## 结论

`ALLOW_CPU_CONFIRMATION_ONLY`

Phase A 正向筛查：`True`。这只放行下一轮 CPU 确认，不构成独立确认，也不允许把数学模态解释为具体亚能级。

## 门控

- G0_data: PASS
- G1_two_mode_stability: PASS
- G2_block_prediction: PASS
- G3_negative_controls: PASS

## 关键数值

- 单调违例：1 个；最大 0.001000 µA。
- 参考设置前两模态累计解释率：0.9778。
- 参考设置 `s2/s3`：5.506。
- 参考设置主周期：11.200 V、11.225 V。
- 敏感性设置最低前两模态累计解释率：0.9335。
- rank-2 改善分块：5/5；相对改善中位数 42.15%。
- 平均 balanced RMSE：rank-1=0.7802，rank-2=0.4779，等复杂度五次趋势=0.9948。
- 最佳错位周期对照平均 balanced RMSE：1.0268。

## 主张上限

可报告为：`stable two-dimensional period-locked coarse retarding-response redistribution`。

不得报告为：已分辨原子亚能级、已识别具体碰撞通道、已恢复校准 EEDF/能谱，或独立确认性证据。
