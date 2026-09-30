# Phase 2 v1 决策记录

## 结果

- G4_unique_retarding_order: PASS
- G5_rank2_exceeds_conditional_null: PASS
- G6_not_driven_by_Vr10: PASS
- G7_NIST_spacing_not_specific: PASS
- G8_0p28eV_not_reliably_recovered: PASS
- G9_phase_result_survives_2pct_gain_stress: PASS

全部门控：`True`。

推荐主张：`robust coarse multichannel retarding-response structure with an explicit sublevel-identifiability boundary`。

## 关键数值

- 正确阻滞电压顺序仅有 1 个单调违例；其他排列的最佳值为 138。
- rank-2 条件代理检验的最大 p 值：0.001000。
- 去掉 Vr=10 V 后：4/5 分块改善，中位改善 31.40%。
- NIST 模板在同均值、同跨度伪模板中的误差百分位：0.748；嵌套 ridge 敏感性百分位：0.500。
- NIST 原始四频设计条件数：26369.9。
- 0.280 eV 注入模板的正确恢复率范围：0.0%–0.8%。
- ±2% 曲线级增益压力下，phase screen 通过率：100.0%。

## 物理解释

数据稳定支持粗粒度、周期锁定且跨阻滞条件非可分的响应结构；但同参数量模板不能选择出 NIST 4s 的内部间隔，0.280 eV 注入也不能稳定恢复。因此，多通道宏观响应与亚能级分辨必须作为两个不同层级的结论。

以上结果仍是同一归档数据上的内部方法证据，不替代装置元数据、重复扫描或独立实验。
