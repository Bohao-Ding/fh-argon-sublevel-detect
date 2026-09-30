# v2i 有限角孔径实验计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan → run → validate
- Origin Date: 2026-08-13
- Verification Status: UNVERIFIED
- Version Label: v2i-finite-angular-aperture

## v2h 剩余误差与代码复查

v2h 已通过截止、零区、低压、峰谷位置和漂移等 12 项门槛，仅后段谷值电流 MAE 0.2098 μA 略高于 0.205 μA。代码复查发现 v2d–v2h 把零次碰撞状态错误地放入各向同性角积分；同时角积分被归一化为总概率 1，不能表达大角度散射电子离开有限收集孔径的损失。

## 两个关联修正

1. 零次碰撞状态保持纯前向；仅碰撞状态使用能量依赖的各向同性到前向混合。
2. 各向同性分量乘不归一化高斯角孔径：`A(theta)=exp[-theta^2/(2*sigma_a^2)]`，其中 `0.15 <= sigma_a <= pi/2` rad。

两项均属于同一个角输运语义修正。供给、碰撞危险率、到达可达率、高斯能量门和冻结的 13 项判据保持不变。只运行 H1；全部通过后才允许 H4s。
