# v2j 碰撞场程分布实验计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: diagnose → plan → run → validate
- Origin Date: 2026-08-13
- Verification Status: UNVERIFIED
- Version Label: v2j-collision-field-ensemble

## 研究问题

v2i 已通过 13 项单能级门，但在 `Vr=8,10 V` 的 24.5/25.0 V 产生观测中不存在的早期谷。状态分解表明，零次碰撞电流在约 20 V 过快衰减，而一次碰撞电流到约 28–32 V 才接续。完整末次碰撞位置分布、固定/动态收集门展宽以及随机总光学深度都不能在保留后段形态时消除该谷。

## 唯一结构修正

实际加速电压继续决定阴极供给和电子到达网格时的总静电功。仅在推进非弹性碰撞历史时，对有效场程使用五点高斯分布：

`V_collision = max(Va + z*sigma_c(Va), 0)`，

`sigma_c(Va) = sigma_c0 * exp(-Va/V_decay)`。

它是弹性角散射、空间非局域路径和热阴极管场/密度非均匀性的低阶闭合，不声称替代轨迹 Monte Carlo。v2i 的供给、碰撞危险率、末次碰撞位置、到达可达率、有限角孔径和高斯拒止门保持不变。

## 冻结成功标准

保留 v2i 的全部 13 项门，并新增：

- `trough_precision == 1.0`；
- `Vr in {8,10} V` 且 `15 <= Va <= 30 V` 的 MAE `<= 0.075 μA`。

仅当 15/15 项全部通过，并且图像不产生新的峰谷丢失，才认为 v2j 优于 v2i。H1 失败时不运行 H4s，也不放宽门槛。
