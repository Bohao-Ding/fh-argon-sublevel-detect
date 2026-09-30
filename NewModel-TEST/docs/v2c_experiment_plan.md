# v2c 固定碰撞角分布实验计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan → run → validate
- Origin Date: 2026-08-12
- Verification Status: UNVERIFIED
- Version Label: v2c-fixed-isotropic-scattering

## v2b 反例

v2b 的平均 NRMSE 降至 0.04770，但三个新增图像门均失败。拟合把角宽压到 0.191 rad，使碰撞后电子近似全部保持轴向，从而绕过最后碰撞位置闭合。光学深度、门宽和截面形状也同时逼近上界。

同一 checkpoint 的只读反事实表明：把角分布放宽到前向半球通量权重，可将后段谷值 MAE 从 0.250 降到约 0.041 μA，但未重新训练的峰和包络被过度压低。

## 唯一变化

- 固定碰撞后角分布为前向半球各向同性通量 `p(mu)=2mu`；
- 取消角宽对碰撞后散射的控制；冻结旧角宽参数，不计入有效参数数目；
- 其余 v2b 方程、数据、损失、初始化和 13 项门槛完全不变。

该分布是缺少状态/能量分辨微分截面时的低阶基线，不宣称是氩 4s 激发的精确 DCS。若固定分布仍不能同时恢复峰和谷，下一步应接入能量依赖的差分截面或能量—角度网格，而不是继续增加自由门控参数。
