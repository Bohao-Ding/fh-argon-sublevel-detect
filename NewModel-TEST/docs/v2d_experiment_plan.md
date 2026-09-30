# v2d 能量依赖角散射实验计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan → run → validate
- Origin Date: 2026-08-12
- Verification Status: UNVERIFIED
- Version Label: v2d-energy-dependent-scattering

## v2c 诊断

v2c 的总体 NRMSE 为 0.04739，AIC/BIC 已优于既有 v1，但后段谷值 MAE 仍为 0.244 μA。拟合把热能推到 0.799 eV，并使光学深度、截面幂和截面衰减接近上界，说明完全各向同性散射过度压低峰值，模型再用不可信的热展宽补偿。

后段谷处的主导最后碰撞位置约为间隙的 92–94%，支持“末次近阈值碰撞导致纵向能量不足”的机制，但固定角分布缺少能量依赖。

## 唯一变化

对已发生碰撞的状态，以碰撞后剩余动能 `E_post` 在两种角分布之间连续插值：

`alpha = E_post / (E_post + E_aniso)`

- `alpha→0`：近阈值，采用前向半球各向同性通量；
- `alpha→1`：较高剩余能量，趋于前向散射；
- `E_aniso` 是唯一新增的物理能标。

无碰撞态、最后碰撞位置传播、供给、碰撞截面、损失函数和 13 项门槛均不改变。该式是能量依赖 DCS 的低阶闭合；若失败，应转入状态分辨 DCS 或能量—角度网格。
