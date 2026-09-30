# v2g 广义供给包络实验计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan → run → validate
- Origin Date: 2026-08-13
- Verification Status: UNVERIFIED
- Version Label: v2g-generalized-supply

## v2f 剩余误差

v2f 把平均 NRMSE 降至 0.04361，峰/谷位置 MAE 降至 0.368/0.375 V，60–80 V MAE 降至 0.1001 μA；仅 `Vr=0` 低压 MAE 0.0868 μA 与后段谷值 MAE 0.2122 μA 未过门。

逐点残差显示 `Vr=0` 在 8–15 V 系统性低估 0.05–0.22 μA，而高压包络接近正确。固定 `Va^1.5` 供给律不能提高低压/高压电流比。

## 唯一变化

将固定 Child–Langmuir 指数 1.5 替换为有界有效指数：

`I_supply = I_lim * u^p / (1 + u^p),  u = a*Va/V_scale,  0.8 <= p <= 1.8`。

该指数表达热发射受限和理想空间电荷受限之间的有效过渡。v2f 的碰撞危险率、注入能量范围、角散射、选择门、损失和 13 项门槛均保持不变。H1 通过后才考虑 H4s。
