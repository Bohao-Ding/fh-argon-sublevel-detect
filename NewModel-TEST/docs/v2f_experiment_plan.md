# v2f 饱和阈值碰撞—高斯收集门实验计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan → run → validate
- Origin Date: 2026-08-12
- Verification Status: UNVERIFIED
- Version Label: v2f-threshold-gaussian

## 已排除的路径

- v2b：自由角宽收缩到近轴向，绕过散射机制；
- v2c：固定各向同性后用 0.799 eV 热能补偿；
- v2d：能量依赖散射仍把热能、光学深度及截面形状推到边界；
- v2e：完整末次碰撞层分布仅改变 NRMSE 0.00011，谷深无改善，且训练成本约为 23 分钟/1800 epoch，故不启动正式拟合。

## 结构替换

1. 碰撞危险率使用饱和阈值函数 `1-exp[-(ΔE/E_rise)^p]`，再乘高能衰减；
2. 光学深度、阈值形状和衰减尺度使用新的参数，不继承触边参数的数值映射；
3. 有效注入能量限制为 0.02–0.30 eV，避免用 0.8 eV 冒充热阴极分布；
4. 移除独立指数可达性开关，带符号轴向能量直接进入高斯选择门；`Va=0` 仍由供给项严格为零，低正电压由高斯尾连续抑制。

最后碰撞位置一阶矩和 v2d 能量依赖角散射保持不变。13 项 H1 门槛保持冻结；若仍失败，下一步不再扩充解析计数核，而转入能量—角度网格或带真实截面的 Monte Carlo。
