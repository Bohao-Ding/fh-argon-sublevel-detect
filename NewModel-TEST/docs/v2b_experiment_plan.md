# v2b 最后碰撞位置闭合实验计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan → run → validate
- Origin Date: 2026-08-12
- Verification Status: UNVERIFIED（训练完成后更新）
- Version Label: v2b-last-collision-moment

## 图像诊断

v2a-H1 已解决峰谷位置与低压伪电流，但仍存在：

1. `Vr=0` 在 0–15 V 的 MAE 为 0.0811 μA，起板呈硬切换而不是热发射尾部；
2. 55 V 以上实验谷值平均比 H1 预测低 0.2539 μA；
3. 60–80 V 全曲线 MAE 为 0.1226 μA；
4. v2a 的角度只乘在热能上，碰撞后的主要剩余能量仍被视为完全轴向，角度参数无法表达末次碰撞靠近收集极时的纵向能量损失。

## 唯一结构改动

薄层主方程除概率 `P(n,x)` 外，同步传播概率加权的最后一次非弹性碰撞位置。收集端使用一阶矩闭合：

- 最后碰撞前的剩余动能按出射角重新分配；
- 最后碰撞位置到收集极之间获得的电场能保持轴向；
- 无碰撞电子保留“热发射角 + 全程轴向加速”的原有物理；
- 可达性在每个热能、角度和碰撞历史内部计算，不再用平均场能量硬开关。

该修改不增加可学习参数。它是轨迹输运的一阶矩闭合，不冒充完整 Monte Carlo。

## 冻结 H1 门槛

除 v2a 的峰谷位置、召回、漂移和零区门槛外，新增三项图像幅值门：

| 指标 | 门槛 |
|---|---:|
| 平均曲线 NRMSE | ≤ 0.050 |
| 截止 MAE / 最坏误差 | ≤ 1.0 / 1.5 V |
| `Vr=0, Va≤15 V` MAE | ≤ 0.070 μA |
| `Va≥55 V` 实验谷位置电流 MAE | ≤ 0.205 μA |
| `Va≥60 V` 全曲线 MAE | ≤ 0.110 μA |

H1 未通过时不运行 H4s。最大 epoch 只是上限，平台期早停保持启用。

## 运行与输出

- 工程 smoke：临时目录，完成后删除；
- 正式 H1：新路径 `output/v2b_iteration_*`；
- 初始化：从已通过的 `v2_iteration_03/h1/checkpoint_best.pt` 加载共同参数；
- 对照：v2a-H1、v2a-H4s、v1 物理核；
- 本轮不做鲁棒性或留出验证。
