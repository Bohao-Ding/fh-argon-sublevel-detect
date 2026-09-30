# 下一阶段实验计划：薄层电子输运前向模型

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan + run + validate
- Origin Date: 2026-08-12
- Verification Status: VERIFIED_WITH_LIMITATIONS（详见 `validation_report.md`）
- Version Label: thin_layer_plan_v1 + thin_layer_validation_v1

## 研究问题与硬边界

研究问题是：一个从热电子供给、加速区薄层碰撞和末端能量—角度选择出发的低阶输运模型，能否在单能级条件下同时解释：

1. 低加速电压区的零击板电流；
2. 重复峰谷结构；
3. 峰谷随阻滞电压发生的横向移动；
4. 五条实验曲线的总体电流包络。

单能级模型通过预先冻结的形态门之后，才允许扫描 `K=1..8`。本阶段只评价拟合与上述形态，不运行鲁棒性、bootstrap、留出预测、前向选择器或神经网络基准。

## 证据、推断与模型选择

### 已有证据

- 数据包含五个阻滞电压、每条 161 个点，`Va=0..80 V`、步长 `0.5 V`。
- 旧 `K=1` 核在零电流区给出明显正电流；其低压截止 MAE 约为数伏。
- 旧振荡项直接使用 `round((Va+phase)/E)` 构造周期凹口，阻滞电压影响主要由经验门和对比度项实现。
- 严格 Franck–Hertz 描述涉及非均匀电场、弹性散射、非平衡电子分布和分析器透射；完整处理通常需要 kinetic/Boltzmann 或 Monte Carlo 方法。

### 本阶段推断

对当前 805 点教学数据，完整二维 kinetic 模型会引入无法由数据约束的气压、几何和截面自由度。因而采用一个中间层级：用薄层主方程显式推进重复非弹性碰撞，再对热能量和发射角作数值积分。它比旧经验核更接近常见物理建模，但不声称是第一性原理输运模拟。

### 采用方案

- 热电子供给：Richardson–Dushman 饱和上限与 Child–Langmuir `V^(3/2)` 空间电荷支路的平滑连接。
- 碰撞区：均匀场划分为薄层，按阈值化有效激发截面更新碰撞次数概率；每次碰撞损失一个通道能量。
- 收集区：对热发射能量和余弦角分布积分；阻滞选择采用高斯能量分辨 CDF，角接受度为高斯。
- 多能级：各通道是共享装置参数的碰撞历史子总体，使用非负归一权重组合。

## 单能级硬门

门槛在正式运行前冻结：

| 指标 | 通过条件 |
|---|---:|
| 五曲线平均 NRMSE | `<= 0.060` |
| 低压截止位置 MAE | `<= 1.50 V` |
| 观测零区预测电流 MAE | `<= 0.015 µA` |
| 峰位置 MAE | `<= 1.00 V` |
| 谷位置 MAE | `<= 1.25 V` |
| 峰召回率 | `>= 0.80` |
| 谷召回率 | `>= 0.75` |
| 跨阻滞电压峰漂移 RMSE | `<= 1.00 V` |
| `Vr=0 V` 伪峰谷数 | `0` |

任何一项失败，`K=1..8` 扫描均不得开始；先根据失败项修改单能级物理结构，而不是放宽门槛。

## 动态停止

- 最大 1800 epochs，不要求跑满。
- 最少 300 epochs；15 点平滑损失的相对改善低于 `1e-4` 且持续 120 epochs 时提前停止。
- `ReduceLROnPlateau` 在平台期减半学习率，最低 `2e-4`。
- 保存最优 checkpoint，而不是最后一轮参数。

## 验证顺序

1. 单元测试：概率守恒、非负电流、阻滞门单调性、梯度有限、平台期早停。
2. Smoke test：只验证工程入口和产物，运行后清除临时目录。
3. `K=1` 正式拟合并计算形态门。
4. 若未通过，按失败项迭代物理模型并重新从第 1 步开始。
5. 通过后扫描 `K=1..8`，分别报告拟合指标、AIC/BIC、能量和权重；不把最低拟合误差自动解释为真实能级数。

## 主要参考依据

- G. F. Hanne, *What really happens in the Franck–Hertz experiment with mercury?*, American Journal of Physics 56, 696 (1988), DOI: [10.1119/1.15503](https://doi.org/10.1119/1.15503).
- F. Sigeneger, R. Winkler and R. E. Robson, *What really happens with the electron gas in the famous Franck–Hertz experiment?*, Contributions to Plasma Physics 43, 178–197 (2003), DOI: [10.1002/ctpp.200310014](https://doi.org/10.1002/ctpp.200310014).
- P. Magyar, I. Korolov and Z. Donkó, *Photoelectric Franck–Hertz experiment and its kinetic analysis by Monte Carlo simulation*, Physical Review E 85, 056409 (2012), DOI: [10.1103/PhysRevE.85.056409](https://doi.org/10.1103/PhysRevE.85.056409).
- R. P. McEachran et al., *Near-threshold electron-impact excitation of argon studied with the time-of-flight technique*, PMC Physics B 2, 3 (2009), DOI: [10.1186/1754-0429-2-3](https://doi.org/10.1186/1754-0429-2-3).
- NIST Atomic Spectra Database, SRD 78, DOI: [10.18434/T4W30F](https://doi.org/10.18434/T4W30F).
- S. Dushman, *Electron emission from metals as a function of temperature*, Physical Review 21, 623 (1923), DOI: [10.1103/PhysRev.21.623](https://doi.org/10.1103/PhysRev.21.623).
- I. Langmuir, *The effect of space charge and residual gases on thermionic currents in high vacuum*, Physical Review 2, 450 (1913), DOI: [10.1103/PhysRev.2.450](https://doi.org/10.1103/PhysRev.2.450).
