# v2l 物理核拟合验证报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: diagnose → falsify → constrained-calibrate → validate
- Origin Date: 2026-08-13
- Verification Status: VERIFIED_WITH_LIMITATIONS
- Selected Version: `v2l-calibrated-collision-field / H1`
- Scope: 只检验同一数据集上的拟合形态；未运行鲁棒性、留出预测、bootstrap、前向选择门或神经网络对照

## 结论

当前应选择 **v2l-H1** 作为主物理核。它保留 v2i 的阴极供给、薄层竞争碰撞、末次碰撞位置、能量—角度收集门，并仅为“碰撞历史”增加随加速电压衰减的场进程分布；供给电压和最终收集能量仍严格使用真实 `Va`。在同一 32 薄层、6 次可达碰撞上限、4 点热/角度求积下，v2l-H1 通过冻结的全部 15 项门槛。

用户指出的三个核心问题得到实质缓解：

1. 五条曲线的起板截止预测为 `5.5, 7.0, 9.0, 10.5, 13.0 V`，观察值为 `5.5, 7.0, 9.0, 11.0, 14.5 V`；截止 MAE 为 0.4 V，最坏误差为 1.5 V；
2. 观测峰和谷全部召回，预测谷与观测谷一一对应，谷 precision 从 v2i/v2j 的 0.889 提升至 1.0；`Vr=8,10 V` 在约 24–27 V 的独立假首谷不再出现；
3. 峰/谷位置 MAE 为 0.447/0.469 V，跨拒斥电压峰漂移 RMSE 为 0.482 V，60–80 V MAE 为 0.0678 μA。

图像上 `Vr=8,10 V` 的首周期仍有轻微肩部，高拒斥早段 MAE 为 0.0696 μA，接近 0.075 μA 门槛；`Vr=10 V` 起板仍提前 1.5 V。因此结论是“当前低阶核已能表达主要形态”，不是“完整输运已被唯一恢复”。

## 当前问题的根因

### v2i/v2j 的零次—一次碰撞交接

v2i 在 `Vr=8,10 V` 的 24.5/25.0 V 附近产生观测中没有的谷。状态分解表明，约 20 V 时电流几乎完全来自零次碰撞分量；该分量在一次碰撞分量于 28–32 V 接续之前下降过快，形成假的首周期凹口。

v2j 把高斯分布只施加于碰撞历史推进，整体 NRMSE 从 v2i 的 0.04086 降到 0.02943，高拒斥早段 MAE 降到 0.0674，后段谷值 MAE 降到 0.1339。但拟合出的低压展宽只有 5.47 eV，仍留下两个假谷，并使 `Vr=10 V` 起板提前 2 V。7–31 点 Hermite 求积均收敛到同一结果，排除了五点求积离散伪影。

### 被否决的 v2k 能量门

v2k 将普通高斯 CDF 改为严格亚阈值为零的半高斯累计门：

`T_E = erf(max(E_axial - E_barrier, 0) / (sqrt(2) * sigma_E))`。

该修改在物理边界上更严格，但正式 H1 只通过 11/15：平均 NRMSE 0.03349、截止 MAE 1.8 V、最坏误差 2.5 V、谷 precision 0.889、高拒斥早段 MAE 0.0919；1800 epoch 仍未平台。图像显示它反而把各拒斥曲线的首周期刻得更深。因此“高斯门的亚阈值尾”不是假首谷的主因，v2k 被否决。

## v2l 的物理闭合

令碰撞历史看到的有效场进程为

`Va_collision(q) = max(Va + z_q * sigma_c(Va), 0)`，

其中

`sigma_c(Va) = 10.5 eV * exp(-Va / 32 V)`，

`z_q` 和权重由五点 Gauss–Hermite 求积给出。该分布只用于推进薄层内的碰撞概率与末次碰撞位置；阴极供给和收集器门仍使用实际 `Va`。因此它表示弹性角散射、非局域路径长度和装置场不均匀对“何时发生碰撞”的低阶综合展宽，不等价于给真实加速电压加噪声，也不是完整 Boltzmann/轨迹 Monte Carlo。

63 点冻结网格扫描中有 21 点通过 15/15。选择规则是在可行点中取平均 NRMSE 最低者，得到 `sigma0=10.5 eV, Vdecay=32 V`；其邻域存在连续可行带，不是单一边界巧合。这两个量被固定为 H1/H4s 共享的装置闭合参数，不随能级通道单独变化。

普通训练损失与离散形态门并不等价：在冻结上述展宽后，H1 初始化时通过 15/15，但再下降 25 epoch 后总损失更低、假谷却重新出现；50 epoch 后最坏起板误差也重新达到 2 V。因此正式 v2l 采用两阶段约束拟合：先继承 v2j 在 375 epoch 的平台最优连续参数，再做预先列出的装置坐标校准。该选择使用了同一数据集和形态门，属于约束内拟合，不是独立验证。

## H1 冻结门结果

| 指标 | 门槛 | v2l-H1 | 结果 |
|---|---:|---:|---|
| 五曲线平均 NRMSE | ≤ 0.050 | 0.030528 | 通过 |
| 截止 MAE | ≤ 1.0 V | 0.4 V | 通过 |
| 最坏截止误差 | ≤ 1.5 V | 1.5 V | 通过（边界） |
| 零区 MAE | ≤ 0.015 μA | 0.001025 μA | 通过 |
| 峰位置 MAE | ≤ 1.0 V | 0.447 V | 通过 |
| 谷位置 MAE | ≤ 1.25 V | 0.469 V | 通过 |
| 峰召回率 | ≥ 0.80 | 1.0 | 通过 |
| 谷召回率 | ≥ 0.75 | 1.0 | 通过 |
| 峰漂移 RMSE | ≤ 1.0 V | 0.482 V | 通过 |
| `Vr=0` 伪特征数 | = 0 | 0 | 通过 |
| `Vr=0` 低压 MAE | ≤ 0.070 μA | 0.056679 μA | 通过 |
| 后段谷值电流 MAE | ≤ 0.205 μA | 0.117420 μA | 通过 |
| 60–80 V MAE | ≤ 0.110 μA | 0.067835 μA | 通过 |
| 谷 precision | = 1.0 | 1.0 | 通过 |
| `Vr>=8, 15<=Va<=30 V` MAE | ≤ 0.075 μA | 0.069639 μA | 通过 |

### 与前一版本比较

| 模型 | 平均 NRMSE | 最坏截止误差 (V) | 谷 precision | 高拒斥早段 MAE (μA) | 后段谷 MAE (μA) | 判定 |
|---|---:|---:|---:|---:|---:|---|
| v2i-H1 | 0.040864 | 1.5 | 0.889 | 0.10076 | 0.20494 | 旧 13 项通过，新形态门失败 |
| v2j-H1 | **0.029427** | 2.0 | 0.889 | **0.06741** | 0.13388 | 13/15，通过总误差但保留假谷 |
| v2k-H1 | 0.033493 | 2.5 | 0.889 | 0.09192 | 0.11915 | 11/15，否决 |
| **v2l-H1** | 0.030528 | **1.5** | **1.0** | 0.06964 | **0.11742** | **15/15，选定** |

v2l 的平均 NRMSE 略高于无约束 v2j，但这是消除假首谷并恢复起板边界的显式代价；选择依据是冻结的形态约束，而不是只比较均方误差。

## 多能级如何自然映入多通道

H4s 状态不是四条独立电流曲线的末端叠加，而是碰撞计数向量 `n=(n1,n2,n3,n4)`。累计损失为 `sum(nk*Ek)`；每一薄层内四个通道以归一化危险率竞争，允许 `(1,1,0,0)` 等混合历史。四个能级固定 NIST Ar I 4s 相对间隔，只学习公共平移和通道率；供给、角孔径、收集门与 v2l 装置展宽全部共享。

| 指标 | H1 | H4s | H4s−H1 |
|---|---:|---:|---:|
| 平均 NRMSE | 0.030528 | **0.030306** | -0.000222 |
| SSE (μA²) | 4.49863 | **4.32112** | -0.17751 |
| 条件 AIC | -4137.59 | **-4164.00** | -26.41 |
| 条件 BIC | -4048.47 | **-4060.80** | -12.34 |
| 最坏截止误差 (V) | **1.5** | 2.0 | +0.5 |
| 谷 precision | **1.0** | 0.941 | -0.059 |
| `Vr=0` 伪特征数 | **0** | 1 | +1 |

H4s 的拟合能量为 `[11.2896, 11.3648, 11.4644, 11.5693] eV`，通道率为 `[0.9136, 0.0547, 0.0146, 0.0171]`。最低通道占 91.36%，其余三通道合计仅 8.64%。H4s 的条件 AIC/BIC 更好，说明在固定装置闭合和同一训练数据上，多能级自由度可以吸收一部分残差；但它又产生 `Vr=0` 的 76.5 V 伪峰、`Vr=10 V` 的 27 V 假谷，并把该曲线起板提前 2 V，后验应用同一形态门仅通过 12/15。

此外 H4s 在 1800 epoch 仍持续下降，最佳点就是最后一轮，未进入平台。故本轮结论是：**多能级竞争历史是自然且可运行的映射，条件拟合给出有限支持，但当前解未通过主形态约束、通道分配高度偏斜，也未达到平台，不能替代 H1，更不能声称四个 4s 子能级被独立分辨。**

绝对 AIC/BIC 还需注意：两项装置展宽常数由同一 H1 数据的 63 点网格选出，却未计入标准连续参数数目；H1/H4s 的条件差值仍共享该闭合，但不能当作留出证据或唯一模型证据。

## 数值收敛与计算裁剪

- 将碰撞上限从 8 降到 6，H1 最大电流差仅 `3.58e-7 μA`；
- 从 6 降到 5，最大差增至 `0.662 μA`，主要破坏 60–80 V 后段，因此 6 是最小可接受上限；
- 热/角度求积从 6 点降到 4 点，最大差 `0.00211 μA`、平均差 `0.000409 μA`，H1 的 15/15 判定不变；
- 正式公平比较采用 32 薄层、6 次碰撞、4 点求积：H1 为 7 个状态，H4s 为 210 个混合历史状态；
- 首次 8 次上限 H4s 有 495 状态，运行 689 s 仍未完成且没有输出文件，核验后终止并删除空目录；6 次/6 点版本又受外层 1200 s 上限终止且未产出文件，随后以数值等价的 4 点配置完成正式比较；
- 正式 H4s 用时 2280.1 s，1800/1800 epoch，`stop_reason=max_epochs`；H1 的连续源拟合在 375 epoch 因平台提前停止，随后只做确定性坐标校准。

## 参数边界与证据限制

H1 的能量门宽、碰撞截面上升尺度、高能衰减尺度和供给尺度接近上界，加速比例接近下界。这说明多个有效参数仍共同承担真实截面、场分布、气体密度和仪器响应，不应逐个解释为独立测量。

Magyar、Korolov 与 Donkó 的 Frank–Hertz 动力学研究使用 Monte Carlo，并指出栅极场穿透和电势不均匀会平滑实验曲线；其模型还显式包含弹性散射、多激发能级和电离。当前 v2l 只把其中“碰撞进程分散”压缩成两参数闭合，因此是可微的低阶近似，不是第一性原理输运求解器。氩的角散射与激发微分截面本身也具有能量依赖性，完整解释需要相应截面数据或轨迹级计算。

本轮只证明 v2l-H1 与当前 805 个拟合点在预设形态约束下兼容。没有新的留出、重复实验或压力/温度扫描，故不检验参数唯一性、跨装置迁移或多能级可辨识性。

## 可复现产物

- 选定 H1 结果：`output/v2l_q4_iteration_01/h1/result.json`
- 选定 H1 图像：`output/v2l_q4_iteration_01/h1/fit.png`
- H1 装置校准网格：`output/v2l_q4_iteration_01/h1/calibration_scan.csv`
- H1 checkpoint：SHA-256 `D1311914232C8C21620BA9E3EB300325EC4096F6FA2632AE770B12E46B7B8EFC`
- H4s 结果：`output/v2l_q4_h4s_iteration_01/h4s/result.json`
- H4s 图像：`output/v2l_q4_h4s_iteration_01/h4s/fit.png`
- H4s 后验形态审计：`output/v2l_q4_h4s_iteration_01/h4s_morphology_audit.json`
- H1/H4s 比较：`output/v2l_q4_h4s_iteration_01/comparison.json`
- H4s checkpoint：SHA-256 `9E1C2DDE699164DFB08BE936169A66C34A877B7391314EF26E73CF550659F228`
- 总结：`output/v2l_experiment_summary.json`
- 输入数据：SHA-256 `FF4B316B7D11C84B99E07B9AE0E32AFD03EEFD33698CF9DE3D02062D7F1902F3`
- 环境：Python 3.13.7、PyTorch 2.8.0+cu128、NVIDIA GeForce RTX 4060 Laptop GPU

复现 H1 装置坐标校准：

```powershell
python -B calibrate_v2l.py --device cuda --max-collisions 6 `
  --quadrature-order 4 --output output/v2l_q4_reproduction
```

H1 通过后复现 H4s：

```powershell
python -B run_v2.py --kernel v2l --stage h4s --device cuda `
  --gate-result output/v2l_q4_reproduction/h1/result.json `
  --max-collisions 6 --quadrature-order 4 `
  --output output/v2l_q4_h4s_reproduction
```

## 参考依据

- P. Magyar, I. Korolov and Z. Donkó, *Photoelectric Franck–Hertz experiment and its kinetic analysis by Monte Carlo simulation*, Physical Review E 85, 056409 (2012), [author-hosted full text](https://plasma.szfki.kfki.hu/~zoli/pdfs/PRE_2012_Magyar_Franck-Hertz_experiment.pdf), [PubMed record](https://pubmed.ncbi.nlm.nih.gov/23004881/).
- G. Okhrimovskyy et al., *Electron anisotropic scattering in gases: a formula for Monte Carlo simulations*, Physical Review E 65, 037402 (2002), [PubMed record](https://pubmed.ncbi.nlm.nih.gov/11909325/).
- M. Allan, O. Zatsarinny and K. Bartschat, *Near-threshold electron-impact excitation of argon: Differential cross sections and spin asymmetries*, Physical Review A 74, 030701 (2006), [APS record](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.74.030701).
- NIST Atomic Spectra Database, Ar I energy levels, [NIST Handbook table](https://physics.nist.gov/PhysRefData/Handbook/Tables/argontable5.htm).
