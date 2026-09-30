> Integration note (2026-10-01): this is a preserved module record. Current questions, corrections and publication status are indexed in [the research guide](../docs/research_overview.md) and [reproduction instructions](../docs/research_reproduction.md). Frozen injection-recovery claims are superseded by A5/A14; A4 is not executed.

# NewModel-TEST

该目录是 Frank–Hertz 下一阶段的隔离工作区。它只从旧工程复制实验数据和旧 `K=1` 预测作为只读基线；不修改主论文、旧源码或其他实验结果。

v1 核心模型把阴极供给、薄层重复非弹性碰撞和末端能量—角度选择串成一条可微前向链。当前选定的 v2l 在 v2i 基础上增加只作用于碰撞历史的低压场进程展宽，并把它固定为 H1/H4s 共享的装置闭合；H1 必须先通过冻结的 15 项曲线形态门，H4s 才会运行。

## 命令

```powershell
# 单能级正式拟合
python run.py --stage single --device cuda --output output/single_v1

# 自动硬门；通过后继续 K=1..8
python run.py --stage auto --device cuda --output output/formal_v1

# 工程 smoke（结果不是科学证据，运行后应删除输出目录）
python run.py --stage single --smoke --device cpu --output .smoke

# v2a 单能级硬门
python run_v2.py --stage h1 --device cuda --output output/v2_iteration_03

# H1 通过后运行固定 NIST 4s 四能级竞争模型
python run_v2.py --stage h4s --gate-result output/v2_iteration_03/h1/result.json `
  --device cuda --max-collisions 6 --max-epochs 1200 --output output/v2_h4s_01

# 当前 v2i 单能级模型
python run_v2.py --kernel v2i --stage h1 --device cuda `
  --output output/v2i_reproduction

# v2i-H1 通过后运行同一物理核的 H4s 竞争历史
python run_v2.py --kernel v2i --stage h4s --device cuda `
  --gate-result output/v2i_reproduction/h1/result.json `
  --max-epochs 1200 --output output/v2i_h4s_reproduction

# 当前 v2l-H1：继承平台停止的 v2j 连续参数并校准装置展宽
python -B calibrate_v2l.py --device cuda --max-collisions 6 `
  --quadrature-order 4 --output output/v2l_q4_reproduction

# v2l-H1 通过后运行同离散配置的 H4s
python -B run_v2.py --kernel v2l --stage h4s --device cuda `
  --gate-result output/v2l_q4_reproduction/h1/result.json `
  --max-collisions 6 --quadrature-order 4 `
  --output output/v2l_q4_h4s_reproduction
```

默认采用平台期早停；最大 epoch 只是上限。详细假设、门槛和方程见 [实验计划](docs/experiment_plan.md) 与 [模型规格](docs/model_specification.md)。

## 本轮正式结果

第三轮 `K=1` 已通过全部冻结形态门，随后完成 `K=1..8` 扫描；拟合误差和 BIC 都选择 `K=1`。新模型显著改善低压截止和零区泄漏，但总体 NRMSE 与峰漂移 RMSE 未优于旧现象学基线。证据、反例和适用边界见 [验证报告](docs/validation_report.md)。

- 单能级结果：`output/iteration_03/k1/`
- 多能级扫描：`output/level_scan_01/`
- 复现环境与输入哈希：`data/provenance.json`

## v2a 结构性验证

v2a-H1 在第三轮通过全部 10 项门槛：平均 NRMSE 0.0501、截止 MAE 0.6 V、最坏单曲线截止误差 1.0 V、零区 MAE 0.000361 μA，峰谷召回均为 1。H4s 将平均 NRMSE 降到 0.0483，但四通道危险率偏向最低能级且门宽远大于四能级间隔，因此只支持竞争式多能级建模，不证明四能级被独立分辨。

完整的迭代证据、反例、K=1–8 边界与 Material Passport 见 [v2a 验证报告](docs/v2_validation_report.md)。

## 历史选定的 v2i 结果

v2i-H1 在 395 epoch 触发平台提前停止，并通过全部 13 项冻结门槛：平均 NRMSE 0.04086、截止 MAE 0.4 V、`Vr=0` 低压 MAE 0.05282 μA、峰/谷位置 MAE 0.421/0.406 V。H4s 的 NRMSE 与 AIC/BIC 均差于 H1，且最低通道危险率占 93.7%，因此保留 H1，不宣称分辨出四个子能级。

- 正式报告：[v2i 验证报告](docs/v2i_validation_report.md)
- 汇总：`output/v2i_experiment_summary.json`
- H1：`output/v2i_iteration_01/h1/`
- H4s：`output/v2i_h4s_01/h4s/`

## 当前选定的 v2l 结果

v2l-H1 使用 H1 形态扫描校准并固定共享装置展宽 `10.5*exp(-Va/32) eV`。它通过全部 15 项门槛：平均 NRMSE 0.03053、截止 MAE 0.4 V、峰/谷位置 MAE 0.447/0.469 V、谷 precision 1.0、高拒斥早段 MAE 0.06964 μA、后段谷值 MAE 0.11742 μA。`Vr=8,10 V` 的独立假首谷不再出现。

H4s 的平均 NRMSE 和条件 AIC/BIC 略优，但只通过 12/15 的同形态审计，重新产生 `Vr=10 V` 的 27 V 假谷和 2 V 起板误差；最低通道占 91.4%，且 1800 epoch 未平台。因此保留 H1 为主模型，不宣称四个 4s 子能级被独立分辨。

## v2m 源码候选（尚无正式科学结果）

v2m 保持 v2l 的供给、薄层历史、装置展宽与收集门，但用同一份外部部分截面构造匹配的 H1/H4s：H1 把四条截面求和为一个有效碰撞通道，H4s 使用同一张表拆成四个竞争危险率，并固定四个 NIST 激发损失能量，不再训练公共“原子能级平移”或常数 softmax 权重。这样比较不会把“截面来源变化”和“能级数变化”混在一起。为避免把合成数据误当证据，没有经过来源核验的四通道截面 CSV 和元数据侧车时，v2m 会拒绝启动 H1/H4s 正式训练。输入格式见 `data/cross_sections/README.md`。

```powershell
# CPU 工程 smoke；测试后删除输出目录
python -B run_v2.py --kernel v2m --stage h1 --smoke --device cpu `
  --cross-section-table data/cross_sections/argon_4s_verified.csv `
  --output .smoke_v2m

# GPU 空闲且四通道截面核验完成后，才运行正式 H4s
python -B run_v2.py --kernel v2m --stage h4s --device cuda `
  --gate-result output/v2m_h1_formal/h1/result.json `
  --cross-section-table data/cross_sections/argon_4s_verified.csv `
  --max-collisions 6 --quadrature-order 4 --output output/v2m_h4s_formal
```

- 正式报告：[v2l 验证报告](docs/v2l_validation_report.md)
- 汇总：`output/v2l_experiment_summary.json`
- H1：`output/v2l_q4_iteration_01/h1/`
- H4s：`output/v2l_q4_h4s_iteration_01/h4s/`

严格物理前向模型与论文综合模型的证据边界、H4s 不可辨识原因、论文影响及 v2m 修改路线，见 [综合自比较与修订方案](docs/strict_forward_model_comparative_analysis.md)。
