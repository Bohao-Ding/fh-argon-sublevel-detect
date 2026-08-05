# Franck-Hertz 补充验证实验与进度台账

## 1. 当前研究问题

本验证层回答两个不同问题：

1. 现有自由 K 扫描中的 (K=4) 选择对 selector、复杂度惩罚和相关噪声是否稳定。
2. 已知的 Ar I 最低 (4s) 四能级组，能否在相同物理前向核下比单一激发能假设更好地预测完全留出的阻滞电压曲线。

第二个问题由针对性的四臂物理假设比较承担，不再使用 semi-synthetic recovery 或固定预算 MLP。删除旧实验不改变已完成 selector、block-bootstrap 和 holdout 结果；Git 历史仍保留旧设计的实现过程。

本阶段不修改论文、补充材料、图件、production selector、默认模型或 `source_data_package/`。技术完成仅表示代码、身份记录、失败语义和 smoke 链路可复现，不代表任何物理假设已经得到支持。

## 2. CLI

正式整合命令：

```powershell
python run.py --mode fullscan --validation --device cpu
```

当前整合顺序为 `selector`、`bootstrap`、`holdout`、`h4s`。分阶段命令：

```powershell
python run.py --mode fullscan --validation-only selector --device cpu
python run.py --mode fullscan --validation-only bootstrap --device cuda
python run.py --mode fullscan --validation-only holdout --device cuda
python run.py --mode fullscan --validation-only h4s --device cpu
```

H4s 功能 smoke：

```powershell
$smoke = Join-Path $env:TEMP 'fh_h4s_smoke'
python run.py --mode smoke --exclude hpopt --validation-only h4s --output $smoke --device cpu
Remove-Item -LiteralPath $smoke -Recurse -Force
```

Smoke 只检查运行链路和输出 schema，固定写入 `claim_evaluable=false`，不得用于论文结论。

## 3. Baseline 与身份隔离

1. 若指定输出根含完整且身份兼容的 `main/fullscan`，验证使用该本地 baseline。
2. 否则只读使用已提交 `source_data_package/` 中的 config、decision、scan table、forward evidence、selected scorecard、checkpoint 和 prediction points。
3. 每阶段记录输入、baseline、矩阵、seed、设备及源码哈希。已有阶段身份不一致时拒绝复用或覆盖。
4. 单元失败保留在预定义分母中；任一单元失败均令阶段及 CLI 返回非零。

## 4. 保留的验证阶段

### 4.1 Selector audit

冻结 production selector，对逐指标移除、分组移除及 fit-complexity-only 情景重新计分，不重新训练。当前正式结果为 15/15 情景完成：13个选择 (K=4)，移除 BIC 或整个复杂度组的2个情景选择 (K=8)。该结果说明 (K=4) 是复杂度约束下的条件性折中，不是无条件唯一解。

### 4.2 Circular moving-block residual bootstrap

在每条曲线内部对中心化残差作循环移动块采样。当前60/60单元完成且零失败；主条件 block=7、seed=0 中 (K=4) 为8/30，Wilson 95%区间为 `[0.1418, 0.4445]`，众数为 (K=5)。该结果是 exact-(K=4) 稳定性的限制证据。

### 4.3 Leave-one-(V_r)-out prediction

五折训练 K=1..8、seeds 0/1/2，以完全留出曲线上的零样本 RMSE、MAE、NRMSE 为主要终点。当前120/120单元完成且零失败；平均零样本 NRMSE 为 K1 `0.12547`、K4 `0.12269`、K8 `0.11695`。该阶段支持多通道预测改善，但不单独确定真实通道数。

## 5. 针对性 H4s 四臂比较

### 5.1 四个假设

| 键 | 物理含义 | 4s 能量 | 平滑高能背景 |
|---|---|---|---|
| `h1` | 单一有效激发能 | 一个自由能量 | 关闭 |
| `h1_background` | 单一有效激发能加背景 | 一个自由能量 | 开启 |
| `h4s` | 最低 Ar I 4s 四能级组 | NIST 相对间隔固定 | 关闭 |
| `h4s_background` | 四能级组加背景 | NIST 相对间隔固定 | 开启 |

NIST Ar I 4s 能量固定为：

```text
11.54835442, 11.62359272, 11.72316039, 11.82807116 eV
```

来源：[NIST Atomic Spectra Database](https://physics.nist.gov/cgi-bin/ASD/energy1.pl?spectrum=Ar+I&units=1&level_out=on&conf_out=on&term_out=on&j_out=on)。四个能量保持相对间隔不变，只允许共同微调 ([-0.25,0.25]) eV；现有 `phase` 参数继续表达曲线相位或接触电势相关偏移。通道权重通过 softmax 保证非负且归一化，四通道共享展宽。

`B` 直接复用模型现有的平滑 `high_energy_loss` 强度、起点和宽度。无背景模型将这三个参数冻结，并把强度严格置零；它不是第五个原子能级或振荡通道。

### 5.2 实验矩阵

- Smoke：1个留出 (V_r) × 4个假设 × 1个 seed，共4单元，使用2 epochs。
- Fullscan：5个留出 (V_r) × 4个假设 × seeds 0/1/2，共60单元。
- Fullscan 统一使用 `init_jitter_scale=0.05`。seed 仅检查优化起点敏感性，不作为独立统计样本。
- 每个单元只在其余4条曲线上训练，留出曲线只参与 neutral-nuisance 零样本评价。

### 5.3 终点和预定义对照

逐单元报告 RMSE、MAE、NRMSE、训练时间、可训练参数数、四能级共同偏移及逐点预测。预定义配对差异为：

1. (H_{4s}-H_1)
2. ((H_{4s}+B)-(H_1+B))
3. ((H_1+B)-H_1)
4. ((H_{4s}+B)-H_{4s})

负的误差差值表示候选模型误差更低。程序先计算同一 fold、同一 seed 内的配对差值，再取每个 fold 的 seed 中位数；方向一致性只以这5个 fold-level 中位差为分母。失败 seed 或 fold 仍计入预期分母。程序不自动生成“支持”或“否定”结论。

## 6. H4s 输出

`output/validation/h4s_comparison/` 包含：

- `hypothesis_manifest.json`：四个假设、NIST 数据、背景定义和预定义对照。
- `unit_status.csv`：全部成功、失败和复用单元。
- `zero_shot_metrics.csv`：逐折逐 seed 指标。
- `zero_shot_predictions.csv`：全部留出点预测。
- `paired_contrasts.csv`：四类配对差异。
- `fold_seed_medians.csv`：每个 fold 与假设的 seed 中位数及预期、可用、失败 seed 数。
- `fold_median_contrasts.csv`：每个 fold 的配对差值中位数及失败传播。
- `h4s_comparison_summary.json`：描述性汇总、fold median 和 claim 边界。
- `stage_manifest.json`、`stage_result.json`、`stage_status.json`：阶段身份和完成状态。

## 7. 当前状态

| 阶段 | 状态 | 正式结果 |
|---|---|---|
| selector audit | `complete` | 15/15，失败0 |
| block bootstrap | `complete` | 60/60，失败0 |
| held-out prediction | `complete` | 120/120，失败0 |
| targeted H4s comparison | `smoke_passed` | 尚无正式结果；CPU 与 CUDA smoke 均为4/4单元完成、0失败，schema/checkpoint 哈希完整，`claim_evaluable=false` |

已退役验证的结果目录与旧五阶段顶层进度清单已删除。保留的三组正式结果不因本次源码替换而重算。

## 8. 正式训练前性能与正确性审计

- 正式 H4s 单元使用四曲线批量前向与批量损失，但保持逐曲线损失定义、权重、3500 epochs 和早停规则不变。
- Fullscan checkpoint 改为至多每60秒周期保存；epoch 1、早停、正常结束和异常退出仍强制保存。production 默认仍为每10 epochs 保存。
- 300-epoch、两次独立重复的墙钟中位数：CPU 基线 `20.1578 s`，优化 CPU-4 `15.7539 s`，加速 `21.85%`；CUDA 基线 `83.5604 s`，优化 CUDA-4 `29.3465 s`，加速 `64.88%`。
- 优化 CPU 比优化 CUDA 快 `46.32%`，且通过跨设备数值准入门，因此正式 H4s 设备确定为 CPU，worker 数为4、每 worker 单线程。
- 同设备预测、指标、能量、共同偏移、权重及四类方向均通过有界等价门。性能数字只用于工程设备选择，不构成物理论文证据。完整审计见 `docs/h4s_performance_audit.md`。

正式 H4s 命令冻结为：

```powershell
python run.py --mode fullscan --validation-only h4s --device cpu
```

## 9. 临时文件规则

Smoke 输出必须写入 `C:\tmp` 或 pytest 临时目录。断言完成后删除整个 smoke 根；不得在仓库中遗留临时训练目录、缓存副本或 smoke 结果。正式训练必须等待新的明确指令。
