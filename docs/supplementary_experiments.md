# Franck-Hertz 补充验证实验预注册与进度台账

## Material Passport

- Origin Skill: academic-research-suite/experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-12
- Verification Status: VERIFIED_AND_FROZEN
- Version Label: preregistration_v1
- Execution Workflow: superpowers using-superpowers, brainstorming, writing-plans, using-git-worktrees, subagent-driven-development

## 1. 文档用途与冻结规则

本文档同时承担补充验证实验的设计定稿、预注册协议、执行台账和结果索引。本文档提交并经用户审阅确认后，下列研究问题、实验矩阵、随机种子、主要终点、解释阈值和失败规则正式冻结。冻结后只能更新执行状态，并在第 14 节追加“证据—推断—限制”记录，不得依据结果回改预注册定义。

技术完成标准是：预注册单元全部具有成功或明确失败记录，输入与配置可追溯，失败单元保留在分母中，汇总可由同一 CLI 重跑。技术完成不以 K=4 被恢复、物理模型胜出或任何结果方向为条件。

## 2. 研究目标与范围边界

### 2.1 目标

1. 审计 production selector 对重复指标和训练内诊断的依赖。
2. 在保留曲线内相关性的条件下重新评估 residual-bootstrap selected-K 分布。
3. 测量模型对完全留出的阻滞电压曲线的零样本预测能力，以及有限 nuisance 校准后的迁移能力。
4. 在已知真值的 semi-synthetic 数据中测量通道数、近邻间隔、能量和权重的恢复能力。
5. 以参数预算匹配的单一 MLP 检查预测性能与参数量比较，避免对所有黑盒模型作泛化结论。

### 2.2 冻结边界

- 当前 production K=4、production forward prior 和默认 selector 只作为冻结参考，不由本轮验证自动重定义。
- 新实验写入独立 `output/validation/`，不覆盖 `output/main/`、`output/robustness/` 或 `output/sensitivity/`。
- 本阶段不修改论文、补充材料、论文图件或 `source_data_package/`。
- 新实验可以产生与当前中心结论不一致的结果；所有负结果和失败记录均须保留。
- 普通运行输出继续由 `.gitignore` 排除，只有本文档及后续源码、测试和复现文档进入版本控制。

## 3. 环境与执行入口

- Repository: `fh-argon-sublevel-detect`
- Isolated worktree: `D:\VScode_repository\FrankHertz_Experiment\.worktrees\fh-validation`
- Branch: `feature/supplementary-validation-experiments`
- Language/framework: Python 3.13, PyTorch 2.8
- Preferred formal device: NVIDIA GeForce RTX 4060 Laptop GPU through explicit `--device cuda`
- CPU fallback: allowed only when CUDA is unavailable；设备、驱动和运行时必须写入 manifest，CPU/GPU wall time 不得混合比较。

正式整合命令：

```powershell
python run.py --mode fullscan --validation --device cuda
```

Smoke 命令：

```powershell
python run.py --mode smoke --exclude hpopt --validation --output C:\tmp\fh_validation_smoke --device cpu
```

分阶段命令：

```powershell
python run.py --mode fullscan --validation-only selector --device cuda
python run.py --mode fullscan --validation-only bootstrap --device cuda
python run.py --mode fullscan --validation-only holdout --device cuda
python run.py --mode fullscan --validation-only synthetic --device cuda
python run.py --mode fullscan --validation-only benchmark --device cuda
```

`--validation-only` 自动启用 validation。正式论文复现采用不带 `--validation-only` 的整合命令；分阶段命令仅用于逐项推进、诊断和显式重跑。

## 4. Baseline 解析与证据隔离

Baseline 解析遵循以下不可交换的优先级：

1. 若指定 `--output` 下存在完整的 `main/fullscan` 与 selected-model artifact，则整套使用该本地 baseline。
2. 若本地 baseline 不完整，则整套回退到已提交 `source_data_package` 中的 production config、decision、scan tables、forward evidence、K=4 scorecard/checkpoint 和 prediction points。

同一次 validation run 使用的 decision、config、model-selection table、scan summary、forward evidence、selected scorecard、checkpoint 和 prediction points 必须来自同一 baseline root。每个必需文件记录 SHA-256；禁止跨 root 拼接。若已有完成阶段的代码版本、输入、baseline、实验矩阵或 seed 哈希不匹配，流程拒绝覆盖并要求使用新的 `--output`。

## 5. Selector decontamination audit

该实验只读取冻结 scan rows 并重算 decision，不重新训练、不修改 production selector。共冻结 15 个情景：

1. `production`
2. `leave_one_rmse_out`
3. `leave_one_legacy_summary_out`
4. `leave_one_d1_out`
5. `leave_one_d2_out`
6. `leave_one_structure_out`
7. `leave_one_physical_out`
8. `leave_one_bic_out`
9. `leave_one_aic_out`
10. `leave_one_degeneracy_out`
11. `remove_fit_group`：移除 RMSE 与 legacy summary
12. `remove_shape_group`：移除 d1、d2 与 structure
13. `remove_physical_group`：移除 physical
14. `remove_complexity_group`：移除 BIC、AIC 与 degeneracy
15. `fit_complexity_only`：仅保留 RMSE、BIC、AIC 与 degeneracy

所有移除情景都把被移除分量的权重设为0，未移除分量严格沿用 production 权重，不重新归一化。原始指标按升序生成 competition rank：数值在 `1e-12` 内相同则共享最小名次，下一名次保留位置跳跃；非有限值排在全部有限值之后。composite score并列时选择较小K。`fit_complexity_only` 同样沿用 RMSE=1、BIC=1、AIC=0.5、degeneracy=1 的production权重。

主要终点：

- 去掉 legacy summary 后 selected K 是否变化。
- 去掉训练内 shape/physical 诊断后 selected K 如何变化。
- 15 个情景的 selected-K 分布。
- 各 rank 分量在 K=1..8 上的 Spearman 相关矩阵。

该审计不加入 no-forward、prior-off、±25% rank-weight 或既有复合偏重情景；这些属于已有 ablation/robustness 证据，不在本实验重复计算。

## 6. Correlation-preserving residual bootstrap

### 6.1 生成协议

- 残差源：冻结 production K=4 `prediction_points.csv`。
- 每条曲线分别中心化残差，禁止跨曲线采样。
- 使用 circular moving-block bootstrap，保持块内相邻顺序。
- 主 block length：7 个采样点；在0.5 V等间隔网格上，块内首尾坐标跨度为3.0 V。
- 敏感性 block length：5 和 9 个采样点。

### 6.2 预注册矩阵

- block=7：30 个 data replicate，训练 seed 0。
- block=7 的 replicate 0–4：追加训练 seed 1 和 2。
- block=5：10 个 data replicate，训练 seed 0。
- block=9：10 个 data replicate，训练 seed 0。

### 6.3 终点

- 主要终点：block=7 的 selected-K 分布，以及 K=4 比例的 Wilson 95% 区间。
- 次要终点：block length 与 optimizer seed 对 selected K 的影响。
- 现有 pointwise residual bootstrap 只作为独立对照；两种 bootstrap 的行和结论不得合并。

## 7. Leave-one-\(V_r\)-out prediction

### 7.1 训练折

- 依次留出 `Vr=0,4,6,8,10 V`，共 5 折。
- 每折只使用其余 4 条曲线训练 K=1..8。
- 每个 K 使用 seeds 0、1、2，固定 production hyperparameters，不重新执行 hpopt。

### 7.2 主要零样本终点

对完全留出的曲线使用 `nuisance_mode=neutral`，不读取或拟合该曲线的 nuisance 参数。报告：

- RMSE
- MAE
- range-normalized RMSE

分别保留 K=1、K=4、K=8 和训练折 selected-K 模型的结果。

训练折 selected K 只使用该折4条训练曲线生成的 K=1..8 scan summaries，并调用冻结 production selector。forward evidence、训练损失、structure、physical、AIC/BIC和退化指标全部由训练折计算；在 selected K 固定之前不得读取留出曲线的观测值或预测指标。

### 7.3 稀疏校准次要终点

将留出曲线按 \(V_a\) 排序，固定索引 `0,8,16,...,160` 为 21 个校准点，其余 140 点为测试点。全局物理核和通道参数保持冻结，只允许拟合 gain、bias 和 \(\Delta V_a\) 三个 nuisance 参数。校准点和测试点不得重叠；报告只在 140 个测试点上计算。

零样本结果是主要预测证据；稀疏校准结果是条件性迁移诊断，二者不得互换称谓。

全文统一将 range-normalized RMSE 记为 NRMSE，定义为 `RMSE / (max(I_heldout)-min(I_heldout))`。分母固定使用该留出曲线全部161个观测点的电流范围；零样本和稀疏校准测试使用同一分母。该观测范围只参与事后指标归一化，不进入训练、模型选择或nuisance拟合。

## 8. Synthetic recovery and detectability

### 8.1 生成器

生成器继承冻结 production K=4 checkpoint 的共享物理核与曲线 nuisance，再显式注入已知通道：

- K=1：energies=`[11.5]`，weights=`[1.0]`。
- K=2：energies=`[11.5, 11.5+delta]`，weights=`[0.5, 0.5]`。
- K=4：energies=`[11.5, 11.5+delta, 12.594, 13.965]`；weights=`[0.359396, 0.396105, 0.086224, 0.158275]` 后归一化。
- K=8：能量与权重逐项读取已打包 `channel_parameters.csv` 的 K=8 行，并记录该文件 SHA-256。

`delta={0.25,0.5,1.0} V` 仅用于 K=2 和 K=4。噪声使用 block=7 的真实残差块，尺度为 `{0.5,1.0,1.5}`。该实验是 semi-synthetic、in-family 恢复上界，不构成真实仪器的独立分辨率证明。

对K=4，`delta`只定义并检验 `E2-E1`，不是四通道集合的全局最小间隔。特别是 `delta=1.0 V` 时，固定的 `E3=12.594 V` 使 `E3-E2=0.094 V`；该单元仍归类为 `E2-E1=1.0 V` 情景，并单独报告全部相邻间隔，禁止把它解释为“所有通道至少相隔1.0 V”。

### 8.2 Epoch gate

在固定代表数据上比较 800、1600 与 3500 epochs。只有当 1600 相对 3500 满足以下全部条件时，正式矩阵才使用 1600：

1. 所有 K 的 RMSE 相对差均小于 1%。
2. selected K 一致。

否则正式矩阵统一使用 3500 epochs。选择结果写入 manifest，后续不随实验方向改变。

### 8.3 Replicate 与 seed

- 全矩阵筛查：每个 truth/gap/noise 单元 3 个 replicate，训练 seed 0。
- 五个确认单元扩展为 30 个 replicate：
  - K=1，noise=1x
  - K=2，delta=0.25 V，noise=1x
  - K=4，delta=0.25 V，noise=1x
  - K=4，delta=0.5 V，noise=1x
  - K=8，noise=1x
- 每个确认单元的 replicate 0–4 追加训练 seeds 1、2。
- K=4、delta=0.25 V、noise=1x 追加相同 replicate/seed 结构的 prior-off 对照。

### 8.4 终点

- 主要终点：exact-K recovery rate 与 Wilson 95% 区间。
- 只有 Wilson 区间下界不低于 0.80，才允许称该情景下“可靠恢复”。
- 次要终点：over-selection、under-selection、Hungarian matching energy RMSE、weight MAE 与 near-pair recovery rate。
- 失败训练保留在预注册分母中，不得通过删除失败单元提高恢复率。

## 9. Matched-budget MLP benchmark

- 输入仅为 \(V_a,V_r\)，不使用 curve ID 或 embedding。
- 标准化参数只由每折训练数据计算，再原样应用于测试折。
- 网络：`2 -> 11 -> 1`，hidden activation 为 Tanh，输出为 Softplus。
- 可训练参数严格为 45，与 production K=4 参数数一致。
- 使用与 leave-one-\(V_r\)-out 完全相同的 5 折和 seeds 0、1、2。
- Optimizer：AdamW，learning rate=0.0025，weight decay=1e-4。
- Max epochs=3500；early-stop min epochs/warmup/patience=`300/300/45`。
- 早停只读取训练 loss，不查看测试折，不执行额外调参。

报告零样本 NRMSE、MAE、参数量和训练时间。结论只能表述为“这一固定参数预算 MLP 基线下”的比较，不能外推到所有深度学习模型。

## 10. 输出契约

验证根目录固定为 `output/validation/`，至少包含：

- `experiment_manifest.json`
- `progress.json`
- `validation_summary.json`
- `validation_summary.md`
- 每阶段 `stage_status.json`
- `selector_audit/` 的 scenario table、rank correlation 与 summary
- `block_bootstrap/` 的 replicate manifest、selection rows 与 block-length summary
- `holdout/` 的 fold config、zero-shot/calibrated predictions、fold metrics 与 summary
- `synthetic_recovery/` 的 truth manifest、生成数据哈希、selection rows、channel matching 与 recovery summary
- `benchmark/` 的 MLP config、fold metrics 与 paired comparison

`experiment_manifest.json` 必须记录命令、Git commit/dirty 状态、Python/PyTorch/CUDA版本、设备、输入哈希、baseline来源、场景、seed、主要终点和epoch-gate选择。

## 11. 状态、断点和失败规则

阶段状态只允许：

- `not_started`
- `smoke_passed`
- `formal_running`
- `complete`
- `incomplete`
- `blocked`

规则：

1. 只有代码、输入、baseline、矩阵和seed哈希全部匹配，已完成阶段才能复用。
2. 哈希不匹配时拒绝覆盖；流程不自动删除旧实验。
3. 数据或配置错误立即停止。
4. 单个随机训练失败时记录原因并继续其余预注册单元；阶段最终标记 `incomplete`，CLI 返回非零。
5. 失败单元保留在汇总分母中，不得静默删除。
6. 流程不自动重试；用户再次运行命令才会显式重跑缺失或失败单元。
7. 每个 fit 完成后更新 `progress.json`，记录完成数、失败数、耗时和 ETA。

## 12. 测试与验收协议

- CLI：默认关闭、`validation-only` 隐式启用、stage路由、依赖解析和旧命令兼容。
- Baseline：优先级、同根约束、必需哈希和 mismatch 拒绝复用。
- Selector：默认权重不可原地修改；15个情景及相关矩阵字段准确。
- Bootstrap：同seed可复现、行数不变、曲线隔离、block连续、残差中心化、噪声尺度正确。
- Holdout：训练集中不存在留出 \(V_r\)；校准点和测试点不重叠；仅3个nuisance参数变化。
- Synthetic：真值注入、block noise、Wilson区间和Hungarian matching正确。
- MLP：参数数严格45、输入仅 \(V_a,V_r\)、训练/测试无泄漏。
- Integration：两次独立smoke均生成最小完整输出并清理临时目录；全测试和compileall通过。
- Final audit：manifest场景数、seed、哈希、失败分母和output schema完整；论文与source-data package无改动。

## 13. 执行顺序与状态台账

| 顺序 | 阶段 | 状态 | 完成判据 |
|---:|---|---|---|
| 1 | 隔离worktree、基线测试、预注册文档 | complete | 基线测试、文档自检、提交及用户审阅确认均已完成 |
| 2 | validation基础设施与selector audit | complete | 单测、smoke、正式审计与摘要完成 |
| 3 | block residual bootstrap | not_started | 主/敏感性矩阵及区间汇总完成 |
| 4 | leave-one-\(V_r\)-out prediction | not_started | 5折零样本与稀疏校准输出完成 |
| 5 | synthetic recovery | not_started | epoch gate、筛查、确认与prior-off完成 |
| 6 | matched-budget MLP | not_started | 5折三seed及paired comparison完成 |
| 7 | 整体验证与交付 | not_started | 全测试、两次smoke、manifest与清理审计通过 |

## 14. 追加式结果记录

本节只允许在每阶段完成后追加记录，采用以下固定结构：

### 阶段 1：隔离worktree、基线测试与预注册文档

- 状态：`complete`
- 证据：已在 `feature/supplementary-validation-experiments` 隔离worktree中确认依赖全部满足；完整基线测试为28项通过、1项按环境条件跳过；原始checkout的source-data package未提交改动未被带入或修改；本文档禁用词扫描和空白错误检查通过；提交 `7878144` 经用户明确确认。
- 推断：隔离环境、现有测试基线与冻结的预注册协议满足开始补充验证开发的前置条件；该结论只涉及代码与研究设计基线，不涉及任何科学结果。
- 限制：其余实验阶段均未开始；后续只允许更新状态并在本节追加结果，不依据结果回改冻结定义。

### 后续阶段记录格式

- 状态：使用第11节定义的状态值。
- 证据：记录命令、输入/配置哈希、完成与失败分母以及主要数值输出。
- 推断：只陈述证据直接支持的结论，不把条件性结果外推为真实能级发现。
- 限制：记录数据、模型、先验、计算预算和可复现性边界。

### 阶段 2：Validation 基础设施与 selector decontamination audit

- 状态：`complete`
- 证据：正式命令 `python run.py --mode fullscan --validation-only selector --device cuda` 在干净提交 `e6ec9314fdea9daa3c70d661adbede0c3bebe2af` 上返回0；manifest记录 `dirty=false`、package baseline identity SHA-256=`5E5B921CC22BCAE3BFDD58BCCBE8BABE5435EF034B95BA7E738FF789012C49CA`、model-selection table SHA-256=`1C5517D8FD4821431B613457688B79BDCFC1B05AAE7FD982A0A1E242BF3479A7`。15个预注册情景全部完成且无失败：13/15选择K=4，`leave_one_bic_out`与`remove_complexity_group`选择K=8；移除legacy summary后仍选择K=4。相关矩阵含81个有序分量对；RMSE与legacy summary的Spearman相关为1.0，d1与d2、d1与structure均为0.95238。CLI smoke同样完成15个情景与81个相关单元，临时目录已删除；完整测试为48项通过、1项按环境条件跳过。
- 推断：在冻结scan rows上，K=4并非由legacy summary这一重复分量单独造成，也不因逐项移除shape或physical分量而改变；但选择对BIC复杂度惩罚具有实质依赖，去掉BIC或整个复杂度组时转向K=8。RMSE与legacy summary完全同序证实二者在当前表中没有独立排序信息。
- 限制：这是对既有训练结果的确定性重计分，不是新的独立数据或外推预测验证；13/15情景的K=4比例不能解释为统计置信度。该结果只定位selector的依赖结构，不证明K=4是真实能级数，也不支持重新定义production selector。

## 15. 临时文件清理

Smoke测试产生的目录必须位于 `C:\tmp` 或 pytest 临时目录，并在断言完成后删除。正式扰动输入和运行结果只进入被忽略的 `output/validation/`。任何用于诊断的额外临时目录在证据收集完成后立即清除；工作区中不得遗留smoke输出、缓存副本或未登记的实验结果。
