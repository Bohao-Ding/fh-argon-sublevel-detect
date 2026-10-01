# SubLevel Detect 中文说明

## 核心方法：完整曲线的有效能量反演

物理约束下的完整曲线建模与有效能量反演是核心方法贡献。神经网络式训练承担响应参数估计；早期自由 K=4 案例是方法基础与比较基线。`spectrum` 流程拟合非负归一化的平滑有效激发能量分布，定位集中区并将主峰与氩第一激发范围比较；已完成的局部离散假设作为补充探索。它不把原子能级改为连续能级，也不将其混作收集电子的剩余动能谱。

协议只用 0/4/6/8 V 的 644 点训练，完成嵌套整曲线留出、核宽剖面、30 次条件性残差重采样及 20 组合成回收，最后预测原样保留的 10 V 曲线，作为已知异常条件压力测试。见[冻结协议](docs/SPECTRAL_PROTOCOL.md)与[执行和参数复播说明](docs/spectral_reproduction.md)。正式矩阵已完成。H1/G1/C 的四条件平均留出 NRMSE 为 0.10833/0.10824/0.12242，连续表示没有整体预测优势；局部四态没有稳定胜过简单表示及等间距对照。

```powershell
python run.py --experiment spectrum --mode smoke --device cpu --output output/spectral_smoke
python run.py --experiment spectrum --mode fullscan --device cpu --output output/spectral_v1
```

正式结果给出 H1 有效尺度 11.7636 eV、C 条件性峰位 12.04 eV；最终选择局部单能量 d1。30 次条件性重采样和 20 组合成回收已完成，没有得到四态间隔支持。最终九个第一阶段拟合均达到 3500 轮上限，未称收敛；外层 6/8 V 折的内层局部选择为空，候选诊断评分保留。

[独立最小证据](source_data_package/spectral_evidence_v1/README.md)包含 252 文件、实际字节 SHA-256 与来源映射。0.01 eV 网格全部 1343 个拟合测度通过数值门；参数复播 43,470 点最大差 2.38×10⁻⁷ μA。Smoke 只验证工程；正式输出、PDF 和完整新图留在本地。

## 当前文章重点：能量集中区

连续反演首先定位有效能区，不要求产生四个分立态。C 的连通半高区间为 11.48–12.62 eV，包含有效质量 48.7%，覆盖已知 4s 范围 11.548–11.828 eV。峰顶 12.04 eV 比该组上界高 0.212 eV，整个分布落入四态窄区间的质量仅为 11.2%；这是条件于核的粗尺度相容，不是精确恢复或原子占据比例。G1 在同一区间的有效质量为 76.0%。见 `concentration.csv` 和 `scripts/summarize_spectral_concentration.py`；已完成的离散比较保留为补充探索。

## 固定增益与零点的二轮检验

新增训练数据限定的读出校准检验：四个外层 C 训练集及最终训练集均保留原读出。共同电流增益/零点使未选入正式方案的 C 诊断平均由 0.12242 降至 0.11570，但 4 V 一折变差，未采用。电压比例与自由激发尺度存在混合解释，归档不能独立确定仪器误差。物理源码哈希及能量谱保持不变。随 Vr 变化的探索性读出改善较大，表示条件响应差异，不能直接作为仪器校准。见[诊断与物理讨论](docs/AFFINE_CALIBRATION_AUDIT.md)。

```powershell
python -B scripts/audit_affine_calibration.py --output output/calibration_audit_replay
```

## 历史整合研究（805 点口径）

物理结构约束的可微模型能够描述归档曲线，并在指定先验与筛选规则下提出多成分候选。本仓库现已纳入碰撞历史模型、阻滞响应分析及后续诊断，依次检验预测作用、解释必要性、宏观信息和能级专一性。入口：[研究总览](docs/research_overview.md)、[独立复现](docs/research_reproduction.md)、[详细中文主线](docs/research_storyline.zh-CN.md)。新增最小数值证据在 [research_evidence](source_data_package/research_evidence/README.md)；文章 PDF 与新增完整图件留在本地。

自由 K=4 是条件性候选，与固定相对间隔 H4s 不同。真正的 H4s 留出有 60 个完成单元，平均 NRMSE 改善 2.36%，端点反转。自由 K 导出的旧 `cv_rmse_mean` 是拟合误差。selected-model residual bootstrap 的 30 次均未选 K=4，K=5/6 各10次；后期 A5 跨度 argmin 是另一选择器。A4 公平截面比较仍未执行。

[English README](README.md)

这是用于检验氩 Franck-Hertz 教学实验中“四个最低 Ar I 4s 能级共同参与”解释并复现论文结果的源码项目。

本仓库在常规复现层面采用“代码优先”策略：仓库包含源码、输入数据表、测试与说明文档；新的运行输出仍写入 `output/`，不作为普通源码变更提交。为便于手稿核验，当前版本单独提交了整理后的 `source_data_package/`，其中包含当前论文草稿使用的图像资产、manuscript-facing CSV/JSON 表、保留的 K=1/K=4 运行记录和校验清单。

## 技术报告

实现细节记录在：

- `Techique_Report.md`：英文技术报告。
- `Techique_Report_zh.md`：中文技术报告。

报告覆盖完整工作流、损失函数设计、物理核解析式、优化器分流、超参搜索、K-neutral selector、forward-prior 逻辑、扰动结构、消融试验、稳健性试验和 source-data package validation。

## 项目复现内容

代码以物理前向核拟合并评估 Frank-Hertz 氩响应。论文面对的主要问题是：相较一个有效激发阈值，四个最低 NIST Ar I 4s 能级能否更好地预测完全留出的阻滞电压曲线。正式复现流程包含：

- 主基线：构建正向证据、可选超参调节、候选 K 扫描训练、自动后评估。
- 消融基线：selector-only 消融，以及关闭 forward anchor gap 的 retrain，用于检验最终选择对正向锚点的依赖程度。
- 稳健性基线：selector 权重扰动，以及固定主基线超参后的 leave-one-retarding-voltage-out 重训。
- 敏感性补充实验：forward-anchor prior-strength 扫描，以及 seed jitter、残差 bootstrap、噪声扰动和峰谷窗口半径扰动下的两类 K=4 不确定度汇总。`conditional_k4_all_fits` 是所有 K=4 条件拟合的 stress-test drift；`production_anchor_matched_k4` 将扰动后的 K=4 通道匹配回 production K=4 四个锚定通道。
- 补充验证套件：selector 去污染审计、循环移动块残差 bootstrap、leave-one-retarding-voltage-out 预测，以及针对 NIST Ar I 4s 能级组的四假设比较。

正式 H4s 比较已经完成：60/60 单元完成、0 失败（5个留出阻滞电压 × 4个假设 × 3个优化重启）。`H4s` 在5个留出条件中的4个比 `H1` 具有更低的折级中位 NRMSE，主要改善位于4、6和8 V；10 V 条件下方向反转。该结果说明指定相对间隔在受检模型中具有条件性预测作用。后续碰撞历史与专一性检验进一步限定其物理解释：它不能据此确认四个原子能级共同生成响应、逐级分辨这些能级，或排除同复杂度替代解释。

物理响应审核只保留两项 caveat：late-bias 与 high-retarding-voltage valley-depth。

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 快速复现

先用 CPU 对 H4s 针对性比较进行 smoke check：

```powershell
$smoke = Join-Path $env:TEMP 'fh_validation_smoke'
python run.py --mode smoke --exclude hpopt --validation-only h4s --output $smoke --device cpu
Remove-Item -LiteralPath $smoke -Recurse -Force
```

使用经性能审计选定的 CPU 配置运行预注册正式验证套件：

```powershell
python run.py --mode fullscan --validation --device cpu
```

正式命令会按照依赖顺序运行 `selector`、`bootstrap`、`holdout` 和 `h4s`。只复现针对性物理比较时使用：

```powershell
python run.py --mode fullscan --validation-only h4s --device cpu
```

`--validation-only` 会自动启用 validation 并解析必要的前置阶段。程序优先复用所选输出根下兼容且已完成的主基线；若不存在，则只读使用已提交的 `source_data_package/`。所有输入、配置、checkpoint、源码和基线哈希都会写入清单，身份不兼容的结果不会被静默混用或覆盖。

正式验证的计算量较大。新运行产物写入 `output/validation/`，由 Git 忽略，不属于本次源码发布内容。

## 如果你是 AI agent

请按以下流程快速、稳定地复现实验，不要猜测目录结构：

1. 先阅读 `README.md`、`README.zh-CN.md`、`docs/reproduction.md` 和 `docs/outputs.md`。
2. 确认 `data/argon/FHdata.xlsx` 存在。做审核复算时先归档旧运行输出，再将正式运行写入 `output/`。
3. 运行功能级 smoke check：

```powershell
python run.py --mode smoke --exclude hpopt --ablation
```

4. smoke 输出只用于确认程序能跑通，不能作为论文结论；正式分析前请删除或忽略 smoke 输出。
5. 将论文主基线写入复算目录：

```powershell
python run.py --mode fullscan --output output
```

6. 运行包含消融和稳健性证据的完整流程：

```powershell
python run.py --mode fullscan --output output --ablation --robustness
```

7. 基于复算主基线运行 forward-prior 和 uncertainty sensitivity：

```powershell
python run.py --mode fullscan --output output --exclude hpopt --sensitivity --device cpu
```

8. `output/` 现在就是正式默认输出根；做新旧核对时不要混用已归档旧证据和当前 `output/` 证据。

9. 若需要更快但不含超参调节的复现，可运行：

```powershell
python run.py --mode fullscan --exclude hpopt --ablation
```

10. 结论应只从所选输出根下的 `main/fullscan/decision.json`、`main/fullscan/model_selection_table.csv`、`main/paper_summary.json`、`ablation/ablation_summary.csv`、`robustness/robustness_summary.json` 和 `sensitivity/` CSV/JSON 文件汇总。

11. 写不确定度结论时，production K=4 候选区间使用 `sensitivity/uncertainty/channel_uncertainty_anchor_matched.csv`。`channel_uncertainty_conditional_k4.csv` 及兼容别名 `channel_uncertainty_summary.csv` 只能解释为条件性 stress-test drift，不能当作 production 能级置信区间。

## 审核复算命令

```powershell
python run.py --mode fullscan --output output
python run.py --mode fullscan --output output --ablation --robustness
python run.py --mode fullscan --output output --exclude hpopt --sensitivity --device cpu
```

smoke 命令只用于功能检查。除非测试本身需要保留输出，否则 smoke 输出目录用完后应删除。

## 试验设计

输入数据：

- 默认文件：`data/argon/FHdata.xlsx`
- 预期内容：Frank-Hertz 氩实验的电流-电压曲线，包含加速电压、拒止电压、曲线编号和测量电流列。
- 读取方式：代码会解析常见列名；常规 Excel 依赖不可用时，也包含 `.xlsx` 后备读取逻辑。

主基线流程：

- `python run.py --mode fullscan` 会先从实验曲线的振荡结构中构建 forward evidence。
- fullscan 默认开启超参调节；可用 `--exclude hpopt` 跳过。
- 程序在配置的 K 范围内扫描候选能级数。
- 每个候选 K 会在 `output/main/fullscan/` 下写出 per-level metrics、checkpoint、scorecard、scan table 与 selector diagnostics。
- 自动后评估会在 `output/main/` 下写出论文汇总文件。

消融流程：

- `--ablation` 会先运行主基线，再运行消融分析。
- selector 消融会在移除不同 selector 组件后重新计算选择结果。
- no-forward-anchor-gap 条件会关闭 forward anchor priors 后重新训练。
- 消融输出写入 `output/ablation/`。

稳健性流程：

- `--robustness` 在主基线已经生成 sweep table 之后运行。
- selector 扰动在固定 rank-weight 网格下重新计算选择结果，不重新训练。
- leave-one-Vr-out 每次排除一条阻滞电压曲线，复用主基线超参配置重训 K 候选，不在每折重新运行 hyperopt。
- 稳健性输出写入 `output/robustness/`。

敏感性补充实验：

- `--sensitivity` 在已有主 fullscan 时复用主基线；与 `--exclude hpopt` 配合时不重新运行超参调节。
- prior-strength 扫描使用 `0`、`0.25`、`0.5`、`1`、`2`、`4` 六档 forward-anchor 强度，并在每档扫描 K=1..8。`1x` 严格定义为 `main/fullscan/config_used.json` 中记录的实际 production prior 权重，其他倍率直接缩放这些记录权重。
- 主流程会输出 `main/k_selected_full/prediction_points.csv`，字段为 `curve_id,Vr,Va,observed,predicted,residual`。residual bootstrap 只从这些 selected-model 残差采样；若该表不存在，sensitivity 会直接失败并提示先运行主流程。
- uncertainty 扫描包含 seed-dependent 初始化扰动、残差 bootstrap、曲线级噪声扰动和 peak-window-radius 扰动，同时保留 K=1..8 scorecard。
- `channel_uncertainty_conditional_k4.csv` 汇总所有 K=4 条件拟合，是 stress-test drift 表。
- `channel_uncertainty_anchor_matched.csv` 将扰动 K=4 拟合匹配回 production K=4 四个锚定通道，是 production K=4 不确定度表述的候选依据。
- 敏感性输出写入 `output/sensitivity/`。

预注册验证流程：

- Selector audit 冻结 production selector，依次移除单项与分组诊断，加入 fit-complexity-only 对照，并报告 rank correlation 和 selected-K 分布。
- 循环移动块残差 bootstrap 在每条曲线内部对中心化残差重采样。正式设计以 7 点块为主条件，以 5 点和 9 点块检查敏感性，并报告 selected-K 分布及 K=4 比例的 Wilson 区间。
- Leave-one-retarding-voltage-out prediction 在未见曲线上评估 K=1..8 和训练折 selected model。主要终点为零样本 RMSE、MAE 和 range-normalized RMSE；次要的 21 点校准只拟合 gain、bias 与加速电压偏移。
- H4s 针对性比较在相同五个留出阻滞电压上评估 `H1`、`H1+B`、`H4s` 和 `H4s+B`。`H4s` 使用四个 NIST Ar I 4s 能量并允许一个受限共同能量微调；`B` 是现有平滑高能损失项，不是第五个激发通道。
- H4s 阶段报告零样本 RMSE、MAE、NRMSE 的配对差异，不自动生成支持或否定结论。seed 仅代表优化起点；正式结果以5个 fold-level 中位数汇总，smoke 结果不可用于论文声称。
- H4s 汇总先在同一 seed 内形成配对差值，再对每个 fold 取 seed 中位数；失败 seed 与失败 fold 仍保留在预期分母中。固定性能审计选定 CPU 与4个单线程 worker，详见 `docs/h4s_performance_audit.md`。
- 每个阶段均写出 manifest、progress、逐单元记录、stage status 和 `stage_result.json`。失败单元保留在预注册统计分母中，未完成或失败的验证最终返回非零退出码。

设备和路径控制：

```powershell
python run.py --mode fullscan --input data/argon/FHdata.xlsx --output output --device cpu
```

`--device` 支持 `cpu`、`cuda` 或 `auto`。项目默认使用 CPU 调度；`auto` 也按 CPU 处理，只有显式传入 `--device cuda` 时才会使用 CUDA。

## 数据分析方式

主结论文件：

- `output/main/fullscan/decision.json`：最终选择的 K 与决策诊断。
- `output/main/fullscan/model_selection_table.csv`：用于模型选择的候选 K 评分表。
- `output/main/fullscan/scan_summary.csv`：按候选 K 汇总拟合、保留的 seed-summary diagnostic、结构和物理响应指标。
- `output/main/k_selected_full/prediction_points.csv`：selected model 的逐点预测和残差，供 residual-bootstrap sensitivity 使用。
- `output/main/paper_summary.json`：面向论文写作的紧凑结论摘要。

结构分析：

- `structure_metrics.csv` 汇总 peak-valley 保真度和 flatline guard。
- `peak_valley_segments.csv`、`curve_structure_summary.csv` 与 `class_structure_summary.csv` 提供分段、曲线和类别级诊断。

物理响应分析：

- `vr_physical_response.csv` 包含逐曲线物理响应诊断。
- `vr_physical_response.json` 保存物理响应汇总和分组结果。
- 主文中报告的物理 caveat 只使用 late-bias 与 high-retarding-voltage valley-depth。

消融分析：

- `output/ablation/ablation_summary.csv` 比较主基线、no-forward-anchor-gap retrain 和 selector-only 变体。
- `output/ablation/selector_ablation_decision.json` 保存各消融组的详细 selector 决策。
- `output/ablation/ablation_report.md` 是可直接阅读的消融报告。

稳健性分析：

- `output/robustness/selector_weight_perturbation.csv` 报告每个 rank-weight 扰动情景及其 selected K。
- `output/robustness/selector_weight_perturbation_summary.csv` 汇总扰动情景下的 selected-K 分布。
- `output/robustness/leave_one_vr_out_summary.csv` 报告每个留出阻滞电压曲线对应的 selected K 和关键指标。
- `output/robustness/robustness_summary.json` 是面向论文写作的紧凑稳健性摘要。

补充验证分析：

- `output/validation/experiment_manifest.json` 记录请求阶段以及所有身份相关输入的哈希。
- `output/validation/progress.json` 记录可断点复用的逐单元进度，不改变预注册分母。
- `output/validation/<stage>/stage_manifest.json` 和 `stage_result.json` 将每项结果绑定到精确的阶段身份。
- 全部请求阶段完成后，`output/validation/validation_summary.json` 和 `validation_summary.md` 汇总证据、推断边界与限制。

论文写作时应引用生成的 JSON/CSV 表，而不是中间 checkpoint。已提交的 `source_data_package/` 是当前手稿对应的整理归档；该目录之外新生成的 checkpoint 和运行输出仍由 `.gitignore` 排除。

## Source Data Package

`source_data_package/` 目录保存当前手稿的 source-data package：

- `manuscript_source_tables/`：29 个 CSV/JSON 文件，用于支撑手稿图、模型选择表、物理响应审核、消融讨论、稳健性检查和修复后的 sensitivity/uncertainty 检查。
- `figures/main/` 与 `figures/supplementary/`：71 个生成图像文件，包括 R/GGPlot2 sensitivity 的 PNG/PDF/SVG/TIFF 导出。
- `run_records/k1_full/` 与 `run_records/k_selected_full/`：16 个 K=1 和 selected K=4 的保留运行记录文件，包括 metrics、parameters、prediction points、logs、scorecards、status files 和 checkpoints。
- `output_results/`：从 `output/` 精选出的 decision table、summary、prediction residual points、robustness 输出和 sensitivity 输出，用于支撑手稿结论。
- `FILE_INDEX.csv`、`SHA256SUMS.txt`、`validation_report.json` 与 `source_data_package_manifest.md`：记录包内路径、源路径、文件大小、SHA256 校验值、校验结果和可读清单。

该数据包可从已有正式输出根重建：

```powershell
$env:SUBLEVEL_OUTPUT='output'
python scripts/build_source_data_package.py
```

若 manuscript visualization 的 source table 或 figure 目录位于仓库外部，请通过 `SUBLEVEL_SOURCE_TABLES` 和 `SUBLEVEL_FIGURES` 显式指定。

当前数据包包含 309 个文件。校验已通过 29 个源表、71 个图件、`prior_strength=1x selected_k=4`、四个 production anchor energy，以及 residual-bootstrap scale check。

## 常用命令

smoke check：

```powershell
python run.py --mode smoke --exclude hpopt
python run.py --mode smoke --exclude hpopt --ablation
```

完整主基线：

```powershell
python run.py --mode fullscan
```

跳过超参调节：

```powershell
python run.py --mode fullscan --exclude hpopt
```

主基线加消融：

```powershell
python run.py --mode fullscan --ablation
```

主基线加消融与稳健性分析：

```powershell
python run.py --mode fullscan --ablation --robustness
```

forward-prior 与通道不确定度补充实验：

```powershell
python run.py --mode fullscan --exclude hpopt --sensitivity --device cpu
```

预注册补充验证：

```powershell
python run.py --mode fullscan --validation --device cpu
python run.py --mode fullscan --validation-only selector --device cpu
python run.py --mode fullscan --validation-only bootstrap --device cuda
python run.py --mode fullscan --validation-only holdout --device cuda
python run.py --mode fullscan --validation-only h4s --device cpu
```

## 输出文件

主基线证据：

- `output/main/fullscan/decision.json`
- `output/main/fullscan/model_selection_table.csv`
- `output/main/fullscan/scan_summary.csv`
- `output/main/fullscan/vr_physical_response.csv`
- `output/main/paper_summary.json`
- `output/main/paper_summary.md`

消融基线证据：

- `output/ablation/ablation_summary.csv`
- `output/ablation/ablation_report.md`
- `output/ablation/selector_ablation_decision.json`

稳健性基线证据：

- `output/robustness/robustness_summary.json`
- `output/robustness/selector_weight_perturbation.csv`
- `output/robustness/selector_weight_perturbation_summary.csv`
- `output/robustness/leave_one_vr_out_summary.csv`

敏感性补充实验证据：

- `output/sensitivity/prior_strength/prior_strength_selection.csv`
- `output/sensitivity/prior_strength/prior_strength_channel_drift.csv`
- `output/sensitivity/uncertainty/channel_uncertainty_samples.csv`
- `output/sensitivity/uncertainty/channel_uncertainty_conditional_k4.csv`
- `output/sensitivity/uncertainty/channel_uncertainty_anchor_matched.csv`
- `output/sensitivity/uncertainty/channel_uncertainty_summary.csv`
- `output/sensitivity/uncertainty/uncertainty_selection_summary.csv`
- `output/sensitivity/sensitivity_summary.json`

补充验证证据：

- `output/validation/experiment_manifest.json`
- `output/validation/progress.json`
- `output/validation/selector_audit/stage_result.json`
- `output/validation/block_bootstrap/stage_result.json`
- `output/validation/holdout/stage_result.json`
- `output/validation/h4s_comparison/stage_result.json`
- `output/validation/h4s_comparison/h4s_comparison_summary.json`
- `output/validation/h4s_comparison/paired_contrasts.csv`
- `output/validation/h4s_comparison/fold_seed_medians.csv`
- `output/validation/h4s_comparison/fold_median_contrasts.csv`
- `output/validation/validation_summary.json`
- `output/validation/validation_summary.md`

## 测试

```powershell
python -m compileall -q run.py src/sublevel_detect
python -m pytest
```

smoke 测试会在仓库内创建临时输出目录，并在测试结束前删除。

