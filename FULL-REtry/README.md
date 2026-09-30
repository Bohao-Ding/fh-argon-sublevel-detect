> Integration note (2026-10-01): this is a preserved module record. Current questions, corrections and publication status are indexed in [the research guide](../docs/research_overview.md) and [reproduction instructions](../docs/research_reproduction.md). Frozen injection-recovery claims are superseded by A5/A14; A4 is not executed.

# FULL-REtry：Franck–Hertz 阻滞响应层析

这是一个与既有稿件隔离的新实验工作区。唯一复用的实验资源是五条氩 Franck–Hertz 曲线的原始工作簿；旧模型代码、旧拟合结果和现有稿件均不作为本项目输入。

## 当前研究问题

五个阻滞电压条件是否支持一个比“固定形状、统一相位的单模态响应”更丰富、但仍保持粗粒度解释的周期锁定响应结构？

这里的 `mode` 是数学模态，`band` 是离散阻滞响应带。它们不是已分辨的原子亚能级、绝对碰撞截面或经校准的电子能量分布。

## 最小实验

1. 审计数据哈希、结构、共同电压网格和阻滞单调性。
2. 构造守恒差分带：`I0-I4`、`I4-I6`、`I6-I8`、`I8-I10`、`I10`。
3. 用去趋势 SVD 检查稳定的二维响应子空间及其主周期。
4. 固定物理先验周期 `11.55 V`，按完整周期块留出，比较 rank-1 与 rank-2 谐波响应。
5. 用等复杂度五次趋势和错位周期作为负对照。

完整的假设、门控和解释边界见 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)，文献缺口见 [LITERATURE_GAP.md](LITERATURE_GAP.md)。

## 运行

```powershell
python run_minimal.py
& 'D:\R\R-4.6.1\bin\x64\Rscript.exe' scripts\plot_minimal.R
python -m pytest -q
```

数值输出写入 `results/minimal_v1/`。正式图由 R 脚本单源生成；Python 不生成图。

第二阶段的等复杂度模板、条件代理、端点排除、注入恢复与增益压力测试运行：

```powershell
python run_phase2.py
Rscript scripts\plot_phase2.R
```

数值输出写入 `results/phase2_v1/`，主图写入 `figures/phase2_v1/`。第二阶段仍只使用解析线性代数，不训练神经网络；正式配置在当前 CPU 上用时 32.84 s，其中条件代理和注入回收分别占 15.69 s 与 10.20 s。

第二阶段按当时定义的六项开发门控全部通过；Phase 3 使用统一的原 G2 三项规则复审后发现，删去 `Vr=10 V` 时最差块为负且越过 `-5%` 界限，因此第二阶段的端点结论只能保留为“多数块改善”，不能称为完整端点稳健性。正文可整合段落、结果表、图注和教学任务见 [MANUSCRIPT_INTEGRATION.md](MANUSCRIPT_INTEGRATION.md)，图的证据契约见 [FIGURE_CONTRACT.md](FIGURE_CONTRACT.md)。这些文件是新项目产物，不直接修改现有稿件。

第三阶段的组成数据、周期、块边界、绝对充分性、端点规则和校准注入检查运行：

```powershell
python run_phase3.py --workers auto --surrogate-replicates 2000 --injection-replicates 500
```

数值输出写入 `results/phase3_v1/`。程序把 16 个条件代理任务和 48 个注入任务分派给最多 8 个工作进程，并把每个进程的 BLAS/OpenMP 线程限制为 1；正式运行总墙钟约 18 s。完整计划见 [PHASE3_PLAN.md](PHASE3_PLAN.md)，门控与解释见 [results/phase3_v1/RESULTS_INTERPRETATION.md](results/phase3_v1/RESULTS_INTERPRETATION.md)。

Phase 3 的六项门控中四项通过、两项失败。支持的主张收缩为：同一归档数据在多种表示和块划分下主要支持一个二维、非可分的周期响应结构，但该优势并非逐块坐标不变，且依赖 `Vr=10 V` 端点才能满足完整门槛。NIST 4s 跨度在给定模板的条件任务中不可恢复；由于候选选择仍强烈偏向最大跨度，完整注入回收曲线不能转换成仪器能量分辨率。

第四阶段在无法重新采集数据的约束下，运行 12 条合成正反向扫描的方向—滞后应力检查：

```powershell
python run_phase4.py --workers auto
```

数值输出写入 `results/phase4_virtual_scans_v1/`。该阶段使用解析线性代数和线程级任务并行，不进行神经网络训练。`Vr=2 V`、扫描方向和滞后均为明确标记的计算构造，不能写成新的实验重复。完整证据边界见 [PHASE4_PLAN.md](PHASE4_PLAN.md)。

## 证据等级

本数据已经参与过既有项目开发，因此本轮结果只能称为内部重新分析或方法探索。即使最小门控通过，也只允许进入低算力的注入恢复、块自助法和截面模板敏感性分析；不能直接升级为独立确认或亚能级分辨主张。
