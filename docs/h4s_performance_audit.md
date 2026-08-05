# H4s 正式训练前性能与正确性审计

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: implementation and verification
- Date: 2026-08-05
- Verification Status: PASS
- Version: h4s_pretraining_hardening_v1

## 1. 结论与边界

固定性能矩阵、同设备有界等价门和跨设备准入门均通过。正式 H4s 设备选择为 CPU，使用4个 worker，每个 worker 固定1个 PyTorch/MKL 线程：

```powershell
python run.py --mode fullscan --validation-only h4s --device cpu
```

本审计只决定工程运行配置，不评价四能级假设，也不构成论文物理结论。未运行5折×4假设×3 seeds 的60单元正式实验。

## 2. 固定环境

| 项目 | 记录 |
|---|---|
| 操作系统 | Windows 11 `10.0.26200` |
| CPU | Intel Core i7-14650HX，16 cores / 24 logical processors |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Python | 3.13.7, MSC v.1944, 64 bit |
| PyTorch | 2.8.0+cu128 |
| CUDA runtime | 12.8 |
| 基线提交 | `c3b286eca68e1570f6e7c4c53ca78e3b65de7bfa` |
| 审计矩阵 | held-out Vr=0；4 hypotheses；seeds 0/1；300 epochs；2 independent repeats |

最终审计脚本 SHA256 为 `7F375A75B2FCB57B4338C00822E9879B41A521A2FE20B879BD6C14BD72691619`。基线 `model.py`/`validation_h4s.py` SHA256 分别为 `3A9CAD9B...F5C`、`A78F3FF9...39B`；优化版分别为 `62C68BEC...B79`、`861AED6D...F34`。完整哈希由审计脚本写入临时 manifest，并在临时目录删除前人工核验。

## 3. 100-epoch 候选筛查

每个候选均完成8/8单元、零失败。最终源码下的墙钟时间为：

| 设备 | workers | 单元时间中位数 (s) | 墙钟时间 (s) | 设备内选择 |
|---|---:|---:|---:|---|
| CPU | 1 | 1.0346 | 10.7986 |  |
| CPU | 4 | 2.4992 | 10.3007 | selected |
| CPU | 8 | 4.0643 | 10.7785 |  |
| CUDA | 1 | 3.4851 | 29.3429 |  |
| CUDA | 2 | 3.7761 | 22.1634 |  |
| CUDA | 4 | 5.1603 | 16.4852 | selected |

CPU 与 CUDA 的胜出配置分别进入300-epoch、两次独立重复审计。每次重复单独建立 worker pool、单独计时，汇总取两次墙钟的中位数；不再使用把两次重复混入同一进程池的旧口径。

## 4. 300-epoch 性能结果

| 实现 | 设备 / workers | 单元时间中位数 (s) | 重复墙钟 (s) | 墙钟中位数 (s) | 相对同设备基线 |
|---|---|---:|---|---:|---:|
| `c3b286e` baseline | CPU / 8 | 13.3097 | 20.8812, 19.4344 | 20.1578 | reference |
| optimized | CPU / 4 | 5.1233 | 15.3852, 16.1226 | 15.7539 | 21.85% faster |
| `c3b286e` baseline | CUDA / 4 | 38.5394 | 81.6792, 85.4417 | 83.5604 | reference |
| optimized | CUDA / 4 | 11.5110 | 29.5444, 29.1485 | 29.3465 | 64.88% faster |

优化 CPU 比优化 CUDA 快46.32%，超过 CPU 切换所需的20%。优化 CPU 相对 CPU 基线也超过20%，因此最终设备门为 PASS。CUDA 优化版峰值分配显存为380,928 bytes；CUDA 基线为456,192 bytes。CPU 计时与 CUDA 计时未混作训练时间比较以外的物理证据。

## 5. 同设备有界等价

| 门槛 | CPU 最大差异 | CUDA 最大差异 | 限值 | 结果 |
|---|---:|---:|---:|---|
| 完成状态 / epochs / best epoch | 完全一致 | 完全一致 | 完全一致 | PASS |
| prediction max abs | 7.153e-7 | 2.384e-7 | 1e-5 | PASS |
| RMSE/MAE/NRMSE max abs | 7.540e-8 | 5.611e-8 | 1e-6 | PASS |
| energy/common shift max abs | 1.490e-8 eV | 2.235e-8 eV | 1e-5 eV | PASS |
| level weight max abs | 4.470e-8 | 8.941e-8 | 1e-5 | PASS |
| 4 contrast directions | 完全一致 | 完全一致 | 完全一致 | PASS |

## 6. 跨设备准入

| 门槛 | 实测 | 限值 | 结果 |
|---|---:|---:|---|
| NRMSE max relative difference | 3.041e-7 | 1% | PASS |
| energy/common shift max abs | 1.490e-8 eV | 0.02 eV | PASS |
| level weight MAE max | 5.029e-8 | 0.02 | PASS |
| 4 contrast directions | 完全一致 | 完全一致 | PASS |
| failed/OOM/out-of-bounds units | 0 | 0 | PASS |

## 7. 保留的实现与被否决方案

保留：静态训练张量缓存、无背景分支短路、完整四曲线批量前向与损失、按时间门控 checkpoint、原子 checkpoint、终态续算、配置与哈希身份校验、单线程 worker、父进程曲线复用，以及正确的 fold-level seed 汇总。

被否决：CPU 线程池。实测受 Python 调度与 GIL 影响，未优于多进程，因此没有保留。未使用 AMP、减少 epoch、放宽早停或改变损失权重。

性能审计曾发现并修正两个只影响审计口径的脚本 bug：两次重复错误共用一个总墙钟，以及总单元数错误取最后一次重复的8个单元。最终表格只使用修正后、同一审计脚本哈希生成的结果。
