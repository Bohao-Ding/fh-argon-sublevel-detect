# v2h 到达可达率门控实验计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan → run → validate
- Origin Date: 2026-08-13
- Verification Status: UNVERIFIED
- Version Label: v2h-reachability-gated

## v2g 剩余误差与诊断

v2g 通过 13 项 H1 门槛中的 11 项。失败项为 `Vr=0` 的最大截止点误差 2.0 V，以及后段谷值电流 MAE 0.2095 μA。局部灵敏度分析显示，二者均与高斯门在负平均到达能量区域保留非零尾概率有关。

## 唯一结构变化

在高斯拒止势垒门之前恢复独立可达率：

`R(E_z) = 1 - exp[-max(E_z, 0) / E_reach]`。

这里 `E_z` 是末次碰撞后到达选择区的轴向能量，`E_reach` 在 0.05–3.00 eV 内学习。`R` 表示电子能否实际到达选择区；高斯 CDF 继续表示到达电子越过拒止势垒的概率。v2g 的供给律、碰撞危险率、末次碰撞闭合、角散射和冻结的 13 项门槛保持不变。

## 判定

仅运行 H1。只有全部 13 项门槛通过后，才允许进入 H4s 多能级验证。
