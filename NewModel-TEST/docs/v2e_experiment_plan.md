# v2e 精确末次碰撞层分布实验计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan → run → validate
- Origin Date: 2026-08-12
- Verification Status: UNVERIFIED
- Version Label: v2e-exact-last-collision

## 目的

检验 v2b–v2d 使用“最后碰撞位置均值”进入非线性收集门，是否造成高阶谷值被系统性抹平。

## 唯一变化

- 主方程显式传播 `P(碰撞历史, 最后碰撞层)`；
- 收集端对 0（无碰撞）及 32 个薄层位置逐项计算角度—能量门后再加权；
- 不新增参数，不改变 v2d 的能量依赖角散射、碰撞率、供给、损失或 13 项门槛。

这是离散最后碰撞位置的精确积分，但能量和角度仍使用低阶闭合。如果它仍不能通过谷深门，则计数状态本身不足，应转向能量—角度分布或轨迹 Monte Carlo。
