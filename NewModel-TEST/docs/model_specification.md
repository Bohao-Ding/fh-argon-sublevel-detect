# 薄层输运模型规格

## 1. 阴极供给

热电子发射的绝对饱和电流由 Richardson–Dushman 形式控制；低电压抽取区受 Child–Langmuir 空间电荷支路约束。由于阴极温度、面积和极间距未记录，本模型只拟合二者连接后的有效电流尺度：

\[
I_{\rm supply}(V_a)=I_{\rm RD}\frac{z^{3/2}}{1+z^{3/2}},\qquad
z=\frac{(s_a V_a-V_{s0})_+}{V_s}.
\]

这保留 `V^(3/2)` 低压极限与热发射饱和极限，但不把 `I_RD` 解释为独立测得的阴极温度。

## 2. 薄层碰撞主方程

将加速区写成 `L` 个无量纲薄层。对通道 `k`，`p_n^(k)(x)` 表示电子在位置 `x` 前已发生 `n` 次该类非弹性碰撞的概率。层内可用动能为

\[
\varepsilon_{n,k}(x)=(s_aV_a-V_{a0})_+x+\bar\varepsilon_0-nE_k.
\]

有效截面形状为

\[
q_k(\varepsilon)=
\left(\frac{(\varepsilon-E_k)_+}{E_r+(\varepsilon-E_k)_+}\right)^p
\exp\!\left[-\frac{(\varepsilon-E_k)_+}{E_d}\right].
\]

层碰撞概率 `c=1-exp(-tau*q/L)`，随后按

\[
p_n(x+\Delta x)=p_n(x)(1-c_n)+p_{n-1}(x)c_{n-1}
\]

更新。因而振荡来自碰撞次数概率随 `Va` 的跃迁与量子化能量损失，不使用正弦、余弦或 `round` 预置峰谷。

## 3. 能量—角度选择门

热电子初始法向能量用指数分布作 Gauss–Laguerre 积分，发射角使用余弦通量分布 `2 mu dmu`。碰撞 `n` 次后的剩余能量为

\[
\varepsilon_f=(s_aV_a-V_{a0})_++\varepsilon_0-nE_k.
\]

收集器对纵向能量 `epsilon_f mu^2` 的响应为高斯分辨函数的累积分布：

\[
G_E=\Phi\!\left(
\frac{\varepsilon_f\mu^2-(V_g+s_rV_r)}{\sigma_E}
\right),
\]

角接受度为

\[
G_\theta=\exp[-\theta^2/(2\sigma_\theta^2)].
\]

`V_s0` 表示空间电荷受限供给的抽取起点，`V_a0` 表示阴极—加速区的有效接触偏置，`V_g` 表示栅极—收集极侧的剩余势垒。第一轮失败表明，将三种不同位置的电势效应压缩成一个偏置会同时造成低压上升沿错误与零阻滞伪振荡。低压截止由这三部分串联产生，阻滞电压导致的横向移动则由共享的收集门产生，而不是逐曲线平移参数。

## 4. 多通道组合

\[
I(V_a,V_r)=I_{\rm supply}(V_a)
\sum_{k=1}^{K}w_k
\sum_n p_n^{(k)}(V_a)
\langle G_EG_\theta\rangle_{\varepsilon_0,\theta},
\quad w_k\ge0,\quad\sum_kw_k=1.
\]

通道组合是低阶混合近似：它不枚举同一个电子依次激发不同能级的所有序列。该限制必须与拟合出的 `K` 一起报告。

## 5. 明确不包含的物理

- 实际二维/三维电极几何和非均匀电场；
- 能量依赖的弹性微分截面与角扩散；
- 亚稳态、二次电子、离化和空间电荷的自洽反馈；
- 气压、温度、平均自由程和绝对截面的独立标定；
- 完整 Boltzmann 方程或逐电子 Monte Carlo 轨迹。

因此，合适名称是“可微薄层主方程输运近似”，不是第一性原理模型。
