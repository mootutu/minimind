# 2 位置编码 RoPE&YaRN

## 2.1 RoPE (Rotary Positional Embedding)
假设有两个二维向量 $\mathbf{q}$ 与 $\mathbf{k}$，分别表示  **query**  向量与  **key**  向量：

$$
\mathbf{q} = [q_1, q_2], \qquad \mathbf{k} = [k_1, k_2]
$$

为了将向量旋转 $\theta$ 角度，我们引入二维旋转矩阵 $\mathbf{R}(\theta)$，定义为：

$$
\mathbf{R}(\theta) =
\begin{bmatrix}
\cos\theta & -\sin\theta \
\sin\theta & \cos\theta
\end{bmatrix}
$$

其中，$\cos\theta$ 与 $\sin\theta$ 分别是角度 $\theta$ 的余弦与正弦值。

将旋转矩阵与原向量相乘，即可得到旋转后的向量：

$$
\mathbf{q}' = \mathbf{R}(\theta)\mathbf{q}, \qquad
\mathbf{k}' = \mathbf{R}(\theta)\mathbf{k}
$$

其中，$\mathbf{q}'$ 与 $\mathbf{k}'$ 分别表示旋转 $\theta$ 角度后的  **query**  向量与  **key**  向量。

我们对 **query** 向量 $\mathbf{q}$ 施加一次旋转，旋转角度为 $m\theta$；对 **key** 向量 $\mathbf{k}$ 也施加旋转，旋转角度为 $n\theta$。其中，$m$ 与 $n$ 分别是 **query** 与 **key** 的位置索引：$m$ 表示该 **query** 向量 $\mathbf{q}$ 来自序列中的第 $m$ 个位置（第 $m$ 个 token），$n$ 表示该 **key** 向量 $\mathbf{k}$ 来自序列中的第 $n$ 个位置（第 $n$ 个 token）。旋转后的 **query** 与 **key** 向量分别为：

$$
\mathbf{q_m} = [q_1\cos (m\theta) - q_2\sin (m\theta), q_1\sin (m\theta) + q_2\cos (m\theta)] \\
\mathbf{k_n} = [k_1\cos (n\theta) - k_2\sin (n\theta), k_1\sin (n\theta) + k_2\cos (n\theta)]
$$

计算两者的点积，并将其进行逐步展开：

$$
\begin{aligned}
\mathbf{q_m}\cdot \mathbf{k_n}
&= (q_1\cos (m\theta) - q_2\sin (m\theta))(k_1\cos (n\theta) - k_2\sin (n\theta)) \\
&\quad + (q_1\sin (m\theta) + q_2\cos (m\theta))(k_1\sin (n\theta) + k_2\cos (n\theta)) \\
&= q_1k_1\cos (m\theta) \cos (n\theta) - q_1k_2\cos (m\theta) \sin (n\theta) - q_2k_1\sin (m\theta) \cos (n\theta) + q_2k_2\sin (m\theta) \sin (n\theta) \\
&\quad+ q_1k_1\sin (m\theta) \sin (n\theta) + q_1k_2\sin (m\theta) \cos (n\theta) + q_2k_1\cos (m\theta) \sin (n\theta) + q_2k_2\cos (m\theta) \cos (n\theta) \\
&=q_1k_1(\cos (m\theta) \cos (n\theta) + \sin (m\theta) \sin (n\theta)) + q_2k_2(\sin (m\theta) \sin (n\theta) + \cos (m\theta) \cos (n\theta)) \\
&\quad + q_1k_2(\sin (m\theta) \cos (n\theta) - \cos (m\theta) \sin (n\theta)) + q_2k_1(\cos (m\theta) \sin (n\theta) - \sin (m\theta) \cos (n\theta)) \\
&=(q_1k_1 + q_2k_2)\cos((m-n)\theta) + (q_1k_2 - q_2k_1)\sin((m-n)\theta)
\end{aligned}
$$

由于 $q_1$, $k_1$, $q_2$, $k_2$ 都是我们给定的值，且 $\theta$ 也是我们给定的值，也就是说整个式子中只和 $(m-n)$ 相关，也就是说最后的位置信息只由 $(m-n)$ 决定。  

RoPE 之后，用旋转后的 $q'_m, k'_n$ 来计算注意力分数：

$$\text{score}(m, n) = \frac{(q'_m)^\top (k'_n)}{\sqrt{d}}$$

因为 $q$ 和 $k$ 都带着各自位置的旋转角（$m\theta$、$n\theta$），因此点积会自然编码 相对位置信息（与 $m-n$ 强相关）。

## 2.2 RoPE 的实际应用

RoPE 不是直接旋转词向量 embedding，而是在注意力里对每个位置的 Query (Q)、Key (K) 做旋转（通常 Value (V) 不旋转）。

### 2.2.2 从隐藏状态到 Q/K/V

在某一层 Transformer 中，每个位置都会有一个隐藏状态向量：

$$
\mathbf{h}_m = [h_{m1}, h_{m2}, \ldots, h_{d}] \in \mathbb{R}^d
$$

通过线性投影得到注意力所需的 **Query**/**Key**/**Value**（以单头举例）：

$$
q_m = h_m W_q,\quad k_m = h_m W_k,\quad v_m = h_m W_v
$$

其中：

- $q_m, k_m, v_m \in \mathbb{R}^{d}$

- $d$ 是该 head 的维度（head_dim）

### 2.2.3 RoPE 对 Q/K 的“旋转”是怎么做的

将 head_dim 按 2 维一组拆开，假设 $d = 8$，则：

$$
q_m = [q_{m,0}, q_{m,1}, q_{m,2}, q_{m,3}, q_{m,4}, q_{m,5}, q_{m,6}, q_{m,7}]
$$

两两成对分组：

- 第 0 组：$(q_{m,0}, q_{m,1})$

- 第 1 组：$(q_{m,2}, q_{m,3})$

- 第 2 组：$(q_{m,4}, q_{m,5})$

- 第 3 组：$(q_{m,6}, q_{m,7})$

对 $k_m$ 同理。

每一组有自己的频率 $\theta_i$，角度随位置线性增长。对第 $i$ 组（一个 2D pair），旋转角度为：

$$\text{angle}_{m,i} = m \cdot \theta_i$$

其中：

$m$ 是 token 的位置索引（第 $m$ 个 token）

$\theta_i$ 是与维度组 $i$ 对应的频率（通常基于 $10000^{-2i/d}$ 生成）

用旋转矩阵对每个 pair 旋转，对某一组 2D 向量 $(x, y)$，旋转后为 $(x', y')$：

$$\begin{bmatrix}
x' \\ y'
\end{bmatrix}=
\begin{bmatrix}
\cos(m\theta_i) & -\sin(m\theta_i) \\
\sin(m\theta_i) & \cos(m\theta_i)
\end{bmatrix}
\begin{bmatrix}
x \\ y
\end{bmatrix}$$

### 2.2.4 一个例子（只演示某个 pair）

以一句话为例：

> “我(0) 喜欢(1) 吃(2) 苹果(3)”

取 token “吃”，其位置为 $m=2$。

假设在某个 head 的第 0 个 pair 上（仅示意）：

- 原始 $q$ 的 pair：$(x, y) = (0.5, -1.0)$

- 原始 $k$ 的 pair：$(x, y) = (1.2, 0.3)$

- 该 pair 的频率：$\theta_0 = 0.1$

那么旋转角度为：


$$m\theta_0 = 2 \times 0.1 = 0.2 \text{ rad}$$

旋转后（四舍五入）：

- $q' \approx (0.6887, -0.8807)$

- $k' \approx (1.1165, 0.5324)$

**注意：这只是 Q/K 的一个 2D 片段；实际会对所有 pair、所有 head 都做同样操作，只是每个 pair 的 $\theta_i$ 不同。**

## 2.2 YaRN (Yet another RoPE extensioN method)

### 2.2.1 为什么要引入 YaRN 呢？

RoPE 中每一对维度都会对应一个固定的旋转频率，其定义为：

$$
\text{freqs}_i = \frac{1}{\text{rope\_base}^{(2i/\text{dim})}}
$$

从这个公式可以看出，随着维度索引 $i$ 的增大，旋转频率会逐渐减小：低维部分对应较高的旋转频率，高维部分对应较低的旋转频率。

在序列长度不太长的情况（比如 4096）下，这样的频率分布通常是没有问题的。但当我们把上下文长度拉得很长（比如超过模型训练时见过的最大长度 4096， 如5000）时，就会出现问题。

直观来说，对于高频维度，当相对位置变得非常大时，不同距离对应的旋转角度可能会“转到同一个位置”上。也就是说，模型在远处计算得到的位置信息，在经过 RoPE 旋转之后，可能会和某些近处位置产生非常相似的表示。

一旦出现这种情况，模型在注意力计算中就很难区分「这是一个很远的 token」还是「这是一个比较近的 token」，远近关系被混淆，最终会影响模型在长上下文场景下的训练稳定性和效果。这也是为什么原始 RoPE 在长序列外推时容易出现性能下降的问题。

### 2.2.2 如何解决 RoPE 的长度外推问题

主要有两种方法：

1. **方法一：** 把 5000 的长度，压缩到 4096 长度里，这种方法叫做插值（interpolation）。但是这种强硬的方法，会导致损失一些信息

2. **方法二：** 对高低频率使用不同的处理方式：1）对于高频，使用复杂缩放，非线性插值；2）对于低频，使用普通缩放。

### 2.2.3 YaRN

原始的计算公式为：

$$
\text{freqs}_i' = \text{freqs}_i' \cdot t^{-\alpha}\quad(t > 1 为温度参数)
$$

实际计算过程中按以下四步走：

第一步计算哪一个维度超出了我们理解的序列长度，我们用 $2\pi$ 除以频率得到波长，去和原来的我们能够处理的最长序列进行对比，如果他超出来，我们就去最小值，也就是那个分界点就是我们需要去处理的维度了：

$$
\text{corr\_dim} = \text{min}\{i | \frac{2\pi}{\text{freqs}[i]}  > \text{original\_max} \}
$$

第二步，我们需要去计算一个放缩的 power，我们计算出这个 power 值之后，是为了后面更好地辅助我们去对高频和低频的部分去进行缩放的