## 目标

我们的目标是：找到一种资产组合，尽量降低它的 risk。

以 csi500 为例，我们有 $N=500$ 个股，有 $T=252$ 的一年日频数据。

## 解法

假设我们直接就知道股票的协方差矩阵 $\Sigma$，直接考虑降低方差的话就是在解决这个问题：

$$
\Sigma \succ 0\\
\min w^T\Sigma w\\
\text{s.t.} 1^Tw=1
$$

拉格朗日乘子一下：

$$
L=w^T\Sigma w+\mu(1^Tw-1)=0\\
\begin{cases}
1^Tw-1=0\\
2\Sigma w+\mu 1=0
\end{cases}\\
w=-\frac{\mu}{2}\Sigma^{-1}1\\
\mu=-\frac{2}{1^{T}\Sigma^{-1}1}\\
w=\frac{\Sigma^{-1}1}{1^{T}\Sigma^{-1}1}
$$

我们就求出了针对一个协方差矩阵的方差最小配置。

这样看难以提取什么特征，$\Sigma$ 是正定矩阵，我们分解一下：

$$
\Sigma=U\Lambda U^T=\sum \lambda_i u_iu_i^T\\
\Sigma^{-1}=U^T\Lambda^{-1} U=\sum \lambda_i^{-1} u_iu_i^T\\
w=\frac{1}{1^{T}\Sigma^{-1}1}\sum \frac{u_i^T1}{\lambda_i}u_i
$$

我们分解出了独立的方差方向 $u_i$，分配的策略就是 $\lambda_i$ 越小，意味着这个方向的方差小，那么我分配更大的权重。

## 问题

如果我们真的知道协方差矩阵，那么这样做当然没有问题。但是问题在于我们实际上只有一个采样的协方差矩阵 $\hat \Sigma$，求出对应的采样方案就是 $\hat w$，这是一个努力迎合 $\hat \Sigma$ 的方案，那么很显然可以会出现过拟合一个错误的协方差。

当然重要前提是我们认为股市的方差是具有一定稳定性的，如果方差每天都剧烈变化预测就没有意义。而 $\hat \Sigma$ 之所以可能不对，是因为有随机噪声在干扰，淹没了真实的协方差矩阵。

简单来说我们建模出：

$$
\hat \Sigma =\Sigma+E
$$

我们想要找的是 $\Sigma$ 所指示的，方差大或者小的方向 $u$。但是我们对 $\hat \Sigma$ 直接做提取出的 $\lambda_i$ 是经过 $E$ 干扰的随机值，就不能作为预测未来一个方向的方差的依据。

那么目标就很明确：$\hat \Sigma$ 中找出所有的 $\lambda_i,u_i$，判断哪些 $\lambda_i$ 对应真实方差方向。

## 建模白噪声

和假设检验里面做的一样。$H_0:\lambda_i\in E$。我们希望找出哪些偏离白噪声的 $\lambda_i$，它们才是我们需要的东西。至于被噪声淹没的 $\lambda_i$，我们均分资产就可以，不需要管它们的 $\lambda_i$ 大小。

我们认为白噪声：

$$
\Sigma\sim N(0,\sigma^2I)
$$

那么我们实际上想要知道 $\Sigma$ 中采样一个协方差矩阵，它的 $\lambda_i$ 的分布。

我们认为采样结果是 $X\in R^{n\times T},S=\dfrac{1}{n}X^TX$。

问题在于：想要知道 $S$ 的每个 $\lambda_i$ 是困难的，我们想要用一种方式来把 $\lambda$ 压缩处理，再解压。

于是有一个工具：Stieltjes 变换。我们对 $\lambda={\lambda_1,\lambda_2,...,\lambda_n}$ 定义生成函数：

$$
g(z)=\int p(\lambda)\frac{d\lambda}{z-\lambda}=\text{tr}((Iz-S)^{-1})
$$

背后的道理：

$$
\text{tr}((Iz-S)^{-1})=\sum \frac{1}{\lambda_i-z}
$$

而 $g(z)$ 满足可以还原 $p(\lambda)$，那么我们想办法求出 $g(z)$ 就行。

具体分布：

$$
p_\gamma(\lambda) =
\frac{1}{2\pi\,\sigma^2\gamma\,\lambda}\,
\sqrt{(\lambda_+ - \lambda)(\lambda - \lambda_-)},
\qquad \lambda \in [\lambda_-, \lambda_+]
$$

$$
\lambda_\pm = \sigma^2(1 \pm \sqrt{\gamma})^2
$$

只需要记住 $\lambda_{+},\lambda_{-}$ 是整个分布的两个边界，中间是一个单峰分布。

如果当 $\lambda_{-}<0$ 的时候，小的那部分会被截断，累计在 $\lambda=0$ 上面。

最后的分布只和 $\gamma=T/N,\sigma$ 决定。

$\sigma$ 决定了分布整体的缩放。

$\gamma=T/N$ 代表着这个分布的“方差”（$\gamma$ 越大，代表样本量充足，样本的 $\lambda$ 就更接近真实的 $\sigma$）。

