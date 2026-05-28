# 采样方法 (sampling methods)

> 采样是 AI 探索可能性空间的方式。

**类型：** 构建
**语言：** Python
**先修要求：** 第一阶段，第 06-07 课（概率、贝叶斯定理）
**时间：** ~120 分钟

## 学习目标

- 仅使用均匀随机数，从零实现逆累积分布函数 (inverse CDF)、拒绝采样 (rejection sampling) 和重要性采样 (importance sampling)
- 为语言模型的词元生成构建温度采样 (temperature sampling)、Top-k 采样 (top-k sampling) 和 Top-p（核）采样 (top-p / nucleus sampling)
- 解释重参数化技巧 (reparameterization trick) 以及它为何能让变分自编码器 (VAE) 中的采样支持反向传播
- 运行 Metropolis-Hastings 马尔可夫链蒙特卡洛 (Markov Chain Monte Carlo, MCMC)，从未归一化的目标分布中采样

## 问题

语言模型 (language model) 处理完你的提示词后，会输出一个包含 50,000 个对数几率 (logits) 的向量。词表里的每个词元 (token) 都对应一个值。现在它必须选一个。怎么选？

如果它总是选概率最高的词元，每次回答都会一模一样。确定性。无聊。如果它完全均匀随机地选，输出又会变成乱码。真正的答案落在这两个极端之间，而这个“中间地带”正是由采样控制的。

采样并不只用于文本生成。强化学习 (reinforcement learning) 通过对轨迹进行采样来估计策略梯度。变分自编码器 (VAE) 通过从学习到的分布中采样并穿过随机性进行反向传播来学习潜在表示。扩散模型 (diffusion models) 通过对噪声采样并迭代去噪来生成图像。蒙特卡洛方法 (Monte Carlo methods) 用于估计那些没有闭式解的积分。MCMC 算法则探索无法枚举的高维后验分布。

每个生成式 AI 系统本质上都是一个采样系统。采样策略决定了输出的质量、多样性和可控性。本课将从零搭建所有主要的采样方法：从均匀随机数出发，一直到驱动现代 LLM 和生成模型的那些技术。

## 概念

### 为什么采样很重要

采样在 AI 和机器学习中承担四种基础角色：

**生成。** 语言模型、扩散模型和生成对抗网络 (GANs) 都通过采样产生输出。采样算法直接控制创造性、连贯性和多样性。温度采样、Top-k 采样和核采样是工程师每天都在调的旋钮。

**训练。** 随机梯度下降会采样小批量数据。Dropout 会采样要停用的神经元。数据增强会采样随机变换。重要性采样会对样本重新加权，以降低强化学习（PPO、TRPO）中的梯度方差。

**估计。** 机器学习中的许多数值都没有闭式解。例如数据分布上的期望损失、能量模型的配分函数，以及贝叶斯推断中的证据。蒙特卡洛估计通过对样本求平均来近似所有这些量。

**探索。** MCMC 算法用于探索贝叶斯推断中的后验分布。进化策略会采样参数扰动。汤普森采样 (Thompson sampling) 则在多臂老虎机问题中平衡探索与利用。

核心挑战在于：你只能直接从简单分布（均匀分布、正态分布）中采样。对于其他一切分布，你都需要一种方法，把简单样本转换成目标分布的样本。

### 均匀随机采样 (uniform random sampling)

所有采样方法都从这里开始。均匀随机数生成器会在 [0, 1) 中产生数值，其中任意长度相同的子区间都有相同概率。

```
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    for 0 <= a <= b <= 1

Properties:
  E[U] = 0.5
  Var(U) = 1/12
```

如果要从一个包含 n 个元素的离散集合中均匀采样，生成 U 后返回 floor(n * U) 即可。如果要从连续区间 [a, b] 中采样，则计算 a + (b - a) * U。

关键洞见是：一个均匀随机数恰好包含了从任意分布中生成一个样本所需的随机性。诀窍在于找到正确的变换。

### 逆累积分布函数方法 (inverse CDF method，也称 inverse transform sampling)

累积分布函数 (CDF) 会把取值映射成概率：

```
F(x) = P(X <= x)

Properties:
  F is non-decreasing
  F(-inf) = 0
  F(+inf) = 1
  F maps the real line to [0, 1]
```

逆累积分布函数会把概率映射回取值。如果 U ~ Uniform(0, 1)，那么 X = F_inverse(U) 就服从目标分布。

```
Algorithm:
  1. Generate u ~ Uniform(0, 1)
  2. Return F_inverse(u)

Why it works:
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)
```

**指数分布示例：**

```
PDF: f(x) = lambda * exp(-lambda * x),   x >= 0
CDF: F(x) = 1 - exp(-lambda * x)

Solve F(x) = u for x:
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

Since (1 - U) and U have the same distribution:
  x = -ln(u) / lambda
```

当你能写出 F_inverse 的闭式表达时，这个方法就非常完美。对于正态分布，并不存在闭式的逆 CDF，因此我们会使用其他方法（如 Box-Muller 或数值近似）。

**离散版本：** 对于离散分布，先把 CDF 构造成累积和，生成 U，然后找到第一个使累积和超过 U 的索引。这就是第 06 课中 `sample_categorical` 的工作方式。

### 拒绝采样 (rejection sampling)

当你无法求逆 CDF，但能够在差一个常数的意义下计算目标 PDF 时，拒绝采样就能派上用场。

```
Target distribution: p(x)  (can evaluate, possibly unnormalized)
Proposal distribution: q(x)  (can sample from)
Bound: M such that p(x) <= M * q(x) for all x

Algorithm:
  1. Sample x ~ q(x)
  2. Sample u ~ Uniform(0, 1)
  3. If u < p(x) / (M * q(x)), accept x
  4. Otherwise, reject and go to step 1

Acceptance rate = 1/M
```

界 M 越紧，接受率越高。在低维（1-3 维）情况下，拒绝采样表现很好。在高维情况下，接受率会指数级下降，因为提议分布的大部分体积都会被拒掉。这就是拒绝采样中的维度灾难 (curse of dimensionality)。

**示例：从截断正态分布 (truncated normal) 中采样。** 在截断区间上使用均匀提议分布。包络常数 M 就是该区间内正态 PDF 的最大值。

**示例：从半圆中采样。** 在包围矩形内做均匀提议。如果点落在半圆内部，就接受它。这正是蒙特卡洛计算 pi 的方式：接受率等于面积比 pi/4。

### 重要性采样 (importance sampling)

有时你并不需要从目标分布 p(x) 中直接得到样本。你真正需要的是估计 p(x) 下的一个期望，而你手头的样本来自另一个分布 q(x)。

```
Goal: estimate E_p[f(x)] = integral of f(x) * p(x) dx

Rewrite:
  E_p[f(x)] = integral of f(x) * (p(x)/q(x)) * q(x) dx
            = E_q[f(x) * w(x)]

where w(x) = p(x) / q(x)  are the importance weights.

Estimator:
  E_p[f(x)] ~ (1/N) * sum(f(x_i) * w(x_i))    where x_i ~ q(x)
```

这在强化学习中至关重要。在 PPO（近端策略优化，Proximal Policy Optimization）里，你会在旧策略 pi_old 下收集轨迹，但希望优化新策略 pi_new。重要性权重就是 pi_new(a|s) / pi_old(a|s)。PPO 会对这些权重做裁剪，以防新策略偏离旧策略太远。

重要性采样估计器的方差取决于 q 与 p 有多相似。如果 q 和 p 差别很大，少数样本就会获得极大的权重，并主导整个估计。自归一化重要性采样 (self-normalized importance sampling) 通过除以权重总和来缓解这个问题：

```
E_p[f(x)] ~ sum(w_i * f(x_i)) / sum(w_i)
```

### 蒙特卡洛估计 (Monte Carlo estimation)

蒙特卡洛估计通过对随机样本求平均来近似积分。大数定律保证了它会收敛。

```
Goal: estimate I = integral of g(x) dx over domain D

Method:
  1. Sample x_1, ..., x_N uniformly from D
  2. I ~ (Volume of D / N) * sum(g(x_i))

Error: O(1 / sqrt(N))   regardless of dimension
```

它的误差率与维度无关。这也是为什么在高维情况下、当基于网格的积分几乎不可能时，蒙特卡洛方法会占据主导地位。

**估计 pi：**

```
Sample (x, y) uniformly from [-1, 1] x [-1, 1]
Count how many fall inside the unit circle: x^2 + y^2 <= 1
pi ~ 4 * (count inside) / (total count)
```

**估计期望：**

```
E[f(X)] ~ (1/N) * sum(f(x_i))    where x_i ~ p(x)

The sample mean converges to the true expectation.
Variance of the estimator = Var(f(X)) / N
```

### 马尔可夫链蒙特卡洛 (Markov Chain Monte Carlo, MCMC)：Metropolis-Hastings

MCMC 会构造一个马尔可夫链，其平稳分布 (stationary distribution) 就是目标分布 p(x)。经过足够多步之后，从链中得到的样本就会（近似地）服从 p(x)。

```
Target: p(x)  (known up to a normalizing constant)
Proposal: q(x'|x)  (how to propose the next state given the current state)

Metropolis-Hastings algorithm:
  1. Start at some x_0
  2. For t = 1, 2, ..., T:
     a. Propose x' ~ q(x'|x_t)
     b. Compute acceptance ratio:
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. Accept with probability min(1, alpha):
        - If u < alpha (u ~ Uniform(0,1)): x_{t+1} = x'
        - Otherwise: x_{t+1} = x_t
  3. Discard first B samples (burn-in)
  4. Return remaining samples
```

对于对称提议分布（q(x'|x) = q(x|x')），比值会简化为 p(x')/p(x)。这就是最初的 Metropolis 算法。

**为什么可行。** 接受规则保证了细致平衡 (detailed balance)：处于 x 并移动到 x' 的概率，等于处于 x' 并移动到 x 的概率。细致平衡意味着 p(x) 就是这条链的平稳分布。

**实践注意事项：**
- 预热期 (burn-in)：在链达到平衡之前，丢弃前期样本
- 抽稀 (thinning)：每隔 k 个样本保留一个，以减少自相关
- 提议尺度：太小则链移动缓慢（接受率高，但探索慢）；太大则大多数提议都会被拒绝（接受率低，基本卡住）
- 对于高维空间中的高斯提议分布，最优接受率大约是 0.234

### 吉布斯采样 (Gibbs sampling)

吉布斯采样是多元分布场景下 MCMC 的一种特殊情况。它不会一次性在所有维度上提出一个移动，而是每次从条件分布中更新一个变量。

```
Target: p(x_1, x_2, ..., x_d)

Algorithm:
  For each iteration t:
    Sample x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    Sample x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    Sample x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```

吉布斯采样要求你能够从每个条件分布 p(x_i | x_{-i}) 中采样。对于许多模型，这都很直接：
- 贝叶斯网络：条件分布由图结构直接给出
- 高斯混合：条件分布仍然是高斯分布
- Ising 模型：每个自旋的条件分布只依赖于它的邻居

接受率始终是 1（每个提议都会被接受），因为从精确条件分布中采样会自动满足细致平衡。

**局限。** 当变量高度相关时，吉布斯采样的混合速度会很慢，因为一次只更新一个变量，无法沿着分布的对角方向做大步移动。

### 温度采样 (temperature sampling，用于 LLM)

语言模型会为词表中的每个词元输出 logits z_1, ..., z_V。Softmax 会把它们转成概率。温度会在 softmax 之前对 logits 进行缩放：

```
p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0: standard softmax (original distribution)
T -> 0:  argmax (deterministic, always picks highest logit)
T -> inf: uniform (all tokens equally likely)
T < 1.0: sharpens the distribution (more confident, less diverse)
T > 1.0: flattens the distribution (less confident, more diverse)
```

**为什么可行。** 当 T &lt; 1 时，用 T 去除 logits 会放大它们之间的差异。如果 z_1 = 2 且 z_2 = 1，那么除以 T = 0.5 后得到 z_1/T = 4、z_2/T = 2，差距就更大了。经过 softmax 后，logit 最大的词元会占据更大的概率份额。

**在实践中：**
- T = 0.0：贪心解码，最适合事实型问答
- T = 0.3-0.7：略有创意，适合代码生成
- T = 0.7-1.0：较为平衡，适合一般对话
- T = 1.0-1.5：适合创意写作、头脑风暴
- T > 1.5：随机性越来越强，通常没什么用

温度不会改变哪些词元是可能的。它改变的是分配给每个词元的概率质量。

### Top-k 采样 (top-k sampling)

Top-k 采样会把候选集合限制为概率最高的 k 个词元，然后在这个受限集合上重新归一化并进行采样。

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Keep only the top k tokens
  4. Renormalize: p_i' = p_i / sum(p_j for j in top-k)
  5. Sample from the renormalized distribution

k = 1:  greedy decoding
k = V:  no filtering (standard sampling)
k = 40: typical setting, removes long tail of unlikely tokens
```

Top-k 能防止模型选到词表分布长尾中那些极不可能的词元（错别字、胡话）。问题在于：无论上下文如何，k 都是固定的。当模型非常有把握时（某个词元概率达到 95%），k = 40 依然会放进 39 个备选项。当模型非常不确定时（概率分散在 1000 个词元上），k = 40 又会砍掉很多本来合理的选项。

### Top-p（核）采样 (top-p / nucleus sampling)

Top-p 采样会动态调整候选集合的大小。它不保留固定数量的词元，而是保留“累计概率刚刚超过 p”的最小词元集合。

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Find smallest k such that sum of top-k probabilities >= p
  4. Keep only those k tokens
  5. Renormalize and sample

p = 0.9:  keeps tokens covering 90% of probability mass
p = 1.0:  no filtering
p = 0.1:  very restrictive, nearly greedy
```

当模型很有把握时，核采样只会保留很少几个词元（可能只有 2-3 个）。当模型不确定时，它会保留很多词元（可能有 200 个）。这种自适应行为，就是为什么核采样通常比 Top-k 生成更好的文本。

**常见组合：**
- Temperature 0.7 + top-p 0.9：通用场景下的好设置
- Temperature 0.0（greedy）：最适合确定性任务
- Temperature 1.0 + top-k 50：Fan 等人（2018）原始论文中的设置

Top-k 和 Top-p 可以组合使用。先应用 Top-k，再在剩余集合上应用 Top-p。

### 重参数化技巧 (reparameterization trick，用于 VAE)

变分自编码器 (VAEs) 的学习方式是：先把输入编码成潜在空间里的一个分布，再从这个分布中采样，最后把样本解码回来。问题在于：你无法穿过“采样操作”做反向传播。

```
Standard sampling (not differentiable):
  z ~ N(mu, sigma^2)

  The randomness blocks gradient flow.
  d/d_mu [sample from N(mu, sigma^2)] = ???
```

重参数化技巧把随机性和参数分离开来：

```
Reparameterized sampling:
  epsilon ~ N(0, 1)          (fixed random noise, no parameters)
  z = mu + sigma * epsilon   (deterministic function of parameters)

  Now z is a deterministic, differentiable function of mu and sigma.
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  Gradients flow through mu and sigma.
```

这是因为 N(mu, sigma^2) 与 mu + sigma * N(0, 1) 具有相同分布。核心洞见是：把随机性挪到一个不含参数的来源（epsilon）上，再把样本表示成参数的可微变换。

**在 VAE 的训练循环中：**
1. 编码器为每个输入输出 mu 和 log(sigma^2)
2. 采样 epsilon ~ N(0, 1)
3. 计算 z = mu + sigma * epsilon
4. 解码 z 以重建输入
5. 穿过第 4、3、2、1 步反向传播（之所以可行，是因为第 3 步可微）

如果没有重参数化技巧，VAE 就无法使用标准反向传播进行训练。正是这一个洞见，让 VAE 真正变得实用。

### Gumbel-Softmax（可微分类采样）

重参数化技巧适用于连续分布（如高斯分布）。对于离散分类分布，我们需要另一种办法。Gumbel-Softmax 为分类采样提供了一个可微近似。

**Gumbel-Max 技巧（不可微）：**

```
To sample from a categorical distribution with log-probabilities log(p_1), ..., log(p_k):
  1. Sample g_i ~ Gumbel(0, 1) for each category
     (g = -log(-log(u)), where u ~ Uniform(0, 1))
  2. Return argmax(log(p_i) + g_i)

This produces exact categorical samples.
```

**Gumbel-Softmax（可微近似）：**

```
Replace the hard argmax with a soft softmax:
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau (temperature) controls the approximation:
  tau -> 0:  approaches a one-hot vector (hard categorical)
  tau -> inf: approaches uniform (1/k, 1/k, ..., 1/k)
  tau = 1.0: soft approximation
```

Gumbel-Softmax 会把离散样本连续松弛化。输出结果是一个概率向量（软 one-hot），而不是硬 one-hot。梯度可以流过 softmax。在训练时的前向传播中，你可以使用 “straight-through” 估计器：前向时使用硬 argmax，但反向时使用软 Gumbel-Softmax 的梯度。

**应用场景：**
- VAE 中的离散潜变量
- 神经架构搜索（选择离散操作）
- 硬注意力机制
- 具有离散动作的强化学习

### 分层采样 (stratified sampling)

标准蒙特卡洛采样可能会因为随机性而在样本空间中留下空白区域。分层采样通过把空间划分成多个层，并从每一层中采样，来强制实现更均匀的覆盖。

```
Standard Monte Carlo:
  Sample N points uniformly from [0, 1]
  Some regions may have clusters, others gaps

Stratified sampling:
  Divide [0, 1] into N equal strata: [0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  Sample one point uniformly within each stratum
  x_i = (i + u_i) / N   where u_i ~ Uniform(0, 1),  i = 0, ..., N-1
```

与标准蒙特卡洛相比，分层采样的方差总是不大于它：

```
Var(stratified) <= Var(standard Monte Carlo)

The improvement is largest when f(x) varies smoothly.
For piecewise-constant functions, stratified sampling is exact.
```

**应用场景：**
- 数值积分（准蒙特卡洛）
- 训练数据划分（确保每个折中的类别均衡）
- 带分层的重要性采样（结合这两种技术）
- NeRF（Neural Radiance Fields）会沿着相机射线使用分层采样

### 与扩散模型 (diffusion models) 的联系

扩散模型通过一个采样过程来生成图像。前向过程会在 T 个步骤中不断向图像加入高斯噪声，直到它变成纯噪声。反向过程则学习如何去噪，一步步恢复原始图像。

```
Forward process (known):
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  where epsilon ~ N(0, I)

  After T steps: x_T ~ N(0, I)  (pure noise)

Reverse process (learned):
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  where z ~ N(0, I)

  Each denoising step is a sampling step.
```

它与本课方法的联系在于：
- 每一步去噪都使用了重参数化技巧（先采样噪声，再做确定性变换）
- 噪声调度 {alpha_t} 控制了一种温度退火 (temperature annealing)
- 训练会用蒙特卡洛估计来近似 ELBO（evidence lower bound，证据下界）
- 扩散模型中的祖先采样 (ancestral sampling) 本质上是一条马尔可夫链（每一步只依赖当前状态）

整个图像生成过程就是迭代采样：从噪声开始，然后在每一步中，根据学到的去噪模型，采样出一个噪声稍微更少的版本。

## 动手构建

### 第 1 步：均匀采样与逆 CDF 采样

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

生成 10,000 个指数分布样本，并验证其均值是否为 1/lambda。

### 第 2 步：拒绝采样

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

使用拒绝采样从截断正态分布中抽样。通过样本直方图验证其形状。

### 第 3 步：重要性采样

```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

在正态分布下，使用均匀提议分布估计 E[X^2]。将结果与已知答案（mu^2 + sigma^2）比较。

### 第 4 步：用蒙特卡洛估计 pi

```python
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```

### 第 5 步：Metropolis-Hastings MCMC

```python
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```

从双峰分布（两个高斯的混合）中采样。把链的轨迹可视化出来。

### 第 6 步：吉布斯采样

```python
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```

### 第 7 步：温度采样

```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

展示温度如何改变一组 token logits 的输出分布。

### 第 8 步：Top-k 与 Top-p 采样

```python
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```

### 第 9 步：重参数化技巧

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

演示梯度可以穿过重参数化后的样本流动，但无法穿过直接采样流动。

### 第 10 步：Gumbel-Softmax

```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

展示随着温度降低，输出如何逐渐逼近 one-hot 向量。

包含全部可视化的完整实现位于 `code/sampling.py`。

## 实际使用

使用 NumPy 和 SciPy 时，生产环境版本如下：

```python
import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"Exponential mean: {exponential_samples.mean():.4f} (expected 2.0)")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"CDF at 1.96: {normal.cdf(1.96):.4f}")
print(f"Inverse CDF at 0.975: {normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"Sampled token index: {token}")
```

对于大规模 MCMC，可以使用专门的库：
- PyMC：带 NUTS（自适应 HMC）的完整贝叶斯建模
- emcee：集成式 MCMC 采样器
- NumPyro/JAX：GPU 加速的 MCMC

这些你都已经亲手实现过了。现在你知道这些库函数背后到底在做什么。

## 练习

1. 为柯西分布实现逆 CDF 采样。其 CDF 为 F(x) = 0.5 + arctan(x)/pi。生成 10,000 个样本，并将直方图与真实 PDF 一起绘制出来。观察它的重尾（远离中心的极端值）。

2. 使用拒绝采样，在 Uniform(0, 1) 提议分布下为 Beta(2, 5) 分布生成样本。将接受到的样本与真实 Beta PDF 一起绘制出来。理论接受率是多少？

3. 使用 1,000、10,000 和 100,000 个样本，采用蒙特卡洛方法估计 sin(x) 在 0 到 pi 上的积分。比较每个样本规模下的误差。验证误差是否按 O(1/sqrt(N)) 缩放。

4. 实现 Metropolis-Hastings，从一个与 exp(-(x^2 * y^2 + x^2 + y^2 - 8*x - 8*y) / 2) 成正比的二维分布 p(x, y) 中采样。绘制样本与链的轨迹。尝试不同的提议标准差。

5. 构建一个完整的文本生成演示：给定一个包含 10 个词的词表及其 logits，分别使用 (a) greedy、(b) temperature=0.7、(c) top-k=3、(d) top-p=0.9 生成长度为 20 个词元的序列。比较 5 次运行下输出的多样性。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| 采样 | “抽取随机值” | 按照某个概率分布生成取值。它是所有生成式 AI 背后的机制 |
| 均匀分布 | “都一样可能” | [a, b] 中的每个取值都具有相同的概率密度 1/(b-a)。这是所有采样方法的起点 |
| 逆 CDF | “概率变换” | F_inverse(U) 会把均匀样本转换成任何已知 CDF 分布的样本。精确且高效 |
| 拒绝采样 | “提议，然后接受/拒绝” | 先从简单提议分布生成样本，再按目标/提议比值成比例的概率接受。精确，但会浪费样本 |
| 重要性采样 | “给样本重新加权” | 使用来自 q(x) 的样本，通过给每个样本乘上 p(x)/q(x) 来估计 p(x) 下的期望。这是强化学习中 PPO 的核心 |
| 蒙特卡洛 | “对随机样本求平均” | 把积分近似为样本均值。无论维度如何，误差都是 O(1/sqrt(N)) |
| MCMC | “会收敛的随机游走” | 构造一条平稳分布就是目标分布的马尔可夫链。Metropolis-Hastings 是其基础算法 |
| Metropolis-Hastings | “上坡接受，下坡有时也接受” | 提议移动，再根据密度比决定是否接受。细致平衡保证它收敛到目标分布 |
| 吉布斯采样 | “一次更新一个变量” | 固定其他变量，从每个变量的条件分布中更新它。接受率始终为 100% |
| 温度 | “控制置信度的旋钮” | 在 softmax 之前用 T 去除 logits。T&lt;1 会变尖锐（更自信），T>1 会变平坦（更多样） |
| Top-k 采样 | “保留最好的 k 个” | 把除最高概率的 k 个词元外其余全部置零，重新归一化后采样。候选集合大小固定 |
| 核采样 (top-p) | “保留那些更可能的” | 保留累计概率超过 p 的最小词元集合。候选集合大小是自适应的 |
| 重参数化技巧 | “把随机性挪到外面” | 写成 z = mu + sigma * epsilon，其中 epsilon ~ N(0,1)。这样采样就可微了，是 VAE 训练的关键 |
| Gumbel-Softmax | “软分类采样” | 使用 Gumbel 噪声加上带温度的 softmax，对分类采样做可微近似 |
| 分层采样 | “强制覆盖” | 把样本空间划分成多个层，并从每一层中采样。其方差总是低于朴素蒙特卡洛 |
| 预热期 | “热身阶段” | 在马尔可夫链达到平稳分布之前，被丢弃的那部分初始 MCMC 样本 |
| 细致平衡 | “可逆性条件” | p(x) * T(x->y) = p(y) * T(y->x)。这是 p 成为马尔可夫链平稳分布的充分条件 |
| 扩散采样 | “迭代去噪” | 从噪声开始，通过不断应用学到的去噪步骤来生成数据。每一步都是条件采样操作 |

## 延伸阅读

- [Holbrook (2023): The Metropolis-Hastings Algorithm](https://arxiv.org/abs/2304.07010) - 关于 MCMC 基础的详细教程
- [Jang, Gu, Poole (2017): Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) - Gumbel-Softmax 原始论文
- [Holtzman et al. (2020): The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - 核采样（top-p）论文
- [Kingma & Welling (2014): Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - 引入重参数化技巧的 VAE 论文
- [Ho, Jain, Abbeel (2020): Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - 将采样与图像生成联系起来的 DDPM 论文
