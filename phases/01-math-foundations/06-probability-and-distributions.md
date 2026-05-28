# 概率与分布 (Probability and Distributions)

> 概率是人工智能表达不确定性的语言。

**类型：** 学习
**语言：** Python
**前置知识：** 第一阶段，第01-04课
**时间：** ~75 分钟

## 学习目标

- 从头实现伯努利分布、分类分布、泊松分布、均匀分布和正态分布的概率质量函数（PMF，probability mass function）和概率密度函数（PDF，probability density function）
- 计算期望值（expected value）和方差（variance），并利用中心极限定理（Central Limit Theorem）解释为何高斯分布无处不在
- 实现带数值稳定技巧（减去最大 logit）的 softmax 和 log-softmax 函数
- 从 logits 计算交叉熵损失（cross-entropy loss），并将其与负对数似然联系起来

## 问题所在

分类器输出 `[0.03, 0.91, 0.06]`，语言模型从 50,000 个候选词中选下一个词，扩散模型通过从学习到的分布中采样生成图像——这些都是概率的应用。

模型的每一个预测都是一个概率分布。每个损失函数衡量预测分布与真实分布之间的差距。每个训练步骤调整参数，使一个分布更像另一个。不懂概率，你就无法读懂任何一篇机器学习论文，无法调试任何一个模型，也无法理解为什么训练损失会变成 NaN。

## 概念讲解

### 事件、样本空间与概率 (Events, Sample Spaces, and Probability)

样本空间（sample space）S 是所有可能结果的集合，事件（event）是样本空间的子集，概率将事件映射到 0 到 1 之间的数。

```
Coin flip:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

Single die roll:
  S = {1, 2, 3, 4, 5, 6}
  P(even) = P({2, 4, 6}) = 3/6 = 0.5
```

三条公理定义了全部概率论：
1. 对任意事件 A，P(A) >= 0
2. P(S) = 1（某件事总会发生）
3. 当 A 与 B 不能同时发生时，P(A 或 B) = P(A) + P(B)

其他一切（贝叶斯定理、期望值、分布）都从这三条规则推导而来。

### 条件概率与独立性 (Conditional Probability and Independence)

P(A|B) 是在 B 已发生的条件下 A 发生的概率。

```
P(A|B) = P(A and B) / P(B)

Example: deck of cards
  P(King | Face card) = P(King and Face card) / P(Face card)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

当两个事件互相独立（independent）时，知道其中一个对另一个毫无帮助：

```
Independent:   P(A|B) = P(A)
Equivalent to: P(A and B) = P(A) * P(B)
```

掷硬币是独立的，不放回抽牌不是。

### 概率质量函数与概率密度函数 (Probability Mass Functions vs Probability Density Functions)

离散随机变量具有概率质量函数（PMF）。每个结果都有一个可以直接读取的具体概率。

```
PMF: P(X = k)

Fair die:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  Sum of all probabilities = 1
```

连续随机变量具有概率密度函数（PDF）。单个点处的密度不是概率，概率来自对密度在区间上的积分。

```
PDF: f(x)

P(a <= X <= b) = integral of f(x) from a to b

f(x) can be greater than 1 (density, not probability)
integral from -inf to +inf of f(x) dx = 1
```

这一区别在机器学习中很重要。分类输出是 PMF（离散选择），VAE 的潜在空间使用 PDF（连续型）。

### 常见分布 (Common Distributions)

**伯努利分布（Bernoulli）：** 一次试验，两种结果。用于建模二分类。

```
P(X = 1) = p
P(X = 0) = 1 - p
Mean = p,  Variance = p(1-p)
```

**分类分布（Categorical）：** 一次试验，k 种结果。用于建模多分类（softmax 输出）。

```
P(X = i) = p_i,  where sum of p_i = 1
Example: P(cat) = 0.7,  P(dog) = 0.2,  P(bird) = 0.1
```

**均匀分布（Uniform）：** 所有结果等可能。用于随机初始化。

```
Discrete: P(X = k) = 1/n for k in {1, ..., n}
Continuous: f(x) = 1/(b-a) for x in [a, b]
```

**正态分布/高斯分布（Normal/Gaussian）：** 钟形曲线。由均值（mu）和方差（sigma^2）参数化。

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

Standard normal: mu = 0, sigma = 1
  68% of data within 1 sigma
  95% within 2 sigma
  99.7% within 3 sigma
```

**泊松分布（Poisson）：** 固定时间间隔内稀有事件的计数。用于建模事件频率。

```
P(X = k) = (lambda^k * e^(-lambda)) / k!
Mean = lambda,  Variance = lambda
```

### 期望值与方差 (Expected Value and Variance)

期望值是加权平均结果。

```
Discrete:   E[X] = sum of x_i * P(X = x_i)
Continuous: E[X] = integral of x * f(x) dx
```

方差衡量均值附近的离散程度。

```
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
Standard deviation = sqrt(Var(X))
```

在机器学习中，期望值以损失函数的形式出现（数据分布上的平均损失），方差反映模型的稳定性——梯度方差高意味着训练噪声大。

### 联合分布与边缘分布 (Joint and Marginal Distributions)

联合分布（joint distribution）P(X, Y) 同时描述两个随机变量。

联合 PMF 示例（X = 天气，Y = 雨伞）：

| | Y=0（无雨伞） | Y=1（有雨伞） | 边缘 P(X) |
|---|---|---|---|
| X=0（晴天） | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1（雨天） | 0.05 | 0.45 | P(X=1) = 0.50 |
| **边缘 P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

边缘分布（marginal distribution）对另一个变量求和：

```
P(X = x) = sum over all y of P(X = x, Y = y)
```

上表的行和列的合计就是边缘分布。

### 正态分布无处不在的原因 (Why the Normal Distribution Shows Up Everywhere)

中心极限定理（Central Limit Theorem，CLT）：许多独立随机变量的和（或平均值）会收敛到正态分布，无论原始分布是什么。

```
Roll 1 die:  uniform distribution (flat)
Average of 2 dice:  triangular (peaked)
Average of 30 dice: nearly perfect bell curve

This works for ANY starting distribution.
```

这就是为什么：
- 测量误差近似服从正态分布（来自许多小的独立来源）
- 神经网络中的权重初始化使用正态分布
- 随机梯度下降（SGD）中的梯度噪声近似服从正态分布（大量样本梯度的求和）
- 正态分布是给定均值和方差条件下的最大熵分布

### 对数概率 (Log Probabilities)

原始概率会导致数值问题。将许多小概率相乘很快就会下溢为零。

```
P(sentence) = P(word1) * P(word2) * ... * P(word_n)
            = 0.01 * 0.003 * 0.02 * ...
            -> 0.0 (underflow after ~30 terms)
```

对数概率（log probabilities）解决了这个问题——乘法变为加法。

```
log P(sentence) = log P(word1) + log P(word2) + ... + log P(word_n)
                = -4.6 + -5.8 + -3.9 + ...
                -> finite number (no underflow)
```

规则：
- log(a * b) = log(a) + log(b)
- 对数概率始终 &lt;= 0（因为 0 &lt; P &lt;= 1）
- 越负 = 越不可能
- 交叉熵损失是正确类别的负对数概率

### Softmax 作为概率分布 (Softmax as a Probability Distribution)

神经网络输出原始分数（logits，对数几率）。Softmax 将其转换为有效的概率分布。

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

Properties:
  - All outputs are in (0, 1)
  - All outputs sum to 1
  - Preserves relative ordering of inputs
  - exp() amplifies differences between logits
```

Softmax 技巧：在指数运算前减去最大 logit，防止溢出。

```
z = [100, 101, 102]
exp(102) = overflow

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  (safe)

Same result, no overflow.
```

Log-softmax 将 softmax 和 log 合并以提高数值稳定性。PyTorch 在计算交叉熵损失时内部使用这一方法。

### 采样 (Sampling)

采样（sampling）指从分布中随机抽取值。在机器学习中：
- Dropout 随机采样要置零的神经元
- 数据增强采样随机变换
- 语言模型从预测分布中采样下一个词元
- 扩散模型采样噪声并逐步去噪

从任意分布中采样需要逆变换采样、拒绝采样或重参数化技巧（用于 VAE）等技术。

## 动手实现

### 第一步：概率基础

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

### 第二步：从头实现 PMF 和 PDF

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### 第三步：期望值与方差

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### 第四步：从分布中采样

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### 第五步：Softmax 与对数概率

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### 第六步：中心极限定理演示

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### 第七步：可视化

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

包含所有可视化的完整实现见 `code/probability.py`。

## 实际使用

使用 NumPy 和 SciPy，上述一切都是一行代码：

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

你已从头实现了这些。现在你知道这些库函数在做什么了。

## 练习

1. 实现指数分布的逆变换采样。通过采样 10,000 个值并将直方图与真实 PDF 进行比较来验证结果。

2. 为两个有偏骰子构建联合分布表。计算边缘分布，并检验两个骰子是否独立。

3. 计算一个 5 类分类器在正确类别为索引 3 时，对 logits `[2.0, 0.5, -1.0, 3.0, 0.1]` 的交叉熵损失。然后用 PyTorch 的 `nn.CrossEntropyLoss` 验证你的答案。

4. 编写一个函数，接受对数概率列表，返回最可能的序列、总对数概率及其等效原始概率。用一个每个词概率为 0.01 的 50 个词的句子测试。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| 样本空间 (sample space) | "所有可能性" | 实验每个可能结果的集合 S |
| PMF（概率质量函数）(PMF) | "概率函数" | 给出每个离散结果精确概率的函数，总和为 1 |
| PDF（概率密度函数）(PDF) | "概率曲线" | 连续变量的密度函数，在区间上积分即得概率 |
| 条件概率 (conditional probability) | "给定某条件下的概率" | P(A\|B) = P(A 和 B) / P(B)，贝叶斯思维和贝叶斯定理的基础 |
| 独立性 (independence) | "互不影响" | P(A 和 B) = P(A) * P(B)，知道一个事件对另一个毫无帮助 |
| 期望值 (expected value) | "平均值" | 所有结果的概率加权求和，损失函数就是期望值 |
| 方差 (variance) | "离散程度" | 关于均值的期望平方偏差，方差高 = 估计嘈杂、不稳定 |
| 正态分布 (normal distribution) | "钟形曲线" | f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2))，因中心极限定理而普遍存在 |
| 中心极限定理 (central limit theorem) | "平均值趋向正态" | 无论来源分布如何，大量独立样本的均值收敛到正态分布 |
| 联合分布 (joint distribution) | "两个变量一起" | P(X, Y) 描述 X 和 Y 各种组合的概率 |
| 边缘分布 (marginal distribution) | "对另一个变量求和" | P(X) = sum_y P(X, Y)，从联合分布中恢复单个变量的分布 |
| 对数概率 (log probability) | "概率的对数" | log P(x)，将乘积转为求和，防止长序列中的数值下溢 |
| Softmax | "将分数转为概率" | softmax(z_i) = exp(z_i) / sum(exp(z_j))，将实值 logits 映射为有效概率分布 |
| 交叉熵 (cross-entropy) | "损失函数" | -sum(p_true * log(p_predicted))，衡量两个分布的差异，越低越好 |
| Logits（对数几率） | "模型原始输出" | softmax 之前的未归一化分数，得名于逻辑斯蒂函数 |
| 采样 (sampling) | "抽取随机值" | 按照概率分布生成值，模型生成输出的方式 |

## 延伸阅读

- [3Blue1Brown：什么是中心极限定理？](https://www.youtube.com/watch?v=zeJD6dqJ5lo) —— 为什么平均值趋向正态的可视化证明
- [斯坦福 CS229 概率复习](https://cs229.stanford.edu/section/cs229-prob.pdf) —— 涵盖本课所有内容及更多内容的简明参考
- [Log-Sum-Exp 技巧](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) —— 数值稳定性的重要性及实现方法
