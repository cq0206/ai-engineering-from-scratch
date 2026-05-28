# 机器学习中的统计学 (Statistics for Machine Learning)

> 统计学 (statistics) 帮助你判断模型到底是真的有效，还是只是运气好。

**类型：** 构建
**语言：** Python
**前置要求：** 第 1 阶段，第 06 课（概率与分布，Probability and Distributions），第 07 课（贝叶斯定理，Bayes' Theorem）
**时间：** ~120 分钟

## 学习目标

- 从零计算描述性统计、Pearson/Spearman 相关性以及协方差矩阵
- 进行假设检验（t 检验、卡方检验），并正确解释 p 值和置信区间
- 使用 bootstrap 重采样在不依赖分布假设的情况下为任意指标构造置信区间
- 使用效应量区分统计显著性与实际显著性

## 问题

你训练了两个模型。模型 A 在测试集上得到 0.87，模型 B 得到 0.89。你部署了模型 B。三周后，生产环境指标反而比之前更差。发生了什么？

模型 B 实际上并没有优于模型 A。那 0.02 的差异只是噪声。要么你的测试集太小，要么方差太高，或者两者兼而有之。你上线的并不是改进，而是被伪装成改进的随机性。

这种事一直都在发生。Kaggle 榜单频繁洗牌。论文无法复现。只基于几百个样本就宣布赢家的 A/B 测试。根本原因总是一样：有人跳过了统计学。

统计学给你提供了区分信号与噪声的工具。它告诉你差异何时真实存在、你应该有多大把握，以及在信任结果之前到底需要多少数据。每一条机器学习流水线、每一次模型比较、每一项实验都需要统计学。没有它，你就是在猜。

## 核心概念

### 描述性统计 (Descriptive Statistics)：总结你的数据

在建立任何模型之前，你必须先了解数据长什么样。描述性统计把一个数据集压缩成少量数字，用来概括它的整体形态。

**集中趋势度量**回答的是“中间位置在哪里？”

```
Mean:   sum of all values / count
        mu = (1/n) * sum(x_i)

Median: middle value when sorted
        Robust to outliers. If you have [1, 2, 3, 4, 1000], the mean is 202
        but the median is 3.

Mode:   most frequent value
        Useful for categorical data. For continuous data, rarely informative.
```

均值是平衡点，中位数是正中间的位置。当两者差异很大时，说明你的分布是偏斜的。收入分布通常满足 mean >> median（因为亿万富翁导致右偏）。训练过程中的损失分布则常常是 mean &lt;&lt; median（因为容易样本导致左偏）。

**离散程度度量**回答的是“数据有多分散？”

```
Variance:   average squared deviation from the mean
            sigma^2 = (1/n) * sum((x_i - mu)^2)

Standard deviation:  square root of variance
                     sigma = sqrt(sigma^2)
                     Same units as the data, so more interpretable.

Range:      max - min
            Sensitive to outliers. Almost never useful alone.

IQR:        Q3 - Q1 (interquartile range)
            The range of the middle 50% of the data.
            Robust to outliers. Used for box plots and outlier detection.
```

**百分位数 (percentiles)**把排序后的数据分成 100 个相等部分。第 25 百分位数（Q1）表示有 25% 的数值低于这个点。第 50 百分位数就是中位数。第 75 百分位数是 Q3。

```
For latency monitoring:
  P50 = median latency        (typical user experience)
  P95 = 95th percentile       (bad but not worst case)
  P99 = 99th percentile       (tail latency, often 10x the median)
```

在机器学习中，你会关心百分位数来分析推理延迟、预测置信度分布以及误差分布。一个平均误差很低但 P99 误差非常糟糕的模型，在安全关键场景里可能毫无用处。

**样本统计量与总体统计量。** 当你从样本计算方差时，分母应该用 (n-1) 而不是 n。这叫作贝塞尔校正 (Bessel's correction)。它用于补偿“样本均值不是真实总体均值”这一事实。如果分母用 n，你会系统性低估真实方差；如果用 (n-1)，这个估计就是无偏的。

```
Population variance: sigma^2 = (1/N) * sum((x_i - mu)^2)
Sample variance:     s^2     = (1/(n-1)) * sum((x_i - x_bar)^2)
```

在实践中，如果 n 很大（几千个样本），差异几乎可以忽略；如果 n 很小（几十个样本），这就很重要。

### 相关性 (Correlation)：变量如何一起变化

相关性衡量两个变量之间线性关系的强度和方向。

**Pearson 相关系数 (Pearson correlation coefficient)**用于衡量线性关联：

```
r = sum((x_i - x_bar)(y_i - y_bar)) / (n * s_x * s_y)

r = +1:  perfect positive linear relationship
r = -1:  perfect negative linear relationship
r =  0:  no linear relationship (but there might be a nonlinear one!)

Range: [-1, 1]
```

Pearson 相关假设这种关系是线性的，并且两个变量都近似服从正态分布。它对离群点很敏感。单个极端点就可能把 r 从 0.1 拉到 0.9。

**Spearman 秩相关 (Spearman rank correlation)**用于衡量单调关联：

```
1. Replace each value with its rank (1, 2, 3, ...)
2. Compute Pearson correlation on the ranks

Spearman catches any monotonic relationship, not just linear.
If y = x^3, Pearson gives r < 1 but Spearman gives rho = 1.
```

**什么时候用哪一种：**

```
Pearson:    Both variables are continuous and roughly normal.
            You care about the linear relationship specifically.
            No extreme outliers.

Spearman:   Ordinal data (rankings, ratings).
            Data is not normally distributed.
            You suspect a monotonic but not linear relationship.
            Outliers are present.
```

**黄金法则：**相关性不代表因果性。冰淇淋销量和溺水死亡人数相关，是因为它们都会在夏天上升。你的模型准确率和参数数量可能相关，但增加参数并不会自动提升准确率（参见：过拟合）。

### 协方差矩阵 (Covariance Matrix)

两个变量之间的协方差衡量的是它们如何共同变化：

```
Cov(X, Y) = (1/n) * sum((x_i - x_bar)(y_i - y_bar))

Cov(X, Y) > 0:  X and Y tend to increase together
Cov(X, Y) < 0:  when X increases, Y tends to decrease
Cov(X, Y) = 0:  no linear co-movement
```

对于 d 个特征，协方差矩阵 C 是一个 d x d 的矩阵，其中 C[i][j] = Cov(feature_i, feature_j)。对角线元素 C[i][i] 是各个特征的方差。

```
C = | Var(x1)      Cov(x1,x2)  Cov(x1,x3) |
    | Cov(x2,x1)  Var(x2)      Cov(x2,x3) |
    | Cov(x3,x1)  Cov(x3,x2)  Var(x3)     |

Properties:
  - Symmetric: C[i][j] = C[j][i]
  - Positive semi-definite: all eigenvalues >= 0
  - Diagonal = variances
  - Off-diagonal = covariances
```

**它与 PCA 的联系。** PCA 会对协方差矩阵做特征分解。特征向量是主成分（方差最大的方向），特征值告诉你每个成分捕获了多少方差。这正是第 10 课讲过的内容；现在你也明白了为什么协方差矩阵是正确的分解对象：它编码了数据中所有成对的线性关系。

**它与相关性的联系。** 相关矩阵是标准化变量（每个变量都除以自己的标准差）对应的协方差矩阵。相关性把协方差归一化，所以所有值都落在 [-1, 1] 内。

### 假设检验 (Hypothesis Testing)

假设检验是在不确定性下做决策的框架。你从一个主张开始，收集数据，然后判断这些数据是否与该主张一致。

**基本设置：**

```
Null hypothesis (H0):        the default assumption, usually "no effect"
Alternative hypothesis (H1): what you are trying to show

Example:
  H0: Model A and Model B have the same accuracy
  H1: Model B has higher accuracy than Model A
```

**p 值 (p-value)**是在 H0 为真的前提下，观察到像当前这样极端数据的概率。它**不是** H0 为真的概率。这是统计学里最常见的误解。

```
p-value = P(data this extreme | H0 is true)

If p-value < alpha (typically 0.05):
    Reject H0. The result is "statistically significant."
If p-value >= alpha:
    Fail to reject H0. You do not have enough evidence.
    This does NOT mean H0 is true.
```

**置信区间 (confidence intervals)**给出了某个参数的一组合理取值范围：

```
95% confidence interval for the mean:
    x_bar +/- z * (s / sqrt(n))

where z = 1.96 for 95% confidence

Interpretation: if you repeated this experiment many times, 95% of the
computed intervals would contain the true mean. It does NOT mean there
is a 95% probability the true mean is in this specific interval.
```

置信区间的宽度体现了估计的精度。区间越宽，不确定性越高；区间越窄，说明你的估计越精确（但如果数据本身有偏，它不一定更准确）。

### t 检验 (t-test)

t 检验用于比较均值。有几种常见形式。

**单样本 t 检验：**总体均值是否与某个假设值不同？

```
t = (x_bar - mu_0) / (s / sqrt(n))

degrees of freedom = n - 1
```

**双样本 t 检验（独立样本）：**两个组的均值是否不同？

```
t = (x_bar_1 - x_bar_2) / sqrt(s1^2/n1 + s2^2/n2)

This is Welch's t-test, which does not assume equal variances.
Always use Welch's unless you have a specific reason for equal variances.
```

**配对 t 检验：**当测量成对出现时（例如同一个模型在同样的数据划分上被评估）：

```
Compute d_i = x_i - y_i for each pair
Then run a one-sample t-test on the d_i values against mu_0 = 0
```

在机器学习中，配对 t 检验很常见：你会在相同的 10 个交叉验证折 (cross-validation folds) 上运行两个模型，然后逐对比较它们的分数。

### 卡方检验 (Chi-squared Test)

卡方检验用于检查观测频数是否与期望频数一致。它对分类数据尤其有用。

```
chi^2 = sum((observed - expected)^2 / expected)

Example: does a language model's output distribution match the
training distribution across categories?

Category    Observed   Expected
Positive       120        100
Negative        80        100
chi^2 = (120-100)^2/100 + (80-100)^2/100 = 4 + 4 = 8

With 1 degree of freedom, chi^2 = 8 gives p < 0.005.
The difference is significant.
```

### 面向机器学习模型的 A/B 测试 (A/B Testing for ML Models)

机器学习中的 A/B 测试和网页产品中的 A/B 测试并不一样。模型比较有其特定挑战：

```
1. Same test set:    Both models must be evaluated on identical data.
                     Different test sets make comparison meaningless.

2. Multiple metrics: Accuracy alone is not enough. You need precision,
                     recall, F1, latency, and fairness metrics.

3. Variance:         Use cross-validation or bootstrap to estimate
                     the variance of each metric, not just point estimates.

4. Data leakage:     If the test set was used during model selection,
                     your comparison is biased. Hold out a final test set.
```

**流程如下：**

```
1. Define your metric and significance level (alpha = 0.05)
2. Run both models on the same k-fold cross-validation splits
3. Collect paired scores: [(a1, b1), (a2, b2), ..., (ak, bk)]
4. Compute differences: d_i = b_i - a_i
5. Run a paired t-test on the differences
6. Check: is the mean difference significantly different from 0?
7. Compute a confidence interval for the mean difference
8. Compute effect size (Cohen's d) to judge practical significance
```

### 统计显著性 (Statistical Significance) 与实际显著性 (Practical Significance)

一个结果可能在统计上显著，但在实际中毫无意义。只要数据量足够大，再微不足道的差异也会变成统计显著。

```
Example:
  Model A accuracy: 0.9234
  Model B accuracy: 0.9237
  n = 1,000,000 test samples
  p-value = 0.001

Statistically significant? Yes.
Practically significant? A 0.03% improvement is not worth the
engineering cost of deploying a new model.
```

**效应量 (effect size)**用于量化差异到底有多大，并且不受样本量影响：

```
Cohen's d = (mean_1 - mean_2) / pooled_std

d = 0.2:  small effect
d = 0.5:  medium effect
d = 0.8:  large effect
```

请始终同时报告 p 值和效应量。p 值告诉你差异是否真实存在，效应量告诉你这种差异是否值得在实践中关注。

### 多重比较问题 (Multiple Comparison Problem)

当你同时检验很多假设时，其中一些结果会纯粹因为运气而“显著”。如果你用 alpha = 0.05 去检验 20 个东西，那么即便实际上什么都没有，也预期会出现 1 个假阳性。

```
P(at least one false positive) = 1 - (1 - alpha)^m

m = 20 tests, alpha = 0.05:
P(false positive) = 1 - 0.95^20 = 0.64

You have a 64% chance of at least one false positive.
```

**Bonferroni 校正 (Bonferroni correction)：**把 alpha 除以检验次数。

```
Adjusted alpha = alpha / m = 0.05 / 20 = 0.0025

Only reject H0 if p-value < 0.0025.
Conservative but simple. Works when tests are independent.
```

在机器学习里，这在以下场景非常重要：你在多个指标上比较同一个模型、测试很多超参数配置，或者在多个数据集上做评估。

### Bootstrap 方法 (Bootstrap Methods)

bootstrap 通过“有放回重采样”来估计某个统计量的抽样分布。它不要求你对底层分布做任何假设。

**算法如下：**

```
1. You have n data points
2. Draw n samples WITH replacement (some points appear multiple times,
   some not at all)
3. Compute your statistic on this bootstrap sample
4. Repeat B times (typically B = 1000 to 10000)
5. The distribution of bootstrap statistics approximates the
   sampling distribution
```

**Bootstrap 置信区间（百分位法）：**

```
Sort the B bootstrap statistics
95% CI = [2.5th percentile, 97.5th percentile]
```

**为什么 bootstrap 对机器学习很重要：**

```
- Test set accuracy is a point estimate. Bootstrap gives you
  confidence intervals.
- You cannot assume metric distributions are normal (especially
  for AUC, F1, precision at k).
- Bootstrap works for ANY statistic: median, ratio of two means,
  difference in AUC between two models.
- No closed-form formula needed.
```

**用 bootstrap 比较模型：**

```
1. You have predictions from Model A and Model B on the same test set
2. For each bootstrap iteration:
   a. Resample test indices with replacement
   b. Compute metric_A and metric_B on the resampled set
   c. Store diff = metric_B - metric_A
3. 95% CI for the difference:
   [2.5th percentile of diffs, 97.5th percentile of diffs]
4. If the CI does not contain 0, the difference is significant
```

这比配对 t 检验更稳健，因为它不依赖任何分布假设。

### 参数检验 (Parametric Tests) 与非参数检验 (Non-parametric Tests)

**参数检验**假设数据服从某种特定分布（通常是正态分布）：

```
t-test:         assumes normally distributed data (or large n by CLT)
ANOVA:          assumes normality and equal variances
Pearson r:      assumes bivariate normality
```

**非参数检验**不做分布假设：

```
Mann-Whitney U:     compares two groups (replaces independent t-test)
Wilcoxon signed-rank: compares paired data (replaces paired t-test)
Spearman rho:       correlation on ranks (replaces Pearson)
Kruskal-Wallis:     compares multiple groups (replaces ANOVA)
```

**什么时候使用非参数检验：**

```
- Small sample size (n < 30) and data is clearly non-normal
- Ordinal data (ratings, rankings)
- Heavy outliers you cannot remove
- Skewed distributions
```

**什么时候使用参数检验：**

```
- Large sample size (CLT makes the test statistic approximately normal)
- Data is roughly symmetric without extreme outliers
- More statistical power (better at detecting real differences)
```

在机器学习实验中，n 通常很小（例如 5 个或 10 个交叉验证折），因此像 Wilcoxon signed-rank 这样的非参数检验往往比 t 检验更合适。

### 中心极限定理 (Central Limit Theorem, CLT)：实践意义

CLT 指出：随着 n 增长，样本均值的分布会趋近于正态分布，而不管总体分布本身是什么样。

```
If X_1, X_2, ..., X_n are iid with mean mu and variance sigma^2:

    X_bar ~ Normal(mu, sigma^2 / n)    as n -> infinity

Works for n >= 30 in most cases.
For highly skewed distributions, you might need n >= 100.
```

**为什么这对机器学习很重要：**

```
1. Justifies confidence intervals and t-tests on aggregated metrics
2. Explains why averaging over cross-validation folds gives stable
   estimates even when individual folds vary wildly
3. Mini-batch gradient descent works because the average gradient
   over a batch approximates the true gradient (CLT in action)
4. Ensemble methods: averaging predictions from many models gives
   more stable output than any single model
```

**CLT 不能做什么：**

```
- Does NOT make your data normal. It makes the MEAN of samples normal.
- Does NOT work for heavy-tailed distributions with infinite variance
  (Cauchy distribution).
- Does NOT apply to dependent data (time series without correction).
```

### 机器学习论文中常见的统计错误

1. **在训练集上测试。** 这必然导致过拟合。一定要留出模型在训练期间从未见过的数据。

2. **没有置信区间。** 只报告一个准确率数字而不给出不确定性，会让结果既不可复现，也无法验证。

3. **忽略多重比较。** 测试 50 种配置却不做校正，只报告最好的那个，会抬高假阳性率。

4. **混淆统计显著性和实际显著性。** 在准确率只提升 0.01% 的情况下得到 p 值 0.001，并不意味着结果有意义。

5. **在类别不平衡数据上使用准确率。** 如果一个数据集里 99% 都是负类，那么 99% 的准确率可能只意味着模型什么都没学到。应该使用 precision、recall、F1 或 AUC。

6. **挑指标报喜。** 只报告你的模型表现最好的那个指标。诚实的评估应该报告所有相关指标。

7. **训练/测试划分之间发生信息泄漏。** 比如先归一化再划分，或者用未来数据预测过去。

8. **测试集太小且没有方差估计。** 在 100 个样本上评估，然后宣称提升了 2%，这通常是噪声，不是信号。

9. **在数据并不独立时仍假设独立。** 例如同一位病人的多张医学图像，或者同一篇文档中的多个句子。组内观测是相关的。

10. **P-hacking。** 不断尝试不同检验、不同子集或不同排除标准，直到得到 p &lt; 0.05。这样的结果只是搜索过程制造出来的假象。

## 动手实现

你将实现：

1. **从零实现描述性统计**（均值、中位数、众数、标准差、百分位数、IQR）
2. **相关性函数**（Pearson 和 Spearman，以及协方差矩阵）
3. **假设检验**（单样本 t 检验、双样本 t 检验、卡方检验）
4. **Bootstrap 置信区间**（适用于任意统计量，无需分布假设）
5. **A/B 测试模拟器**（生成数据、做检验，并检查第一类错误和第二类错误）
6. **统计显著性 vs 实际显著性演示**（展示为什么只要 n 足够大，几乎一切都会“显著”）

全部从零开始，只使用 `math` 和 `random`。不用 numpy，也不用 scipy。

## 关键术语

| 术语 | 定义 |
|---|---|
| 均值 (Mean) | 数值之和除以数量。对离群点敏感。 |
| 中位数 (Median) | 排序后位于中间的值。对离群点稳健。 |
| 标准差 (Standard deviation) | 方差的平方根。用原始单位衡量离散程度。 |
| 百分位数 (Percentile) | 有一定比例数据落在其下方的那个值。 |
| 四分位距 (IQR) | 四分位距。即 Q3 减 Q1，表示中间 50% 数据的跨度。 |
| Pearson 相关性 (Pearson correlation) | 衡量两个变量之间的线性关联。范围为 [-1, 1]。 |
| Spearman 相关性 (Spearman correlation) | 使用秩来衡量单调关联。 |
| 协方差矩阵 (Covariance matrix) | 所有特征两两协方差组成的矩阵。 |
| 零假设 (Null hypothesis) | 默认认为没有效应或没有差异的假设。 |
| p 值 (p-value) | 在零假设为真时，观察到当前这么极端数据的概率。 |
| 置信区间 (Confidence interval) | 在给定置信水平下，参数可能取值的范围。 |
| t 检验 (t-test) | 检验均值差异是否显著，使用 t 分布。 |
| 卡方检验 (Chi-squared test) | 检验观测频数是否与期望频数不同。 |
| 效应量 (Effect size) | 差异幅度的大小，与样本量无关。常见形式是 Cohen's d。 |
| Bonferroni 校正 (Bonferroni correction) | 用检验次数去除显著性阈值，以控制假阳性。 |
| Bootstrap | 通过有放回重采样来估计抽样分布。 |
| 第一类错误 (Type I error) | 假阳性。H0 为真时却拒绝了 H0。 |
| 第二类错误 (Type II error) | 假阴性。H0 为假时却未能拒绝 H0。 |
| 统计功效 (Statistical power) | 正确拒绝错误 H0 的概率。功效 = 1 减去第二类错误率。 |
| 中心极限定理 (Central limit theorem) | 随着样本量增加，样本均值会收敛到正态分布。 |
| 参数检验 (Parametric test) | 假设数据服从某种特定分布（通常是正态分布）。 |
| 非参数检验 (Non-parametric test) | 不做分布假设，通常基于秩或符号。 |
