# 范数 (Norms) 与距离 (Distances)

> 你的距离函数 (distance function) 定义了“相似”意味着什么。选错了，后面所有环节都会出问题。

**类型：** 构建
**语言：** Python
**先修内容：** 第 1 阶段，第 01 课（线性代数直觉 / Linear Algebra Intuition）、第 02 课（向量、矩阵与运算 / Vectors, Matrices & Operations）
**时间：** ~90 分钟

## 学习目标

- 从零实现 L1、L2、余弦、马哈拉诺比斯、杰卡德和编辑距离函数
- 为给定的机器学习任务选择合适的距离度量，并解释为什么其他方案会失效
- 将 L1 和 L2 范数与 LASSO 和 Ridge 正则化及其几何约束区域联系起来
- 展示同一数据集在不同度量下如何产生不同的最近邻

## 问题

你有两个向量。它们也许是词嵌入，也许是用户画像，也许是像素数组。你需要知道：它们到底有多接近？

答案完全取决于你选择了哪一种距离函数。两个数据点在一种度量 (metric) 下可能互为最近邻，而在另一种度量下却相距很远。你的 KNN 分类器、推荐引擎、向量数据库、聚类算法、损失函数——它们全都依赖这个选择。选错了，你的模型优化的就是错误的目标。

不存在放之四海而皆准的最佳距离。L2 适用于空间数据。余弦相似度主导 NLP。杰卡德处理集合。编辑距离处理字符串。马哈拉诺比斯距离考虑相关性。Wasserstein 距离移动概率质量。它们每一种都编码了对“相似”含义的不同假设。

本课将从零构建各种主要的距离函数，说明每一种在什么场景下才是正确工具，并演示同一份数据在不同度量下会得到完全不同的最近邻。

## 概念

### 范数：衡量向量大小

范数 (norm) 衡量向量的“大小”。两个向量之间的每个距离函数都可以写成它们之差的范数：d(a, b) = ||a - b||。因此，理解范数，也就是理解距离。

### L1 范数 (L1 norm)（曼哈顿距离 / Manhattan distance）

L1 范数把所有分量的绝对值相加。

```
||x||_1 = |x_1| + |x_2| + ... + |x_n|
```

它之所以叫曼哈顿距离，是因为它衡量的是你在城市网格中、只能沿坐标轴移动时需要走多远。不能走对角线。

```
Point A = (1, 1)
Point B = (4, 5)

L1 distance = |4-1| + |5-1| = 3 + 4 = 7

On a grid, you walk 3 blocks east and 4 blocks north.
```

L1 的适用场景：
- 高维稀疏数据（文本特征、one-hot 编码）
- 当你希望对异常值更稳健时（单个巨大的差异不会主导结果）
- 特征选择问题（L1 正则化会促进稀疏性）

与 L1 正则化 (Lasso) 的联系：在损失函数中加入 ||w||_1，会惩罚权重绝对值之和。这会把较小的权重直接压到 0，从而自动完成特征选择。L1 惩罚会在权重空间中形成菱形约束区域，而菱形的角恰好落在坐标轴上，也就是某些权重为 0 的位置。

与损失函数的联系：平均绝对误差 (Mean Absolute Error, MAE) 是预测值与目标值之间 L1 距离的平均值。它对所有误差都做线性惩罚，因此相较于 MSE 对异常值更稳健。

### L2 范数 (L2 norm)（欧几里得距离 / Euclidean distance）

L2 范数就是直线距离，即各分量平方和的平方根。

```
||x||_2 = sqrt(x_1^2 + x_2^2 + ... + x_n^2)
```

这就是你在几何课上学过的距离。把毕达哥拉斯定理推广到 n 维即可。

```
Point A = (1, 1)
Point B = (4, 5)

L2 distance = sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = sqrt(25) = 5.0

The straight line, cutting diagonally through the grid.
```

L2 的适用场景：
- 低维到中维的连续数据
- 当各特征尺度可比较时
- 物理距离（空间数据、传感器读数）
- 像素级图像相似度

与 L2 正则化 (Ridge) 的联系：在损失函数中加入 ||w||_2^2，会惩罚较大的权重。与 L1 不同，它不会把权重直接压到 0，而是按比例把所有权重向 0 收缩。L2 惩罚会形成圆形约束区域，因此坐标轴上没有尖角。权重会变小，但很少恰好等于 0。

与损失函数的联系：平均平方误差 (Mean Squared Error, MSE) 是 L2 距离平方的平均值。平方会让大误差比小误差受到更重的惩罚。

```
MAE (L1 loss):  |y - y_hat|         Linear penalty. Robust to outliers.
MSE (L2 loss):  (y - y_hat)^2       Quadratic penalty. Sensitive to outliers.
```

### Lp 范数：一般家族

L1 和 L2 都是 Lp 范数 (Lp norm) 的特例：

```
||x||_p = (|x_1|^p + |x_2|^p + ... + |x_n|^p)^(1/p)
```

不同的 p 会产生不同形状的“单位球” (unit balls)（即与原点距离为 1 的所有点的集合）：

```
p=1:    Diamond shape      (corners on axes)
p=2:    Circle/sphere      (the usual round ball)
p=3:    Superellipse       (rounded square)
p=inf:  Square/hypercube   (flat sides along axes)
```

### L-无穷范数 (L-infinity norm)（切比雪夫距离 / Chebyshev distance）

当 p 趋近无穷大时，Lp 范数会收敛为绝对值最大的那个分量。

```
||x||_inf = max(|x_1|, |x_2|, ..., |x_n|)
```

两点之间的距离由它们差异最大的那个维度决定。其他维度都会被忽略。

```
Point A = (1, 1)
Point B = (4, 5)

L-inf distance = max(|4-1|, |5-1|) = max(3, 4) = 4
```

L-无穷的适用场景：
- 当任意单个维度上的最坏偏差最重要时
- 棋盘游戏（国际象棋中的国王移动对应 L-infinity：任意方向走一步代价都为 1）
- 制造公差控制（每个维度都必须满足规格）

### 余弦相似度 (cosine similarity) 与余弦距离 (cosine distance)

余弦相似度衡量两个向量之间的夹角，忽略它们的大小。

```
cos_sim(a, b) = (a . b) / (||a||_2 * ||b||_2)
```

它的取值范围是 -1（方向相反）到 +1（方向相同）。互相垂直的向量，其余弦相似度为 0。

余弦距离把它转换成距离：cosine_distance = 1 - cosine_similarity。它的范围是 0（方向完全一致）到 2（方向完全相反）。

```
a = (1, 0)    b = (1, 1)

cos_sim = (1*1 + 0*1) / (1 * sqrt(2)) = 1/sqrt(2) = 0.707
cos_dist = 1 - 0.707 = 0.293
```

为什么余弦在 NLP 和嵌入中占主导地位：在文本里，文档长度不应影响相似度。一篇关于猫的文档，即使长度是另一篇关于猫的文档的两倍，也仍然应该算“相似”。余弦相似度忽略大小（长度），只关心方向。词分布相同但长度不同的两篇文档会指向同一方向，因此余弦相似度为 1.0。

余弦相似度的适用场景：
- 文本相似度（TF-IDF 向量、词嵌入、句向量）
- 任意“大小是噪声、方向才是信号”的领域
- 推荐系统（用户偏好向量）
- 嵌入检索（向量数据库几乎总是使用余弦或点积）

### 点积相似度 (dot product similarity) vs 余弦相似度

两个向量的点积 (dot product) 为：

```
a . b = a_1*b_1 + a_2*b_2 + ... + a_n*b_n
      = ||a|| * ||b|| * cos(angle)
```

余弦相似度就是把点积再除以两个向量的大小完成归一化。当两个向量都已经做了单位归一化（大小 = 1）时，点积和余弦相似度完全相同。

```
If ||a|| = 1 and ||b|| = 1:
    a . b = cos(angle between a and b)
```

它们何时不同：点积包含大小信息。大小更大的向量会得到更高的点积分数。在一些检索系统中，这一点很重要，因为你可能希望“热门”项目排得更高。向量大小会充当一种隐式的质量或重要性信号。

```
a = (3, 0)    b = (1, 0)    c = (0, 1)

dot(a, b) = 3     dot(a, c) = 0
cos(a, b) = 1.0   cos(a, c) = 0.0

Both agree on direction, but dot product also reflects magnitude.
```

在实践中：
- 当你想要纯粹的方向相似性时，用余弦相似度
- 当大小本身携带有意义的信息时，用点积
- 许多向量数据库（Pinecone、Weaviate、Qdrant）都允许你在两者之间选择
- 如果你的嵌入已经做了 L2 归一化，那么两者没有区别

### 马哈拉诺比斯距离 (Mahalanobis distance)

欧几里得距离把所有维度一视同仁。但如果特征之间存在相关性，或者尺度不同，L2 就会给出误导性的结果。

马哈拉诺比斯距离会考虑数据的协方差结构。

```
d_M(x, y) = sqrt((x - y)^T * S^(-1) * (x - y))
```

其中 S 是数据的协方差矩阵。

直观地说：马哈拉诺比斯距离会先对数据做去相关并归一化（白化 / whitening），然后在变换后的空间里计算 L2 距离。如果 S 是单位矩阵（特征不相关且方差为 1），马哈拉诺比斯距离就会退化为欧几里得距离。

```
Example: height and weight are correlated.
Someone 6'2" and 180 lbs is not unusual.
Someone 5'0" and 180 lbs is unusual.

Euclidean distance might say they are equally far from the mean.
Mahalanobis distance correctly identifies the second as an outlier
because it accounts for the height-weight correlation.
```

马哈拉诺比斯距离的适用场景：
- 异常值检测（与均值的马哈拉诺比斯距离很大的点就是异常值）
- 特征尺度不同且彼此相关时的分类问题
- 当你有足够的数据来估计可靠的协方差矩阵时
- 制造业质量控制（多变量过程监控）

### 杰卡德相似度 (Jaccard similarity)（用于集合）

杰卡德相似度衡量两个集合的重叠程度。

```
J(A, B) = |A intersect B| / |A union B|
```

它的范围是 0（完全不重叠）到 1（完全相同的集合）。杰卡德距离 = 1 - 杰卡德相似度。

```
A = {cat, dog, fish}
B = {cat, bird, fish, snake}

Intersection = {cat, fish}         size = 2
Union = {cat, dog, fish, bird, snake}  size = 5

Jaccard similarity = 2/5 = 0.4
Jaccard distance = 0.6
```

杰卡德的适用场景：
- 比较标签、类别或特征集合
- 基于词是否出现（而不是出现频率）的文档相似度
- 近重复检测（杰卡德的 MinHash 近似）
- 比较二值特征向量（存在/不存在数据）
- 评估分割模型（Intersection over Union = Jaccard）

### 编辑距离 (Edit distance)（Levenshtein Distance）

编辑距离统计把一个字符串变成另一个字符串所需的最少单字符操作数。允许的操作包括：插入、删除或替换。

```
"kitten" -> "sitting"

kitten -> sitten  (substitute k -> s)
sitten -> sittin  (substitute e -> i)
sittin -> sitting (insert g)

Edit distance = 3
```

它通常通过动态规划 (dynamic programming) 计算。我们填充一个矩阵，其中位置 (i, j) 表示字符串 A 的前 i 个字符与字符串 B 的前 j 个字符之间的编辑距离。

```
        ""  s  i  t  t  i  n  g
    ""   0  1  2  3  4  5  6  7
    k    1  1  2  3  4  5  6  7
    i    2  2  1  2  3  4  5  6
    t    3  3  2  1  2  3  4  5
    t    4  4  3  2  1  2  3  4
    e    5  5  4  3  2  2  3  4
    n    6  6  5  4  3  3  2  3
```

编辑距离的适用场景：
- 拼写检查与纠错
- DNA 序列比对（使用带权操作）
- 模糊字符串匹配
- 脏文本数据去重

### KL 散度 (KL divergence)（不是距离，但经常被当作距离使用）

KL 散度衡量一个概率分布与另一个概率分布有多大差异。它在第 09 课中已经讲过，但它也属于这里的讨论范围，因为人们虽然常把它当“距离”来用，它其实并不是距离。

```
D_KL(P || Q) = sum(p(x) * log(p(x) / q(x)))
```

关键性质：KL 散度不是对称的。

```
D_KL(P || Q) != D_KL(Q || P)
```

这意味着它不满足距离度量最基本的要求。它也不满足三角不等式。它是散度，不是距离。

前向 KL（D_KL(P || Q)）是“均值寻优”的：Q 会尝试覆盖 P 的所有模态。
反向 KL（D_KL(Q || P)）是“模态寻优”的：Q 会聚焦于 P 的某一个模态。

你通常会在这些地方看到 KL 散度：
- VAE（ELBO 中的 KL 项会把潜变量分布推向先验分布）
- 知识蒸馏（学生模型尝试匹配教师模型的分布）
- RLHF（KL 惩罚让微调后的模型保持接近基础模型）
- 策略梯度方法（约束策略更新）

### Wasserstein 距离 (Wasserstein distance)（Earth Mover's Distance）

Wasserstein 距离衡量把一个概率分布变成另一个概率分布所需的最小“工作量”。你可以把它想成：如果一个分布是一堆土，另一个分布是一个坑，那么你需要把多少土搬多远？

```
W(P, Q) = inf over all transport plans gamma of E[d(x, y)]
```

对于一维分布，它可以简化为两个累积分布函数之差的绝对值积分：

```
W_1(P, Q) = integral |CDF_P(x) - CDF_Q(x)| dx
```

为什么 Wasserstein 很重要：
- 它是真正的度量（对称，并满足三角不等式）
- 即使两个分布没有重叠，它也能提供梯度（而 KL 散度会变成无穷大）
- 正是这个性质，让它成为 Wasserstein GAN（WGAN）的核心，并解决了原始 GAN 训练不稳定的问题

```
Distributions with no overlap:

P: [1, 0, 0, 0, 0]    Q: [0, 0, 0, 0, 1]

KL divergence: infinity (log of zero)
Wasserstein: 4 (move all mass 4 bins)

Wasserstein gives a meaningful gradient. KL does not.
```

Wasserstein 的适用场景：
- GAN 训练（WGAN、WGAN-GP）
- 比较可能没有重叠的分布
- 最优传输问题
- 图像检索（比较颜色直方图）

### 为什么不同任务需要不同距离

| 任务 | 最佳距离 | 原因 |
|------|--------------|-----|
| 文本相似度 | 余弦 | 大小是噪声，方向才是语义 |
| 图像像素比较 | L2 | 空间关系重要，而且特征尺度可比较 |
| 稀疏高维特征 | L1 | 更稳健，不会放大少数罕见的大差异 |
| 集合重叠（标签、类别） | 杰卡德 | 数据天然是集合值，而不是向量值 |
| 字符串匹配 | 编辑距离 | 操作定义符合人类编辑直觉 |
| 异常值检测 | 马哈拉诺比斯 | 考虑了特征相关性和尺度 |
| 比较分布 | KL 散度 | 衡量用 Q 代替 P 时丢失了多少信息 |
| GAN 训练 | Wasserstein | 即使分布不重叠也能提供梯度 |
| 嵌入（向量 DB） | 余弦或点积 | 嵌入被训练为把语义编码进方向 |
| 推荐 | 点积 | 大小可以编码流行度或置信度 |
| DNA 序列 | 带权编辑距离 | 不同核苷酸对的替换代价不同 |
| 制造质量控制 | L-infinity | 任意维度上的最坏偏差都很重要 |

### 与损失函数的联系

损失函数本质上就是作用在预测值和目标值之间的距离函数。

```
Loss function       Distance it uses       Behavior
MSE                 L2 squared             Penalizes large errors heavily
MAE                 L1                     Penalizes all errors equally
Huber loss          L1 for large errors,   Best of both: robust to outliers,
                    L2 for small errors    smooth gradient near zero
Cross-entropy       KL divergence          Measures distribution mismatch
Hinge loss          max(0, margin - d)     Only penalizes below margin
Triplet loss        L2 (typically)         Pulls positives close, pushes
                                           negatives away
Contrastive loss    L2                     Similar pairs close, dissimilar
                                           pairs beyond margin
```

### 与正则化 (regularization) 的联系

正则化会在损失函数上额外加入一个针对权重的范数惩罚项。

```
L1 regularization (Lasso):   loss + lambda * ||w||_1
  -> Sparse weights. Some weights become exactly zero.
  -> Automatic feature selection.
  -> Solution has corners (non-differentiable at zero).

L2 regularization (Ridge):   loss + lambda * ||w||_2^2
  -> Small weights. All weights shrink toward zero.
  -> No feature selection (nothing goes to exactly zero).
  -> Smooth solution everywhere.

Elastic Net:                  loss + lambda_1 * ||w||_1 + lambda_2 * ||w||_2^2
  -> Combines sparsity of L1 with stability of L2.
  -> Groups of correlated features are kept or dropped together.
```

为什么 L1 会产生稀疏性而 L2 不会：想象二维权重空间中的约束区域。L1 是菱形，L2 是圆形。损失函数的等高线（椭圆）最有可能在菱形的角上与其相切，而角点正对应某个权重为 0。相反，它们会在圆的光滑位置相切，此时两个权重通常都非零。

### 最近邻搜索 (Nearest Neighbor Search)

每一种距离函数都会对应一个最近邻搜索问题：给定一个查询点，在数据集中找出离它最近的点。

在包含 n 个点、每个点有 d 个维度的数据集中，精确最近邻搜索对每次查询的复杂度是 O(n * d)。对大规模数据集来说，这太慢了。

近似最近邻 (Approximate Nearest Neighbor, ANN) 算法用少量精度损失换来巨大的速度提升：

```
Algorithm         Approach                      Used by
KD-trees          Axis-aligned space partition   scikit-learn (low-dim)
Ball trees        Nested hyperspheres            scikit-learn (medium-dim)
LSH               Random hash projections        Near-duplicate detection
HNSW              Hierarchical navigable         FAISS, Qdrant, Weaviate
                  small-world graph
IVF               Inverted file index with       FAISS (billion-scale)
                  cluster-based search
Product quant.    Compress vectors, search       FAISS (memory-constrained)
                  in compressed space
```

HNSW（Hierarchical Navigable Small World）是现代向量数据库中的主流算法。它构建了一个多层图结构，其中每个节点都连接到它的近似最近邻。搜索从顶层开始（稀疏、长跳跃），再逐层下降到底层（稠密、短跳跃）。

## 动手构建

### 第 1 步：所有范数与距离函数

完整实现请看 `code/distances.py`。每个函数都只使用最基础的 Python 数学操作从零构建。

### 第 2 步：同样的数据，不同的距离，不同的邻居

`distances.py` 中的演示会创建一个数据集，选取一个查询点，并展示最近邻会如何随着距离度量不同而改变。在 L1 下“最近”的点，在 L2 或余弦下未必仍然最近。

### 第 3 步：嵌入相似度搜索

代码中还包含一个模拟的嵌入相似度搜索：它分别使用余弦相似度和 L2 距离来寻找与查询最相似的“文档”，并展示两种方法的排序可能不同。

## 使用方式

最常见的实际用途：在向量数据库中查找相似项。

```python
import numpy as np

def cosine_similarity_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_normalized = X / norms
    return X_normalized @ X_normalized.T

embeddings = np.random.randn(1000, 768)

sim_matrix = cosine_similarity_matrix(embeddings)

query_idx = 0
similarities = sim_matrix[query_idx]
top_k = np.argsort(similarities)[::-1][1:6]
print(f"Top 5 most similar to item 0: {top_k}")
print(f"Similarities: {similarities[top_k]}")
```

当你调用 `model.encode(text)` 然后再去搜索向量数据库时，底层发生的就是这件事。嵌入模型把文本映射为向量，向量数据库再计算查询向量与每个已存储向量之间的余弦相似度（或点积），并使用 ANN 算法来避免逐个比对全部向量。

## 练习

1. 计算 (1, 2, 3) 与 (4, 0, 6) 之间的 L1、L2 和 L-infinity 距离。验证对于任意一对点，都始终有 L-inf &lt;= L2 &lt;= L1。证明为什么这种顺序一定成立。

2. 构造两个向量，使它们的余弦相似度很高 (> 0.9)，但 L2 距离很大 (> 10)。从几何角度解释发生了什么。然后再构造两个向量，使它们的余弦相似度很低 (&lt; 0.3)，但 L2 距离很小 (&lt; 0.5)。

3. 实现一个函数，输入一个数据集和一个查询点，返回该查询点在 L1、L2、余弦和马哈拉诺比斯距离下的最近邻。找一个数据集，使这四种度量对“最近的是哪个点”给出完全不同的答案。

4. 用 CDF 方法手算 [0.5, 0.5, 0, 0] 与 [0, 0, 0.5, 0.5] 之间的 Wasserstein 距离。然后再计算 [0.25, 0.25, 0.25, 0.25] 与 [0, 0, 0.5, 0.5] 之间的距离。哪一个更大？为什么？

5. 实现用于近似杰卡德相似度的 MinHash。生成 100 个随机集合，计算所有成对组合的精确杰卡德值，再分别使用 50、100 和 200 个哈希函数的 MinHash 近似进行比较。画出近似误差图。

## 关键术语

| 术语 | 人们常说的说法 | 它真正的含义 |
|------|----------------|----------------------|
| 范数 (Norm) | “向量的大小” | 一个把向量映射到非负标量的函数，并满足三角不等式、绝对齐次性，且只有零向量的值为 0 |
| L1 范数 | “曼哈顿距离” | 各分量绝对值之和。在优化中会产生稀疏性，并且对异常值更稳健 |
| L2 范数 | “欧几里得距离” | 各分量平方和的平方根。即欧几里得空间中的直线距离 |
| Lp 范数 | “广义范数” | 各分量绝对值 p 次方之和再开 p 次根。L1 和 L2 都是它的特例 |
| L-infinity 范数 | “最大范数”或“切比雪夫距离” | 分量绝对值中的最大值。是 p 趋向无穷大时 Lp 的极限 |
| 余弦相似度 | “向量之间的夹角” | 将点积除以两个向量大小后的结果。范围从 -1 到 +1，忽略向量长度 |
| 余弦距离 | “1 减去余弦相似度” | 把余弦相似度转换成距离。范围从 0 到 2 |
| 点积 | “未归一化的余弦” | 各分量对应乘积之和。等于余弦相似度再乘上两个向量的大小 |
| 马哈拉诺比斯距离 | “考虑相关性的距离” | 在使用数据协方差矩阵对白化（去相关并归一化）后的空间中计算的 L2 距离 |
| 杰卡德相似度 | “集合重叠度” | 交集大小除以并集大小。适用于集合，而不是向量 |
| 编辑距离 | “Levenshtein 距离” | 把一个字符串变成另一个字符串所需的最少插入、删除和替换次数 |
| KL 散度 | “分布之间的距离” | 它不是真正的距离（不对称）。它衡量的是用 Q 编码 P 时额外需要的比特数 |
| Wasserstein 距离 | “Earth mover's distance” | 把质量从一个分布运输到另一个分布所需的最小工作量。它是真正的度量 |
| 近似最近邻 | “ANN 搜索” | 比精确搜索快得多、用于找到近似最近点的算法（HNSW、LSH、IVF） |
| HNSW | “向量 DB 算法” | Hierarchical Navigable Small World 图。用于快速近似最近邻搜索的多层图结构 |
| L1 正则化 | “Lasso” | 在损失中加入权重的 L1 范数。会把权重压到 0（稀疏） |
| L2 正则化 | “Ridge”或“weight decay” | 在损失中加入权重的 L2 范数平方。会把权重向 0 收缩，但不会产生稀疏性 |
| Elastic Net | “L1 + L2” | 结合 L1 与 L2 正则化。比单独使用任一种更擅长处理相关特征组 |

## 延伸阅读

- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss) - Meta 用于十亿级 ANN 搜索的库
- [Wasserstein GAN (Arjovsky et al., 2017)](https://arxiv.org/abs/1701.07875) - 将 Earth Mover's distance 引入 GAN 的论文
- [Locality-Sensitive Hashing (Indyk & Motwani, 1998)](https://dl.acm.org/doi/10.1145/276698.276876) - 奠基性的 ANN 算法
- [Efficient Estimation of Word Representations (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781) - Word2Vec 论文，余弦相似度也是从这里开始成为嵌入默认选择
- [sklearn.neighbors documentation](https://scikit-learn.org/stable/modules/neighbors.html) - scikit-learn 中距离度量与邻居算法的实用指南
