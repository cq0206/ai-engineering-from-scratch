# 线性代数直觉

> 每一个 AI 模型，不过是披着华丽外衣的矩阵运算。

**类型（Type）：** 学习
**语言（Languages）：** Python、Julia
**前置知识（Prerequisites）：** 第 0 阶段
**预计用时（Time）：** 约 60 分钟

## 学习目标

- 用 Python 从零实现向量和矩阵运算（加法、点积、矩阵乘法）
- 用几何直觉解释点积、投影和 Gram-Schmidt 正交化的含义
- 利用行变换判断向量组的线性无关性（linear independence）、秩（rank）和基（basis）
- 将线性代数概念与 AI 应用相连接：词嵌入（embeddings）、注意力分数（attention scores）和 LoRA

## 问题背景

打开任意一篇机器学习论文，第一页就会出现向量、矩阵、点积和变换。如果缺乏线性代数直觉，这些不过是符号；一旦理解了它，你就能看清神经网络在做什么——在空间中移动点。

你不需要成为数学家。你需要从几何角度理解这些运算的含义，然后亲手编写代码。

## 核心概念

### 向量是点（也是方向）

向量不过是一列数字。但这些数字有意义——它们是空间中的坐标。

**二维向量 [3, 2]：**

| x | y | 含义 |
|---|---|-------|
| 3 | 2 | 该向量从原点 (0,0) 指向平面上的点 (3, 2) |

该向量的模为 sqrt(3^2 + 2^2) = sqrt(13)，方向朝右上方。

在 AI 中，向量可以表示一切：
- 一个词 → 一个 768 维的向量（它在嵌入空间中的"语义"）
- 一张图片 → 一个包含数百万像素值的向量
- 一位用户 → 一个表示偏好的向量

### 矩阵是变换

矩阵将一个向量变换为另一个向量，可以执行旋转、缩放、拉伸或投影。

```mermaid
graph LR
    subgraph Before["变换前"]
        A["点 A"]
        B["点 B"]
    end
    subgraph Matrix["矩阵乘法"]
        M["M（变换）"]
    end
    subgraph After["变换后"]
        A2["点 A'"]
        B2["点 B'"]
    end
    A --> M
    B --> M
    M --> A2
    M --> B2
```

在 AI 中，矩阵就是模型本身：
- 神经网络权重 → 将输入变换为输出的矩阵
- 注意力分数 → 决定关注什么的矩阵
- 词嵌入 → 将词映射为向量的矩阵

### 点积度量相似度

两个向量的点积（dot product）表示它们的相似程度。

```
a · b = a₁×b₁ + a₂×b₂ + ... + aₙ×bₙ

方向相同：      a · b > 0  （相似）
垂直：          a · b = 0  （无关）
方向相反：      a · b < 0  （相异）
```

这正是搜索引擎、推荐系统和 RAG 的工作原理——找到点积较大的向量。

### 线性无关

如果一组向量中，没有任何向量可以用其余向量的线性组合表示，则称这些向量线性无关（linearly independent）。若 v1、v2、v3 线性无关，它们张成一个三维空间；若其中一个是其余向量的线性组合，它们只能张成一个平面。

为何对 AI 重要：特征矩阵的列向量应当线性无关。若两个特征完全相关（线性相关），模型无法区分它们的影响。这会造成回归中的多重共线性（multicollinearity）——权重矩阵变得不稳定，微小的输入变化会引起输出剧烈波动。

**具体示例：**

```
v1 = [1, 0, 0]
v2 = [0, 1, 0]
v3 = [2, 1, 0]   # v3 = 2*v1 + v2
```

v1 和 v2 线性无关——两者都不是对方的标量倍或线性组合。但 v3 = 2*v1 + v2，因此 {v1, v2, v3} 是一个线性相关的向量组。这三个向量都位于 xy 平面内。无论如何组合，都无法到达 [0, 0, 1]。你有三个向量，却只有两个自由度。

在数据集中：若 feature_3 = 2*feature_1 + feature_2，添加 feature_3 不会给模型带来任何新信息。更糟的是，法方程（normal equations）会变成奇异矩阵——权重没有唯一解。

### 基与秩

基（basis）是能张成整个空间的最小线性无关向量组，基向量的个数即为空间的维数。

三维空间的标准基是 {[1,0,0], [0,1,0], [0,0,1]}，但任意三个线性无关的三维向量都构成合法的基。选择基就是选择坐标系。

矩阵的秩（rank）= 线性无关列向量的数量 = 线性无关行向量的数量。若秩 &lt; min(行数, 列数)，则矩阵是秩亏（rank-deficient）的，这意味着：
- 方程组有无穷多解（或无解）
- 变换过程中信息丢失
- 矩阵不可逆

| 情形 | 秩 | 对机器学习的含义 |
|-----------|------|---------------------|
| 满秩（rank = min(m, n)） | 最大可能值 | 最小二乘解唯一，模型条件良好。 |
| 秩亏（rank &lt; min(m, n)） | 低于最大值 | 特征冗余，权重有无穷多解，需要正则化。 |
| 秩为 1 | 1 | 每列都是同一向量的缩放副本，所有数据分布在一条直线上。 |
| 接近秩亏（小奇异值） | 数值意义上较低 | 矩阵病态，微小输入噪声导致输出剧变。建议使用 SVD 截断或岭回归。 |

### 投影

将向量 **a** 投影（projection）到向量 **b** 上，得到 **a** 在 **b** 方向上的分量：

```
proj_b(a) = (a dot b / b dot b) * b
```

残差（a - proj_b(a)）与 b 垂直。这种正交分解是最小二乘拟合的基础。

投影在机器学习中无处不在：
- 线性回归最小化观测值到列空间的距离——其解本质上就是投影
- 主成分分析（PCA）将数据投影到方差最大的方向
- Transformer 中的注意力机制计算查询向量（queries）在键向量（keys）上的投影

```mermaid
graph LR
    subgraph Projection["a 在 b 上的投影"]
        direction TB
        O["原点"] --> |"b（方向）"| B["b"]
        O --> |"a（原始向量）"| A["a"]
        O --> |"proj_b(a)"| P["投影"]
        A -.-> |"残差（垂直）"| P
    end
```

**示例：** a = [3, 4]，b = [1, 0]

proj_b(a) = (3*1 + 4*0) / (1*1 + 0*0) * [1, 0] = 3 * [1, 0] = [3, 0]

投影去掉了 y 分量。这是最简单形式的降维——丢弃你不关心的方向。

### Gram-Schmidt 正交化

将任意一组线性无关向量转换为正交归一基（orthonormal basis）。正交归一意味着每个向量的模为 1，且每对向量相互垂直。

算法步骤：
1. 取第一个向量并归一化
2. 取第二个向量，减去其在第一个向量上的投影，然后归一化
3. 取第三个向量，减去其在所有已处理向量上的投影，然后归一化
4. 对剩余向量重复以上步骤

```
Input:  v1, v2, v3, ... (linearly independent)

u1 = v1 / |v1|

w2 = v2 - (v2 dot u1) * u1
u2 = w2 / |w2|

w3 = v3 - (v3 dot u1) * u1 - (v3 dot u2) * u2
u3 = w3 / |w3|

Output: u1, u2, u3, ... (orthonormal basis)
```

这就是 QR 分解的内部实现：Q 是正交归一基，R 存储投影系数。QR 分解用于：
- 求解线性方程组（比高斯消元法更稳定）
- 计算特征值（QR 算法）
- 最小二乘回归（标准数值方法）

## 动手实现

### 第 1 步：用 Python 从零实现向量

```python
class Vector:
    def __init__(self, components):
        self.components = list(components)
        self.dim = len(self.components)

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.components, other.components)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.components, other.components)])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.components, other.components))

    def magnitude(self):
        return sum(x**2 for x in self.components) ** 0.5

    def normalize(self):
        mag = self.magnitude()
        return Vector([x / mag for x in self.components])

    def cosine_similarity(self, other):
        return self.dot(other) / (self.magnitude() * other.magnitude())

    def __repr__(self):
        return f"Vector({self.components})"


a = Vector([1, 2, 3])
b = Vector([4, 5, 6])

print(f"a + b = {a + b}")
print(f"a · b = {a.dot(b)}")
print(f"|a| = {a.magnitude():.4f}")
print(f"cosine similarity = {a.cosine_similarity(b):.4f}")
```

### 第 2 步：用 Python 从零实现矩阵

```python
class Matrix:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]
        self.shape = (len(self.rows), len(self.rows[0]))

    def __matmul__(self, other):
        if isinstance(other, Vector):
            return Vector([
                sum(self.rows[i][j] * other.components[j] for j in range(self.shape[1]))
                for i in range(self.shape[0])
            ])
        rows = []
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
                row.append(sum(
                    self.rows[i][k] * other.rows[k][j]
                    for k in range(self.shape[1])
                ))
            rows.append(row)
        return Matrix(rows)

    def transpose(self):
        return Matrix([
            [self.rows[j][i] for j in range(self.shape[0])]
            for i in range(self.shape[1])
        ])

    def __repr__(self):
        return f"Matrix({self.rows})"


rotation_90 = Matrix([[0, -1], [1, 0]])
point = Vector([3, 1])

rotated = rotation_90 @ point
print(f"Original: {point}")
print(f"Rotated 90°: {rotated}")
```

### 第 3 步：为何这对 AI 重要

```python
import random

random.seed(42)
weights = Matrix([[random.gauss(0, 0.1) for _ in range(3)] for _ in range(2)])
input_vector = Vector([1.0, 0.5, -0.3])

output = weights @ input_vector
print(f"Input (3D): {input_vector}")
print(f"Output (2D): {output}")
print("This is what a neural network layer does -- matrix multiplication.")
```

### 第 4 步：Julia 版本

```julia
a = [1.0, 2.0, 3.0]
b = [4.0, 5.0, 6.0]

println("a + b = ", a + b)
println("a · b = ", a ⋅ b)       # Julia supports unicode operators
println("|a| = ", √(a ⋅ a))
println("cosine = ", (a ⋅ b) / (√(a ⋅ a) * √(b ⋅ b)))

# Matrix-vector multiplication
W = [0.1 -0.2 0.3; 0.4 0.5 -0.1]
x = [1.0, 0.5, -0.3]
println("Wx = ", W * x)
println("This is a neural network layer.")
```

### 第 5 步：用 Python 从零实现线性无关性检验与投影

```python
def is_linearly_independent(vectors):
    n = len(vectors)
    dim = len(vectors[0].components)
    mat = Matrix([v.components[:] for v in vectors])
    rows = [row[:] for row in mat.rows]
    rank = 0
    for col in range(dim):
        pivot = None
        for row in range(rank, len(rows)):
            if abs(rows[row][col]) > 1e-10:
                pivot = row
                break
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        scale = rows[rank][col]
        rows[rank] = [x / scale for x in rows[rank]]
        for row in range(len(rows)):
            if row != rank and abs(rows[row][col]) > 1e-10:
                factor = rows[row][col]
                rows[row] = [rows[row][j] - factor * rows[rank][j] for j in range(dim)]
        rank += 1
    return rank == n


def project(a, b):
    scalar = a.dot(b) / b.dot(b)
    return Vector([scalar * x for x in b.components])


def gram_schmidt(vectors):
    orthonormal = []
    for v in vectors:
        w = v
        for u in orthonormal:
            proj = project(w, u)
            w = w - proj
        if w.magnitude() < 1e-10:
            continue
        orthonormal.append(w.normalize())
    return orthonormal


v1 = Vector([1, 0, 0])
v2 = Vector([1, 1, 0])
v3 = Vector([1, 1, 1])
basis = gram_schmidt([v1, v2, v3])
for i, u in enumerate(basis):
    print(f"u{i+1} = {u}")
    print(f"  |u{i+1}| = {u.magnitude():.6f}")

print(f"u1 · u2 = {basis[0].dot(basis[1]):.6f}")
print(f"u1 · u3 = {basis[0].dot(basis[2]):.6f}")
print(f"u2 · u3 = {basis[1].dot(basis[2]):.6f}")
```

## 实际应用

用 NumPy 完成同样的工作——这才是实践中的常规做法：

```python
import numpy as np

a = np.array([1, 2, 3], dtype=float)
b = np.array([4, 5, 6], dtype=float)

print(f"a + b = {a + b}")
print(f"a · b = {np.dot(a, b)}")
print(f"|a| = {np.linalg.norm(a):.4f}")
print(f"cosine = {np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)):.4f}")

W = np.random.randn(2, 3) * 0.1
x = np.array([1.0, 0.5, -0.3])
print(f"Wx = {W @ x}")
```

### 用 NumPy 计算秩、投影与 QR 分解

```python
import numpy as np

A = np.array([[1, 2], [2, 4]])
print(f"Rank: {np.linalg.matrix_rank(A)}")

a = np.array([3, 4])
b = np.array([1, 0])
proj = (np.dot(a, b) / np.dot(b, b)) * b
print(f"Projection of {a} onto {b}: {proj}")

Q, R = np.linalg.qr(np.random.randn(3, 3))
print(f"Q is orthogonal: {np.allclose(Q @ Q.T, np.eye(3))}")
print(f"R is upper triangular: {np.allclose(R, np.triu(R))}")
```

### PyTorch——带自动微分的张量（Tensors）

```python
import torch

x = torch.randn(3, requires_grad=True)
y = torch.tensor([1.0, 0.0, 0.0])

similarity = torch.dot(x, y)
similarity.backward()

print(f"x = {x.data}")
print(f"y = {y.data}")
print(f"dot product = {similarity.item():.4f}")
print(f"d(dot)/dx = {x.grad}")
```

点积对 x 的梯度正好是 y。PyTorch 自动完成了这一计算。神经网络中的每个运算都是由类似的操作构建而成——矩阵乘法、点积、投影——而自动微分会追踪所有操作的梯度。

你刚刚从零实现了 NumPy 一行就能完成的功能，现在你知道底层发生了什么。

## 交付物

本节课产出：
- `outputs/prompt-linear-algebra-tutor.md` —— 一个用于 AI 助手通过几何直觉讲授线性代数的提示词

## 知识连接

本节课的每个概念都与现代 AI 的具体部分相关联：

| 概念 | 在 AI 中的应用 |
|---------|------------------|
| 点积 | Transformer 中的注意力分数，RAG 中的余弦相似度 |
| 矩阵乘法 | 每个神经网络层，每个线性变换 |
| 线性无关性 | 特征选择，避免多重共线性 |
| 秩 | 判断方程组是否有解，LoRA（低秩自适应） |
| 投影 | 线性回归（投影到列空间），PCA |
| Gram-Schmidt / QR | 数值求解器，特征值计算 |
| 正交归一基 | 稳定数值计算，白化变换 |

LoRA 值得特别说明。它通过将权重更新分解为低秩矩阵来微调大型语言模型。LoRA 不直接更新 4096×4096 的权重矩阵（1600 万个参数），而是更新两个尺寸为 4096×16 和 16×4096 的矩阵（13.1 万个参数）。秩为 16 的约束意味着 LoRA 假设权重更新存在于完整 4096 维空间的 16 维子空间中。这就是线性代数在实际工作中发挥作用的体现。

## 练习

1. 实现 `Vector.angle_between(other)`，返回两个向量之间的夹角（以度为单位）
2. 创建一个 2D 缩放矩阵，使 x 坐标翻倍、y 坐标变为三倍，然后对向量 [1, 1] 应用该矩阵
3. 给定 5 个随机的类词向量（维度为 50），用余弦相似度找出最相似的两个
4. 验证 Gram-Schmidt 输出确实是正交归一的：检验每对向量的点积为 0，且每个向量的模为 1
5. 创建一个秩为 2 的 3×3 矩阵，用 `rank()` 方法验证，并解释其列向量张成的几何对象
6. 将向量 [1, 2, 3] 投影到 [1, 1, 1] 上。结果在几何上代表什么？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| 向量（Vector） | "一个箭头" | 表示 n 维空间中某一点或方向的数字列表 |
| 矩阵（Matrix） | "一张数字表格" | 将向量从一个空间映射到另一个空间的变换 |
| 点积（Dot product） | "相乘后求和" | 衡量两个向量对齐程度的度量——相似度搜索的核心 |
| 嵌入（Embedding） | "某种 AI 魔法" | 表示某事物（词、图像、用户）语义的向量 |
| 线性无关（Linear independence） | "它们不重叠" | 向量组中没有任何向量可以写成其余向量的线性组合 |
| 秩（Rank） | "有几个维度" | 矩阵中线性无关的列（或行）的数量 |
| 投影（Projection） | "影子" | 一个向量在另一个向量方向上的分量 |
| 基（Basis） | "坐标轴" | 能张成该空间的最小线性无关向量组 |
| 正交归一（Orthonormal） | "相互垂直的单位向量" | 向量两两垂直且每个向量的模均为 1 |
