# 向量、矩阵与运算

> 每个神经网络，不过是带了几步额外操作的矩阵乘法。

**类型（Type）：** 构建
**语言（Languages）：** Python、Julia
**前置知识（Prerequisites）：** 第 1 阶段第 01 课（线性代数直觉）
**预计用时（Time）：** 约 60 分钟

## 学习目标

- 构建一个 Matrix 类，支持逐元素运算、矩阵乘法、转置、行列式和逆矩阵
- 区分逐元素乘法与矩阵乘法，并解释各自的适用场景
- 仅使用自制 Matrix 类实现单个全连接神经网络层（`relu(W @ x + b)`）
- 解释广播规则及偏置加法在神经网络框架中的工作原理

## 问题背景

你想构建一个神经网络，阅读代码时看到：

```
output = activation(weights @ input + bias)
```

其中 `@` 是矩阵乘法，`weights` 是矩阵，`input` 是向量。如果你不理解这些操作，这行代码就是魔法；如果你理解，它就是一个神经网络层完整前向传播的三步操作。

你的模型处理的每张图片都是像素值矩阵，每个词嵌入都是向量，每个神经网络层都是一次矩阵变换。无法熟练运用矩阵操作，就像不理解变量却想写代码——根本无从构建 AI 系统。

本节课将从零开始培养这种熟练度。

## 核心概念

### 向量：有序数字列表

向量（vector）是一列有方向和大小的数字。在 AI 中，向量代表数据点、特征或参数。

```
v = [3, 4]        -- 一个二维向量
w = [1, 0, -2]    -- 一个三维向量
```

二维向量 `[3, 4]` 指向平面上坐标 (3, 4) 的位置，其长度（模）为 5（即 3-4-5 直角三角形）。

### 矩阵：数字网格

矩阵（matrix）是一个二维网格，由行和列组成。m × n 矩阵有 m 行 n 列。

```
A = | 1  2  3 |     -- 2x3 矩阵（2 行 3 列）
    | 4  5  6 |
```

在神经网络中，权重矩阵将输入向量变换为输出向量。具有 784 个输入和 128 个输出的层，使用一个 128×784 的权重矩阵。

### 形状为何至关重要

矩阵乘法有严格规则：`(m × n) @ (n × p) = (m × p)`，内维必须匹配。

```
(128 x 784) @ (784 x 1) = (128 x 1)
   权重          输入        输出

内维：784 = 784  -- 合法
```

若在 PyTorch 中遇到形状不匹配错误，原因就在于此。

### 运算一览

| 运算 | 功能 | 在神经网络中的用途 |
|-----------|-------------|-------------------|
| 加法 | 逐元素相加 | 在输出上加偏置 |
| 标量乘法 | 每个元素乘以同一标量 | 学习率 × 梯度 |
| 矩阵乘法 | 变换向量 | 层的前向传播 |
| 转置 | 交换行与列 | 反向传播 |
| 行列式 | 单一数值汇总 | 检验可逆性 |
| 逆矩阵 | 撤销变换 | 求解线性方程组 |
| 单位矩阵 | 什么都不改变的矩阵 | 初始化，残差连接 |

### 逐元素乘法 vs 矩阵乘法

这个区别常让初学者困惑。

逐元素乘法（element-wise multiplication）：对应位置相乘，两个矩阵形状必须相同。

```
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

矩阵乘法（matrix multiplication）：行与列的点积，内维必须匹配。

```
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

不同的运算，不同的结果，不同的规则。

### 广播

将偏置向量加到输出矩阵时，形状并不匹配。广播（broadcasting）会将较小的数组"拉伸"以适配较大的数组。

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

广播将向量沿行方向展开：

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

所有现代框架都会自动处理广播。理解它能防止因形状看似异常但代码仍正常运行而产生困惑。

## 动手实现

### 第 1 步：向量类

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### 第 2 步：Matrix 类及核心运算

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### 第 3 步：验证效果

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### 第 4 步：与神经网络相连接

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")
```

这就是单个全连接层：`output = relu(W @ x + b)`。每个神经网络中的每个全连接层都精确地执行这一操作。

## 实际应用

NumPy 用更少的代码完成上述所有操作，且速度快几个数量级。

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (element-wise) =\n", A * B)
print("A @ B (matrix multiply) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\nNeural network layer: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"Output:\n{output}")
```

Python 中的 `@` 运算符调用 `__matmul__`。NumPy 通过用 C 和 Fortran 编写的优化 BLAS 例程实现该运算符。数学相同，速度快 100 倍。

NumPy 中的广播：

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy 会自动将一维偏置广播到两行。这就是每个神经网络框架中偏置加法的工作原理。

## 交付物

本节课产出一个用于通过几何直觉讲授矩阵运算的提示词，详见 `outputs/prompt-matrix-operations.md`。

本节课构建的 Matrix 类是第 3 阶段第 10 课迷你神经网络框架的基础。

## 练习

1. **验证逆矩阵。** 计算 `A @ A.inverse_2x2()` 并确认结果为单位矩阵。用三个不同的 2×2 矩阵进行测试。行列式为零时会发生什么？

2. **实现 3×3 逆矩阵。** 扩展 Matrix 类，使用伴随矩阵法计算 3×3 矩阵的逆，并用 NumPy 的 `np.linalg.inv` 验证结果。

3. **构建两层网络。** 仅使用你的 Matrix 类（不使用 NumPy），创建一个两层神经网络：输入层（3）→ 隐藏层（4）→ 输出层（2）。初始化随机权重，运行一次前向传播，并验证所有形状正确。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| 向量（Vector） | "一个箭头" | 有序的数字列表。在 AI 中：高维空间中的一个点。 |
| 矩阵（Matrix） | "一张数字表格" | 线性变换，将向量从一个空间映射到另一个空间。 |
| 矩阵乘法（Matrix multiply） | "就是数字相乘" | 第一个矩阵的每一行与第二个矩阵的每一列做点积。顺序至关重要。 |
| 转置（Transpose） | "把它翻转" | 交换行和列，将 m × n 矩阵变为 n × m。反向传播中不可或缺。 |
| 行列式（Determinant） | "矩阵里出来的某个数" | 衡量矩阵对面积（二维）或体积（三维）的缩放比例。为零意味着变换压缩了某个维度。 |
| 逆矩阵（Inverse） | "撤销矩阵" | 能逆转变换的矩阵，仅当行列式不为零时存在。 |
| 单位矩阵（Identity matrix） | "没意思的矩阵" | 等同于乘以 1 的矩阵，用于残差连接（ResNets）。 |
| 广播（Broadcasting） | "神奇的形状修复" | 通过沿缺失维度重复，将较小数组拉伸以匹配较大数组。 |
| 逐元素（Element-wise） | "普通乘法" | 对应位置相乘，两个数组形状必须相同（或可广播）。 |

## 延伸阅读

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) - 本节课涉及每个运算的视觉直觉
- [NumPy documentation on broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) - NumPy 遵循的精确广播规则
- [Stanford CS229 Linear Algebra Review](http://cs229.stanford.edu/section/cs229-linalg.pdf) - 面向机器学习的线性代数简明参考
