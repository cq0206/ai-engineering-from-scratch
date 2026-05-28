# 线性方程组 (linear systems)

> 求解 Ax = b 是数学中最古老、至今仍在驱动你的神经网络的问题。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 1，第 01 课（线性代数直觉），第 02 课（向量与矩阵），第 03 课（矩阵变换）
**时间：** ~120 分钟

## 学习目标

- 使用带部分主元选取 (partial pivoting) 和回代 (back substitution) 的高斯消元 (Gaussian elimination) 求解 Ax = b
- 用 LU 分解 (LU decomposition)、QR 分解 (QR decomposition) 和 Cholesky 分解 (Cholesky decomposition) 对矩阵进行分解，并解释各自适用的场景
- 推导最小二乘 (least squares) 的正规方程 (normal equations)，并将其与线性回归 (linear regression) 和岭回归 (ridge regression) 联系起来
- 用条件数 (condition number) 诊断病态系统 (ill-conditioned systems)，并应用正则化 (regularization) 来稳定它们

## 问题

每当你训练一个线性回归模型时，你就在求解一个线性方程组。每当你计算一次最小二乘拟合时，你也在求解一个线性方程组。每当神经网络的一层计算 `y = Wx + b` 时，它其实是在计算一个线性方程组的一侧。加入正则化时，你是在修改这个系统。使用高斯过程 (Gaussian processes) 时，你是在分解一个矩阵。为了计算马氏距离 (Mahalanobis distance) 而对协方差矩阵 (covariance matrix) 求逆时，你同样是在求解一个线性方程组。

方程 Ax = b 无处不在。A 是由已知系数组成的矩阵，b 是由已知输出组成的向量，x 是你想要求出的未知量向量。在线性回归中，A 是你的数据矩阵，b 是目标向量，x 是权重向量。整个模型都可以归结为：找到 x，使得 Ax 尽可能接近 b。

本课将从零开始构建求解这个方程的每一种主要方法。你会理解为什么有些方法速度快，而另一些方法更稳定；为什么有些方法只适用于方阵系统，而另一些可以处理超定系统；以及为什么矩阵的条件数会决定你的答案是否真的有意义。

## 概念

### 从几何角度理解 Ax = b

线性方程组有一个几何解释。每个方程都定义一个超平面 (hyperplane)。解就是所有超平面相交的那个点（或那组点）。

```
2x + y = 5          Two lines in 2D.
x - y  = 1          They intersect at x=2, y=1.
```

```mermaid
graph LR
    A["2x + y = 5"] --- S["解：(2, 1)"]
    B["x - y = 1"] --- S
```

可能出现三种情况：

```mermaid
graph TD
    subgraph "唯一解"
        A1["直线在单一点相交"]
    end
    subgraph "无解"
        A2["直线平行——没有交点"]
    end
    subgraph "无穷多解"
        A3["直线重合——每个点都是解"]
    end
```

在线性代数的矩阵形式里，“唯一解”表示 A 可逆；“无解”表示系统不相容；“无穷多解”表示 A 有零空间 (null space)。大多数机器学习 (ML) 问题都属于“没有精确解”这一类，因为你的方程个数（数据点）多于未知数个数（参数）。这正是最小二乘发挥作用的地方。

### 列视角与行视角

理解 Ax = b 有两种方式。

**行视角 (row picture)。** A 的每一行定义一个方程。每个方程都是一个超平面。解就是它们全部相交的位置。

**列视角 (column picture)。** A 的每一列都是一个向量。问题会变成：A 的各列向量做怎样的线性组合，才能得到 b？

```
A = | 2  1 |    b = | 5 |
    | 1 -1 |        | 1 |

Row picture: solve 2x + y = 5 and x - y = 1 simultaneously.

Column picture: find x1, x2 such that:
  x1 * [2, 1] + x2 * [1, -1] = [5, 1]
  2 * [2, 1] + 1 * [1, -1] = [4+1, 2-1] = [5, 1]   check.
```

列视角更为根本。如果 b 位于 A 的列空间 (column space) 中，系统就有解；如果 b 不在其中，你就要在列空间中找到离它最近的点。那个最近点对应的就是最小二乘解。

### 高斯消元 (Gaussian elimination)

高斯消元把 Ax = b 转换成一个上三角系统 Ux = c，然后你再通过回代求解它。这是最直接的方法。

算法如下：

```
1. For each column k (the pivot column):
   a. Find the largest entry in column k at or below row k (partial pivoting).
   b. Swap that row with row k.
   c. For each row i below k:
      - Compute multiplier m = A[i][k] / A[k][k]
      - Subtract m times row k from row i.
2. Back substitute: solve from the last equation upward.
```

示例：

```
Original:
| 2  1  1 | 8 |       R2 = R2 - (2)R1     | 2  1   1 |  8 |
| 4  3  3 |20 |  -->  R3 = R3 - (1)R1 --> | 0  1   1 |  4 |
| 2  3  1 |12 |                            | 0  2   0 |  4 |

                       R3 = R3 - (2)R2     | 2  1   1 |  8 |
                                       --> | 0  1   1 |  4 |
                                           | 0  0  -2 | -4 |

Back substitute:
  -2 * x3 = -4    -->  x3 = 2
  x2 + 2  = 4     -->  x2 = 2
  2*x1 + 2 + 2 = 8 --> x1 = 2
```

高斯消元的计算成本是 O(n^3)。对于一个 1000x1000 的系统，这大约是一十亿次浮点运算。它很快，但如果你需要用同一个 A 求解多个系统，还有更好的办法。

### 部分主元选取：为什么它很重要

如果不做主元选取，高斯消元可能会失败，或者产生垃圾结果。若主元为零，你会除以零；若主元很小，你会放大舍入误差。

```
Bad pivot:                       With partial pivoting:
| 0.001  1 | 1.001 |            Swap rows first:
| 1      1 | 2     |            | 1      1 | 2     |
                                 | 0.001  1 | 1.001 |
m = 1/0.001 = 1000              m = 0.001/1 = 0.001
R2 = R2 - 1000*R1               R2 = R2 - 0.001*R1
| 0.001  1     | 1.001   |      | 1      1     | 2     |
| 0     -999   | -999.0  |      | 0      0.999 | 0.999 |

x2 = 1.000 (correct)            x2 = 1.000 (correct)
x1 = (1.001 - 1)/0.001          x1 = (2 - 1)/1 = 1.000 (correct)
   = 0.001/0.001 = 1.000        Stable because the multiplier is small.
```

在精度有限的浮点运算中，不做主元选取的版本可能会丢失大量有效数字。部分主元选取总是选择当前可用的最大主元，以尽量减少误差放大。

### LU 分解 (LU decomposition)

LU 分解把 A 分解成一个下三角矩阵 L 和一个上三角矩阵 U：A = LU。L 矩阵保存了高斯消元中的乘子，U 矩阵则是消元后的结果。

```
A = L @ U

| 2  1  1 |   | 1  0  0 |   | 2  1   1 |
| 4  3  3 | = | 2  1  0 | @ | 0  1   1 |
| 2  3  1 |   | 1  2  1 |   | 0  0  -2 |
```

为什么要分解，而不是直接消元？因为一旦你有了 L 和 U，针对任意新的 b，求解 Ax = b 只需要 O(n^2)：

```
Ax = b
LUx = b
Let y = Ux:
  Ly = b    (forward substitution, O(n^2))
  Ux = y    (back substitution, O(n^2))
```

O(n^3) 的成本只在分解时支付一次。之后每次求解都只需 O(n^2)。如果你需要对同一个 A、不同的 b 向量求解 1000 次，LU 会让总工作量减少大约 1000/3 倍。

配合部分主元选取后，你得到的是 PA = LU，其中 P 是记录行交换的置换矩阵。

### QR 分解 (QR decomposition)

QR 分解把 A 分解成一个正交矩阵 Q 和一个上三角矩阵 R：A = QR。

正交矩阵满足 Q^T Q = I。它的列是标准正交向量。用 Q 相乘会保持长度和角度不变。

```
A = Q @ R

Q has orthonormal columns: Q^T Q = I
R is upper triangular

To solve Ax = b:
  QRx = b
  Rx = Q^T b    (just multiply by Q^T, no inversion needed)
  Back substitute to get x.
```

在求解最小二乘问题时，QR 在数值上比 LU 更稳定。Gram-Schmidt 过程会按列构造 Q：

```
Given columns a1, a2, ... of A:

q1 = a1 / ||a1||

q2 = a2 - (a2 . q1) * q1        (subtract projection onto q1)
q2 = q2 / ||q2||                (normalize)

q3 = a3 - (a3 . q1) * q1 - (a3 . q2) * q2
q3 = q3 / ||q3||

R[i][j] = qi . aj    for i <= j
```

每一步都会去掉沿着之前所有 q 向量的分量，只保留新的正交方向。

### Cholesky 分解 (Cholesky decomposition)

当 A 是对称的（A = A^T）且正定的（所有特征值都为正）时，你可以把它分解为 A = L L^T，其中 L 是下三角矩阵。这就是 Cholesky 分解。

```
A = L @ L^T

| 4  2 |   | 2  0 |   | 2  1 |
| 2  5 | = | 1  2 | @ | 0  2 |

L[i][i] = sqrt(A[i][i] - sum(L[i][k]^2 for k < i))
L[i][j] = (A[i][j] - sum(L[i][k]*L[j][k] for k < j)) / L[j][j]    for i > j
```

Cholesky 的速度是 LU 的两倍，而且只需要一半存储空间。它只适用于对称正定矩阵，但这类矩阵非常常见：

- 协方差矩阵是对称半正定的（加上正则化后可变为正定）。
- 高斯过程中的核矩阵 (kernel matrix) 是对称正定的。
- 凸函数在极小值点处的 Hessian 矩阵是对称正定的。
- A^T A 总是对称半正定的。

在高斯过程中，你会先用 Cholesky 分解核矩阵 K，再求解 K alpha = y 来得到预测均值。Cholesky 因子还可以用来计算边际似然的对数行列式：log det(K) = 2 * sum(log(diag(L)))。

### 最小二乘：当 Ax = b 没有精确解时

如果 A 是一个 m x n 矩阵且 m > n（方程比未知数多），那么这个系统就是超定的 (overdetermined)。它没有精确解。此时你转而最小化平方误差：

```
minimize ||Ax - b||^2

This is the sum of squared residuals:
  sum((A[i,:] @ x - b[i])^2 for i in range(m))
```

最优解满足正规方程：

```
A^T A x = A^T b
```

推导如下：展开 ||Ax - b||^2 = (Ax - b)^T (Ax - b) = x^T A^T A x - 2 x^T A^T b + b^T b。对 x 求梯度并令其为零：2 A^T A x - 2 A^T b = 0。

```
Original system (overdetermined, 4 equations, 2 unknowns):
| 1  1 |         | 3 |
| 1  2 | x     = | 5 |       No exact x satisfies all 4 equations.
| 1  3 |         | 6 |
| 1  4 |         | 8 |

Normal equations:
A^T A = | 4  10 |    A^T b = | 22 |
        | 10 30 |            | 63 |

Solve: x = [1.5, 1.7]

This is linear regression. x[0] is the intercept, x[1] is the slope.
```

### 正规方程 = 线性回归

这种联系是完全精确的。在线性回归中，你的数据矩阵 X 每一行对应一个样本，每一列对应一个特征。目标向量 y 每个样本有一个值。权重向量 w 满足：

```
X^T X w = X^T y
w = (X^T X)^(-1) X^T y
```

这就是线性回归的闭式解 (closed-form solution)。每次调用 `sklearn.linear_model.LinearRegression.fit()`，本质上都在计算它（或通过 QR、SVD 计算其等价形式）。

给矩阵加上一个 lambda * I 的正则项，就得到岭回归：

```
(X^T X + lambda * I) w = X^T y
w = (X^T X + lambda * I)^(-1) X^T y
```

正则化会让矩阵的条件更好（更容易被准确求逆），并通过把权重压向零来防止过拟合。当 lambda > 0 时，矩阵 X^T X + lambda * I 总是对称正定的，因此你可以用 Cholesky 来求解它。

### 伪逆 (pseudoinverse, Moore-Penrose)

伪逆 A+ 把矩阵求逆推广到了非方阵和奇异矩阵。对于任意矩阵 A：

```
x = A+ b

where A+ = V Sigma+ U^T    (computed via SVD)
```

Sigma+ 的构造方式是：对每个非零奇异值取倒数，再将结果转置。如果 A = U Sigma V^T，那么 A+ = V Sigma+ U^T。

```
A = U Sigma V^T        (SVD)

Sigma = | 5  0 |       Sigma+ = | 1/5  0  0 |
        | 0  2 |                | 0  1/2  0 |
        | 0  0 |

A+ = V Sigma+ U^T
```

伪逆给出的是最小范数最小二乘解。如果系统有：
- 一个解：A+ b 会给出它。
- 无解：A+ b 会给出最小二乘解。
- 无穷多个解：A+ b 会给出其中 ||x|| 最小的那个。

NumPy 的 `np.linalg.lstsq` 和 `np.linalg.pinv` 内部都使用 SVD。

### 条件数 (condition number)

条件数衡量的是：输入发生微小变化时，解会有多敏感。对于矩阵 A，条件数定义为：

```
kappa(A) = ||A|| * ||A^(-1)|| = sigma_max / sigma_min
```

其中 sigma_max 和 sigma_min 分别是最大的和最小的奇异值。

```
Well-conditioned (kappa ~ 1):        Ill-conditioned (kappa ~ 10^15):
Small change in b -->                Small change in b -->
small change in x                    huge change in x

| 2  0 |   kappa = 2/1 = 2          | 1   1          |   kappa ~ 10^15
| 0  1 |   safe to solve            | 1   1+10^(-15) |   solution is garbage
```

经验法则：
- kappa < 100：通常是安全的，解会比较准确。
- kappa ~ 10^k：你的浮点运算大约会损失 k 位精度。
- kappa ~ 10^16（对于 float64）：这个解已经没有意义了。矩阵在数值上等同于奇异。

在机器学习里，当特征几乎共线时，就会出现病态。正则化（加上 lambda * I）会把条件数从 sigma_max / sigma_min 改善为 (sigma_max + lambda) / (sigma_min + lambda)。

### 迭代方法：共轭梯度 (conjugate gradient)

对于非常大的稀疏系统（有数百万个未知数），LU 或 Cholesky 这样的直接方法太昂贵了。迭代方法会从一个猜测开始，经过多轮迭代不断改进近似解。

当 A 是对称正定矩阵时，共轭梯度法 (CG) 可以求解 Ax = b。在精确算术下，它最多 n 次迭代就能找到精确解；但如果 A 的特征值聚得比较紧，它通常会收敛得快得多。

```
Algorithm sketch:
  x0 = initial guess (often zero)
  r0 = b - A x0           (residual)
  p0 = r0                 (search direction)

  For k = 0, 1, 2, ...:
    alpha = (rk . rk) / (pk . A pk)
    x_{k+1} = xk + alpha * pk
    r_{k+1} = rk - alpha * A pk
    beta = (r_{k+1} . r_{k+1}) / (rk . rk)
    p_{k+1} = r_{k+1} + beta * pk
    if ||r_{k+1}|| < tolerance: stop
```

CG 常用于：
- 大规模优化（Newton-CG 方法）
- 求解 PDE 离散化后的系统
- 核方法中核矩阵太大、无法直接分解的情况
- 作为其他迭代求解器的预条件步骤

收敛速度取决于条件数。系统条件越好，收敛越快，这也是正则化有帮助的另一个原因。

### 全景图：什么时候用哪种方法

| 方法 | 要求 | 成本 | 适用场景 |
|--------|-------------|------|----------|
| 高斯消元 | 方阵，且 A 非奇异 | O(n^3) | 一次性求解一个方阵系统 |
| LU 分解 | 方阵，且 A 非奇异 | O(n^3) 分解 + O(n^2) 求解 | 用相同的 A 重复求解多个系统 |
| QR 分解 | 任意 A（m >= n） | O(mn^2) | 最小二乘，数值稳定 |
| Cholesky | 对称正定的 A | O(n^3/3) | 协方差矩阵、高斯过程、岭回归 |
| 正规方程 | 超定（m > n） | O(mn^2 + n^3) | 线性回归（n 较小时） |
| SVD / 伪逆 | 任意 A | O(mn^2) | 秩亏系统、最小范数解 |
| 共轭梯度 | 对称正定、稀疏的 A | O(n * k * nnz) | 大型稀疏系统，其中 k = 迭代次数 |

### 与机器学习的联系

本课里的每一种方法都会出现在生产级机器学习系统中：

**线性回归。** 闭式解就是求解正规方程 X^T X w = X^T y。通常会通过 Cholesky（如果 n 很小）、QR（如果更关注数值稳定性）或 SVD（如果矩阵可能秩亏）来完成。

**岭回归。** 给 X^T X 加上 lambda * I。正则化后的系统 (X^T X + lambda * I) w = X^T y 总能通过 Cholesky 求解，因为当 lambda > 0 时，X^T X + lambda * I 是对称正定的。

**高斯过程。** 预测均值需要求解 K alpha = y，其中 K 是核矩阵。对 K 做 Cholesky 分解是标准做法。对数边际似然会用到 log det(K) = 2 sum(log(diag(L)))。

**神经网络初始化。** 正交初始化会使用 QR 分解来构造列向量标准正交的权重矩阵。这可以防止深层网络中的信号塌缩。

**预条件。** 大规模优化器会使用不完全 Cholesky 或不完全 LU 作为共轭梯度求解器的预条件器。

**特征工程。** X^T X 的条件数会告诉你特征是否共线。如果 kappa 很大，就删掉一些特征，或者加入正则化。

## 动手构建

### 第 1 步：带部分主元选取的高斯消元

```python
import numpy as np

def gaussian_elimination(A, b):
    n = len(b)
    Ab = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])

    for k in range(n):
        max_row = k + np.argmax(np.abs(Ab[k:, k]))
        Ab[[k, max_row]] = Ab[[max_row, k]]

        if abs(Ab[k, k]) < 1e-12:
            raise ValueError(f"Matrix is singular or nearly singular at pivot {k}")

        for i in range(k + 1, n):
            m = Ab[i, k] / Ab[k, k]
            Ab[i, k:] -= m * Ab[k, k:]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - Ab[i, i+1:n] @ x[i+1:n]) / Ab[i, i]

    return x
```

### 第 2 步：LU 分解

```python
def lu_decompose(A):
    n = A.shape[0]
    L = np.eye(n)
    U = A.astype(float).copy()
    P = np.eye(n)

    for k in range(n):
        max_row = k + np.argmax(np.abs(U[k:, k]))
        if max_row != k:
            U[[k, max_row]] = U[[max_row, k]]
            P[[k, max_row]] = P[[max_row, k]]
            if k > 0:
                L[[k, max_row], :k] = L[[max_row, k], :k]

        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]

    return P, L, U

def lu_solve(P, L, U, b):
    n = len(b)
    Pb = P @ b.astype(float)

    y = np.zeros(n)
    for i in range(n):
        y[i] = Pb[i] - L[i, :i] @ y[:i]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]

    return x
```

### 第 3 步：Cholesky 分解

```python
def cholesky(A):
    n = A.shape[0]
    L = np.zeros_like(A, dtype=float)

    for i in range(n):
        for j in range(i + 1):
            s = A[i, j] - L[i, :j] @ L[j, :j]
            if i == j:
                if s <= 0:
                    raise ValueError("Matrix is not positive definite")
                L[i, j] = np.sqrt(s)
            else:
                L[i, j] = s / L[j, j]

    return L
```

### 第 4 步：通过正规方程求最小二乘

```python
def least_squares_normal(A, b):
    AtA = A.T @ A
    Atb = A.T @ b
    return gaussian_elimination(AtA, Atb)

def ridge_regression(A, b, lam):
    n = A.shape[1]
    AtA = A.T @ A + lam * np.eye(n)
    Atb = A.T @ b
    L = cholesky(AtA)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (Atb[i] - L[i, :i] @ y[:i]) / L[i, i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - L.T[i, i+1:] @ x[i+1:]) / L.T[i, i]
    return x
```

### 第 5 步：条件数

```python
def condition_number(A):
    U, S, Vt = np.linalg.svd(A)
    return S[0] / S[-1]
```

## 使用它

把这些部分组合起来，在真实数据上做线性回归和岭回归：

```python
np.random.seed(42)
X_raw = np.random.randn(100, 3)
w_true = np.array([2.0, -1.0, 0.5])
y = X_raw @ w_true + np.random.randn(100) * 0.1

X = np.column_stack([np.ones(100), X_raw])

w_ols = least_squares_normal(X, y)
print(f"OLS weights (ours):    {w_ols}")

w_np = np.linalg.lstsq(X, y, rcond=None)[0]
print(f"OLS weights (numpy):   {w_np}")
print(f"Max difference: {np.max(np.abs(w_ols - w_np)):.2e}")

w_ridge = ridge_regression(X, y, lam=1.0)
print(f"Ridge weights (ours):  {w_ridge}")

from sklearn.linear_model import Ridge
ridge_sk = Ridge(alpha=1.0, fit_intercept=False)
ridge_sk.fit(X, y)
print(f"Ridge weights (sklearn): {ridge_sk.coef_}")
```

## 交付成果

本课将产出：
- `code/linear_systems.py`，其中包含从零实现的高斯消元、LU 分解、Cholesky 分解、最小二乘和岭回归
- 一个可运行的演示，用来说明正规方程与 sklearn 的 LinearRegression 会产生相同的权重

## 练习

1. 使用你自己实现的高斯消元、你自己的 LU 求解器和 `np.linalg.solve` 来求解系统 `[[1,2,3],[4,5,6],[7,8,10]] x = [6, 15, 27]`。验证三者在浮点误差容忍范围内给出相同答案。

2. 生成一个 50x5 的随机矩阵 X，以及目标 `y = X @ w_true + noise`。分别用正规方程、QR（通过 `np.linalg.qr`）、SVD（通过 `np.linalg.svd`）和 `np.linalg.lstsq` 求解 w。比较这四个解。测量 X^T X 的条件数，并解释它如何影响你更信任哪种方法。

3. 构造一个接近奇异的矩阵，让两列几乎完全相同（例如，第 2 列 = 第 1 列 + 1e-10 * noise）。计算它的条件数。在有和没有正则化（加上 0.01 * I）的情况下分别求解 Ax = b。比较两组解和残差，并解释为什么正则化有帮助。

4. 对一个 100x100 的随机对称正定矩阵实现共轭梯度算法。统计它收敛到容差 1e-8 需要多少次迭代，并与理论上的 n 次迭代上限进行比较。

5. 比较你的 Cholesky 求解器、LU 求解器和 `np.linalg.solve` 在大小为 10、50、200、500 的对称正定矩阵上的运行时间。把结果画出来，并验证 Cholesky 大致比 LU 快 2 倍。

## 关键术语

| 术语 | 人们常说的话 | 它真正的意思 |
|------|----------------|----------------------|
| 线性方程组 | “求 x” | 一组线性方程 Ax = b。求 x，就是找到在变换 A 下会产生输出 b 的那个输入。 |
| 高斯消元 | “做行化简” | 用行变换系统性地把对角线以下的元素消成零，得到一个可通过回代求解的上三角系统。O(n^3)。 |
| 部分主元选取 | “换行以保持稳定” | 在第 k 列消元前，把该列中绝对值最大的那一行换到主元位置。这样可以避免除以很小的数。 |
| LU 分解 | “分解成两个三角形” | 把 A 写成 A = LU，其中 L 是下三角矩阵（保存乘子），U 是上三角矩阵（消元后的矩阵）。它把 O(n^3) 的成本摊到了多次求解上。 |
| QR 分解 | “正交分解” | 把 A 写成 A = QR，其中 Q 的列两两正交且范数为 1，R 是上三角矩阵。在最小二乘问题中，它比 LU 更稳定。 |
| Cholesky 分解 | “矩阵的平方根” | 对于对称正定矩阵 A，把它写成 A = LL^T。成本只有 LU 的一半。常用于协方差矩阵、核矩阵和岭回归。 |
| 最小二乘 | “精确解不存在时的最佳拟合” | 当系统是超定的（方程多于未知数）时，最小化残差平方和 ||Ax - b||^2。 |
| 正规方程 | “微积分捷径” | A^T A x = A^T b。它来自把 ||Ax - b||^2 的梯度设为零。这就是线性回归的闭式解。 |
| 伪逆 | “给非方阵做求逆” | 通过 SVD 得到 A+ = V Sigma+ U^T。对任意矩阵——无论方阵还是长方阵、奇异还是非奇异——它都能给出最小范数最小二乘解。 |
| 条件数 | “这个答案到底靠不靠谱” | kappa = sigma_max / sigma_min。它衡量输入扰动带来的敏感程度。大约会损失 log10(kappa) 位精度。 |
| 岭回归 | “带正则的最小二乘” | 求解 (X^T X + lambda I) w = X^T y。加入 lambda I 会改善条件并把权重压向零，从而防止过拟合。 |
| 共轭梯度 | “大矩阵上的迭代版 Ax=b” | 一种用于对称正定系统的迭代求解器。最多 n 步收敛。对于无法承受分解成本的大型稀疏系统非常实用。 |
| 超定系统 | “数据比参数多” | 在一个 m 行 n 列的系统里，m > n。此时不存在精确解。最小二乘会给出最佳近似。这就是所有回归问题的本质。 |
| 回代 | “从底部开始往上解” | 对于上三角系统，先解最后一个方程，再逐步向上代回。O(n^2)。 |
| 前代 | “从顶部开始往下解” | 对于下三角系统，先解第一个方程，再逐步向前推进。O(n^2)。它用于 LU 求解中的 L 这一步。 |

## 延伸阅读

- [MIT 18.06: Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)（Gilbert Strang）—— 关于线性方程组和矩阵分解的权威课程
- [Numerical Linear Algebra](https://people.maths.ox.ac.uk/trefethen/text.html)（Trefethen & Bau）—— 理解数值稳定性、条件性以及算法为何会失败的标准参考
- [Matrix Computations](https://www.cs.cornell.edu/cv/GolubVanLoan4/golubandvanloan.htm)（Golub & Van Loan）—— 几乎涵盖所有矩阵算法的百科全书式参考书
- [3Blue1Brown: Inverse Matrices](https://www.3blue1brown.com/lessons/inverse-matrices) —— 从可视化角度理解求解 Ax = b 的几何意义
