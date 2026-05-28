# 矩阵变换

> 矩阵是重塑空间的机器。理解它对每个点的作用，你就理解了整个变换。

**类型（Type）：** 构建
**语言（Languages）：** Python、Julia
**前置知识（Prerequisites）：** 第 1 阶段第 01-02 课（线性代数直觉、向量与矩阵运算）
**预计用时（Time）：** 约 75 分钟

## 学习目标

- 构建旋转（rotation）、缩放（scaling）、错切（shearing）和反射（reflection）矩阵，并将其应用于二维和三维坐标点
- 通过矩阵乘法组合多个变换，并验证顺序的重要性
- 利用特征方程计算 2×2 矩阵的特征值（eigenvalues）和特征向量（eigenvectors）
- 解释为何特征值决定 PCA 的方向、RNN 的稳定性以及谱聚类的行为

## 问题背景

你读到 PCA 相关资料，看到"求协方差矩阵的特征向量"；你读到模型稳定性资料，看到"检验所有特征值的模是否小于 1"；你读到数据增强资料，看到"应用随机旋转"。在理解矩阵对空间的几何作用之前，这些都毫无意义。

矩阵不只是数字网格，它们是空间机器：旋转矩阵让点旋转，缩放矩阵让点拉伸，错切矩阵让点倾斜。神经网络对数据施加的每一个变换，都是这些操作或它们的组合。本节课将这些操作变得具体可感。

## 核心概念

### 用矩阵表示变换

二维中的每个线性变换都可以写成一个 2×2 矩阵。矩阵告诉你基向量 [1, 0] 和 [0, 1] 最终到达的位置，其余的一切由此推导而来。

```mermaid
graph LR
    subgraph Before["标准基"]
        e1["e1 = [1, 0]（沿 x 轴）"]
        e2["e2 = [0, 1]（沿 y 轴）"]
    end
    subgraph Transform["矩阵 M"]
        M["M = 列为新基向量"]
    end
    subgraph After["经矩阵 M 变换后"]
        e1p["e1' = 新的 x 基"]
        e2p["e2' = 新的 y 基"]
    end
    e1 --> M --> e1p
    e2 --> M --> e2p
```

### 旋转

二维旋转角度 theta 保持距离和角度不变，每个点沿圆弧移动。

```mermaid
graph LR
    subgraph Before["旋转前"]
        A["A(2, 1)"]
        B["B(0, 2)"]
    end
    subgraph Rot["旋转 45 度"]
        R["R(θ) = [[cos θ, -sin θ], [sin θ, cos θ]]"]
    end
    subgraph After["旋转后"]
        Ap["A'(0.71, 2.12)"]
        Bp["B'(-1.41, 1.41)"]
    end
    A --> R --> Ap
    B --> R --> Bp
```

在三维空间中，旋转围绕某个轴进行，每个轴有对应的旋转矩阵：

```
Rz(theta) = | cos  -sin  0 |     Rotate around z-axis
            | sin   cos  0 |     (x-y plane spins, z stays)
            |  0     0   1 |

Rx(theta) = | 1   0     0    |   Rotate around x-axis
            | 0  cos  -sin   |   (y-z plane spins, x stays)
            | 0  sin   cos   |

Ry(theta) = |  cos  0  sin |     Rotate around y-axis
            |   0   1   0  |     (x-z plane spins, y stays)
            | -sin  0  cos |
```

### 缩放

缩放沿每个轴独立地拉伸或压缩。

```mermaid
graph LR
    subgraph Before["缩放前"]
        A["A(2, 1)"]
        B["B(0, 2)"]
    end
    subgraph Scale["缩放 sx=2, sy=0.5"]
        S["S = [[2, 0], [0, 0.5]]"]
    end
    subgraph After["缩放后"]
        Ap["A'(4, 0.5)"]
        Bp["B'(0, 1)"]
    end
    A --> S --> Ap
    B --> S --> Bp
```

### 错切

错切使一个轴倾斜，同时保持另一个轴不变，将矩形变为平行四边形。

```mermaid
graph LR
    subgraph Before["错切前"]
        A["A(1, 0)"]
        B["B(0, 1)"]
    end
    subgraph Shear["沿 x 方向错切，k=1"]
        Sh["Shx = [[1, k], [0, 1]]"]
    end
    subgraph After["错切后"]
        Ap["A(1, 0) 不变"]
        Bp["B'(1, 1) 偏移"]
    end
    A --> Sh --> Ap
    B --> Sh --> Bp
```

错切矩阵：
- `Shx = [[1, k], [0, 1]]`：x 偏移 k × y
- `Shy = [[1, 0], [k, 1]]`：y 偏移 k × x

### 反射

反射将点关于某个轴或直线做镜像。

```mermaid
graph LR
    subgraph Before["反射前"]
        A["A(2, 1)"]
    end
    subgraph Reflect["关于 y 轴反射"]
        R["[[-1, 0], [0, 1]]"]
    end
    subgraph After["反射后"]
        Ap["A'(-2, 1)"]
    end
    A --> R --> Ap
```

反射矩阵：
- 关于 y 轴反射：`[[-1, 0], [0, 1]]`
- 关于 x 轴反射：`[[1, 0], [0, -1]]`

### 组合：链式变换

先应用变换 A，再应用变换 B，等同于矩阵相乘：`result = B @ A @ point`。顺序至关重要：先旋转后缩放与先缩放后旋转的结果不同。

```mermaid
graph LR
    subgraph Path1["先旋转 90°，再缩放 (2, 0.5)"]
        P1["(1, 0)"] -->|"旋转 90°"| P2["(0, 1)"] -->|"缩放"| P3["(0, 0.5)"]
    end
```

组合后：`S @ R = [[0, -2], [0.5, 0]]`

```mermaid
graph LR
    subgraph Path2["先缩放 (2, 0.5)，再旋转 90°"]
        Q1["(1, 0)"] -->|"缩放"| Q2["(2, 0)"] -->|"旋转 90°"| Q3["(0, 2)"]
    end
```

组合后：`R @ S = [[0, -0.5], [2, 0]]`

结果不同。矩阵乘法不满足交换律。

### 特征值与特征向量

大多数向量在被矩阵作用后会改变方向。特征向量是特殊的：矩阵只对它们进行缩放，而不旋转。缩放因子即为特征值。

```
A @ v = lambda * v

v 是特征向量（方向得以保留）
lambda 是特征值（拉伸的幅度）

Example: A = | 2  1 |
             | 1  2 |

Eigenvector [1, 1] with eigenvalue 3:
  A @ [1,1] = [3, 3] = 3 * [1, 1]     (same direction, scaled by 3)

Eigenvector [1, -1] with eigenvalue 1:
  A @ [1,-1] = [1, -1] = 1 * [1, -1]  (same direction, unchanged)
```

该矩阵沿 [1, 1] 方向将空间拉伸 3 倍，沿 [1, -1] 方向保持不变。所有其他方向都是这两个方向的混合。

### 特征分解

若一个矩阵有 n 个线性无关的特征向量，则可以对其进行特征分解（eigendecomposition）：

```
A = V @ D @ V^(-1)

V = matrix whose columns are eigenvectors
D = diagonal matrix of eigenvalues
V^(-1) = inverse of V

This says: rotate into eigenvector coordinates, scale along each axis, rotate back.
```

### 为何特征值重要

**主成分分析（PCA）。** 协方差矩阵的特征向量就是主成分（principal components），特征值告诉你每个主成分捕获了多少方差。按特征值排序后，保留前 k 个，即可实现降维。

**稳定性。** 在循环神经网络（RNN）和动力系统中，模大于 1 的特征值会导致输出爆炸，模小于 1 会导致输出消失。这就是梯度消失/爆炸问题（vanishing/exploding gradient problem）的一句话描述。

**谱方法。** 图神经网络使用邻接矩阵的特征值，谱聚类（spectral clustering）使用拉普拉斯矩阵的特征值。特征向量揭示图的结构。

### 行列式作为体积缩放因子

变换矩阵的行列式（determinant）告诉你它对面积（二维）或体积（三维）的缩放比例。

```
det = 1:   面积不变（旋转）
det = 2:   面积翻倍
det = 0:   空间被压缩到低维（奇异矩阵）
det = -1:  面积不变但方向翻转（反射）

| det(旋转矩阵) | = 1        （恒成立）
| det(缩放矩阵 sx, sy) | = sx * sy
| det(错切矩阵) | = 1        （面积不变）
| det(反射矩阵) | = -1       （方向翻转）
```

## 动手实现

### 第 1 步：用 Python 从零实现变换矩阵

```python
import math

def rotation_2d(theta):
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s], [s, c]]

def scaling_2d(sx, sy):
    return [[sx, 0], [0, sy]]

def shearing_2d(kx, ky):
    return [[1, kx], [ky, 1]]

def reflection_x():
    return [[1, 0], [0, -1]]

def reflection_y():
    return [[-1, 0], [0, 1]]

def mat_vec_mul(matrix, vector):
    return [
        sum(matrix[i][j] * vector[j] for j in range(len(vector)))
        for i in range(len(matrix))
    ]

def mat_mul(a, b):
    rows_a, cols_b = len(a), len(b[0])
    cols_a = len(a[0])
    return [
        [sum(a[i][k] * b[k][j] for k in range(cols_a)) for j in range(cols_b)]
        for i in range(rows_a)
    ]

point = [1.0, 0.0]
angle = math.pi / 4

rotated = mat_vec_mul(rotation_2d(angle), point)
print(f"Rotate (1,0) by 45 deg: ({rotated[0]:.4f}, {rotated[1]:.4f})")

scaled = mat_vec_mul(scaling_2d(2, 3), [1.0, 1.0])
print(f"Scale (1,1) by (2,3): ({scaled[0]:.1f}, {scaled[1]:.1f})")

sheared = mat_vec_mul(shearing_2d(1, 0), [1.0, 1.0])
print(f"Shear (1,1) kx=1: ({sheared[0]:.1f}, {sheared[1]:.1f})")

reflected = mat_vec_mul(reflection_y(), [2.0, 1.0])
print(f"Reflect (2,1) across y: ({reflected[0]:.1f}, {reflected[1]:.1f})")
```

### 第 2 步：组合变换

```python
R = rotation_2d(math.pi / 2)
S = scaling_2d(2, 0.5)

rotate_then_scale = mat_mul(S, R)
scale_then_rotate = mat_mul(R, S)

point = [1.0, 0.0]
result1 = mat_vec_mul(rotate_then_scale, point)
result2 = mat_vec_mul(scale_then_rotate, point)

print(f"Rotate 90 then scale: ({result1[0]:.2f}, {result1[1]:.2f})")
print(f"Scale then rotate 90: ({result2[0]:.2f}, {result2[1]:.2f})")
print(f"Same? {result1 == result2}")
```

### 第 3 步：从零计算特征值（2×2 矩阵）

对于 2×2 矩阵 `[[a, b], [c, d]]`，特征值通过特征方程（characteristic equation）求解：`lambda^2 - (a+d)*lambda + (ad - bc) = 0`。

```python
def eigenvalues_2x2(matrix):
    a, b = matrix[0]
    c, d = matrix[1]
    trace = a + d
    det = a * d - b * c
    discriminant = trace ** 2 - 4 * det
    if discriminant < 0:
        real = trace / 2
        imag = (-discriminant) ** 0.5 / 2
        return (complex(real, imag), complex(real, -imag))
    sqrt_disc = discriminant ** 0.5
    return ((trace + sqrt_disc) / 2, (trace - sqrt_disc) / 2)

def eigenvector_2x2(matrix, eigenvalue):
    a, b = matrix[0]
    c, d = matrix[1]
    if abs(b) > 1e-10:
        v = [b, eigenvalue - a]
    elif abs(c) > 1e-10:
        v = [eigenvalue - d, c]
    else:
        if abs(a - eigenvalue) < 1e-10:
            v = [1, 0]
        else:
            v = [0, 1]
    mag = (v[0] ** 2 + v[1] ** 2) ** 0.5
    return [v[0] / mag, v[1] / mag]

A = [[2, 1], [1, 2]]
vals = eigenvalues_2x2(A)
print(f"Matrix: {A}")
print(f"Eigenvalues: {vals[0]:.4f}, {vals[1]:.4f}")

for val in vals:
    vec = eigenvector_2x2(A, val)
    result = mat_vec_mul(A, vec)
    scaled = [val * vec[0], val * vec[1]]
    print(f"  lambda={val:.1f}, v={[round(x,4) for x in vec]}")
    print(f"    A@v = {[round(x,4) for x in result]}")
    print(f"    l*v = {[round(x,4) for x in scaled]}")
```

### 第 4 步：行列式作为体积缩放因子

```python
def det_2x2(matrix):
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

print(f"det(rotation 45) = {det_2x2(rotation_2d(math.pi/4)):.4f}")
print(f"det(scale 2,3)   = {det_2x2(scaling_2d(2, 3)):.1f}")
print(f"det(shear kx=1)  = {det_2x2(shearing_2d(1, 0)):.1f}")
print(f"det(reflect y)   = {det_2x2(reflection_y()):.1f}")

singular = [[1, 2], [2, 4]]
print(f"det(singular)     = {det_2x2(singular):.1f}")
print("Singular: columns are proportional, space collapses to a line.")
```

## 实际应用

NumPy 通过优化例程处理上述所有操作。

```python
import numpy as np

theta = np.pi / 4
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

point = np.array([1.0, 0.0])
print(f"Rotate (1,0) by 45 deg: {R @ point}")

S = np.diag([2.0, 3.0])
composed = S @ R
print(f"Scale(2,3) after Rotate(45): {composed @ point}")

A = np.array([[2, 1], [1, 2]], dtype=float)
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"\nEigenvalues: {eigenvalues}")
print(f"Eigenvectors (columns):\n{eigenvectors}")

for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    print(f"  A @ v{i} = {A @ v}, lambda * v{i} = {lam * v}")

print(f"\ndet(R) = {np.linalg.det(R):.4f}")
print(f"det(S) = {np.linalg.det(S):.1f}")

B = np.array([[3, 1], [0, 2]], dtype=float)
vals, vecs = np.linalg.eig(B)
D = np.diag(vals)
V = vecs
reconstructed = V @ D @ np.linalg.inv(V)
print(f"\nEigendecomposition A = V @ D @ V^-1:")
print(f"Original:\n{B}")
print(f"Reconstructed:\n{reconstructed}")
```

### 用 NumPy 实现三维旋转

```python
def rotation_3d_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def rotation_3d_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

point_3d = np.array([1.0, 0.0, 0.0])
rotated_z = rotation_3d_z(np.pi / 2) @ point_3d
rotated_x = rotation_3d_x(np.pi / 2) @ point_3d

print(f"\n3D point: {point_3d}")
print(f"Rotate 90 around z: {np.round(rotated_z, 4)}")
print(f"Rotate 90 around x: {np.round(rotated_x, 4)}")
```

## 交付物

本节课为 PCA（第 2 阶段）和神经网络权重分析奠定了几何基础。此处构建的特征值/特征向量代码，与生产级机器学习系统中用于降维、谱聚类和稳定性分析的算法完全一致。

## 练习

1. 将旋转、缩放和错切分别应用于单位正方形（顶角为 [0,0]、[1,0]、[1,1]、[0,1]）。逐一打印变换后的顶角坐标，并验证旋转保留了顶角之间的距离。

2. 利用特征方程手动求矩阵 [[4, 2], [1, 3]] 的特征值，再用你的从零实现函数和 NumPy 分别验证。

3. 组合三个变换（旋转 30 度、缩放 [1.5, 0.8]、错切 kx=0.3），并将其应用于圆上均匀分布的 8 个点。打印变换前后的坐标，计算组合矩阵的行列式，验证其等于各个单独行列式的乘积。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| 旋转矩阵（Rotation matrix） | "让东西旋转" | 正交矩阵，使点沿圆弧移动，同时保持距离和角度不变。行列式恒为 1。 |
| 缩放矩阵（Scaling matrix） | "让东西变大" | 对角矩阵，沿每个轴独立地拉伸或压缩。行列式等于各缩放因子的乘积。 |
| 错切矩阵（Shearing matrix） | "让东西倾斜" | 将一个坐标按比例偏移另一个坐标，使矩形变为平行四边形。行列式为 1。 |
| 反射（Reflection） | "镜像" | 关于某个轴或平面翻转空间。行列式为 -1。 |
| 组合（Composition） | "做两件事" | 通过矩阵相乘来链接操作。顺序至关重要：B @ A 表示先应用 A，再应用 B。 |
| 特征向量（Eigenvector） | "特殊方向" | 矩阵仅对其进行缩放而不旋转的方向，是变换的"指纹"。 |
| 特征值（Eigenvalue） | "拉伸了多少" | 矩阵对其特征向量进行缩放的标量因子，可以为负数（翻转）或复数（旋转）。 |
| 特征分解（Eigendecomposition） | "把矩阵拆开" | 将矩阵写成 V @ D @ V^(-1)，分离出其基本缩放方向和大小。 |
| 行列式（Determinant） | "矩阵的一个数" | 变换对面积（二维）或体积（三维）的缩放因子。为零意味着变换不可逆。 |
| 特征方程（Characteristic equation） | "特征值从哪来" | det(A - lambda * I) = 0，以特征值为根的多项式。 |

## 延伸阅读

- [3Blue1Brown: Linear Transformations](https://www.3blue1brown.com/lessons/linear-transformations) -- 矩阵如何重塑空间的视觉直觉
- [3Blue1Brown: Eigenvectors and Eigenvalues](https://www.3blue1brown.com/lessons/eigenvalues) -- 关于特征向量几何含义的最佳视觉解释
- [MIT 18.06 Lecture 21: Eigenvalues and Eigenvectors](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) -- Gilbert Strang 的经典讲解
