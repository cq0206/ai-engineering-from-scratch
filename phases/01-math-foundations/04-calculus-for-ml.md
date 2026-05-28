# 机器学习中的微积分

> 导数告诉你哪个方向是下坡。这就是神经网络学习所需的全部。

**类型（Type）：** 学习
**语言（Language）：** Python
**前置知识（Prerequisites）：** 第 1 阶段第 01-03 课
**预计用时（Time）：** 约 60 分钟

## 学习目标

- 计算常见机器学习函数（x^2、sigmoid、交叉熵）的数值导数和解析导数
- 从零实现梯度下降，最小化一维和二维损失函数
- 推导线性回归模型的梯度，并通过手动权重更新进行训练
- 解释 Hessian 矩阵（Hessian matrix）、Taylor 级数近似（Taylor series approximation）及其与优化方法的联系

## 问题背景

你有一个包含数百万个权重的神经网络，每个权重都是一个旋钮。你需要弄清楚应该朝哪个方向转动每一个旋钮，才能让模型的误差稍微减小一点。微积分给了你这个方向。

没有微积分，训练神经网络只能靠随机尝试碰运气。有了导数，你就能精确知道每个权重如何影响误差，每次都能往正确的方向转动所有旋钮。

## 核心概念

### 什么是导数？

导数（derivative）衡量变化率。对于函数 y = f(x)，导数 f'(x) 告诉你：若 x 微小扰动，y 会改变多少？

从几何角度看，导数是某点处切线的斜率。

**f(x) = x^2：**

| x | f(x) | f'(x)（斜率） |
|---|------|---------------|
| 0 | 0    | 0（平坦，处于底部） |
| 1 | 1    | 2 |
| 2 | 4    | 4（该点切线斜率） |
| 3 | 9    | 6 |

在 x=2 处，斜率为 4。若 x 向右微小移动，y 大约增加该移动量的 4 倍。在 x=0 处，斜率为 0，此时处于碗状函数的底部。

形式化定义：

```
f'(x) = lim   f(x + h) - f(x)
        h->0  -----------------
                     h
```

在代码中，你省略极限，只使用一个很小的 h，这就是数值导数（numerical derivative）。

### 偏导数：每次只对一个变量求导

实际函数有很多输入，神经网络的损失依赖于成千上万个权重。偏导数（partial derivative）固定所有其他变量，只对其中一个变量求导。

```
f(x, y) = x^2 + 3xy + y^2

df/dx = 2x + 3y     (treat y as a constant)
df/dy = 3x + 2y     (treat x as a constant)
```

每个偏导数回答这个问题：如果只微调这一个权重，损失会如何变化？

### 梯度：所有偏导数组成的向量

梯度（gradient）将所有偏导数收集为一个向量。对于函数 f(x, y, z)，梯度为：

```
grad f = [ df/dx, df/dy, df/dz ]
```

梯度指向函数增长最快的方向。若要最小化函数，则沿反方向移动。

**f(x,y) = x^2 + y^2 的等高线图：**

该函数形成一个碗状，等高线为一系列同心圆，最小值在 (0, 0) 处。

| 点 | grad f | -grad f（下降方向） |
|-------|--------|----------------------------|
| (1, 1) | [2, 2]（指向上坡，远离最小值） | [-2, -2]（指向下坡，朝向最小值） |
| (0, 0) | [0, 0]（平坦，处于最小值） | [0, 0] |

这就是梯度下降的图示：计算梯度，取反，迈出一步。

### 与优化的联系

训练神经网络本质上是优化问题。你有一个损失函数 L(w1, w2, ..., wn) 衡量模型的误差，目标是最小化它。

```
Gradient descent update rule:

  w_new = w_old - learning_rate * dL/dw

For every weight:
  1. Compute the partial derivative of loss with respect to that weight
  2. Subtract a small multiple of it from the weight
  3. Repeat
```

学习率（learning rate）控制步长。太大会导致振荡发散，太小则收敛缓慢。

**损失曲面（一维截面）：**

损失函数 L(w) 随权重 w 的变化形成一条有峰有谷的曲线。

| 特征 | 描述 |
|---------|-------------|
| 全局最小值（Global minimum） | 整条曲线的最低点——最优解 |
| 局部最小值（Local minimum） | 比其邻域低但非全局最低的谷 |
| 斜率 | 梯度下降从任意起点沿斜坡向下移动 |

梯度下降沿斜坡向下移动，可能陷入局部最小值，但在高维空间（数百万个权重）中这在实践中很少成为问题。

### 数值导数 vs 解析导数

计算导数有两种方法。

解析法（analytical）：手动应用微积分规则。对于 f(x) = x^2，导数为 f'(x) = 2x。精确且快速。

数值法（numerical）：利用定义近似计算。对很小的 h，计算 f(x+h) 和 f(x-h)，再用差分求导。

```
Numerical (central difference):

f'(x) ~= f(x + h) - f(x - h)
          -----------------------
                  2h

h = 0.0001 works well in practice
```

数值导数较慢，但适用于任意函数；解析导数快速，但需要推导公式。神经网络框架使用第三种方法：自动微分（automatic differentiation），能以机械化方式精确计算导数。你将在第 3 阶段看到这一点。

### 手动推导简单函数的导数

以下是机器学习中反复出现的导数。

```
Function        Derivative       Used in
--------        ----------       -------
f(x) = x^2     f'(x) = 2x      Loss functions (MSE)
f(x) = wx + b  f'(w) = x        Linear layer (gradient w.r.t. weight)
                f'(b) = 1        Linear layer (gradient w.r.t. bias)
                f'(x) = w        Linear layer (gradient w.r.t. input)
f(x) = e^x     f'(x) = e^x     Softmax, attention
f(x) = ln(x)   f'(x) = 1/x     Cross-entropy loss
f(x) = 1/(1+e^-x)  f'(x) = f(x)(1-f(x))   Sigmoid activation
```

对于 f(x) = x^2：

```
f(x) = x^2    f'(x) = 2x

  x    f(x)   f'(x)   meaning
  -2    4      -4      slope tilts left (decreasing)
  -1    1      -2      slope tilts left (decreasing)
   0    0       0      flat (minimum!)
   1    1       2      slope tilts right (increasing)
   2    4       4      slope tilts right (increasing)
```

对于 f(w) = wx + b，其中 x=3，b=1：

```
f(w) = 3w + 1    f'(w) = 3

The derivative with respect to w is just x.
If x is big, a small change in w causes a big change in output.
```

### 链式法则

对于复合函数，链式法则（chain rule）告诉你如何求导。

```
If y = f(g(x)), then dy/dx = f'(g(x)) * g'(x)

Example: y = (3x + 1)^2
  outer: f(u) = u^2       f'(u) = 2u
  inner: g(x) = 3x + 1    g'(x) = 3
  dy/dx = 2(3x + 1) * 3 = 6(3x + 1)
```

神经网络是一系列复合函数：输入 → 线性层 → 激活函数 → 线性层 → 激活函数 → 损失。反向传播（backpropagation）就是从输出到输入反复应用链式法则的过程——整个算法仅此而已。

### Hessian 矩阵

梯度告诉你斜率，Hessian 矩阵告诉你曲率（curvature）。

Hessian 是二阶偏导数的矩阵。对于函数 f(x1, x2, ..., xn)，Hessian 的第 (i, j) 项为：

```
H[i][j] = d^2f / (dx_i * dx_j)
```

对于二元函数 f(x, y)：

```
H = | d^2f/dx^2    d^2f/dxdy |
    | d^2f/dydx    d^2f/dy^2 |
```

**Hessian 在临界点（梯度为零处）的含义：**

| Hessian 的性质 | 含义 | 曲面示例 |
|-----------------|---------|-----------------|
| 正定（所有特征值 > 0） | 局部最小值 | 开口朝上的碗 |
| 负定（所有特征值 &lt; 0） | 局部最大值 | 开口朝下的碗 |
| 不定（特征值有正有负） | 鞍点 | 马鞍形曲面 |

**示例：** f(x, y) = x^2 - y^2（鞍点函数）

```
df/dx = 2x       df/dy = -2y
d^2f/dx^2 = 2    d^2f/dy^2 = -2    d^2f/dxdy = 0

H = | 2   0 |
    | 0  -2 |

Eigenvalues: 2 and -2 (one positive, one negative)
--> Saddle point at (0, 0)
```

与 f(x, y) = x^2 + y^2（碗形函数）对比：

```
H = | 2  0 |
    | 0  2 |

Eigenvalues: 2 and 2 (both positive)
--> Local minimum at (0, 0)
```

**Hessian 在机器学习中为何重要：**

牛顿法（Newton's method）利用 Hessian 采取比梯度下降更好的优化步骤。它不只是顺着坡走，还考虑了曲率：

```
Newton's update:    w_new = w_old - H^(-1) * gradient
Gradient descent:   w_new = w_old - lr * gradient
```

牛顿法收敛更快，因为 Hessian 对梯度进行了"重新缩放"——陡峭方向的步长更小，平坦方向的步长更大。

代价是：对于有 N 个参数的神经网络，Hessian 矩阵大小为 N × N。拥有 100 万个参数的模型需要一个 1 万亿项的矩阵，这就是为什么我们使用近似方法。

| 方法 | 使用的信息 | 代价 | 收敛速度 |
|--------|-------------|------|-------------|
| 梯度下降 | 仅一阶导数 | O(N) 每步 | 慢（线性） |
| 牛顿法 | 完整 Hessian | O(N^3) 每步 | 快（二次） |
| L-BFGS | 基于梯度历史的近似 Hessian | O(N) 每步 | 中等（超线性） |
| Adam | 自适应逐参数学习率（对角 Hessian 近似） | O(N) 每步 | 中等 |
| 自然梯度 | Fisher 信息矩阵（统计 Hessian） | O(N^2) 每步 | 快 |

在实践中，Adam 是深度学习的默认优化器。它通过追踪每个参数梯度的历史均值和方差，以低廉的代价近似二阶信息。

### Taylor 级数近似

任何光滑函数都可以在局部用多项式近似：

```
f(x + h) = f(x) + f'(x)*h + (1/2)*f''(x)*h^2 + (1/6)*f'''(x)*h^3 + ...
```

包含的项越多，近似越精确——但仅在点 x 附近有效。

**Taylor 级数对机器学习的意义：**

- **一阶 Taylor = 梯度下降。** 使用 f(x + h) ~ f(x) + f'(x)*h 时，你在做线性近似。梯度下降通过最小化这个线性模型来选择步长 h = -lr * f'(x)。

- **二阶 Taylor = 牛顿法。** 使用 f(x + h) ~ f(x) + f'(x)*h + (1/2)*f''(x)*h^2 时，你得到一个二次模型。最小化该模型给出 h = -f'(x)/f''(x)——即牛顿步。

- **损失函数设计。** 均方误差（MSE）和交叉熵都是光滑的，意味着它们的 Taylor 展开行为良好。这并非偶然，光滑损失函数使优化过程可预测。

```
Approximation order    What it captures    Optimization method
-------------------    -----------------   -------------------
0th order (constant)   Just the value      Random search
1st order (linear)     Slope               Gradient descent
2nd order (quadratic)  Curvature           Newton's method
Higher orders          Finer structure     Rarely used in ML
```

关键洞见：所有基于梯度的优化，本质上都是在局部近似损失函数，然后朝该近似的最小值迈步。

### 积分在机器学习中的应用

导数告诉你变化率，积分（integral）计算累积量——曲线下的面积。

在机器学习中，你很少手动计算积分，但这一概念无处不在：

**概率（Probability）。** 对于密度函数为 p(x) 的连续随机变量：
```
P(a < X < b) = integral from a to b of p(x) dx
```
概率密度曲线在 a 到 b 之间的面积，就是落在该区间内的概率。

**期望值（Expected value）。** 按概率加权的平均结果：
```
E[f(X)] = integral of f(x) * p(x) dx
```
数据分布上的期望损失是一个积分，训练最小化的是其经验近似。

**KL 散度（KL divergence）。** 衡量两个分布之间的差异：
```
KL(p || q) = integral of p(x) * log(p(x) / q(x)) dx
```
用于变分自编码器（VAE）、知识蒸馏和贝叶斯推断。

**归一化常数（Normalization constants）。** 在贝叶斯推断中：
```
p(w | data) = p(data | w) * p(w) / integral of p(data | w) * p(w) dw
```
分母是对所有可能参数值的积分，通常难以计算，这就是为什么我们使用 MCMC 和变分推断等近似方法。

| 积分概念 | 在机器学习中的应用 |
|-----------------|----------------------|
| 曲线下面积 | 从密度函数得到概率 |
| 期望值 | 损失函数，风险最小化 |
| KL 散度 | VAE，策略优化，知识蒸馏 |
| 归一化 | 贝叶斯后验，softmax 分母 |
| 边际似然 | 模型比较，证据下界（ELBO） |

### 计算图中的多变量链式法则

链式法则不只适用于直线上的标量函数。在神经网络中，变量会分叉和合并。下面是导数如何在一个简单前向传播中流动：

```mermaid
graph LR
    x["x（输入）"] -->|"*w"| z1["z1 = w*x"]
    z1 -->|"+b"| z2["z2 = w*x + b"]
    z2 -->|"sigmoid"| a["a = sigmoid(z2)"]
    a -->|"损失函数"| L["L = -(y*log(a) + (1-y)*log(1-a))"]
```

反向传播从右到左计算梯度：

```mermaid
graph RL
    dL["dL/dL = 1"] -->|"dL/da"| da["dL/da = -y/a + (1-y)/(1-a)"]
    da -->|"da/dz2 = a(1-a)"| dz2["dL/dz2 = dL/da * a(1-a)"]
    dz2 -->|"dz2/dw = x"| dw["dL/dw = dL/dz2 * x"]
    dz2 -->|"dz2/db = 1"| db["dL/db = dL/dz2 * 1"]
```

每条箭头乘以局部导数。任意参数的梯度，等于从损失到该参数路径上所有局部导数的乘积。当路径分叉后合并时，将各分支的贡献求和（多变量链式法则）。

这就是反向传播的全部：在计算图中从输出到输入系统地应用链式法则。

### Jacobian 矩阵

当函数将向量映射到向量（如神经网络的一层）时，其导数是一个矩阵。Jacobian 矩阵（Jacobian matrix）包含每个输出对每个输入的所有偏导数。

对于 f: R^n -> R^m，Jacobian J 是 m × n 的矩阵：

| | x1 | x2 | ... | xn |
|---|---|---|---|---|
| f1 | df1/dx1 | df1/dx2 | ... | df1/dxn |
| f2 | df2/dx1 | df2/dx2 | ... | df2/dxn |
| ... | ... | ... | ... | ... |
| fm | dfm/dx1 | dfm/dx2 | ... | dfm/dxn |

你不需要手动计算神经网络的 Jacobian，PyTorch 会处理。但了解它的存在有助于理解反向传播中的形状：若一层将 R^n 映射到 R^m，其 Jacobian 为 m × n；梯度通过该矩阵的转置向后流动。

### 为何这对神经网络重要

神经网络中每个权重都有一个梯度，梯度告诉你如何调整该权重以减小损失。

```mermaid
graph LR
    subgraph Forward["前向传播"]
        I["输入"] --> W1["W1"] --> R["relu"] --> W2["W2"] --> S["softmax"] --> L["损失"]
    end
```

```mermaid
graph RL
    subgraph Backward["反向传播"]
        dL["dL/dloss"] --> dW2["dL/dW2"] --> d2["..."] --> dW1["dL/dW1"]
    end
```

每次权重更新：
- `W1 = W1 - lr * dL/dW1`
- `W2 = W2 - lr * dL/dW2`

前向传播计算预测值和损失，反向传播计算损失对每个权重的梯度，然后每个权重沿下坡方向小步移动。重复数百万次——这就是深度学习。

## 动手实现

### 第 1 步：从零实现数值导数

```python
def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)

def f(x):
    return x ** 2

for x in [-2, -1, 0, 1, 2]:
    numerical = numerical_derivative(f, x)
    analytical = 2 * x
    print(f"x={x:2d}  f'(x) numerical={numerical:.6f}  analytical={analytical:.1f}")
```

数值导数与解析导数在多位小数上吻合。

### 第 2 步：偏导数与梯度

```python
def numerical_gradient(f, point, h=1e-7):
    gradient = []
    for i in range(len(point)):
        point_plus = list(point)
        point_minus = list(point)
        point_plus[i] += h
        point_minus[i] -= h
        partial = (f(point_plus) - f(point_minus)) / (2 * h)
        gradient.append(partial)
    return gradient

def f_multi(point):
    x, y = point
    return x**2 + 3*x*y + y**2

grad = numerical_gradient(f_multi, [1.0, 2.0])
print(f"Numerical gradient at (1,2): {[f'{g:.4f}' for g in grad]}")
print(f"Analytical gradient at (1,2): [2*1+3*2, 3*1+2*2] = [{2*1+3*2}, {3*1+2*2}]")
```

### 第 3 步：用梯度下降求 f(x) = x^2 的最小值

```python
x = 5.0
lr = 0.1
for step in range(20):
    grad = 2 * x
    x = x - lr * grad
    print(f"step {step:2d}  x={x:8.4f}  f(x)={x**2:10.6f}")
```

从 x=5 出发，每步都更接近 x=0（最小值）。

### 第 4 步：二维函数上的梯度下降

```python
def f_2d(point):
    x, y = point
    return x**2 + y**2

point = [4.0, 3.0]
lr = 0.1
for step in range(30):
    grad = numerical_gradient(f_2d, point)
    point = [p - lr * g for p, g in zip(point, grad)]
    loss = f_2d(point)
    if step % 5 == 0 or step == 29:
        print(f"step {step:2d}  point=({point[0]:7.4f}, {point[1]:7.4f})  f={loss:.6f}")
```

### 第 5 步：数值导数与解析导数的对比

```python
import math

test_functions = [
    ("x^2",      lambda x: x**2,          lambda x: 2*x),
    ("x^3",      lambda x: x**3,          lambda x: 3*x**2),
    ("sin(x)",   lambda x: math.sin(x),   lambda x: math.cos(x)),
    ("e^x",      lambda x: math.exp(x),   lambda x: math.exp(x)),
    ("1/x",      lambda x: 1/x,           lambda x: -1/x**2),
]

x = 2.0
print(f"{'Function':<12} {'Numerical':>12} {'Analytical':>12} {'Error':>12}")
print("-" * 50)
for name, f, df in test_functions:
    num = numerical_derivative(f, x)
    ana = df(x)
    err = abs(num - ana)
    print(f"{name:<12} {num:12.6f} {ana:12.6f} {err:12.2e}")
```

### 第 6 步：数值计算 Hessian 矩阵

```python
def hessian_2d(f, x, y, h=1e-5):
    fxx = (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / (h ** 2)
    fyy = (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / (h ** 2)
    fxy = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ** 2)
    return [[fxx, fxy], [fxy, fyy]]

def saddle(x, y):
    return x ** 2 - y ** 2

def bowl(x, y):
    return x ** 2 + y ** 2

H_saddle = hessian_2d(saddle, 0.0, 0.0)
H_bowl = hessian_2d(bowl, 0.0, 0.0)
print(f"Saddle Hessian: {H_saddle}")  # [[2, 0], [0, -2]] -- mixed signs
print(f"Bowl Hessian:   {H_bowl}")    # [[2, 0], [0, 2]]  -- both positive
```

鞍点函数的 Hessian 特征值为 2 和 -2（符号相反，确认为鞍点），碗形函数的特征值为 2 和 2（均为正，确认为最小值）。

### 第 7 步：Taylor 近似实战

```python
import math

def taylor_approx(f, f_prime, f_double_prime, x0, h, order=2):
    result = f(x0)
    if order >= 1:
        result += f_prime(x0) * h
    if order >= 2:
        result += 0.5 * f_double_prime(x0) * h ** 2
    return result

x0 = 0.0
for h in [0.1, 0.5, 1.0, 2.0]:
    true_val = math.sin(h)
    t1 = taylor_approx(math.sin, math.cos, lambda x: -math.sin(x), x0, h, order=1)
    t2 = taylor_approx(math.sin, math.cos, lambda x: -math.sin(x), x0, h, order=2)
    print(f"h={h:.1f}  sin(h)={true_val:.4f}  order1={t1:.4f}  order2={t2:.4f}")
```

在 x0=0 附近，sin(x) ~ x（一阶 Taylor）。近似在较小的 h 时效果极好，但在较大的 h 时失效。这就是梯度下降在较小学习率下效果最好的原因——每步都假设线性近似是准确的。

### 第 8 步：为何这对神经网络重要

```python
import random

random.seed(42)

w = random.gauss(0, 1)
b = random.gauss(0, 1)
lr = 0.01

xs = [1.0, 2.0, 3.0, 4.0, 5.0]
ys = [3.0, 5.0, 7.0, 9.0, 11.0]

for epoch in range(200):
    total_loss = 0
    dw = 0
    db = 0
    for x, y in zip(xs, ys):
        pred = w * x + b
        error = pred - y
        total_loss += error ** 2
        dw += 2 * error * x
        db += 2 * error
    dw /= len(xs)
    db /= len(xs)
    total_loss /= len(xs)
    w -= lr * dw
    b -= lr * db
    if epoch % 40 == 0 or epoch == 199:
        print(f"epoch {epoch:3d}  w={w:.4f}  b={b:.4f}  loss={total_loss:.6f}")

print(f"\nLearned: y = {w:.2f}x + {b:.2f}")
print(f"Actual:  y = 2x + 1")
```

每个基于梯度的训练循环都遵循这一模式：预测、计算损失、计算梯度、更新权重。

## 实际应用

使用 NumPy，相同操作更快更简洁：

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([3, 5, 7, 9, 11], dtype=float)

w, b = np.random.randn(), np.random.randn()
lr = 0.01

for epoch in range(200):
    pred = w * x + b
    error = pred - y
    loss = np.mean(error ** 2)
    dw = np.mean(2 * error * x)
    db = np.mean(2 * error)
    w -= lr * dw
    b -= lr * db

print(f"Learned: y = {w:.2f}x + {b:.2f}")
```

你刚刚从零实现了梯度下降。PyTorch 自动化了梯度计算，但更新循环完全相同。

## 练习

1. 实现 `numerical_second_derivative(f, x)`，通过两次调用 `numerical_derivative` 实现。验证 x^3 在 x=2 处的二阶导数为 12。
2. 用梯度下降求 f(x, y) = (x - 3)^2 + (y + 1)^2 的最小值，从 (0, 0) 出发。答案应收敛至 (3, -1)。
3. 在梯度下降循环中加入动量（momentum）：维护一个累积过去梯度的速度向量。在 f(x) = x^4 - 3x^2 上比较有无动量的收敛速度。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| 导数（Derivative） | "斜率" | 函数在某点的变化率，告诉你输出随输入单位变化改变多少。 |
| 偏导数（Partial derivative） | "对某一变量的导数" | 将所有其他变量视为常数，对某一变量求导。 |
| 梯度（Gradient） | "最陡上坡方向" | 所有偏导数组成的向量，指向函数增长最快的方向。 |
| 梯度下降（Gradient descent） | "沿下坡走" | 从参数中减去梯度（乘以学习率），以减小损失。神经网络训练的核心。 |
| 学习率（Learning rate） | "步长" | 控制每次梯度下降步长的标量。太大会发散，太小则收敛缓慢。 |
| 链式法则（Chain rule） | "导数相乘" | 复合函数求导规则：df/dx = df/dg * dg/dx。反向传播的数学基础。 |
| Jacobian 矩阵（Jacobian） | "导数矩阵" | 当函数将向量映射到向量时，Jacobian 是所有输出对所有输入的偏导数矩阵。 |
| 数值导数（Numerical derivative） | "有限差分" | 通过在两个临近点评估函数值并计算斜率来近似导数。 |
| 反向传播（Backpropagation） | "反向自动微分" | 从输出到输入逐层用链式法则计算梯度，神经网络的学习方式。 |
| Hessian 矩阵（Hessian） | "二阶导数矩阵" | 所有二阶偏导数组成的矩阵，描述函数的曲率。临界点处正定 Hessian 意味着局部最小值。 |
| Taylor 级数（Taylor series） | "多项式近似" | 利用函数的导数在某点附近近似该函数：f(x+h) ~ f(x) + f'(x)h + (1/2)f''(x)h^2 + ...。理解梯度下降和牛顿法为何有效的基础。 |
| 积分（Integral） | "曲线下面积" | 某个量在某个区间上的累积。在机器学习中，积分定义了概率、期望值和 KL 散度。 |

## 延伸阅读

- [3Blue1Brown: Essence of Calculus](https://www.3blue1brown.com/topics/calculus) - 导数、积分和链式法则的视觉直觉
- [Stanford CS231n: Backpropagation](https://cs231n.github.io/optimization-2/) - 梯度如何流经神经网络层
