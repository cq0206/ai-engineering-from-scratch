# 链式法则与自动微分 (Chain Rule & Automatic Differentiation)

> 链式法则是每个能学习的神经网络背后的引擎。

**类型：** 构建
**语言：** Python
**前置知识：** 第一阶段，第04课（导数与梯度）
**时间：** ~90 分钟

## 学习目标

- 构建一个最小化的自动求导引擎（Value 类），记录运算过程并通过反向模式自动微分（reverse-mode autodiff）计算梯度
- 利用拓扑排序（topological sort）在计算图（computational graph）上实现前向传播和反向传播
- 仅使用从头实现的自动求导引擎，构建并训练一个解决 XOR 问题的多层感知机（MLP，multi-layer perceptron）
- 通过与数值有限差分法对比，验证自动微分的正确性

## 问题所在

你已经能够计算简单函数的导数。但神经网络（neural network）不是简单函数，它是数百个函数的复合：矩阵乘法、加偏置、激活函数、再次矩阵乘法、softmax、交叉熵损失。输出是函数套函数套函数的结果。

要训练网络，你需要求损失对每个权重的梯度（gradient）。对数百万个参数手动推导是不可能的，数值方法（有限差分法）又太慢。

链式法则提供了数学工具，自动微分（automatic differentiation）提供了算法。两者合力，让你能在与单次前向传播相当的时间内，计算任意函数复合的精确梯度。

PyTorch、TensorFlow 和 JAX 就是这样工作的。你将从头构建一个迷你版本。

## 概念讲解

### 链式法则 (Chain Rule)

若 `y = f(g(x))`，则 `y` 关于 `x` 的导数为：

```
dy/dx = dy/dg * dg/dx = f'(g(x)) * g'(x)
```

沿链条相乘各局部导数，每一节贡献自己的局部导数。

示例：`y = sin(x^2)`

```
g(x) = x^2       g'(x) = 2x
f(g) = sin(g)     f'(g) = cos(g)

dy/dx = cos(x^2) * 2x
```

对于更深层的复合，链条继续延伸：

```
y = f(g(h(x)))

dy/dx = f'(g(h(x))) * g'(h(x)) * h'(x)
```

神经网络中的每一层都是这条链上的一节。

### 计算图 (Computational Graph)

计算图将链式法则可视化。每个运算成为一个节点，数据在图中前向流动，梯度则反向传播。

**前向传播（计算数值）：**

```mermaid
graph TD
    x1["x1 = 2"] --> mul["* (乘法)"]
    x2["x2 = 3"] --> mul
    mul -->|"a = 6"| add["+ (加法)"]
    b["b = 1"] --> add
    add -->|"c = 7"| relu["relu"]
    relu -->|"y = 7"| y["输出 y"]
```

**反向传播（计算梯度）：**

```mermaid
graph TD
    dy["dy/dy = 1"] -->|"relu'(c)=1 因为 c>0"| dc["dy/dc = 1"]
    dc -->|"dc/da = 1"| da["dy/da = 1"]
    dc -->|"dc/db = 1"| db["dy/db = 1"]
    da -->|"da/dx1 = x2 = 3"| dx1["dy/dx1 = 3"]
    da -->|"da/dx2 = x1 = 2"| dx2["dy/dx2 = 2"]
```

反向传播在每个节点应用链式法则，将梯度从输出传播到输入。

### 前向模式与反向模式 (Forward Mode vs Reverse Mode)

通过计算图应用链式法则有两种方式。

**前向模式（forward mode）**：从输入出发，向前推导数。从 `dx/dx = 1` 开始，逐步传播到每个运算。适用于输入少、输出多的情况。

```
Forward mode: seed dx/dx = 1, propagate forward

  x = 2       (dx/dx = 1)
  a = x^2     (da/dx = 2x = 4)
  y = sin(a)  (dy/dx = cos(a) * da/dx = cos(4) * 4 = -2.615)
```

**反向模式（reverse mode）**：从输出出发，向后拉取梯度。从 `dy/dy = 1` 开始，逆序传播到每个运算。适用于输入多、输出少的情况。

```
Reverse mode: seed dy/dy = 1, propagate backward

  y = sin(a)  (dy/dy = 1)
  a = x^2     (dy/da = cos(a) = cos(4) = -0.654)
  x = 2       (dy/dx = dy/da * da/dx = -0.654 * 4 = -2.615)
```

神经网络有数百万个输入（权重）和一个输出（损失）。反向模式只需一次反向传播就能计算所有梯度，这正是反向传播（backpropagation）使用反向模式的原因。

| 模式 | 种子 | 方向 | 适用场景 |
|------|------|-----------|-----------|
| 前向模式 | `dx_i/dx_i = 1` | 输入到输出 | 输入少，输出多 |
| 反向模式 | `dy/dy = 1` | 输出到输入 | 输入多，输出少（神经网络） |

### 双重数实现前向模式 (Dual Numbers for Forward Mode)

前向模式可以用双重数（dual numbers）优雅地实现。双重数具有 `a + b*epsilon` 的形式，其中 `epsilon^2 = 0`。

```
Dual number: (value, derivative)

(2, 1) means: value is 2, derivative w.r.t. x is 1

Arithmetic rules:
  (a, a') + (b, b') = (a+b, a'+b')
  (a, a') * (b, b') = (a*b, a'*b + a*b')
  sin(a, a')         = (sin(a), cos(a)*a')
```

将输入变量的导数种子设为 1，导数会自动通过每个运算传播。

### 构建自动求导引擎 (Building an Autograd Engine)

自动求导引擎需要三件事：

1. **值的封装。** 将每个数字封装在一个对象中，存储其数值和梯度。
2. **图的记录。** 每个运算记录其输入和局部梯度函数。
3. **反向传播。** 对图进行拓扑排序，再逆序遍历，在每个节点应用链式法则。

这正是 PyTorch 的 `autograd` 所做的事情。`torch.Tensor` 类封装数值，在 `requires_grad=True` 时记录运算，调用 `.backward()` 时计算梯度。

### PyTorch 自动求导的底层原理 (How PyTorch Autograd Works Under the Hood)

当你编写 PyTorch 代码时：

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1
y.backward()
print(x.grad)  # 7.0 = 2*x + 3 = 2*2 + 3
```

PyTorch 内部会：

1. 为 `x` 创建一个 `requires_grad=True` 的 `Tensor` 节点
2. 每个运算（`**`、`*`、`+`）创建一个新节点并记录反向传播函数
3. `y.backward()` 通过已记录的图触发反向模式自动微分
4. 每个节点的 `grad_fn` 计算局部梯度并传递给父节点
5. 梯度通过加法（而非替换）累积在 `.grad` 属性中

计算图是动态的（按运行定义）。每次前向传播都会重新构建新图，这也是 PyTorch 支持在模型中使用 Python 控制流（if/else、循环）的原因。

## 动手实现

### 第一步：Value 类

```python
class Value:
    def __init__(self, data, children=(), op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(children)
        self._op = op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
```

每个 `Value` 存储其数值、梯度（初始为零）、反向函数，以及指向产生它的子节点的指针。

### 第二步：带梯度追踪的算术运算

```python
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'relu')
        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out
```

每个运算创建一个闭包，知道如何计算局部梯度并与上游梯度（`out.grad`）相乘。`+=` 处理一个值被多个运算使用的情况。

### 第三步：反向传播

```python
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
```

拓扑排序确保在梯度向子节点传播之前，每个节点的梯度已完全计算。种子梯度为 1.0（dy/dy = 1）。

### 第四步：更多运算，构建完整引擎

基础 Value 类处理加法、乘法和 relu。真正的自动求导引擎需要更多运算。以下是构建神经网络所需的运算：

```python
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return other + (-self)

    def __pow__(self, n):
        out = Value(self.data ** n, (self,), f'**{n}')
        def _backward():
            self.grad += n * (self.data ** (n - 1)) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * (other ** -1) if isinstance(other, Value) else self * (Value(other) ** -1)

    def exp(self):
        import math
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')
        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    def log(self):
        import math
        out = Value(math.log(self.data), (self,), 'log')
        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        import math
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out
```

**各运算的意义：**

| 运算 | 反向传播规则 | 用途 |
|-----------|--------------|---------|
| `__sub__` | 复用加法 + 取负 | 损失计算（预测值 - 目标值） |
| `__pow__` | n * x^(n-1) | 多项式激活、MSE（误差²） |
| `__truediv__` | 复用乘法 + pow(-1) | 归一化、学习率缩放 |
| `exp` | exp(x) * 上游梯度 | Softmax、对数似然 |
| `log` | (1/x) * 上游梯度 | 交叉熵损失、对数概率 |
| `tanh` | (1 - tanh²) * 上游梯度 | 经典激活函数 |

精妙之处在于：`__sub__` 和 `__truediv__` 是用已有运算定义的，它们通过底层的加/乘/幂运算的链式法则自动获得正确梯度。

### 第五步：从头构建迷你 MLP

有了完整的 Value 类，你可以构建神经网络。不用 PyTorch，不用 NumPy，只用 Value 和链式法则。

```python
import random

class Neuron:
    def __init__(self, n_inputs):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(0.0)

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, n_inputs, n_outputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, sizes):
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

`Neuron` 计算 `tanh(w1*x1 + w2*x2 + ... + b)`，`Layer` 是神经元列表，`MLP` 将各层堆叠。每个权重都是 `Value`，调用 `loss.backward()` 即可将梯度传播到每个参数。

**在 XOR 上训练：**

```python
random.seed(42)
model = MLP([2, 4, 1])  # 2 inputs, 4 hidden neurons, 1 output

xs = [[0, 0], [0, 1], [1, 0], [1, 1]]
ys = [-1, 1, 1, -1]  # XOR pattern (using -1/1 for tanh)

for step in range(100):
    preds = [model(x) for x in xs]
    loss = sum((p - y) ** 2 for p, y in zip(preds, ys))

    for p in model.parameters():
        p.grad = 0.0
    loss.backward()

    lr = 0.05
    for p in model.parameters():
        p.data -= lr * p.grad

    if step % 20 == 0:
        print(f"step {step:3d}  loss = {loss.data:.4f}")

print("\nPredictions after training:")
for x, y in zip(xs, ys):
    print(f"  input={x}  target={y:2d}  pred={model(x).data:6.3f}")
```

这就是 micrograd——用纯 Python 实现的完整神经网络训练循环，具备自动微分能力。所有商业深度学习框架都在大规模地做同样的事情。

### 第六步：梯度检验 (Gradient Checking)

如何确认自动微分的正确性？将其与数值导数进行比较，这就是梯度检验（gradient checking）。

```python
def gradient_check(build_expr, x_val, h=1e-7):
    x = Value(x_val)
    y = build_expr(x)
    y.backward()
    autodiff_grad = x.grad

    y_plus = build_expr(Value(x_val + h)).data
    y_minus = build_expr(Value(x_val - h)).data
    numerical_grad = (y_plus - y_minus) / (2 * h)

    diff = abs(autodiff_grad - numerical_grad)
    return autodiff_grad, numerical_grad, diff
```

在复杂表达式上测试：

```python
def expr(x):
    return (x ** 3 + x * 2 + 1).tanh()

ad, num, diff = gradient_check(expr, 0.5)
print(f"Autodiff:  {ad:.8f}")
print(f"Numerical: {num:.8f}")
print(f"Difference: {diff:.2e}")
# Difference should be < 1e-5
```

在实现新运算时，梯度检验至关重要。若反向传播存在 bug，数值检验能及时发现。每个严肃的深度学习实现在开发阶段都会运行梯度检验。

**何时使用梯度检验：**

| 场景 | 是否做梯度检验？ |
|-----------|-------------------|
| 向自动求导引擎添加新运算 | 是，始终如此 |
| 调试无法收敛的训练循环 | 是，先检查梯度 |
| 生产训练 | 否，太慢（每个参数需 2 次前向传播） |
| 自动求导代码的单元测试 | 是，自动化执行 |

### 第七步：与手动计算对比验证

```python
x1 = Value(2.0)
x2 = Value(3.0)
a = x1 * x2          # a = 6.0
b = a + Value(1.0)    # b = 7.0
y = b.relu()          # y = 7.0

y.backward()

print(f"y = {y.data}")          # 7.0
print(f"dy/dx1 = {x1.grad}")   # 3.0 (= x2)
print(f"dy/dx2 = {x2.grad}")   # 2.0 (= x1)
```

手动验证：`y = relu(x1*x2 + 1)`。由于 `x1*x2 + 1 = 7 > 0`，relu 为恒等函数。`dy/dx1 = x2 = 3`，`dy/dx2 = x1 = 2`。引擎结果吻合。

## 实际使用

### 与 PyTorch 对比验证

```python
import torch

x1 = torch.tensor(2.0, requires_grad=True)
x2 = torch.tensor(3.0, requires_grad=True)
a = x1 * x2
b = a + 1.0
y = torch.relu(b)
y.backward()

print(f"PyTorch dy/dx1 = {x1.grad.item()}")  # 3.0
print(f"PyTorch dy/dx2 = {x2.grad.item()}")  # 2.0
```

梯度相同。你的引擎与 PyTorch 计算出相同的结果，因为数学是一样的：通过链式法则进行的反向模式自动微分。

### 更复杂的表达式

```python
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
f = (a * b + c).relu()  # relu(2*(-3) + 10) = relu(4) = 4

f.backward()
print(f"df/da = {a.grad}")  # -3.0 (= b)
print(f"df/db = {b.grad}")  #  2.0 (= a)
print(f"df/dc = {c.grad}")  #  1.0
```

## 交付成果

本课产出：
- `outputs/skill-autodiff.md` —— 构建和调试自动求导系统的技能说明
- `code/autodiff.py` —— 可扩展的最小化自动求导引擎

此处构建的 Value 类是第三阶段神经网络训练循环的基础。

## 练习

1. 向 Value 类添加 `__pow__`，使其能计算 `x ** n`。验证 `d/dx(x^3)` 在 `x=2` 处等于 `12.0`。

2. 将 `tanh` 添加为激活函数。验证 `tanh'(0) = 1` 且 `tanh'(2) = 0.0707`（近似值）。

3. 为单个神经元构建计算图：`y = relu(w1*x1 + w2*x2 + b)`。计算全部五个梯度，并与 PyTorch 结果对比。

4. 使用双重数实现前向模式自动微分。创建一个 `Dual` 类，验证其与反向模式引擎给出相同的导数。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| 链式法则 (chain rule) | "相乘各导数" | 复合函数的导数等于每个函数局部导数在相应点处的乘积 |
| 计算图 (computational graph) | "网络示意图" | 有向无环图，节点为运算，边传递数值（前向）或梯度（反向） |
| 前向模式 (forward mode) | "向前推导数" | 从输入向输出传播导数的自动微分，每个输入变量需一次传播 |
| 反向模式 (reverse mode) | "反向传播" | 从输出向输入传播梯度的自动微分，每个输出变量需一次传播 |
| 自动求导 (autograd) | "自动梯度" | 记录值上的运算、构建计算图并通过链式法则计算精确梯度的系统 |
| 双重数 (dual numbers) | "值加导数" | 形如 a + b*epsilon（epsilon^2 = 0）的数，在算术运算中携带导数信息 |
| 拓扑排序 (topological sort) | "依赖顺序" | 对图节点排序，使每个节点都在其所有依赖之后出现，梯度正确传播的必要条件 |
| 梯度累积 (gradient accumulation) | "相加而非替换" | 当一个值被多个运算使用时，其梯度是所有流入梯度贡献的总和 |
| 动态图 (dynamic graph) | "按运行定义" | 每次前向传播都重新构建的计算图，允许在模型内部使用 Python 控制流（PyTorch 风格） |
| 梯度检验 (gradient checking) | "数值验证" | 将自动微分梯度与数值有限差分梯度对比，验证正确性，调试必备工具 |
| MLP（多层感知机）(multi-layer perceptron) | "多层感知机" | 包含一个或多个隐藏层的神经网络，每个神经元计算加权和加偏置后应用激活函数 |
| 神经元 (neuron) | "加权求和 + 激活" | 基本单元：输出 = 激活函数(w1*x1 + w2*x2 + ... + b)，权重和偏置为可学习参数 |

## 延伸阅读

- [3Blue1Brown：反向传播微积分](https://www.youtube.com/watch?v=tIeHLnjs5U8) —— 神经网络中链式法则的可视化解释
- [PyTorch Autograd 机制](https://pytorch.org/docs/stable/notes/autograd.html) —— 真实系统的工作原理
- [Baydin 等人，机器学习中的自动微分综述](https://arxiv.org/abs/1502.05767) —— 全面的参考资料
