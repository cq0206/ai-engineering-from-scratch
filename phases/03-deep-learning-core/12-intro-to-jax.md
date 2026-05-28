# JAX 入门

> PyTorch 会原地修改张量。TensorFlow 构建计算图。JAX 编译纯函数。最后这一点，会改变你对深度学习的思考方式。

**类型：** 构建
**语言：** Python
**前置知识：** 第03阶段第01-10课，NumPy 基础
**时长：** 约90分钟

## 学习目标

- 使用 JAX 的函数式 API（jax.numpy、jax.grad、jax.jit、jax.vmap）编写纯函数神经网络代码
- 解释 PyTorch 的即时可变（eager mutation）模式与 JAX 的函数式编译（functional compilation）模型之间的核心设计差异
- 应用 jit 编译和 vmap 向量化，相比朴素 Python 加速训练循环
- 在 JAX 中训练一个简单网络，并将显式状态管理与 PyTorch 的面向对象方式进行对比

## 问题背景

你已经知道如何在 PyTorch 中构建神经网络：定义一个 `nn.Module`，调用 `.backward()`，执行优化器步进。它能工作，数百万人都在使用它。

但 PyTorch 的 DNA 中刻有一个约束：它以即时模式（eagerly）逐个追踪操作，在 Python 中一次处理一个。每个 `tensor + tensor` 都是一次独立的内核（kernel）启动，每个训练步骤都要重新解释同样的 Python 代码。这在大多数情况下没问题，直到你需要跨2048块 TPU 训练一个5400亿参数的模型——那时开销会把你压垮。

Google DeepMind 用 JAX 训练 Gemini，Anthropic 用 JAX 训练 Claude。这些都不是小操作——它们是地球上规模最大的神经网络训练任务。他们选择 JAX，是因为 JAX 将你的训练循环视为一个可编译的程序，而非一系列 Python 调用。

JAX 是具备三项超能力的 NumPy：自动微分（automatic differentiation）、JIT 编译到 XLA，以及自动向量化。你写一个处理单个样本的函数，JAX 就能给你一个处理整个批次、计算梯度、编译为机器码、并在多设备上运行的函数——所有这些都不需要修改原始函数。

## 核心概念

### JAX 的设计理念

JAX 是一个函数式（functional）框架。没有类，没有可变状态，没有 `.backward()` 方法。取而代之：

| PyTorch | JAX |
|---------|-----|
| 带状态的 `nn.Module` 类 | 纯函数：`f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| 即时执行 | 通过 XLA 进行 JIT 编译 |
| `for x in batch:` 手动循环 | `jax.vmap(f)` 自动向量化 |
| `DataParallel` / `FSDP` | `jax.pmap(f)` 自动并行 |
| 可变的 `model.parameters()` | 不可变的数组 pytree |

这不是风格偏好，而是编译器约束。JIT 编译要求纯函数——相同的输入始终产生相同的输出，没有副作用。正是这个限制，让100倍的加速成为可能。

### jax.numpy：熟悉的接口

JAX 在加速器上重新实现了 NumPy API：

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

相同的函数名，相同的广播规则，相同的切片语义。但数组存在于 GPU/TPU 上，每个操作都可被编译器追踪。

一个关键区别：JAX 数组是不可变的。没有 `a[0] = 5`。取而代之：`a = a.at[0].set(5)`。这在刚开始时会感到别扭，但一周后就会豁然开朗——不可变性正是让 `grad`、`jit`、`vmap` 等变换可以自由组合的原因。

### jax.grad：函数式自动微分

PyTorch 将梯度附加到张量（`.grad`），JAX 将梯度附加到函数。

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad` 接受一个函数，返回一个计算梯度的新函数。不需要调用 `.backward()`，不需要在张量上存储计算图。梯度只是另一个可以调用、组合或 JIT 编译的函数。

它可以任意组合：

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

二阶导数、三阶导数、雅可比矩阵（Jacobian）、黑塞矩阵（Hessian）——全部通过组合 `grad` 实现。PyTorch 也能做到（`torch.autograd.functional.hessian`），但那是后加上去的功能。在 JAX 中，这是基础。

约束：`grad` 只对纯函数有效。函数内部不能有 print 语句（它们在追踪时执行，而非运行时）。不能有对外部状态的修改，不能有不显式管理随机数密钥的随机数生成。

### jit：编译到 XLA

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

第一次调用时，JAX 会追踪该函数——它记录发生了哪些操作，但不实际执行。然后将该追踪结果交给 XLA（Accelerated Linear Algebra，加速线性代数），即 Google 面向 TPU 和 GPU 的编译器。XLA 会融合操作，消除多余的内存拷贝，并生成优化后的机器码。

后续调用会完全跳过 Python，编译后的代码以 C++ 速度在加速器上运行。

JIT 有帮助的场景：
- 训练步骤（相同计算重复数千次）
- 推理（相同模型，不同输入）
- 任何以相似形状输入多次调用的函数

JIT 有害的场景：
- 包含依赖值的 Python 控制流（`if x > 0`，其中 x 是被追踪的数组）
- 一次性计算（编译开销超过运行时间）
- 调试（追踪隐藏了实际执行过程）

控制流限制是真实存在的。`jax.lax.cond` 替代 `if/else`，`jax.lax.scan` 替代 `for` 循环。这不是可选的——这是编译的代价。

### vmap：自动向量化

你写一个处理单个样本的函数：

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap` 将其提升为处理整个批次：

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)` 的含义：不对 `params` 进行批处理（共享），对 `x` 的轴0进行批处理。无需手动 `for` 循环，无需重塑，无需在批次维度上穿针引线。JAX 会自行推断批次维度并对整个计算进行向量化。

这不是语法糖。`vmap` 生成融合的向量化代码，比 Python 循环快10到100倍。它还可以与 `jit` 和 `grad` 组合：

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

逐样本梯度，一行代码。这在 PyTorch 中几乎不可能不用 hack 实现。

### pmap：跨设备数据并行

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap` 将函数复制到所有可用设备（GPU/TPU）并拆分批次。在函数内部，`jax.lax.pmean` 和 `jax.lax.psum` 负责跨设备同步梯度。

Google 使用 `pmap`（及其继任者 `shard_map`）在数千块 TPU v5e 芯片上训练 Gemini。编程模型：写单设备版本，用 `pmap` 包装，完成。

### Pytree：通用数据结构

JAX 在"pytree"上操作——由列表、元组、字典和数组的嵌套组合构成的数据结构。你的模型参数就是一个 pytree：

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

每个 JAX 变换——`grad`、`jit`、`vmap`——都知道如何遍历 pytree。`jax.tree.map(f, tree)` 将 `f` 应用于每个叶节点。优化器就是这样一次性更新所有参数的：

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

没有 `.parameters()` 方法，没有参数注册，树结构即是模型。

### 函数式 vs 面向对象

PyTorch 将状态存储在对象内部：

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX 使用带有显式状态的纯函数：

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

参数以参数形式传入，没有任何内容被存储，也没有任何内容被修改。这使每个函数都可测试、可组合、可编译。同时也意味着你需要自己管理参数——或者使用 Flax、Equinox 等库。

### JAX 生态系统

JAX 提供基础组件，各库提供人机工程学：

| 库 | 定位 | 风格 |
|---|------|------|
| **Flax**（Google） | 神经网络层 | 带显式状态的 `nn.Module` |
| **Equinox**（Patrick Kidger） | 神经网络层 | 基于 pytree，Pythonic |
| **Optax**（DeepMind） | 优化器 + 学习率调度 | 可组合的梯度变换 |
| **Orbax**（Google） | 检查点（Checkpointing） | 保存/恢复 pytree |
| **CLU**（Google） | 指标 + 日志 | 训练循环工具 |

Optax 是标准优化器库。它将梯度变换（Adam、SGD、梯度裁剪）与参数更新解耦，使组合变得轻而易举：

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### 何时使用 JAX vs PyTorch

| 因素 | JAX | PyTorch |
|------|-----|---------|
| TPU 支持 | 一流（Google 同时开发了两者） | 社区维护（torch_xla） |
| GPU 支持 | 良好（通过 XLA 使用 CUDA） | 最佳（原生 CUDA） |
| 调试 | 困难（追踪 + 编译） | 简单（即时，逐行） |
| 生态系统 | 研究导向（Flax, Equinox） | 庞大（HuggingFace, torchvision 等） |
| 招聘 | 小众（Google/DeepMind/Anthropic） | 主流（随处可见） |
| 大规模训练 | 卓越（XLA, pmap, mesh） | 良好（FSDP, DeepSpeed） |
| 原型开发速度 | 较慢（函数式开销） | 较快（可变即走） |
| 生产推理 | TensorFlow Serving, Vertex AI | TorchServe, Triton, ONNX |
| 使用者 | DeepMind（Gemini），Anthropic（Claude） | Meta（Llama），OpenAI（GPT），Stability AI |

诚实的答案：除非有特定理由，否则使用 PyTorch。这些理由包括——访问 TPU、需要逐样本梯度、超大规模多设备训练，或者你在 Google/DeepMind/Anthropic 工作。

### JAX 中的随机数

JAX 没有全局随机状态。每个随机操作都需要一个显式的 PRNG 密钥（key）：

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

一开始这会让人烦恼。但它保证了跨设备和编译的可重复性——这是 PyTorch 的 `torch.manual_seed` 在多 GPU 场景下无法保证的。

## 动手构建

### 第一步：准备环境与数据

我们将使用 JAX 和 Optax 在 MNIST 上训练一个3层 MLP（多层感知机）：784个输入，两个隐藏层分别有256和128个神经元，10个输出类别。

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### 第二步：初始化参数

没有类，只有一个返回 pytree 的函数：

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

手动进行 He 初始化（He-initialization）。从一个种子拆分出三个 PRNG 密钥，每个权重都是嵌套字典中的一个不可变数组。

### 第三步：前向传播

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

纯函数，参数进、预测出，没有 `self`，没有存储状态。`loss_fn` 从零开始计算交叉熵——softmax、取对数、取负均值。

### 第四步：JIT 编译的训练步骤

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad` 在一次传播中同时返回损失值和梯度。`@jax.jit` 装饰器将两个函数都编译到 XLA。第一次调用后，每个训练步骤都不再接触 Python。

### 第五步：训练循环

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

训练10个轮次，测试准确率约97%。第一个轮次较慢（JIT 编译），第2至10轮较快。

注意缺失的内容：没有 `.zero_grad()`，没有 `.backward()`，没有 `.step()`。整个更新是一次组合的函数调用。梯度的计算、Adam 变换和参数应用——全部在 `train_step` 内完成。

## 使用示例

### Flax：Google 标准库

Flax 是最常用的 JAX 神经网络库，它重新引入了 `nn.Module`，但带有显式状态管理：

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

结构与 PyTorch 相同，但 `params` 与模型分离。`model.init()` 创建参数，`model.apply(params, x)` 执行前向传播。模型对象不持有状态。

### Equinox：Pythonic 的替代方案

Equinox（由 Patrick Kidger 开发）将模型表示为 pytree：

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

模型本身就是一个 pytree，不需要 `.apply()`，参数就是模型的叶节点。这更接近 JAX 的思维方式。

### Optax：可组合的优化器

Optax 将梯度变换与更新解耦：

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

梯度裁剪、学习率预热（warmup）、权重衰减——全部以变换链的形式组合。每个变换接收梯度、修改它，然后传给下一个。没有单体优化器类。

## 部署上线

**安装：**

```bash
pip install jax jaxlib optax flax
```

GPU 支持：

```bash
pip install jax[cuda12]
```

TPU（Google Cloud）：

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**性能注意事项：**

- 第一次 JIT 调用较慢（编译阶段）。基准测试前需要先预热。
- 避免在 JIT 内部对 JAX 数组使用 Python 循环，改用 `jax.lax.scan` 或 `jax.lax.fori_loop`。
- `jax.debug.print()` 在 JIT 内部有效，普通 `print()` 无效。
- 用 `jax.profiler` 或 TensorBoard 进行性能分析，XLA 编译可能隐藏瓶颈。
- JAX 默认预分配75%的 GPU 内存，设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 可禁用。

**检查点（Checkpointing）：**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**本课产出：**
- `outputs/prompt-jax-optimizer.md` — 用于选择正确 JAX 优化器配置的提示词
- `outputs/skill-jax-patterns.md` — 涵盖 JAX 函数式模式的技能文档

## 练习

1. 在 MLP 中添加 Dropout。在 JAX 中，Dropout 需要 PRNG 密钥——将密钥穿过前向传播，并为每个 Dropout 层拆分密钥。对比加 Dropout 和不加 Dropout 的测试准确率。

2. 使用 `jax.vmap` 计算32张 MNIST 图像的逐样本梯度（per-example gradients）。计算每个样本的梯度范数。哪些样本的梯度最大，为什么？

3. 将手动 forward 函数替换为通用的 `mlp_forward(params, x)`，使其适用于任意层数。使用 `jax.tree.leaves` 自动确定深度。

4. 对带和不带 `@jax.jit` 的训练步骤分别计时，各运行100步。在你的硬件上加速比是多少？第一次调用的编译开销是多少？

5. 通过组合 `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))` 实现梯度裁剪。分别使用带裁剪和不带裁剪的方式训练，绘制训练过程中的梯度范数曲线，观察效果。

## 关键术语

| 术语 | 通常的说法 | 实际含义 |
|------|-----------|---------|
| XLA | "让 JAX 变快的东西" | Accelerated Linear Algebra——一个编译器，能融合操作并从计算图中为 GPU/TPU 生成优化内核 |
| JIT | "即时编译" | JAX 在第一次调用时追踪函数，编译到 XLA，然后在后续调用时运行编译后的版本 |
| 纯函数（Pure function） | "无副作用" | 输出仅取决于输入的函数——没有全局状态，没有可变操作，没有不显式使用密钥的随机数 |
| vmap | "自动批处理" | 将处理单个样本的函数变换为处理整个批次的函数，无需重写 |
| pmap | "自动并行" | 将函数复制到多个设备并拆分输入批次 |
| Pytree | "数组的嵌套字典" | 列表、元组、字典和数组的任意嵌套结构，JAX 可以遍历和变换它 |
| 追踪（Tracing） | "记录计算过程" | JAX 用抽象值执行函数以构建计算图，而不计算真实结果 |
| 函数式自动微分（Functional autodiff） | "对函数求梯度" | 通过变换函数来计算导数，而非将梯度存储附加到张量上 |
| Optax | "JAX 的优化器库" | 一个可组合的梯度变换库——Adam、SGD、梯度裁剪、学习率调度——可以链式组合 |
| Flax | "JAX 的 nn.Module" | Google 为 JAX 开发的神经网络库，增加了层抽象，同时保持状态的显式性 |

## 延伸阅读

- JAX 文档：https://jax.readthedocs.io/ — 官方文档，包含关于 grad、jit 和 vmap 的优秀教程
- "JAX: composable transformations of Python+NumPy programs"（Bradbury 等，2018）——解释设计理念的原始论文
- Flax 文档：https://flax.readthedocs.io/ — Google 为 JAX 开发的神经网络库
- Patrick Kidger，"Equinox: neural networks in JAX via callable PyTrees and filtered transformations"（2021）——Flax 的 Pythonic 替代方案
- DeepMind，"Optax: composable gradient transformation and optimisation"——标准优化器库
- "You Don't Know JAX"（Colin Raffel，2020）——JAX 坑点与模式的实用指南，来自 T5 论文的作者之一
