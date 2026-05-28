# 数值稳定性 (numerical stability)

> 浮点数 (floating point) 是一种会泄漏实现细节的抽象。它会在训练过程中坑到你，而你往往毫无预警。

**类型：** 构建
**语言：** Python
**先修内容：** 第 1 阶段，第 01-04 课
**时间：** ~120 分钟

## 学习目标

- 实现数值稳定的 softmax 和 log-sum-exp，并使用最大值减法技巧 (max-subtraction trick)
- 识别浮点计算中的溢出 (overflow)、下溢 (underflow) 和灾难性消除 (catastrophic cancellation)
- 使用中心有限差分 (centered finite differences) 将解析梯度与数值梯度对照验证
- 解释为什么训练时 bfloat16 比 float16 更合适，以及损失缩放 (loss scaling) 如何防止梯度下溢

## 问题

你的模型训练了三个小时，随后损失变成了 NaN。你加了一条 print 语句。第 9,000 步时 logits 还正常。到第 9,001 步，它们变成了 `inf`。到第 9,002 步，每个梯度都成了 `nan`，训练彻底死掉。

或者：你的模型训练顺利跑完了，但准确率比论文宣称的低 2%。你把一切都检查了一遍。架构一致。超参数一致。数据一致。问题在于论文用的是 float32，而你用的是 float16，却没有做正确的缩放。32 位累计舍入误差悄无声息地吞掉了你的准确率。

或者：你从零实现交叉熵损失 (cross-entropy loss)。在较小 logits 上它能工作。当 logits 超过 100 时，它返回 `inf`。softmax 溢出了，因为 `exp(100)` 比 float32 能表示的范围还大。每个 ML 框架都用一个两行的小技巧处理这个问题，而你之前并不知道这个技巧的存在。

数值稳定性不是理论层面的担忧。它决定了一次训练是成功完成，还是悄无声息地失败。你最终调试到的每一个严肃 ML bug，归根结底都与浮点数有关。

## 概念

### IEEE 754：计算机如何存储实数

计算机按照 IEEE 754 标准，以浮点值的形式存储实数。一个浮点数由三部分组成：符号位 (sign bit)、指数 (exponent) 和尾数 (mantissa，也称 significand)。

```
Float32 layout (32 bits total):
[1 sign] [8 exponent] [23 mantissa]

Value = (-1)^sign * 2^(exponent - 127) * 1.mantissa
```

尾数决定精度 (precision)，也就是能保留多少有效数字。指数决定范围 (range)，也就是数字能有多大或多小。

```
Format     Bits   Exponent  Mantissa  Decimal digits  Range (approx)
float64    64     11        52        ~15-16          +/- 1.8e308
float32    32     8         23        ~7-8            +/- 3.4e38
float16    16     5         10        ~3-4            +/- 65,504
bfloat16   16     8         7         ~2-3            +/- 3.4e38
```

float32 大约提供 7 位十进制精度。这意味着它能区分 1.0000001 和 1.0000002，却区分不了 1.00000001 和 1.00000002。超过 7 位之后，剩下的基本都是舍入噪声。

float16 大约只有 3 位精度。它能表示的最大值是 65,504。对于机器学习来说，这个上限小得惊人，因为 logits、梯度和激活值经常会超过它。

bfloat16 是 Google 对 float16 范围问题给出的答案。它和 float32 一样有 8 位指数（范围相同，最高到 3.4e38），但只有 7 位尾数（精度比 float16 更低）。在神经网络训练中，范围通常比精度更重要，所以 bfloat16 往往更胜一筹。

### 为什么 0.1 + 0.2 != 0.3

数字 0.1 无法在二进制浮点表示中被精确表示。在 2 进制里，它是一个循环小数：

```
0.1 in binary = 0.0001100110011001100110011... (repeating forever)
```

float32 会把它截断到 23 位尾数。存储后的值大约是 0.100000001490116。类似地，0.2 存储后大约是 0.200000002980232。两者相加得到 0.300000004470348，而不是 0.3。

```
In Python:
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

这在 ML 中很重要，因为：

1. 像 `if loss &lt; threshold` 这样的损失比较可能得到错误结果
2. 累加许多小值（例如数千步中的梯度更新）会逐渐偏离真实和
3. 如果你用 `==` 比较浮点数，校验和与可复现性测试都会失败

解决办法：永远不要用 `==` 比较浮点数。请使用 `abs(a - b) &lt; epsilon` 或 `math.isclose()`。

### 灾难性消除 (catastrophic cancellation)

当你相减两个几乎相等的浮点数时，有效数字会彼此抵消，最后留下的往往是被推到高位的舍入噪声。

```
a = 1.0000001    (stored as 1.00000011920929 in float32)
b = 1.0000000    (stored as 1.00000000000000 in float32)

True difference:  0.0000001
Computed:         0.00000011920929

Relative error: 19.2%
```

仅仅一次减法，就产生了 19% 的相对误差。在 ML 中，这种情况会在以下场景出现：

- 用大均值的数据计算方差：当 E[x] 很大时，使用 `E[x^2] - E[x]^2`
- 相减两个几乎相等的对数概率
- 用过小的 epsilon 计算有限差分梯度

解决办法：重排公式，避免相减两个很大且几乎相等的数。对于方差，可以使用 Welford 算法，或者先对数据做中心化。对于对数概率，请全程在对数空间中计算。

### 溢出 (overflow) 与下溢 (underflow)

当结果大到超出可表示范围时，就会发生溢出；当结果小到比最小可表示正数还更接近 0 时，就会发生下溢。

```
Float32 boundaries:
  Maximum:  3.4028235e+38
  Minimum positive (normal): 1.175e-38
  Minimum positive (denorm): 1.401e-45
  Overflow:  anything > 3.4e38 becomes inf
  Underflow: anything < 1.4e-45 becomes 0.0
```

在机器学习 (ML) 中，`exp()` 函数是溢出的主要来源：

```
exp(88.7)  = 3.40e+38   (barely fits in float32)
exp(89.0)  = inf         (overflow)
exp(-87.3) = 1.18e-38   (barely above underflow)
exp(-104)  = 0.0         (underflow to zero)
```

而 `log()` 函数的问题出现在另一端：

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (fine)
log(1e-46) = -inf        (input underflowed to 0, then log(0) = -inf)
```

在 ML 中，`exp()` 出现在 softmax、sigmoid 和各种概率计算里。`log()` 出现在交叉熵、对数似然和 KL 散度中。如果没有正确技巧，`log(exp(x))` 这种组合就是个雷区。

### Log-Sum-Exp 技巧 (log-sum-exp trick)

直接计算 `log(sum(exp(x_i)))` 在数值上很危险。如果某个 `x_i` 很大，`exp(x_i)` 会溢出。如果所有 `x_i` 都非常负，那么每个 `exp(x_i)` 都会下溢为 0，接着 `log(0)` 就变成 `-inf`。

技巧是：在取指数之前，先减去最大值。

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

其原理是：减去 `max(x)` 之后，最大的指数项就是 `exp(0) = 1`。因此不可能溢出。并且求和中至少有一项为 1，所以总和至少是 1，`log(1) = 0`。因此也不会下溢到 `-inf`。

证明：

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    (add and subtract c)
= log(sum(exp(x_i - c) * exp(c)))               (exp(a+b) = exp(a)*exp(b))
= log(exp(c) * sum(exp(x_i - c)))               (factor out exp(c))
= c + log(sum(exp(x_i - c)))                    (log(a*b) = log(a) + log(b))
```

令 `c = max(x)`，就消除了溢出。

这个技巧在 ML 中到处都会出现：
- Softmax 归一化
- 交叉熵损失计算
- 序列模型中的对数概率求和
- 高斯混合模型
- 变分推断

### 为什么 softmax 需要最大值减法技巧

softmax 会把 logits 转成概率：

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

如果不用这个技巧，logits 为 [100, 101, 102] 时就会溢出：

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

These overflow float32 (max ~3.4e38)? No, 2.69e43 < 3.4e38? Actually:
exp(88.7) is already at the float32 limit.
exp(100) = inf in float32.
```

使用这个技巧后，减去 max(x) = 102：

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

概率结果完全相同，计算却变得安全。这不是优化，而是保证正确性的必要条件。

### NaN 与 Inf：检测和预防

`nan`（非数字，Not a Number）和 `inf`（无穷大，infinity）会像病毒一样在计算中传播。梯度更新里只要出现一个 `nan`，对应权重就会变成 `nan`，随后所有输出都会变成 `nan`。训练往往一步之内就宣告死亡。

`inf` 的产生方式：
- 对很大的正数调用 `exp()`
- 除以 0：`1.0 / 0.0`
- 累加过程中发生 `float32` 溢出

`nan` 的产生方式：
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- 对负数调用 `sqrt()`
- 对负数调用 `log()`
- 任何涉及已有 `nan` 的算术运算

检测方法：

```python
import math

math.isnan(x)       # True if x is nan
math.isinf(x)       # True if x is +inf or -inf
math.isfinite(x)    # True if x is neither nan nor inf
```

预防策略：

1. 对 `exp()` 的输入做裁剪：`exp(clamp(x, -80, 80))`
2. 给分母加上 epsilon：`x / (y + 1e-8)`
3. 在 `log()` 内部加上 epsilon：`log(x + 1e-8)`
4. 使用稳定实现（log-sum-exp、稳定 softmax）
5. 使用梯度裁剪，防止权重爆炸
6. 调试时，在每次前向传播后检查 `nan`/`inf`

### 数值梯度检查 (numerical gradient checking)

解析梯度 (analytical gradients，来自反向传播 backpropagation) 可能写错。数值梯度检查会用有限差分 (finite differences) 计算梯度，从而验证它们。

中心差分公式：

```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

它的精度是 O(h^2)，远好于前向差分 `(f(x+h) - f(x)) / h` 的 O(h)。

如何选择 h：太大，近似就不准；太小，灾难性消除会把答案毁掉。常见取值是 `1e-5` 到 `1e-7`。

检查方法：计算解析梯度与数值梯度之间的相对差异。

```
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

经验法则：
- `relative_error &lt; 1e-7`：完美，梯度正确
- `relative_error &lt; 1e-5`：可以接受，很可能是正确的
- `relative_error > 1e-3`：哪里出了问题
- `relative_error > 1`：梯度完全错了

当你实现新的层或损失函数时，一定要做梯度检查。PyTorch 为此提供了 `torch.autograd.gradcheck()`。

### 混合精度训练 (mixed precision training)

现代 GPU 拥有专门的硬件（Tensor Cores），用 float16 做矩阵乘法的速度比 float32 快 2-8 倍。混合精度训练正是利用了这一点：

```
1. Maintain float32 master copy of weights
2. Forward pass in float16 (fast)
3. Compute loss in float32 (prevents overflow)
4. Backward pass in float16 (fast)
5. Scale gradients to float32
6. Update float32 master weights
```

纯 float16 训练的问题在于：梯度往往非常小（1e-8 甚至更小）。float16 会把约 6e-8 以下的值下溢为 0。于是模型会停止学习，因为所有梯度更新都变成了 0。

解决办法是损失缩放 (loss scaling)：

```
1. Multiply loss by a large scale factor (e.g., 1024)
2. Backward pass computes gradients of (loss * 1024)
3. All gradients are 1024x larger (pushed above float16 underflow)
4. Divide gradients by 1024 before updating weights
5. Net effect: same update, but no underflow
```

动态损失缩放会自动调整缩放因子。先从一个较大的值（65536）开始。如果梯度溢出成 `inf`，就把它减半；如果连续 N 步都没有溢出，就把它翻倍。

### bfloat16 vs float16：为什么 bfloat16 更适合训练

```
float16:   [1 sign] [5 exponent]  [10 mantissa]
bfloat16:  [1 sign] [8 exponent]  [7 mantissa]
```

float16 的精度更高（尾数 10 位，对比 bfloat16 的 7 位），但范围有限（最大约为 65,504）。bfloat16 的精度更低，但范围与 float32 相同（最大约为 3.4e38）。

对于神经网络训练：

- 激活值和 logits 在训练波动时经常会超过 65,504。float16 会溢出；bfloat16 可以承受。
- float16 需要损失缩放，而 bfloat16 通常不需要，因为它的范围覆盖了梯度大小的常见区间。
- bfloat16 本质上就是对 float32 的简单截断：去掉尾数最低的 16 位。转换非常简单，而且指数部分无损。

在推理 (inference) 场景中，数值通常有界，而且精度更重要，因此更常用 float16。在训练场景中，范围更重要，因此更偏向 bfloat16。这也是为什么 TPU 和现代 NVIDIA GPU（A100、H100）都原生支持 bfloat16。

### 梯度裁剪 (gradient clipping)

梯度爆炸 (exploding gradients) 会在梯度穿过很多层时呈指数级增长（常见于 RNN、深层网络和 transformer）。一次异常巨大的梯度就足以在一步内破坏所有权重。

两种裁剪方式：

**按值裁剪：** 独立地裁剪每个梯度元素。

```
grad = clamp(grad, -max_val, max_val)
```

简单，但会改变梯度向量的方向。

**按范数裁剪：** 缩放整个梯度向量，使其范数不超过阈值。

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

保留梯度方向。这正是 `torch.nn.utils.clip_grad_norm_()` 所做的，也是标准选择。

常见取值：transformer 用 `max_norm=1.0`，RL 用 `max_norm=0.5`，更简单的网络用 `max_norm=5.0`。

梯度裁剪不是 hack，而是一种安全机制。没有它，一个异常批次就可能产生足够大的梯度，把几周的训练全部毁掉。

### 归一化层作为数值稳定器

批归一化 (batch normalization)、层归一化 (layer normalization) 和 RMS 归一化 (RMS normalization) 通常被介绍为帮助训练收敛的正则化器 (regularizers)。但它们同时也是数值稳定器。

没有归一化时，激活值可能会在层与层之间指数级放大或缩小：

```
Layer 1: values in [0, 1]
Layer 5: values in [0, 100]
Layer 10: values in [0, 10,000]
Layer 50: values in [0, inf]
```

归一化会在每一层对激活值重新居中并缩放：

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

`epsilon`（通常为 1e-5）可以在所有激活值都相同的时候防止除零。可学习参数 `gamma` 和 `beta` 则让网络在需要时恢复任何所需的尺度。

这样可以让整个网络中的数值始终处于安全范围内，同时防止前向传播溢出以及反向传播中的梯度爆炸。

### 常见的 ML 数值 bug

**Bug：训练几个 epoch 后损失变成 NaN。**
原因：logits 变得过大，softmax 溢出了。或者学习率过高，权重发散了。
解决：使用稳定 softmax（最大值减法），降低学习率，并加入梯度裁剪。

**Bug：损失卡在 log(num_classes)。**
原因：模型输出接近均匀概率。这通常意味着梯度正在消失，或者模型根本没有学到任何东西。
解决：检查数据标签是否正确，验证损失函数，并检查是否存在死亡 ReLU。

**Bug：验证准确率比预期低 1-3%。**
原因：使用了混合精度，但没有正确进行损失缩放。梯度下溢会悄悄把小更新清零。
解决：启用动态损失缩放，或者改用 bfloat16。

**Bug：某些层的梯度范数是 0.0。**
原因：出现了死亡 ReLU 神经元（所有输入都为负），或者发生了 float16 下溢。
解决：使用 LeakyReLU 或 GELU，使用梯度缩放，并检查权重初始化。

**Bug：模型在一块 GPU 上正常，在另一块 GPU 上结果不同。**
原因：浮点累加顺序是非确定性的。GPU 并行归约会在不同硬件上以不同顺序求和，而浮点加法不满足结合律。
解决：接受微小差异（1e-6），或者设置 `torch.use_deterministic_algorithms(True)`，同时接受速度损失。

**Bug：损失计算中的 `exp()` 返回 `inf`。**
原因：原始 logits 直接传给了 `exp()`，没有使用最大值减法技巧。
解决：使用 `torch.nn.functional.log_softmax()`，它在内部实现了 log-sum-exp。

**Bug：从 float32 切换到 float16 后训练发散。**
原因：float16 无法表示小于 6e-8 的梯度量级，也无法表示大于 65,504 的激活值。
解决：使用带损失缩放的混合精度（AMP），或者直接改用 bfloat16。

## 动手实现

### 第 1 步：演示浮点精度极限

```python
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```

### 第 2 步：实现朴素版与稳定版 softmax

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) would return [nan, nan, nan]
```

### 第 3 步：实现稳定的 log-sum-exp

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"Stable: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) returns inf
```

### 第 4 步：实现稳定的交叉熵

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
```

### 第 5 步：梯度检查

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## 使用它

### 混合精度模拟

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### 梯度裁剪

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Original norm: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Clipped norm:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"Direction preserved: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### NaN/Inf 检测

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

完整实现以及所有边界情况的演示，请参见 `code/numerical.py`。

## 交付成果

本课将产出：
- `code/numerical.py`，包含稳定 softmax、log-sum-exp、交叉熵、梯度检查和混合精度模拟
- `outputs/prompt-numerical-debugger.md`，用于诊断训练中的 NaN/Inf 和其他数值问题

这些稳定实现会在第 3 阶段构建训练循环时再次出现，也会在第 4 阶段实现注意力机制时再次用到。

## 练习

1. **灾难性消除。** 用朴素公式 `E[x^2] - E[x]^2` 在 float32 中计算 [1000000.0, 1000001.0, 1000002.0] 的方差。然后用 Welford 在线算法再算一次。把两者误差与真实方差（0.6667）做比较。

2. **精度追踪。** 找出最小的正 float32 值 `x`，使得在 Python 中 `1.0 + x == 1.0`。这就是机器 epsilon (machine epsilon)。验证它与 `numpy.finfo(numpy.float32).eps` 一致。

3. **log-sum-exp 边界情况。** 用以下输入测试你的 `logsumexp_stable` 函数：(a) 所有值都相等，(b) 其中一个值远大于其余值，(c) 所有值都非常负（-1000）。验证在朴素版本失效的地方，它仍能给出正确结果。

4. **检查神经网络层的梯度。** 实现一个单层线性层 `y = Wx + b` 及其解析反向传播。使用 `numerical_gradient` 验证一个 3x2 权重矩阵上的正确性。

5. **损失缩放实验。** 模拟 float16 训练：创建取值范围在 [1e-9, 1e-3] 的随机梯度，将其转换为 float16，并测量其中有多少比例变成 0。然后应用损失缩放（乘以 1024），转换为 float16，再缩放回来，重新测量变成 0 的比例。

## 关键术语

| 术语 | 人们常说的话 | 它的实际含义 |
|------|----------------|----------------------|
| IEEE 754 | "浮点标准" | 定义二进制浮点格式、舍入规则和特殊值（inf、nan）的国际标准。所有现代 CPU 和 GPU 都实现了它。 |
| 机器 epsilon (machine epsilon) | "精度极限" | 在给定浮点格式中，使 1.0 + e != 1.0 的最小值 e。对于 float32，它大约是 1.19e-7。 |
| 灾难性消除 (catastrophic cancellation) | "减法导致的精度损失" | 当两个几乎相等的浮点数相减时，有效数字会相互抵消，结果会被舍入噪声主导。 |
| 溢出 (overflow) | "数太大了" | 结果超过最大可表示值并变成 inf。exp(89) 会让 float32 溢出。 |
| 下溢 (underflow) | "数太小了" | 结果比最小可表示正数更接近 0，因此变成 0.0。exp(-104) 会让 float32 下溢。 |
| Log-sum-exp 技巧 | "先减最大值" | 通过提出 exp(max(x)) 来计算 log(sum(exp(x)))，从而避免溢出和下溢。它被用于 softmax、交叉熵和对数概率计算。 |
| 稳定 softmax | "不会炸掉的 softmax" | 在取指数前先减去 max(logits)。数值结果完全相同，但不会溢出。 |
| 梯度检查 | "验证你的反向传播" | 将反向传播得到的解析梯度与有限差分得到的数值梯度进行比较，以捕捉实现 bug。 |
| 混合精度 | "前向用 float16，反向用 float32" | 在速度敏感的操作中使用较低精度浮点数，在数值敏感的操作中使用较高精度浮点数。典型加速约为 2-3 倍。 |
| 损失缩放 (loss scaling) | "防止梯度下溢" | 在反向传播前把损失乘以一个大常数，让梯度保持在 float16 的可表示范围内，然后在权重更新前再除以同一个常数。 |
| bfloat16 | "脑浮点" | Google 的 16 位格式，具有 8 位指数（与 float32 范围相同）和 7 位尾数（精度低于 float16）。它更适合训练。 |
| 梯度裁剪 | "给梯度范数封顶" | 缩放梯度向量，使其范数不超过某个阈值，从而防止梯度爆炸毁掉权重。 |
| NaN | "不是数字" | 来自未定义运算（0/0、inf-inf、sqrt(-1)）的特殊浮点值。它会在后续所有算术运算中传播。 |
| Inf | "无穷大" | 由溢出或除以零产生的特殊浮点值。它与其他值组合后还可能生成 NaN（inf - inf、inf * 0）。 |
| 数值梯度 | "暴力求导" | 通过计算 f(x+h) 和 f(x-h) 再除以 2h 来近似导数。虽然慢，但非常适合做验证。 |

## 延伸阅读

- [每位计算机科学家都应了解的浮点算术知识（Goldberg，1991）](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- 权威参考，内容密集但很完整
- [混合精度训练（Micikevicius 等，2018）](https://arxiv.org/abs/1710.03740) -- NVIDIA 提出的论文，首次为 float16 训练引入损失缩放
- [AMP：自动混合精度（PyTorch 文档）](https://pytorch.org/docs/stable/amp.html) -- PyTorch 中混合精度的实用指南
- [bfloat16 格式（Google Cloud TPU 文档）](https://cloud.google.com/tpu/docs/bfloat16) -- 为什么 Google 为 TPU 选择这种格式
- [Kahan 求和（Wikipedia）](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) -- 用于减少浮点求和舍入误差的算法
