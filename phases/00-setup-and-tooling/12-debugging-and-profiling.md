# 调试与性能分析

> 最糟糕的 AI bug 不会崩溃。它们会悄悄在垃圾数据上训练，还回报一条漂亮的 loss 曲线。

**类型：** 实践
**语言：** Python
**前置要求：** 第 1 课（开发环境），具备基础 PyTorch 知识
**时长：** ~60 分钟

## 学习目标

- 使用条件式 `breakpoint()` 和 `debug_print`，在训练过程中检查张量（tensor）的形状、dtype 和 NaN 值
- 使用 `cProfile`、`line_profiler` 和 `tracemalloc` 对训练循环做性能分析（profiling），找出瓶颈
- 检测常见 AI bug：形状不匹配、NaN loss、数据泄漏和设备错误的张量
- 配置 TensorBoard，可视化 loss 曲线、权重直方图和梯度分布

## 问题

AI 代码的失败方式和普通代码不同。Web 应用会带着堆栈追踪直接崩掉。一个配置错误的训练循环会跑 8 个小时，烧掉 200 美元的 GPU 时间，还产出一个对所有输入都只会预测均值的模型。代码从未报错。bug 可能只是某个张量在错误的设备上、忘记写 `.detach()`，或者标签泄漏进了特征。

你需要能在浪费时间和算力之前，就捕获这些无声失败的调试工具。

## 概念

AI 调试分为三个层次：

```mermaid
graph TD
    L3["3. 训练动态<br/>Loss 曲线、梯度范数、激活值"] --> L2
    L2["2. 张量操作<br/>形状、dtype、设备、NaN/Inf 值"] --> L1
    L1["1. 标准 Python<br/>断点、日志、性能分析、内存"]
```

大多数人会直接跳到第 3 层（盯着 TensorBoard 发呆）。但 80% 的 AI bug 都藏在第 1 层和第 2 层。

## 动手实践

### 第 1 部分：打印调试（没错，它真的有用）

打印调试常常被低估，但它不该如此。对于张量代码，一个有针对性的 print 语句往往比单步调试更有效，因为你需要一次性看到形状、dtype 和数值范围。

```python
def debug_print(name, tensor):
    print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, "
          f"device={tensor.device}, "
          f"min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, "
          f"mean={tensor.mean().item():.4f}, "
          f"has_nan={tensor.isnan().any().item()}")
```

在每个可疑操作后调用它。找到 bug 后，把这些打印删掉。就这么简单。

### 第 2 部分：Python 调试器（pdb 与 breakpoint）

内置调试器在 AI 工作里被低估了。把 `breakpoint()` 丢进训练循环，然后交互式地检查张量。

```python
def training_step(model, batch, criterion, optimizer):
    inputs, labels = batch
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    if loss.item() > 100 or torch.isnan(loss):
        breakpoint()

    loss.backward()
    optimizer.step()
```

进入调试器后，常用命令有：

- `p outputs.shape`：检查形状
- `p loss.item()`：查看 loss 值
- `p torch.isnan(outputs).sum()`：统计 NaN 的数量
- `p model.fc1.weight.grad`：检查梯度
- `c`：继续，`q`：退出

这叫条件调试。只有在看起来不对劲时你才会停下来。对于一次 10,000 步的训练来说，这很重要。

### 第 3 部分：Python Logging

当调试不再只是一次快速确认时，就用 logging 替换 print 语句。

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("Starting training: lr=%.4f, batch_size=%d", lr, batch_size)
logger.warning("Loss spike detected: %.4f at step %d", loss.item(), step)
logger.error("NaN loss at step %d, stopping", step)
```

logging 能给你时间戳、严重级别和文件输出。当训练在凌晨 3 点失败时，你要的是日志文件，而不是早已滚出屏幕的终端输出。

### 第 4 部分：给代码片段计时

知道时间都花在哪，是优化的第一步。

```python
import time

class Timer:
    def __init__(self, name=""):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        print(f"[{self.name}] {elapsed:.4f}s")

with Timer("data loading"):
    batch = next(dataloader_iter)

with Timer("forward pass"):
    outputs = model(batch)

with Timer("backward pass"):
    loss.backward()
```

一种很常见的情况是：数据加载占了 60% 的训练时间。修复方法不是换更快的 GPU，而是把 DataLoader 的 `num_workers` 设为大于 0。

### 第 5 部分：`cProfile` 与 `line_profiler`

当手动计时已经不够用时：

```bash
python -m cProfile -s cumtime train.py
```

这会显示按累计时间排序的每一次函数调用。若要逐行分析：

```bash
pip install line_profiler
```

```python
@profile
def train_step(model, data, target):
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward()
    return loss

# Run with: kernprof -l -v train.py
```

### 第 6 部分：内存分析

#### 用 `tracemalloc` 分析 CPU 内存

```python
import tracemalloc

tracemalloc.start()

# your code here
model = build_model()
data = load_dataset()

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics("lineno")
for stat in top_stats[:10]:
    print(stat)
```

#### 用 `memory_profiler` 分析 CPU 内存

```bash
pip install memory_profiler
```

```python
from memory_profiler import profile

@profile
def load_data():
    raw = read_csv("data.csv")       # watch memory jump here
    processed = preprocess(raw)       # and here
    return processed
```

用 `python -m memory_profiler your_script.py` 运行，即可看到逐行内存使用情况。

#### 用 PyTorch 查看 GPU 内存

```python
import torch

if torch.cuda.is_available():
    print(torch.cuda.memory_summary())

    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

当你遇到 OOM（Out of Memory）时：

1. 先减小 batch size（永远先试这个）
2. 使用 `torch.cuda.empty_cache()` 释放缓存内存
3. 对大型中间结果先 `del tensor`，然后再调用 `torch.cuda.empty_cache()`
4. 使用混合精度（`torch.cuda.amp`）把内存占用减半
5. 对非常深的模型使用 gradient checkpointing

### 第 7 部分：常见 AI bug 及捕捉方法

#### 形状不匹配

这是最常见的 bug。某个张量的 shape 是 `[batch, features]`，而模型期望的是 `[batch, channels, height, width]`。

```python
def check_shapes(model, sample_input):
    print(f"Input: {sample_input.shape}")
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            in_shape = inp[0].shape if isinstance(inp, tuple) else inp.shape
            out_shape = out.shape if hasattr(out, "shape") else type(out)
            print(f"  {name}: {in_shape} -> {out_shape}")
        return hook

    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        model(sample_input)

    for h in hooks:
        h.remove()
```

用一个示例 batch 跑一次它。它会映射出模型中的每一次 shape 变化。

#### NaN loss

NaN loss 表示某个地方炸掉了。常见原因有：

- 学习率过高
- 自定义 loss 中出现除以零
- 对零或负数取对数
- RNN 中梯度爆炸

```python
def detect_nan(model, loss, step):
    if torch.isnan(loss):
        print(f"NaN loss at step {step}")
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"  NaN gradient in {name}")
                if torch.isinf(param.grad).any():
                    print(f"  Inf gradient in {name}")
        return True
    return False
```

#### 数据泄漏

模型在测试集上拿到 99% 的准确率。听起来很棒。其实这是 bug。

```python
def check_data_leakage(train_set, test_set, id_column="id"):
    train_ids = set(train_set[id_column].tolist())
    test_ids = set(test_set[id_column].tolist())
    overlap = train_ids & test_ids
    if overlap:
        print(f"DATA LEAKAGE: {len(overlap)} samples in both train and test")
        return True
    return False
```

还要检查时间泄漏：用未来数据去预测过去。切分前先按时间戳排序。

#### 设备错误

位于不同设备上的张量（CPU 与 GPU）会导致运行时错误。但有时某个张量会悄悄留在 CPU 上，而其他所有东西都在 GPU 上，于是训练虽然能跑，却会变得很慢。

```python
def check_devices(model, *tensors):
    model_device = next(model.parameters()).device
    print(f"Model device: {model_device}")
    for i, t in enumerate(tensors):
        if t.device != model_device:
            print(f"  WARNING: tensor {i} on {t.device}, model on {model_device}")
```

### 第 8 部分：TensorBoard 基础

TensorBoard 会展示训练过程中内部到底发生了什么。

```bash
pip install tensorboard
```

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/experiment_1")

for step in range(num_steps):
    loss = train_step(model, batch)

    writer.add_scalar("loss/train", loss.item(), step)
    writer.add_scalar("lr", optimizer.param_groups[0]["lr"], step)

    if step % 100 == 0:
        for name, param in model.named_parameters():
            writer.add_histogram(f"weights/{name}", param, step)
            if param.grad is not None:
                writer.add_histogram(f"grads/{name}", param.grad, step)

writer.close()
```

启动它：

```bash
tensorboard --logdir=runs
```

重点观察这些信号：

- **Loss 不下降**：学习率太低，或者模型架构有问题
- **Loss 剧烈振荡**：学习率太高
- **Loss 变成 NaN**：数值不稳定（见上面的 NaN 小节）
- **Train loss 下降，但 val loss 上升**：过拟合
- **权重直方图塌缩到零**：梯度消失
- **梯度直方图爆炸**：需要 gradient clipping

### 第 9 部分：VS Code 调试器

如果你要做交互式调试，可以在 VS Code 中配置一个 `launch.json`：

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug Training",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": false
        }
    ]
}
```

点击行号边上的 gutter 即可设置断点。使用 Variables 面板检查张量属性。Debug Console 允许你在执行过程中运行任意 Python 表达式。

这很适合单步查看数据预处理流水线，尤其是你想看清每一步变换时。

## 用起来

下面这套调试流程能抓住大多数 AI bug：

1. **训练前**：用一个示例 batch 运行 `check_shapes`。确认输入和输出维度符合预期。
2. **前 10 步**：对 loss、输出和梯度使用 `debug_print`。确认没有 NaN，且数值范围合理。
3. **训练过程中**：记录 loss、学习率和梯度范数。用 TensorBoard 做可视化。
4. **当出现问题时**：在故障点放一个 `breakpoint()`。交互式检查张量。
5. **为性能问题排查时**：分别给数据加载、forward 和 backward 计时。如果快碰到 OOM，就分析内存。

## 交付

运行调试工具脚本：

```bash
python phases/00-setup-and-tooling/12-debugging-and-profiling/code/debug_tools.py
```

查看 `outputs/prompt-debug-ai-code.md`，其中有一个可帮助诊断 AI 特有 bug 的提示词。

## 练习

1. 运行 `debug_tools.py`，阅读每个小节的输出。修改示例模型，手动引入一个 NaN（提示：在 forward pass 里除以零），然后观察检测器如何抓住它。
2. 用 `cProfile` 分析一个训练循环，并找出最慢的函数。
3. 使用 `tracemalloc` 找出数据加载流水线中哪一行分配了最多内存。
4. 为一次简单训练配置 TensorBoard，并判断模型是否发生过拟合。
5. 在训练循环里使用 `breakpoint()`。练习从调试器提示符里检查张量的形状、设备和梯度值。

