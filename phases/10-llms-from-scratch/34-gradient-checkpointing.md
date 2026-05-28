# 梯度检查点（Gradient Checkpointing）与激活重计算（Activation Recomputation）

> 反向传播会保留每一个中间激活。在 70B 参数、128K 上下文下，这意味着每个 rank 要保存 3 TB 的激活。检查点技术用 FLOPs 换内存：不保存，而是在需要时重算。真正的问题不在于“要不要丢”，而在于“该丢哪些段”；答案绝不是“全部都丢”。

**类型：** Build
**语言：** Python（with numpy, optional torch）
**前置课程：** 第 10 阶段第 04 课（Pre-Training Mini-GPT），第 10 阶段第 05 课（Scaling & Distributed）
**耗时：** ~70 分钟

## 问题

训练一个 Transformer 时，需要为每一层存储所有在反向传播中会被求导操作使用到的输入：attention 的输入、Q/K/V 投影、softmax 输出、FFN 输入、norm 输出，以及残差流。对于一个隐藏维度为 `d`、序列长度为 `L`、batch 为 `B` 的层，这大约相当于每层 `12 * B * L * d` 个浮点数。

当 `d=8192, L=8192, B=1` 时，在 BF16 下这相当于每层 800 MB。一个 64 层模型就是 51 GB 的激活——而这还没乘上 microbatch 大小，还没加上 attention softmax 的中间量（每个头有 `L^2`），也还没考虑 tensor parallel 带来的局部拷贝。

账单的两边都在涨：BF16 权重加优化器状态也许还能塞进 80GB，但激活会把你彻底顶爆。梯度检查点（又称激活重计算）就是标准修复方案。丢掉大部分激活；在反向时重做前向，把它们算回来。代价：额外 FLOPs。收益：内存按“检查点段数 / 总层数”的比例下降。

如果做得很粗暴，检查点大约会让每一步的前向 FLOPs 增加 33%。如果做得好——比如采用 Korthikanti 等人提出的“智能选择（smart selection）”——你可以在 FLOP 开销低于 5% 的情况下，把内存压到原来的五分之一。再叠加 FP8 matmul、FSDP offload 与 expert-parallel MoE，这件事就更加关键：你既负担不起这部分内存，也负担不起无谓的重算。

## 概念

### 反向传播到底需要什么

`output = layer(input)`。反向传播需要 `grad_input` 和 `grad_params`。为了算出它们，需要：

- `input`（在线性层中用来计算 `grad_params = input.T @ grad_output`）
- 某些激活导数中间量（例如 ReLU / GELU / softmax 的导数都依赖激活值本身）

前向传播会自动把这些东西存进 autograd 图里。每一个 `tensor.retain_grad()`，以及每一个需要其输入的操作，都会保留一个引用。

### 朴素的全量检查点

把网络切成 `N` 段。在前向过程中，只保存每一段的*输入*。当反向需要中间量时，就重新执行该段的前向，把它们物化出来，然后再求导。

例子：一个 32 层 Transformer 被切成 32 段，每段 1 层。

- 内存：只保存 32 份 layer input（很小），而不是 `32 * 每层激活体积`（很大）。
- 额外计算：每个 segment 多做 1 次前向，也就是总前向 FLOPs 大约增加 33%（因为反向约等于 2 倍前向，完整一步从 `1 + 2 = 3` 单位变成 `1 + 1 + 2 = 4` 单位）。

这就是 Chen 等人 2016 年的原始方案：每隔 `sqrt(L)` 层放一个检查点，用来平衡内存与计算。若 L=64，就是 8 个检查点。

### 选择性检查点（Selective Checkpointing，Korthikanti 2022）

不是所有激活的代价都一样。attention softmax 输出的大小是 `B*L*L*heads`，会随着序列长度二次增长；FFN 隐层激活是 `B*L*4d`，只线性增长。对于长序列，softmax 才是主导项。

选择性检查点会保留那些便宜的激活（线性投影、残差流），只重算昂贵的部分（attention）。你为重算支付的 FLOPs 很少，却能省下 `O(L^2)` 级别的内存。

Megatron-Core 将其实现为“selective”激活重计算。它已经被大多数 2024+ 的前沿训练运行采用。

### Offload

与其重算，另一个选择是把激活在前向和反向之间转移到 CPU RAM。它依赖 PCIe 带宽；当空闲带宽足以覆盖 rematerialization 成本时，就很划算。混合策略很常见：有些层做检查点，有些层做 offload。

FSDP2 把 offload 作为一等公民支持。它在 GPU 受内存限制、而 CPU-GPU 传输还有余量时尤其有价值。

### 重算成本模型

对于一个有 `L` 层、每 `k` 层做一次朴素检查点的模型，其每步 FLOPs 为：

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # one extra forward per layer in the segment
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

若采用选择性检查点，则只重算 attention kernel，而不是整个层：

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### 内存节省模型

每层激活体积为 `A`。若有 `L` 层，则总激活内存为 `L * A`。

完全检查点（segment size 为 1）：只保存 `L * input_volume`（对标准 Transformer 来说，大约是 `L * 1/10 A`）。节省约 `9 * L * A * 1/10`。

每 `k` 层做一次检查点：需要保存 `L/k * A`，再加上当前活动 segment 中 `k-1` 层的激活量。

当 `k = sqrt(L)` 时，内存与重算成本都按 `sqrt(L)` 缩放——对于各层成本均匀的情况，这是最优折中。

### 什么时候不该做检查点

- 某个流水线 stage 内部已经在飞行中的最内层。反正它们必须完成。
- 如果第一层和最后一层主导了该 stage 的计算（在 Transformer 里很少见），也不适合。
- 已经使用 FlashAttention 的 attention kernel——Flash 本身就会快速重算 softmax，所以在其之上再加整层级检查点，额外收益很小。

### 实现模式

1. **函数包装器：** 用 `torch.utils.checkpoint.checkpoint(fn, input)` 包裹一个 segment。PyTorch 只保存 `input`，其余全部在反向时重算。

2. **基于装饰器：** 给层打上“可检查点”标签；训练器在配置阶段决定哪些 segment 要被包装。

3. **手工显式重算：** 你自己写 backward，并调用自定义的 `recompute_forward`，用保存下来的输入重新执行前向。

三种方式在功能上完全等价。包装器是最标准的写法。

### 与 TP / PP / FP8 的交互

- **Tensor parallel：** 重算时必须重新 gather 或 reshard 检查点输入；需要把通信成本算进去。
- **Pipeline parallel：** 常见模式是为每个 pipeline stage 的前向做检查点，这样逆序 microbatch 可以复用激活内存。
- **FP8 recompute：** 重算时更新的 amax 历史必须与原始前向一致，否则 FP8 scale 会漂移。多数框架会把 scale 一并快照下来。

## 动手实现

### 第 1 步：一个带分段的玩具模型

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### 第 2 步：需要全部激活的朴素反向

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### 第 3 步：每 k 层做一次检查点的内存模型

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### 第 4 步：成本模型

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### 第 5 步：内存估算器

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### 第 6 步：最优 segment size

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### 第 7 步：选择性检查点决策

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## 使用

- **torch.utils.checkpoint**：`from torch.utils.checkpoint import checkpoint` —— PyTorch 里的标准包装器。它包装一个函数；只保存输入，在反向时重算。
- **Megatron-Core activation recomputation**：支持 `selective`、`full` 与 `block` 模式。它是 2024+ 前沿训练中的标准配置。
- **FSDP2 offload**：在 FSDP2 中配合 `offload_policy` 使用 `module.to_empty(device="cpu")`，可以把激活切到 CPU，而不是重算。
- **DeepSpeed ZeRO-Offload**：对优化器状态与激活做 CPU offload，可与检查点互补。

## 交付

本课会产出 `outputs/prompt-activation-recompute-policy.md` —— 一个提示词模板。输入你的模型配置（层数、hidden、seq、batch）和可用 GPU 内存，它会输出逐层的重计算策略（none / selective / full / offload）。

## 练习

1. 验证正确性。运行 `model_forward` + `model_backward`（完整激活）与 `model_forward_checkpointed` + `model_backward_checkpointed`（分段）。参数梯度必须在机器精度下完全一致。

2. 把 segment size `k` 从 1 扫到 `L`。绘制 FLOP 开销与内存曲线。找出曲线的膝点。

3. 实现选择性检查点：保存 attention 模块输入，但不保存其中间量。对一个 32 层、seq=8192 的模型，测量与完整层检查点相比的 FLOP 开销。

4. 加入 offload。把 segment 输入保存到一个模拟的“CPU buffer”（单独的列表）里。测量“PCIe 带宽”（字节/时间），并找出 offload 与重算的盈亏平衡点。

5. 在真实 PyTorch Transformer 上，对比有无 `torch.utils.checkpoint`。测量显存（通过 `torch.cuda.max_memory_allocated`）和 step time。

## 关键术语

| 术语 | 人们常说什么 | 它真正的含义 |
|------|----------------|----------------------|
| 梯度检查点 | “通过重做前向来省内存” | 只保存 segment 输入；在反向时重算中间量，以拿回求梯度所需张量 |
| 激活重计算 | “和 checkpointing 是一回事” | 这是同一技术在 HPC 语境下更常见的名字 |
| Segment size（k） | “每个检查点覆盖多少层” | 一次被丢弃并一起重物化的层数 |
| 选择性检查点 | “Korthikanti 的技巧” | 只重算那些保存代价高的激活（如 attention softmax），保留便宜的 |
| 全量检查点 | “朴素版本” | 对每个 segment 中每层的中间量都进行重算 |
| Block checkpointing | “粗粒度方案” | 以整个 Transformer block 为粒度做检查点 |
| FLOP 开销 | “计算税” | 每步额外 FLOPs = `(recompute FLOPs) / (fwd + bwd FLOPs)`；朴素法 33%，选择性法 5% |
| 激活 offload | “搬到 CPU 去” | 在前向到反向之间，把激活转移到 CPU RAM；是重算的替代方案 |
| sqrt-L 规则 | “经典最优点” | 对于各层成本均匀的情况，最佳检查点间距约为 `sqrt(L)` 层 |
| Attention-softmax 体积 | “那个 O(L^2) 问题” | `L^2 * heads * batch` 个浮点；在长上下文下主导激活内存 |

## 延伸阅读

- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) —— 正式提出梯度检查点的原始论文
- [Korthikanti et al., 2022 -- "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198) —— 选择性激活重计算及其正式成本分析
- [Pudipeddi et al., 2020 -- "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645) —— 通过反向模式 rematerialization 实现常数内存的另一条路线
- [Ren et al., 2021 -- "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840) —— 大规模激活 offload
- [PyTorch torch.utils.checkpoint docs](https://pytorch.org/docs/stable/checkpoint.html) —— 标准 API 文档
- [Megatron-Core activation recomputation documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/memory_optimizations.html) —— `selective`、`full` 与 `block` 模式说明
