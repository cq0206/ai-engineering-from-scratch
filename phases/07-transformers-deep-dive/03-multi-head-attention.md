# 多头注意力 (Multi-Head Attention)

> 一个注意力头一次学一种关系。八个头就学八种。头是自由的。尽管多拿一些。

**类型：** 构建
**语言：** Python
**前置要求：** 第 7 阶段 · 02（从零实现自注意力）
**时长：** ~75 分钟

## 问题

单个自注意力头只会计算一个注意力矩阵。这个矩阵只能捕捉一种关系——通常是哪一种关系最能降低训练信号上的损失。如果你的数据里同时混着主谓一致、共指、长程篇章关系以及句法分块，那么单个头会把这些东西全都抹进同一个 softmax 分布里，最后丢掉一半信号。

2017 年 Vaswani 论文给出的修复方案是：并行运行多个注意力函数，每个函数都有自己的一组 Q、K、V 投影，然后把输出拼接起来。每个头都在一个更小的子空间中工作，这个子空间的维度是 `d_model / n_heads`。总参数量不变，表达能力却提升了。

到 2026 年，多头注意力已经是几乎所有 transformer 的默认配置。唯一还会争论的问题，只剩下*到底要多少个*头，以及键和值是否共享投影（Grouped-Query Attention、Multi-Query Attention、Multi-head Latent Attention）。

## 概念

*多头注意力：先拆分，再关注，后拼接*

**拆分。** 取形状为 `(N, d_model)` 的 `X`。把它投影成形状都为 `(N, d_model)` 的 Q、K、V。然后 reshape 成 `(N, n_heads, d_head)`，其中 `d_head = d_model / n_heads`。再转置为 `(n_heads, N, d_head)`。

**并行关注。** 在每个头内部运行缩放点积注意力 (scaled dot-product attention)。每个头都会产生一个 `(N, d_head)` 的结果。不同头工作在嵌入的不同子空间上，在注意力计算本身这一步里彼此不会交流。

**拼接并投影。** 把所有头重新堆叠回 `(N, d_model)`，再乘上一个可学习的输出矩阵 `W_o`，其形状为 `(d_model, d_model)`。`W_o` 就是各个头开始混合信息的地方。

**为什么有效。** 每个头都可以各自专门化，而不必为了表示容量和其他头争抢预算。2019–2024 年的大量 probing 研究表明，不同的头确实会承担不同角色：位置头、关注前一个 token 的头、复制头、命名实体头、归纳头（induction heads，支撑 in-context learning）。

**到 2026 年为止的变体谱系：**

| 变体 | Q 头数 | K/V 头数 | 使用者 |
|---------|---------|-----------|---------|
| 多头 (MHA) | N | N | GPT-2, BERT, T5 |
| 多查询 (MQA) | N | 1 | PaLM, Falcon |
| 分组查询 (GQA) | N | G（例如 N/8） | Llama 2 70B, Llama 3+, Qwen 2+, Mistral |
| 多头潜变量 (MLA) | N | 压缩到低秩空间 | DeepSeek-V2, V3 |

GQA 是现代默认方案，因为它能把 KV-cache 内存缩小 `N/G` 倍，同时几乎不损失质量。MLA 更进一步：它把 K/V 压缩到一个潜在空间，再在计算时投影回来——会增加 FLOPs，但能换来更多内存节省。

## 动手实现

### 第 1 步：基于现有的单头注意力拆分 heads

拿第 02 课里的 `SelfAttention`，在外面套上一层 split/concat。可以参考 `code/main.py` 中的 numpy 实现；其核心逻辑是：

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

一次 reshape，再一次 transpose。没有循环。这正是 PyTorch 在 `nn.MultiheadAttention` 底层所做的事。

### 第 2 步：对每个头运行缩放点积注意力

每个头都会拿到自己那一片 Q、K、V。注意力于是变成一次 batched matmul：

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

在真实硬件上，`Qh @ Kh.transpose(...)` 就是一条 `bmm`。GPU 看到的是一次统一的 batched matmul，形状为 `(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)`。增加头数几乎是“白送”的。

### 第 3 步：Grouped-Query Attention 变体

变化的只有 key 和 value 投影。Q 仍然有 `n_heads` 组；K 和 V 只有 `n_kv_heads &lt; n_heads` 组，然后通过重复来匹配：

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

在推理阶段，这样可以节省内存，因为 KV cache 里只需要保存 `n_kv_heads` 份副本，而不是 `n_heads` 份。Llama 3 70B 使用 64 个 query heads 和 8 个 KV heads——缓存直接缩小 8 倍。

### 第 4 步：探查每个头学到了什么

用 4 个头在一个短句上运行 MHA。对每个头，打印它的 `(N, N)` 注意力矩阵。即便是随机初始化，你也会看到不同头偏向不同结构——其中一部分是真实信号，另一部分则来自子空间中的旋转对称性。

## 使用它

在 PyTorch 里，一行版写法如下：

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

PyTorch 2.5+ 中的 GQA：

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention auto-dispatches Flash Attention on CUDA.
# For GQA, pass Q of shape (B, n_heads, N, d_head) and K,V of shape
# (B, n_kv_heads, N, d_head). PyTorch handles the repeat.
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**头要设多少？** 来自 2026 年生产模型的一些经验法则：

| 模型规模 | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| 小型（~125M） | 768 | 12 | 64 |
| 基础型（~350M） | 1024 | 16 | 64 |
| 大型（~1B） | 2048 | 16 | 128 |
| 前沿级（~70B） | 8192 | 64 | 128 |

`d_head` 几乎总会落在 64 或 128。它代表一个头到底能“看见”多少信息。低于 32 时，头会开始和缩放因子 `sqrt(d_head)` 对抗；高于 256 时，又会失去“许多小型专家”带来的好处。

## 交付物

参见 `outputs/skill-mha-configurator.md`。这个 skill 会根据参数预算、序列长度和部署目标，为一个新的 transformer 推荐 head 数、kv-head 数以及投影策略。

## 练习

1. **简单。** 取 `code/main.py` 里的 MHA，在固定 `d_model=64` 的情况下，把 `n_heads` 从 1 改到 16。画出一个极小单层模型在合成 copy task 上的 loss 曲线。更多的 heads 是更好、进入平台期，还是反而变差？
2. **中等。** 实现 MQA（所有 query heads 共享一个 KV head）。测量相较完整 MHA，参数量下降了多少。再计算当 N=2048 时，推理阶段 KV-cache 大小缩小了多少。
3. **困难。** 实现一个微型版本的 Multi-head Latent Attention：把 K、V 压缩到秩为 `r` 的潜变量里，把这个潜变量存进 KV cache，再在注意力计算时解压。在哪个 `r` 下，缓存内存会降到完整 MHA 的 1/8 以下，同时质量仍能保持在验证集 ppl 相差 1 bit 以内？

## 关键术语

| 术语 | 人们常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 头 (Head) | “一条独立的注意力回路” | 一组维度为 `d_head = d_model / n_heads` 的 Q/K/V 投影，带有它自己的注意力矩阵。 |
| d_head | “头的维度” | 每个头的隐藏宽度；在生产环境中几乎总是 64 或 128。 |
| 拆分 / 合并 (Split / combine) | “reshape 技巧” | 在注意力前后，对 `(N, d_model) ↔ (n_heads, N, d_head)` 做 reshape + transpose。 |
| W_o | “输出投影” | 在拼接各个头之后应用的 `(d_model, d_model)` 矩阵；也是各个头彼此混合的地方。 |
| MQA | “一个 KV 头” | Multi-Query Attention：共享单个 K/V 投影。KV cache 最小，但会有一些质量损失。 |
| GQA | “自 Llama 2 以来的默认方案” | Grouped-Query Attention，其中 `n_kv_heads &lt; n_heads`；通过重复来匹配 Q。 |
| MLA | “DeepSeek 的技巧” | Multi-head Latent Attention：把 K、V 压缩到低秩潜变量中，再在关注时解压。 |
| 归纳头 (Induction head) | “in-context learning 背后的电路” | 一对能够检测先前出现模式并复制其后续内容的头。 |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) — 原始的多头注意力规范。
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — MQA 论文。
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — 如何在训练后把 MHA 转换成 GQA。
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) — MLA 及其为何在缓存内存上优于 MHA/GQA。
- [Olsson et al. (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — 从机制解释角度看这些 heads 实际在做什么。
