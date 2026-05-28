# 从零实现自注意力 (self-attention)

> 注意力就像一张查找表：每个词都会问“谁对我重要？”——然后自己学会答案。

**类型：** 构建
**语言：** Python
**前置要求：** 第 3 阶段（深度学习核心）、第 5 阶段第 10 课（Sequence-to-Sequence）
**时长：** ~90 分钟

## 学习目标

- 仅使用 NumPy 从零实现缩放点积自注意力 (scaled dot-product self-attention)，包括查询/键/值投影以及基于 softmax 的加权求和
- 构建一个多头注意力 (multi-head attention) 层：将头拆分开来、并行计算注意力，再把结果拼接起来
- 追踪注意力矩阵如何捕捉 token 之间的关系，并解释为什么用 sqrt(d_k) 做缩放可以防止 softmax 饱和
- 应用因果掩码 (causal masking)，把双向注意力转换为自回归（decoder-style）注意力

## 问题

RNN 会按顺序一次处理一个 token。等你处理到第 50 个 token 时，第 1 个 token 的信息已经被压过了 50 次。长程依赖会被挤压进一个固定大小的隐藏状态里——这是一个瓶颈，再多的 LSTM 门控机制也无法彻底解决。

2014 年 Bahdanau 的注意力论文给出了修复思路：让解码器回看编码器中的每一个位置，并决定哪些位置对当前步骤最重要。但它依然是“加装”在 RNN 上的。2017 年的《Attention Is All You Need》提出了一个更尖锐的问题：如果注意力是*唯一*的机制，会怎样？没有循环。没有卷积。只有注意力。

自注意力让序列中的每个位置都能在一次并行步骤中关注到其他所有位置。这正是 transformer 如此快速、可扩展并最终占据主导地位的原因。

## 概念

### 数据库查找类比

把注意力想象成一种“软”的数据库查找：

```
Traditional database:
  Query: "capital of France"  -->  exact match  -->  "Paris"

Attention:
  Query: "capital of France"  -->  similarity to ALL keys  -->  weighted blend of ALL values
```

每个 token 都会生成三个向量：
- **查询 (Query, Q)：** “我在找什么？”
- **键 (Key, K)：** “我包含什么？”
- **值 (Value, V)：** “如果我被选中，我能提供什么信息？”

把一个查询与所有键做点积，就会得到注意力分数。分数高意味着“这个键和我的查询更匹配”。这些分数随后会给值分配权重。输出则是值的加权和。

### Q、K、V 的计算

每个 token 的嵌入都会通过三个可学习的权重矩阵进行投影：

```
Input embeddings (sequence of n tokens, each d-dimensional):

  X = [x1, x2, x3, ..., xn]       shape: (n, d)

Three weight matrices:

  Wq  shape: (d, dk)
  Wk  shape: (d, dk)
  Wv  shape: (d, dv)

Projections:

  Q = X @ Wq    shape: (n, dk)      each token's query
  K = X @ Wk    shape: (n, dk)      each token's key
  V = X @ Wv    shape: (n, dv)      each token's value
```

从可视化角度看，单个 token 的过程如下：

```
             Wq
  x_i ------[*]------> q_i    "What am I looking for?"
       |
       |     Wk
       +----[*]------> k_i    "What do I contain?"
       |
       |     Wv
       +----[*]------> v_i    "What do I offer?"
```

### 注意力矩阵

当你为所有 token 都算出 Q、K、V 后，注意力分数会组成一个矩阵：

```
Scores = Q @ K^T    shape: (n, n)

              k1    k2    k3    k4    k5
        +-----+-----+-----+-----+-----+
   q1   | 2.1 | 0.3 | 0.1 | 0.8 | 0.2 |   <- how much q1 attends to each key
        +-----+-----+-----+-----+-----+
   q2   | 0.4 | 1.9 | 0.7 | 0.1 | 0.3 |
        +-----+-----+-----+-----+-----+
   q3   | 0.2 | 0.6 | 2.3 | 0.5 | 0.1 |
        +-----+-----+-----+-----+-----+
   q4   | 0.9 | 0.1 | 0.4 | 1.7 | 0.6 |
        +-----+-----+-----+-----+-----+
   q5   | 0.1 | 0.3 | 0.2 | 0.5 | 2.0 |
        +-----+-----+-----+-----+-----+

Each row: one token's attention over the entire sequence
```

### 为什么要缩放？

点积的数值会随着维度 dk 增大而变大。如果 dk = 64，那么点积可能落在几十这个量级，从而把 softmax 推到梯度接近消失的区域。解决方法是：除以 sqrt(dk)。

```
Scaled scores = (Q @ K^T) / sqrt(dk)
```

这样可以把数值保持在一个更合适的范围内，让 softmax 产生有用的梯度。

### Softmax 如何把分数变成权重

Softmax 会把每一行的原始分数转换为一个概率分布：

```
Raw scores for q1:   [2.1, 0.3, 0.1, 0.8, 0.2]
                            |
                         softmax
                            |
Attention weights:   [0.52, 0.09, 0.07, 0.14, 0.08]   (sums to ~1.0)
```

这样一来，每个 token 都得到一组权重，用来表示它应该在多大程度上关注其他每个 token。

### 值的加权求和

每个 token 的最终输出，是对所有值向量做加权求和：

```
output_i = sum( attention_weight[i][j] * v_j  for all j )

For token 1:
  output_1 = 0.52 * v1 + 0.09 * v2 + 0.07 * v3 + 0.14 * v4 + 0.08 * v5
```

### 完整流程

```
                    +-------+
  X (input)  ----->|  @ Wq  |-----> Q
                    +-------+
                    +-------+
  X (input)  ----->|  @ Wk  |-----> K
                    +-------+                     +----------+
                    +-------+                     |          |
  X (input)  ----->|  @ Wv  |-----> V ---------->| weighted |----> output
                    +-------+          ^          |   sum    |
                                       |          +----------+
                              +--------+--------+
                              |    softmax      |
                              +---------+-------+
                                        ^
                              +---------+-------+
                              | Q @ K^T / sqrt  |
                              +-----------------+
```

一行公式总结：

```
Attention(Q, K, V) = softmax( Q @ K^T / sqrt(dk) ) @ V
```

## 动手实现

### 第 1 步：从零实现 softmax

Softmax 会把原始 logits 转换成概率。为了数值稳定性，先减去最大值。

```python
import numpy as np

def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

logits = np.array([2.0, 1.0, 0.1])
print(f"logits:  {logits}")
print(f"softmax: {softmax(logits)}")
print(f"sum:     {softmax(logits).sum():.4f}")
```

### 第 2 步：缩放点积注意力

这是核心函数。它接收 Q、K、V 矩阵，返回注意力输出以及权重矩阵。

```python
def scaled_dot_product_attention(Q, K, V):
    dk = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(dk)
    weights = softmax(scores)
    output = weights @ V
    return output, weights
```

### 第 3 步：带可学习投影的自注意力类

这是一个完整的自注意力模块，包含 Wq、Wk、Wv 权重矩阵，并使用类似 Xavier 的缩放方式初始化。

```python
class SelfAttention:
    def __init__(self, d_model, dk, dv, seed=42):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (d_model + dk))
        self.Wq = rng.normal(0, scale, (d_model, dk))
        self.Wk = rng.normal(0, scale, (d_model, dk))
        scale_v = np.sqrt(2.0 / (d_model + dv))
        self.Wv = rng.normal(0, scale_v, (d_model, dv))
        self.dk = dk

    def forward(self, X):
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv
        output, weights = scaled_dot_product_attention(Q, K, V)
        return output, weights
```

### 第 4 步：在一句话上运行它

为一句话创建假的嵌入，然后观察注意力权重。

```python
sentence = ["The", "cat", "sat", "on", "the", "mat"]
n_tokens = len(sentence)
d_model = 8
dk = 4
dv = 4

rng = np.random.default_rng(42)
X = rng.normal(0, 1, (n_tokens, d_model))

attn = SelfAttention(d_model, dk, dv, seed=42)
output, weights = attn.forward(X)

print("Attention weights (each row: where that token looks):\n")
print(f"{'':>6}", end="")
for token in sentence:
    print(f"{token:>6}", end="")
print()

for i, token in enumerate(sentence):
    print(f"{token:>6}", end="")
    for j in range(n_tokens):
        w = weights[i][j]
        print(f"{w:6.3f}", end="")
    print()
```

### 第 5 步：用 ASCII 热力图可视化注意力

把注意力权重映射为字符，快速获得一个直观图。

```python
def ascii_heatmap(weights, tokens, chars=" ░▒▓█"):
    n = len(tokens)
    print(f"\n{'':>6}", end="")
    for t in tokens:
        print(f"{t:>6}", end="")
    print()

    for i in range(n):
        print(f"{tokens[i]:>6}", end="")
        for j in range(n):
            level = int(weights[i][j] * (len(chars) - 1) / weights.max())
            level = min(level, len(chars) - 1)
            print(f"{'  ' + chars[level] + '   '}", end="")
        print()

ascii_heatmap(weights, sentence)
```

## 使用它

PyTorch 的 `nn.MultiheadAttention` 做的正是我们刚刚实现的事情，只不过额外加上了多头拆分和输出投影：

```python
import torch
import torch.nn as nn

d_model = 8
n_heads = 2
seq_len = 6

mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

X_torch = torch.randn(1, seq_len, d_model)

output, attn_weights = mha(X_torch, X_torch, X_torch)

print(f"Input shape:            {X_torch.shape}")
print(f"Output shape:           {output.shape}")
print(f"Attention weight shape: {attn_weights.shape}")
print(f"\nAttn weights (averaged over heads):")
print(attn_weights[0].detach().numpy().round(3))
```

关键区别在于：多头注意力会并行运行多个注意力函数，每个头都有自己的一组 Q、K、V 投影，其大小为 dk = d_model / n_heads，然后再把结果拼接起来。这样模型就可以同时关注不同类型的关系。

## 交付物

本课会产出：
- `outputs/prompt-attention-explainer.md` - 一个用于通过数据库查找类比解释注意力的提示词

## 练习

1. 修改 `scaled_dot_product_attention`，让它接收一个可选的 mask 矩阵，在 softmax 之前把某些位置设为负无穷（这正是因果/decoder 掩码的实现方式）
2. 从零实现多头注意力：把 Q、K、V 切分为 `n_heads` 个块，对每个头分别运行注意力，拼接结果后再通过最终权重矩阵 Wo 投影
3. 取两个长度相同但内容不同的句子，把它们送入同一个 SelfAttention 实例，比较它们的注意力模式。哪些变了？哪些保持不变？

## 关键术语

| 术语 | 人们常怎么说 | 实际含义 |
|------|-------------|---------|
| 查询向量 (Query, Q) | “问题向量” | 输入的一种可学习投影，用来表示这个 token 正在寻找什么信息 |
| 键向量 (Key, K) | “标签向量” | 一种可学习投影，用来表示这个 token 包含什么信息，并与查询进行匹配 |
| 值向量 (Value, V) | “内容向量” | 一种可学习投影，携带真正会被聚合的信息，具体取决于注意力分数 |
| 缩放点积注意力 (Scaled dot-product attention) | “注意力公式” | softmax(QK^T / sqrt(dk)) @ V —— 缩放可以防止高维情况下 softmax 饱和 |
| 自注意力 (Self-attention) | “token 会看自己也看别人” | Q、K、V 都来自同一个序列的注意力机制，因此每个位置都能关注到其他所有位置 |
| 注意力权重 (Attention weights) | “关注多少” | 在位置维度上的概率分布，由缩放点积之后的 softmax 产生 |
| 多头注意力 (Multi-head attention) | “并行注意力” | 用不同投影并行运行多个注意力函数，再把结果拼接起来，以获得更丰富的表示 |

## 延伸阅读

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) - transformer 原始论文
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/) - 对完整架构最好的可视化讲解之一
- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) - 带逐行解释的 PyTorch 实现
