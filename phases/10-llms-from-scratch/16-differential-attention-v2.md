# 差分注意力（Differential Attention，V2）

> Softmax 注意力会把一小部分概率分散到每个不匹配的 token 上。上下文一旦拉到 100k token，这些噪声就会不断累积，最终淹没真正的信号。差分 Transformer（Differential Transformer，Ye 等，ICLR 2025）通过把注意力计算成两个 softmax 的差值来解决这个问题，从而减去共享的噪声底。DIFF V2（Microsoft，2026 年 1 月）则是面向生产栈的重写版本：解码延迟与基线 Transformer 持平、不需要自定义 kernel、兼容 FlashAttention。本课会完整走一遍从 V1 到 V2 的演进，并提供一个可在 stdlib Python 中运行的差值运算玩具实现。

**类型：** Build
**语言：** Python（stdlib）
**前置课程：** 第 7 阶段·02（自注意力），第 7 阶段·15（注意力变体），第 10 阶段·14（架构 walkthrough）
**耗时：** ~60 分钟

## 学习目标

- 准确说明 softmax 注意力为什么存在噪声底（noise floor），以及它为什么会随着上下文长度增长。
- 推导差分注意力公式，并解释为什么减法能消除共享噪声分量，同时保留信号。
- 梳理从 V1 到 V2 的差异：哪里更快了、哪里更简单了、哪里更稳定了，以及为什么每项改动都是生产级预训练所必需的。
- 用纯 Python 从零实现差分注意力，并在一个合成的“信号 + 噪声”查询上经验验证其噪声抵消特性。

## 问题

标准 softmax 注意力有一个数学性质，在大规模场景下会演变成操作层面的麻烦。对于查询 `q`，注意力权重是 `softmax(qK^T / sqrt(d))`。Softmax 永远不会产生精确的 0——每个不匹配的 token 都会分到一点正质量。这部分残余质量就是噪声，而且会随着上下文长度扩张。在 128k token 下，即便每个不匹配 token 只拿到 0.001% 的概率，127,999 个此类 token 合起来也会贡献大约 12% 的总权重。模型不得不学着绕过一个会随上下文增长的噪声底。

经验上，这会表现为注意力头干扰（attention-head interference）：长上下文 RAG 中出现幻觉引用，在 100k token 检索任务里出现 lost-in-the-middle 失败，以及 needle-in-haystack 基准在超过 32k 后出现细微但稳定的准确率下降。Differential Transformer 论文（arXiv:2410.05258，ICLR 2025）量化了这个差距：DIFF Transformer 在相同规模的基线上实现了更低困惑度、更高长上下文准确率，以及更少的幻觉。

DIFF V1 有三个问题，使它无法进入前沿预训练流水线。首先，它的 value cache 在每个解码步都必须加载两次；其次，它依赖自定义 CUDA kernel，破坏了与 FlashAttention 的兼容性；最后，它的逐头 RMSNorm 在 70B 以上规模的长期训练中会引发不稳定。DIFF V2（Microsoft unilm 博客，2026 年 1 月 20 日）把这三点都修掉了。本课会同时讲清两个版本、构建这个差分算子，并在一个玩具查询上评测它的噪声抵消效果。

## 概念

### Softmax 的噪声底

对于查询 `q` 和键 `K = [k_1, ..., k_N]`，注意力权重是：

```
w_i = exp(q . k_i / sqrt(d)) / sum_j exp(q . k_j / sqrt(d))
```

没有任何一个 `w_i` 会严格等于 0。如果 `k_i` 与 `q` 完全无关，分数 `q . k_i` 也不是 0——它会围绕 0 波动，方差为 `||q||^2 / d`。经过 softmax 归一化后，每个无关 token 仍会对加权和贡献 `O(1/N)` 的权重。所有无关 token 的总贡献是 `O((N-1)/N) = O(1)`——这并不是一个小量。

模型真正想要的更像是硬 top-k：匹配 token 拿到高权重，其他位置几乎为零。Softmax 本身太平滑，没法直接做到这一点。

### 差分思路

把每个头的 Q、K 投影拆成两份：Q = (Q_1, Q_2)，K = (K_1, K_2)。计算两张注意力图：

```
A_1 = softmax(Q_1 K_1^T / sqrt(d))
A_2 = softmax(Q_2 K_2^T / sqrt(d))
```

输出为：

```
DiffAttn = (A_1 - lambda * A_2) V
```

这个减法会消掉两张图共享的噪声分布。如果两张图都在那 127k 个无关 token 上分配了近似均匀的权重（随机初始化时通常如此），这些部分就会互相抵消。真正的信号——也就是在少数相关 token 上形成的尖峰权重——只有在两张图里以同样幅度同时出现时才会被完全抵消，而在模型训练之后通常不会发生这种情况。

`lambda` 是每个头上的可学习标量，参数化形式为 `lambda = exp(lambda_q1 dot lambda_k1) - exp(lambda_q2 dot lambda_k2) + lambda_init`。它可以是负值。`lambda_init` 默认是一个较小的正数，比如 0.8。

### 为什么这像“带方向的降噪”

可以把它想成两个带噪麦克风在录同一段人声。两者都会拾取说话者声音，以及相关的背景噪声。把其中一个减掉，公共噪声就会被压低。人声之所以能保留下来，是因为两路信号在相位或幅度上有足够差异，不会被完全相消。逐头的 `lambda` 学到的正是这种平衡。

### V1 与 V2：差异在哪里

V1 试图把参数量维持在与基线 Transformer 相同的水平。为了让每个头拥有两组查询，它把 head dimension 减半了。这牺牲了头的表达能力，更痛的是，每个头的 value cache 也随之减半。解码时每一步都必须加载两次 value cache（每个 softmax 分支一次）。结果是：虽然参数量对齐基线，但解码比基线更慢。

V2 则把查询头数量翻倍，而保持 KV 头数量不变（额外参数从 up-projection 借来）。头维度与基线保持一致。做完减法后，再把多出来的维度投影回去，以匹配基线 Transformer 的 O_W 投影。于是三件事同时发生：

1. 解码速度与基线持平（KV cache 只加载一次）。
2. FlashAttention 可以不改动地直接运行（不再需要自定义 kernel）。
3. 解码时的算术强度（arithmetic intensity）提高了（每从 HBM 加载一个字节，就能做更多计算）。

V2 还移除了 V1 用来稳定减法的逐头 RMSNorm。在 70B 级别的预训练规模下，这个 RMSNorm 会让训练后期变得不稳定。V2 用一个更简单的初始化方案替代它，在不增加额外模块的前提下保持训练稳定。

### 什么时候值得用

| 工作负载 | 收益 |
|----------|---------|
| 长上下文 RAG（64k+） | 更干净的注意力图，更少幻觉引用 |
| Needle-in-haystack 基准 | 在 32k 以上准确率显著提升 |
| 多文档问答 | 更少跨文档干扰 |
| 8k 代码补全 | 收益边际，不值得为此改架构 |
| 短对话（&lt; 4k） | 与基线几乎无法区分 |

它的价值会随着上下文长度增长而上升。在 4k token 时，噪声底还足够小，标准注意力完全够用；在 128k 时，它就真的会拖你后腿。

### 它与 2026 年其他旋钮如何叠加

| 特性 | 与 DIFF V2 兼容吗？ |
|---------|------------------------|
| GQA | 是（V2 增加的是 Q 头，不是 KV 头） |
| MLA（DeepSeek） | 原理上是，但还没有公开论文把两者结合起来 |
| MoE | 是（注意力改动与 MLP 块独立） |
| RoPE | 是（不变） |
| YaRN / 长上下文缩放 | 是（正是 DIFF 最有帮助的场景） |
| FlashAttention | V2 支持（V1 不支持） |
| 推测解码 | 是（注意力改动对 spec-decode 循环是透明的） |

## 动手实现

`code/main.py` 用纯 Python 实现了差分注意力。一个具有已知“信号 + 噪声”结构的玩具查询，可以让你直接测量噪声抵消比例。

### 第 1 步：标准 softmax 注意力

使用 stdlib 风格的矩阵运算：列表嵌套列表、手写 matmul，以及通过减去最大值来保证数值稳定的 softmax。

```python
def softmax(row):
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]
```

### 第 2 步：把 Q、K 分成两半

V1 风格：把 head dimension 减半。V2 风格：保持 head dimension 不变，并把 head 数翻倍。这个玩具实现为了教学清晰，采用 V1 形式——数学完全一样，只是记账方式不同。

### 第 3 步：两个 softmax 分支 + 做差

```python
A1 = [softmax([dot(q1, k) / scale for k in K1]) for q1 in Q1]
A2 = [softmax([dot(q2, k) / scale for k in K2]) for q2 in Q2]
diff_weights = [[a1 - lam * a2 for a1, a2 in zip(r1, r2)] for r1, r2 in zip(A1, A2)]
out = [[sum(w * v[j] for w, v in zip(row, V)) for j in range(d_v)] for row in diff_weights]
```

注意：输出权重可以是负值。这没有问题——value cache 仍然可以处理带符号的贡献。后续的 V 投影会吸收这个符号。

### 第 4 步：测量噪声抵消

构造一个长度为 1024 的合成序列。把信号 token 放在一个已知位置，其余位置填满噪声。分别计算：（a）标准 softmax 注意力在信号位置上的权重；（b）差分注意力在该位置上的权重。再测量两者各自的信噪比。DIFF 注意力会稳定地产生更高的信噪比，通常能比标准 softmax 高出 3x–10x，具体取决于两个分支被训练出多大的差异。

### 第 5 步：V1 与 V2 的参数核算

给定一个配置（hidden=4096、heads=32、d_head=128），打印：

- 基线 Transformer：Q、K、V 各自大小为 `hidden * hidden`，MLP 为 `4 * hidden`。
- DIFF V1：Q、K 各自大小为 `hidden * hidden`，V 大小为 `hidden * hidden`（不变），内部 head dim 减半。额外增加逐头 `lambda` 参数（`O(heads * d_head)`）。
- DIFF V2：Q 的大小为 `2 * hidden * hidden`，K 为 `hidden * hidden`，V 为 `hidden * hidden`。多出来的维度会在 O_W 前投影回去。增加同样的 `lambda` 参数。

这个玩具程序会测量 V2 的额外参数成本（每个注意力块大约多一个 `hidden * hidden`），并把它打印出来。

## 使用

截至 2026 年 4 月，DIFF V2 还没有进入每一个生产级推理服务器，但 vLLM 和 SGLang 的集成已经在推进中。同时，这种模式已经出现在：

- Microsoft 内部长上下文生产模型中。
- 多个面向 256k 以上上下文的开源训练复现项目中。
- 将 DIFF 注意力与滑动窗口注意力交替叠加的混合架构中。

2026 年你会在这些情况下考虑它：

- 从零训练一个目标有效上下文为 64k+ 的新模型。最好从一开始就加入差分注意力；后面再重训会非常昂贵。
- 微调一个长上下文模型，而你的评测主要失败在 lost-in-the-middle。此时可以在 Q 投影上加 LoRA，近似出 DIFF 结构。

而这些场景通常不值得：

- 你正在服务一个已经预训练完成、长上下文表现稳定的稠密模型。为了现有权重重训，通常得不偿失。
- 你的上下文始终低于 16k。噪声底几乎可以忽略。

## 交付

本课会产出 `outputs/skill-diff-attention-integrator.md`。给定模型架构、目标上下文长度、幻觉画像和训练预算，它会生成一份把差分注意力接入新预训练运行或 LoRA 微调的集成计划。

## 练习

1. 运行 `code/main.py`。验证在合成查询上，差分注意力报告出的信噪比高于标准 softmax 注意力。改变噪声幅度，并找出标准注意力开始不可用的拐点。

2. 对一个 7B 级模型（hidden=4096、heads=32、d_head=128、32 层），分别计算基线到 DIFF V1、以及基线到 DIFF V2 的参数量增量。指出哪些组件增加了参数，哪些保持不变。

3. 阅读 DIFF V1 论文（arXiv:2410.05258）第 3 节与 DIFF V2 Hugging Face 博客第 2 节。用两句话解释：为什么 V1 的逐头 RMSNorm 是必要的，以及为什么 V2 能移除它而不导致训练发散。

4. 做一个消融实验：分别用 `lambda = 0`（纯第一支 softmax）和 `lambda = 1`（完整减法）计算差分注意力。在合成查询上，测量信噪比如何随 `lambda` 变化。找出让信噪比最大的 `lambda`。

5. 把这个玩具扩展到 GQA + DIFF V2。选择 8 个 KV 头和 32 个 Q 头。证明它的 KV cache 大小，与拥有相同 `(8, 32)` 配置的基线 GQA 模型一致。

## 关键术语

| 术语 | 人们常说什么 | 它真正的含义 |
|------|----------------|------------------------|
| 差分注意力 | “两个 softmax 相减” | 把 Q、K 分成两半，计算两张 softmax 图，把第二张按 lambda 缩放后从第一张里减掉，再与 V 相乘 |
| 噪声底 | “softmax 的非零尾部” | softmax 分给每个无关 token 的 `O(1/N)` 权重，在长上下文中累加后会变成 `O(1)` |
| lambda | “减法系数” | 逐头可学习标量，参数化为 `exp(lq1.lk1) - exp(lq2.lk2) + lambda_init`；可以为负 |
| DIFF V1 | “ICLR 2025 版本” | 最初版 Differential Transformer；通过减半 head dim 保持参数量，需要自定义 kernel，解码更慢 |
| DIFF V2 | “2026 年 1 月修正版” | 把 Q 头数量翻倍但保持 KV 头不变；解码速度与基线一致，并兼容 FlashAttention |
| 逐头 RMSNorm | “V1 的稳定器” | V1 在做差后应用的额外归一化；V2 为避免训练后期不稳定将其移除 |
| 信噪比 | “多少注意力被浪费了” | 真实信号位置上的权重，与无关位置平均权重之间的比值 |
| Lost in the middle | “长上下文失败模式” | 一种经验现象：长上下文中位于中间的文档检索准确率下降——DIFF 注意力能缓解它 |
| 算术强度 | “每加载一字节做多少 FLOPs” | V2 在解码时通过每次 KV 加载对应更多查询计算而提高的比率；对内存受限解码很重要 |

## 延伸阅读

- [Ye et al. — Differential Transformer (arXiv:2410.05258, ICLR 2025)](https://arxiv.org/abs/2410.05258) —— 原始论文，包含噪声抵消理论与长上下文消融实验
- [Microsoft unilm — Differential Transformer V2 (Hugging Face blog, January 2026)](https://huggingface.co/blog/microsoft/diff-attn-v2) —— 面向生产栈的重写版，解码速度对齐基线且兼容 FlashAttention
- [Understanding Differential Transformer Unchains Pretrained Self-Attentions (arXiv:2505.16333)](https://arxiv.org/abs/2505.16333) —— 从理论上分析为什么减法能恢复预训练注意力结构
- [Shared DIFF Transformer (arXiv:2501.17900)](https://arxiv.org/html/2501.17900) —— 参数共享变体
- [Vaswani et al. — Attention Is All You Need (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762) —— DIFF 所减去的那个基线 Transformer
- [Liu et al. — Lost in the Middle (arXiv:2307.03172)](https://arxiv.org/abs/2307.03172) —— DIFF 注意力瞄准的长上下文基准
