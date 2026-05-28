# KV Cache、Flash Attention 与推理优化

> 训练是并行的，并且受 FLOP 限制。推理是串行的，并且受内存带宽限制。瓶颈不同，技巧也不同。

**类型：** Build
**语言：** Python
**先修要求：** Phase 7 · 02（Self-Attention）, Phase 7 · 05（完整 Transformer）, Phase 7 · 07（GPT）
**时长：** ~75 分钟

## 问题

一个朴素的自回归（autoregressive）解码器在生成 `N` 个 token 时需要做 `O(N²)` 的工作：每一步都会对整个前缀重新计算一次 attention。对于一个 4K token 的回复，这意味着 1600 万次 attention 运算，其中大部分都是重复的。前缀 token 的每个隐藏状态（hidden state）一旦算出就是确定的——你只需要让新 token 的 query 去和之前缓存好的全部 key 与 value 交互即可。

除此之外，attention 本身还会搬运大量数据。标准 attention 会显式生成一个 N×N 的分数矩阵、一个 N×d 的 softmax 输出，以及一个 N×d 的最终输出——对 HBM 的读写实在太多了。当 N≥2K 时，attention 往往会先受内存带宽限制，而不是先受 FLOP 限制。经典 attention kernel 对现代 GPU 的利用率通常低了 4–10×。

两个优化——都来自 Dao 等人——把前沿推理从“慢”推进到了“快”：

1. **KV cache。** 存储每个前缀 token 的 K 和 V 向量。每个新 token 的 attention 只需让一个 query 去访问缓存好的 keys。这样一来，推理在每个生成 step 上就从 `O(N²)` 降到 `O(N)`。
2. **Flash Attention。** 将 attention 计算分块（tile）执行，使完整的 N×N 矩阵永远不会落到 HBM 上。softmax + matmul 全都在 SRAM 中完成。在 A100 上有 2–4× 的实际速度提升；在 H100 + FP8 上可达 5–10×。

到 2026 年，这两者已经成为通用默认配置。所有生产级推理栈（vLLM、TensorRT-LLM、SGLang、llama.cpp）都默认依赖它们。所有前沿模型发布时也都会启用 Flash Attention。

## 概念

*KV cache 的增长与 Flash Attention 的分块计算*

### KV cache 数学

每个 decoder layer、每个 token、每个 head：

```
bytes_per_token_per_layer = 2 * d_head * dtype_size
                          ^
                          K and V
```

对于一个 7B 模型，若有 32 层、32 个头、d_head=128、fp16：

```
per token per layer = 2 * 128 * 2 = 512 bytes
per token (32 layers) = 16 KB
per 32K context = 512 MB
```

对于 Llama 3 70B（80 层、d_head=128、采用 GQA 且有 8 个 KV heads）：

```
per token per layer = 2 * 8 * 128 * 2 = 4096 bytes (4 KB)
per 32K context = 10.4 GB
```

这 10 GB 就是为什么 Llama 3 70B 在 batch size 为 1、上下文长度为 128K 时，仅 KV cache 就会吃掉一张 40 GB A100 上的大部分显存。

**GQA 是 KV cache 的关键收益点。** 如果使用 64 头的 MHA，就会变成 32 GB。MLA 还能压得更低。

### Flash Attention——分块技巧

标准 attention：

```
S = Q @ K^T          (HBM read, N×N, HBM write)
P = softmax(S)       (HBM read, HBM write)
O = P @ V            (HBM read, HBM write)
```

要往返 HBM 三次。在 H100 上，HBM 带宽大约是 3 TB/s；SRAM 是 30 TB/s。每多一次 HBM 往返，相比把数据一直留在芯片上，都会带来约 10 倍的减速。

Flash Attention 实现：

```
for each block of Q (tile size ~128 × 128):
    load Q_tile into SRAM
    for each block of K, V:
        load K_tile, V_tile into SRAM
        compute S_tile = Q_tile @ K_tile^T     (SRAM)
        running softmax aggregation             (SRAM)
        accumulate into O_tile                  (SRAM)
    write O_tile to HBM
```

每个 tile 只需要一次 HBM 往返。总内存占用从 `O(N²)` 降到 `O(N)`。在反向传播中，它还会选择重新计算前向传播里的某些值，而不是把它们存下来——这又带来一次内存收益。

**数值技巧。** 运行中的 softmax 会在多个 tile 之间维护 `(max, sum)`，因此最终归一化是精确的。它不是近似算法——Flash Attention 计算出的输出与标准 attention 在比特级上一致（忽略 fp16 非结合性带来的细微差异）。

**版本演进：**

| 版本 | 年份 | 关键变化 | 参考硬件上的加速比 |
|---------|------|-----------|-------------------------------|
| Flash 1 | 2022 | 基于 SRAM 的分块 kernel | A100 上 2× |
| Flash 2 | 2023 | 更好的并行性、以因果顺序优先的执行方式 | A100 上 3× |
| Flash 3 | 2024 | Hopper 异步机制、FP8 | H100 上 1.5–2×（~740 TFLOPs FP16） |
| Flash 4 | 2026 | Blackwell 五级流水线、software exp2 | 以推理优先（最初仅前向） |

Flash 4 在发布时只支持前向传播。训练仍然使用 Flash 3。Flash 4 对 GQA 和 varlen 的支持还在等待中（2026 年中）。

### Speculative decoding——另一项延迟优化

便宜模型先提议 N 个 token。大模型并行验证这 N 个 token。如果验证接受了其中 k 个，那你只花了 1 次大模型前向传播，就完成了 k 次生成。对于代码和自然语言，典型的 k=3–5。

2026 年默认方案：
- **EAGLE 2 / Medusa。** 集成式 draft heads，与 verifier 共享隐藏状态。可获得 2–3× 加速且无质量损失。
- **带 draft model 的 speculative decoding。** 在消费级硬件上可获得 2–4× 加速。
- **Lookahead decoding。** 基于 Jacobi 迭代；不需要 draft model。小众，但几乎白送。

### 连续批处理（continuous batching）

经典批处理推理：等待最慢的序列完成，然后再开始新的一批。当短回复先结束时，GPU 会被浪费掉。

连续批处理（最早由 Orca 发布，现在已进入 vLLM、TensorRT-LLM、SGLang）：一旦旧请求结束，就立刻把新请求换入批次。对于典型聊天负载，吞吐量可提升 5–10×。

### PagedAttention——把 KV cache 当成虚拟内存

这是 vLLM 的招牌特性。KV cache 以 16-token 为块进行分配；页表（page table）负责把逻辑位置映射到物理块。这样你就可以在并行采样（beam search、parallel sampling）之间共享 KV、为 prompt caching 热切换前缀，并对内存进行碎片整理。相比朴素的连续分配，吞吐量可提升 4×。

## 动手构建

见 `code/main.py`。我们会实现：

1. 一个朴素的 `O(N²)` 增量解码器。
2. 一个使用 KV cache 的 `O(N)` 解码器。
3. 一个分块 softmax，用来模拟 Flash Attention 的 running-max 算法。

### 第 1 步：KV cache

```python
class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)
        self.V[layer][head].append(v)

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]
```

很简单：按层、按头分别维护列表，不断追加每个 token 对应的 K、V 向量。

### 第 2 步：分块 softmax

```python
def tiled_softmax_dot(q, K, V, tile=4):
    """Flash-attention-style softmax(qK^T)V with running max/sum."""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
```

它的输出与一次性计算 `softmax(qK) V` 在比特级上一致，但任意时刻的工作集只是一个 `tile × d_head` 的小块，而不是完整的 `N × d_head`。

### 第 3 步：比较朴素解码与缓存解码在 100-token 生成上的差异

统计 attention 运算次数。朴素方式：`O(N²)` = 5050。缓存方式：`O(N)` = 100。代码会把两者都打印出来。

## 使用

```python
# HuggingFace transformers auto-enables KV cache on decoder-only generate().
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # use FA3 if Hopper
    torch_dtype="bfloat16",
)
# generate() uses KV cache automatically
```

vLLM 生产部署：

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

跨请求的 prefix caching 是 2026 年的一项重要收益——相同的 system prompt、few-shot 示例或长上下文文档都可以在多次调用间复用 KV。对于带有重复工具提示的 agent 工作负载，prefix caching 通常能带来 5× 吞吐量提升。

## 交付

见 `outputs/skill-inference-optimizer.md`。这个 skill 会为新的推理部署选择 attention 实现、KV cache 策略、量化方式以及 speculative decoding 方案。

## 练习

1. **简单。** 运行 `code/main.py`。确认朴素解码器和缓存解码器输出一致，并记录它们的运算次数差异。
2. **中等。** 实现 prefix caching：给定一个提示词 P 和若干补全结果，先对 P 跑一次前向传播填满 KV cache，然后为每个补全分叉。测量它相对于为每个补全都重新编码 P 的加速效果。
3. **困难。** 实现一个玩具版 PagedAttention：KV cache 以固定的 16-token 块存储，并维护一个 free-list。当一个序列结束时，把它占用的块归还给池子。模拟 1000 个不同长度的聊天补全。比较它与连续分配之间的内存碎片情况。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| KV cache | “让解码变快的技巧” | 存储每个前缀 token 的 K 和 V；新 query 直接关注它们，而不是重新计算。 |
| HBM | “GPU 主内存” | High Bandwidth Memory；H100 上为 80 GB，B200 上为 192 GB。带宽约 ~3 TB/s。 |
| SRAM | “片上内存” | 每个 SM 上的高速内存，H100 每个 SM 约 ~256 KB。带宽约 ~30 TB/s。 |
| Flash Attention | “分块 attention kernel” | 在不把 N×N 矩阵显式写入 HBM 的情况下完成 attention 计算。 |
| Continuous batching | “无需等待的批处理” | 把已完成的序列换出、把新序列换入，而无需先清空整个 batch。 |
| PagedAttention | “vLLM 的招牌功能” | KV cache 按固定块分配，并通过页表管理；可消除碎片。 |
| Prefix caching | “复用长提示词” | 在多个请求之间缓存共享前缀的 KV；可大幅降低 agent 成本。 |
| Speculative decoding | “草稿 + 验证” | 便宜的草稿模型先提议 token；大模型一次验证 k 个。 |

## 延伸阅读

- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) —— Flash 1。
- [Dao (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) —— Flash 2。
- [Shah et al. (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) —— Flash 3。
- [FlashAttention-4 release notes (Dao-AILab, 2026)](https://github.com/Dao-AILab/flash-attention) —— Blackwell 五级流水线与 software-exp2 技巧；请阅读仓库 README，了解本课提到的“仅前向发布”限制。
- [Kwon et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) —— vLLM 论文。
- [Leviathan et al. (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) —— speculative decoding。
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) —— EAGLE-1/2 论文，对应本课提到的集成式草稿方案。
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) —— 与 EAGLE 并列提到的 Medusa 方法。
- [vLLM docs — PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html) —— 关于 16-token 分块与页表设计的权威深潜资料。
