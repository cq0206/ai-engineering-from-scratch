# 专家混合（Mixture of Experts, MoE）

> 一个稠密（dense）的 70B Transformer 会为每个 token 激活全部参数。一个 671B 的 MoE 每个 token 只激活 37B 参数，却在所有基准测试上都胜过前者。稀疏性（sparsity）是这个十年里最重要的扩展思路。

**类型：** Build
**语言：** Python
**先修要求：** Phase 7 · 05（完整 Transformer）, Phase 7 · 07（GPT）
**时长：** ~45 分钟

## 问题

稠密 Transformer 在推理时的 FLOPs 等于它的参数量（前向传播再乘以 2）。把稠密模型继续做大，每个 token 都要承担全部计算成本。到 2024 年，前沿模型已经撞上了算力墙：如果想让模型明显更聪明，就需要让每个 token 的 FLOPs 以指数级增长。

MoE 打破了这种绑定关系。把每个 FFN 替换成 `E` 个独立专家（expert）+ 一个路由器（router），后者会为每个 token 选出 `k` 个专家。总参数量 = `E × FFN_size`。每个 token 的激活参数量 = `k × FFN_size`。2026 年的典型配置：`E=256`，`k=8`。存储规模随着 `E` 增长，计算规模随着 `k` 增长。

2026 年的前沿模型几乎清一色都是 MoE：DeepSeek-V3（总计 671B / 激活 37B）、Mixtral 8×22B、Qwen2.5-MoE、Llama 4、Kimi K2、gpt-oss。在 Artificial Analysis 的独立排行榜上，开源模型前 10 名全部都是 MoE。

## 概念

*MoE 层：路由器为每个 token 从 E 个专家中选出 k 个*

### FFN 替换

稠密 Transformer 模块：

```
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

MoE 模块：

```
h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # pick k of E per token
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
```

每个专家都是一个独立的 FFN（通常是 SwiGLU）。路由器则只是一个线性层。每个 token 都会选择属于自己的 `k` 个专家，并获得这些专家输出的门控（gated）混合结果。

### 负载均衡问题

如果路由器把 90% 的 token 都送进 3 号专家，其他专家就会“饿死”。目前尝试过三种修复方式：

1. **辅助负载均衡损失（auxiliary load-balancing loss）**（Switch Transformer、Mixtral）。增加一个与专家使用方差成比例的惩罚项。有效，但会多出一个超参数和第二路梯度信号。
2. **专家容量（expert capacity）+ token 丢弃**（早期 Switch）。每个专家最多处理 `C × N/E` 个 token；溢出的 token 会跳过这一层。会损伤质量。
3. **无辅助损失的均衡（auxiliary-loss-free balancing）**（DeepSeek-V3）。为每个专家加入一个可学习偏置（bias），用于平移路由器的 top-k 选择。偏置在训练损失之外更新。主目标函数没有额外惩罚项。这是 2024 年的重要突破。

DeepSeek-V3 的做法是：每个训练 step 之后，对每个专家检查其使用率是高于还是低于目标值。然后用 `±γ` 轻推偏置。选择时使用 `scores + bias`。但用于门控的专家概率仍然是未改变的原始 `scores`。这样就把“路由选择”和“表达能力”解耦了。

### 共享专家

DeepSeek-V2/V3 还把专家拆分为*共享（shared）*和*路由（routed）*两类。每个 token 都会经过所有共享专家。路由专家则通过 top-k 进行选择。共享专家负责捕获通用知识；路由专家负责专门化。V3 的配置是 1 个共享专家加上 256 个路由专家中的 top-8。

### 细粒度专家

经典 MoE（GShard、Switch）：每个专家都和完整 FFN 一样宽。`E` 较小（8–64），`k` 也较小（1–2）。

现代细粒度（fine-grained）MoE（DeepSeek-V3、Qwen-MoE）：每个专家更窄（约为 1/8 FFN 大小）。`E` 很大（256+），`k` 也更大（8+）。总参数量相同，但组合数量增长快得多。`C(256, 8) = 400 trillion`，也就是每个 token 都有 400 万亿种可能的“专家组合”。质量提升，而延迟保持不变。

### 成本画像

每层、每个 token：

| 配置 | 每个 token 的激活参数 | 总参数量 |
|--------|-----------------------|--------------|
| Mixtral 8×22B | ~39B | 141B |
| Llama 3 70B（稠密） | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |
| Kimi K2（MoE） | ~32B | 1T |

DeepSeek-V3 在几乎所有基准上都优于 Llama 3 70B（稠密），同时它的**每个 token 激活 FLOPs 更少**。更多参数 = 更多知识。更多激活 FLOPs = 每个 token 更多计算。MoE 将这两者解耦。

### 代价：内存

无论哪些专家会被激活，所有专家都必须驻留在 GPU 上。一个 671B 模型若使用 fp16 权重，大约需要 ~1.3 TB VRAM。部署前沿 MoE 时必须使用专家并行（expert parallelism）——把专家切分到多张 GPU 上，再通过网络路由 token。延迟的主导因素不是 matmul，而是 all-to-all 通信。

## 动手构建

见 `code/main.py`。这是一个只用 Python 标准库实现的紧凑版 MoE 层，包含：

- `n_experts=8` 个类 SwiGLU 专家（为便于说明，每个专家只用一个线性层）
- top-k=2 路由
- 经过 softmax 归一化的门控权重
- 通过每专家偏置实现的无辅助损失均衡

### 第 1 步：路由器

```python
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # softmax over ORIGINAL scores of the chosen experts
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

偏置影响选择，不影响门控权重。这就是 DeepSeek-V3 的技巧——偏置只负责修正负载不均衡，而不会去引导模型预测本身。

### 第 2 步：让 100 个 token 通过路由器

追踪每个专家被触发了多少次。没有偏置时，使用率会倾斜。有了偏置更新循环（对过度使用的专家施加 `-γ`，对使用不足的专家施加 `+γ`）后，几轮迭代之内，使用率就会收敛到均匀分布。

### 第 3 步：比较参数量

打印一个 MoE 配置对应的“稠密等价（dense equivalent）”。以 DeepSeek-V3 风格为例：256 个路由专家 + 1 个共享专家，8 个激活专家，d_model=7168。总参数量大得惊人；而激活参数量只有稠密版 Llama 3 70B 的七分之一。

## 使用

HuggingFace 加载方式：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

2026 年的生产推理中：vLLM 原生支持 MoE 路由。SGLang 则拥有最快的专家并行路径。这两者都会自动处理 top-k 选择和专家并行。

**什么时候适合选择 MoE：**
- 你想以更低的单 token 推理成本获得前沿质量。
- 你拥有足够的 VRAM / 专家并行基础设施。
- 你的工作负载是 token 密集型（聊天、代码），而不是上下文密集型（长文档）。

**什么时候不适合选择 MoE：**
- 边缘部署——无论激活多少 FLOP，你都要为全部存储买单。
- 面向单用户的超低延迟服务——专家路由会增加额外开销。
- 小模型（&lt;7B）——MoE 的质量优势只会在超过某个算力阈值（约 6B 激活参数）后显现。

## 交付

见 `outputs/skill-moe-configurator.md`。这个 skill 会根据参数预算、训练 token 数量和部署目标，为新的 MoE 选择 E、k 以及共享专家布局。

## 练习

1. **简单。** 运行 `code/main.py`。观察无辅助损失偏置更新如何在 50 次迭代内把专家使用率拉平。
2. **中等。** 将可学习路由器替换为基于哈希（hash-based）的路由器（确定性、不可学习）。比较质量和均衡性。为什么可学习路由器更好？
3. **困难。** 实现 GRPO 风格的“rollout-matched routing”（DeepSeek-V3.2 的技巧）：记录推理时哪些专家被激活，并在梯度计算时强制使用同样的路由。测量它在一个玩具级 policy-gradient 设置中的效果。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Expert | “多个 FFN 中的一个” | 一个独立的前馈网络；其参数只服务于 FFN 计算中的稀疏切片。 |
| Router | “门” | 一个很小的线性层，用来给每个 token 与每个专家打分；然后执行 top-k 选择。 |
| Top-k routing | “每个 token 激活 k 个专家” | 每个 token 的 FFN 计算只经过恰好 k 个专家，并由 gate 加权。 |
| Auxiliary loss | “负载均衡惩罚” | 一个额外的损失项，用于惩罚倾斜的专家使用率。 |
| Auxiliary-loss-free | “DeepSeek-V3 的技巧” | 只在路由器的选择阶段通过每专家偏置做均衡；没有额外梯度。 |
| Shared expert | “始终开启” | 每个 token 都会经过的额外专家；负责捕获通用知识。 |
| Expert parallelism | “按专家切分” | 把不同专家分布到不同 GPU 上；再通过网络路由 token。 |
| Sparsity | “激活参数 &lt; 总参数” | 比例为 `k × expert_size / (E × expert_size)`；对 DeepSeek-V3 来说是 37/671 ≈ 5.5%。 |

## 延伸阅读

- [Shazeer et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) —— 这一想法的起点。
- [Fedus, Zoph, Shazeer (2022). Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) —— Switch，经典 MoE。
- [Jiang et al. (2024). Mixtral of Experts](https://arxiv.org/abs/2401.04088) —— Mixtral 8×7B。
- [DeepSeek-AI (2024). DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) —— MLA + 无辅助损失 MoE + MTP。
- [Wang et al. (2024). Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) —— 基于偏置的负载均衡论文。
- [Dai et al. (2024). DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) —— 本课路由器采用的细粒度 + 共享专家拆分思路。
- [Kim et al. (2022). DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training](https://arxiv.org/abs/2201.05596) —— 最早的共享专家论文。
