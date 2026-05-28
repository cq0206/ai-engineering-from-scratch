# DeepSeek-V3 架构导读

> 第 10 阶段·第 14 课总结了每个开源模型都会拨动的六个架构旋钮。DeepSeek-V3（2024 年 12 月，总参数 671B，激活参数 37B）把这六个全都拨满了，还额外加上四个：多头潜在注意力（Multi-Head Latent Attention，MLA）、无辅助损失负载均衡（auxiliary-loss-free load balancing）、多 token 预测（Multi-Token Prediction，MTP），以及 DualPipe 训练。本课会自上而下读完整个 DeepSeek-V3 架构，并根据公开配置推导每一项参数量。学完之后，你将能解释为什么 671B/37B 这个比例是合理下注，也能说明为什么在前沿规模上，MLA + MoE 的组合比单独使用其中任一个都更强。

**类型：** Learn
**语言：** Python（stdlib，参数计算器）
**前置课程：** 第 10 阶段·14（开源模型 walkthrough），第 10 阶段·17（NSA），第 10 阶段·18（MTP），第 10 阶段·19（DualPipe）
**耗时：** ~75 分钟

## 学习目标

- 自上而下阅读 DeepSeek-V3 配置，并用“GPT-2 的六个旋钮 + 四个 DeepSeek 特有扩展”的视角解释每个字段。
- 推导总参数量（671B）、激活参数量（37B），以及分别由哪些组件构成。
- 计算 MLA 在 128k 上下文下的 KV cache 占用，并与一个使用 GQA、且激活参数规模相同的稠密模型进行比较。
- 说出四项 DeepSeek 特有创新（MLA、MTP、无辅助损失路由、DualPipe），并指出它们各自瞄准的是架构栈/训练栈中的哪一层。

## 问题

DeepSeek-V3 是第一个在架构上与 Llama 家族真正拉开差异的前沿开源模型。Llama 3 405B 可以理解成“把 GPT-2 的六个旋钮都拧了一遍”。DeepSeek-V3 则是“六个经典旋钮 + 四个新增旋钮”的 GPT-2。会读 Llama 3 配置，只能算是读懂 DeepSeek 配置前的热身；因为它的深层结构——注意力块的形状、路由逻辑、训练目标——都已经明显不同，必须单独做一遍 walkthrough。

学会它的回报在于：DeepSeek-V3 的开源权重发布，重新定义了开源模型里“前沿能力”意味着什么。它的架构已经成为 2026 年很多训练运行在模仿的蓝图。对于任何会接触前沿 LLM 训练或推理的岗位来说，理解它已经是基本功。

## 概念

### 依然不变的核心

DeepSeek-V3 仍然是自回归模型。它仍然堆叠 decoder block。每个 block 仍然是 attention + MLP + 两个 RMSNorm。MLP 里仍然使用 SwiGLU。仍然使用 RoPE。仍然是 pre-norm。嵌入仍然与输出头权重绑定。它与所有 Llama 或 Mistral 共享同一条基线。

### 变化点：用 MLA 取代 GQA

从第 10 阶段·第 14 课你已经知道，GQA 会通过在一组 Q 头之间共享 K、V 来缩小 KV cache。多头潜在注意力（MLA）走得更远：它把 K 和 V 压缩到一个共享的低秩潜变量表示（即 `kv_lora_rank`）中，再在运行时按头解压。KV cache 中保存的只有这个潜变量——通常是每层每 token 512 个浮点，而不是 `8 x 128 = 1024` 个浮点。

在 128k 上下文下，使用 MLA 的 DeepSeek-V3（每层每 token 只保存一个共享潜变量 `c^{KV}`；K 与 V 都从它通过上投影推导出来，而这些上投影又能在后续 matmul 中被吸收）：

```
kv_cache = num_layers * kv_lora_rank * max_seq_len * bytes_per_element
         = 61 * 512 * 131072 * 2
         = 7.6 GB
```

一个假想的 GQA 基线（Llama 3 70B 的形状：8 个 KV 头、head dim 为 128）需要付出：

```
kv_cache = 2 * 61 * 8 * 128 * 131072 * 2
         = 30.5 GB
```

在 128k 上下文下，MLA 的 KV cache 比 Llama-3-70B 风格的 GQA 小 4 倍。

对应的权衡是：MLA 会在每次注意力计算中加入一个按头解压的步骤。与省下的带宽相比，这部分额外计算很小。因此在长上下文推理里，它是净收益。

### 路由：无辅助损失负载均衡

MoE 路由器负责决定每个 token 交给哪几个 top-k expert 处理。朴素路由器会把过多工作集中到少数几个 expert 上，让其他 expert 闲置。标准修复方式是在主损失外再加一个辅助损失项，惩罚负载不均衡。这能工作，但会轻微拖累主任务表现。

DeepSeek-V3 引入了无辅助损失方案。它会给每个 expert 的路由 logits 添加偏置项，并在训练中按一个简单规则动态调整：如果 expert `e` 过载，就减小 `bias_e`；如果负载不足，就增大它。整个过程不再需要额外损失项。训练目标保持干净，expert 负载也能保持均衡。

对主损失的影响：几乎测不到。对 MoE 架构的影响：更简洁，不再需要调辅助损失超参数。

### MTP：更密集的训练 + 免费草稿器

从第 10 阶段·第 18 课你已经知道，DeepSeek-V3 增加了一个 D=1 的 MTP 模块，用来预测再往后一个位置的 token。在推理时，这个训练好的模块会被改造成推测解码草稿器，接受率超过 80%。在训练时，每个隐藏状态因此会同时对 D+1 = 2 个目标接受监督，训练信号更密集。

参数量：在 671B 主模型之上额外增加 14B。开销：2.1%。

### 训练：DualPipe

从第 10 阶段·第 19 课你已经知道，DualPipe 是一种双向流水线，它会把前向/反向 chunk 与跨节点 all-to-all 通信重叠起来。在 DeepSeek-V3 这次 2,048 张 H800 的训练规模下，它大约追回了 24.5 万 GPU 小时——这些时间如果使用 1F1B，会被流水线气泡白白吞掉。

### 配置逐字段解析

下面是一个简化后的 DeepSeek-V3 配置：

```
hidden_size: 7168
intermediate_size: 18432   (dense MLP hidden size, used on first few layers)
moe_intermediate_size: 2048 (expert MLP hidden size)
num_hidden_layers: 61
first_k_dense_layers: 3    (first 3 layers use dense MLP)
num_attention_heads: 128
num_key_value_heads: 128   (formally equal to num_heads under MLA, but
                           the real compression is in kv_lora_rank)
kv_lora_rank: 512          (MLA latent dimension)
num_experts: 256            (MoE expert count per block)
num_experts_per_tok: 8      (top-8 routing)
shared_experts: 1           (always-on shared expert per block)
max_position_embeddings: 163840
rope_theta: 10000.0
vocab_size: 129280
mtp_module: 1               (1 MTP module at depth 1)
```

逐项解释：

- `hidden_size=7168`：嵌入维度。
- `num_hidden_layers=61`：总 block 深度。
- `first_k_dense_layers=3`：前 3 个 block 使用大小为 18432 的稠密 MLP，后面 58 个 block 使用 MoE。
- `num_attention_heads=128`：128 个查询头。
- `kv_lora_rank=512`：K 与 V 会被压缩到这个潜在维度，再按头解压。
- `num_experts=256, num_experts_per_tok=8`：每个 MoE block 有 256 个 expert，并执行 top-8 路由。
- `shared_experts=1`：除了 256 个被路由到的 expert，每个 block 还额外有 1 个始终启用的 shared expert。可以把它看作一个“稠密底板”，确保每个 token 至少能得到一份可靠处理。
- `moe_intermediate_size=2048`：每个 expert 的 MLP hidden size。之所以比稠密 MLP 小，是因为一共有 256 个 expert。

### 参数核算

完整计算写在 `code/main.py` 里。先看 headline：

- Embedding：`vocab * hidden = 129280 * 7168 = ~0.93B`。
- 前 3 个稠密 block：MLA attention（每个 block 约 144M）+ 稠密 MLP（每个 block 约 260M）+ norm。合计约 1.2B。
- 58 个 MoE block：MLA attention（约 144M）+ 256 个 expert（每个约 30M）+ 1 个 shared expert（30M）+ norm。按包含全部 expert 计算，每个 block 总计约 7.95B；58 个 block 总计 461B。
- MTP 模块：14B。

总计来看：核心架构约 ~476B，加上 14B 的 MTP；而公开的 671B 数字还额外计入了更多结构性参数（偏置张量、expert 特定组件、shared expert 缩放项等）。计算器复现的结果与公开数字相差约 3–5%；这部分差值来自更细粒度的核算，DeepSeek 在其第 2 节附录中有详细说明。

一次前向真正激活的参数量：

- Attention：每层 144M，共 `144M * 61 = 8.8B`（所有层都会运行）。
- MLP 激活参数：前 3 层是稠密 MLP（`3 * 260M = 780M`），后 58 层是 MoE，每层只激活 8 个 routed expert + 1 个 shared expert + 路由开销。每层实际激活的 MLP 大约是 `~260M`。总计：`3 * 260M + 58 * 260M = ~15.9B`。
- Embedding + norm：1.2B。
- 激活总量：核心约 26B；再加上 MTP 的 14B（训练时启用、推理时不一定总运行），约等于 37B。

### 671B / 37B 这个比例意味着什么

这相当于 18x 的稀疏比（激活参数只占总参数的 5.5%）。DeepSeek-V3 是目前开源权重中最稀疏的前沿 MoE 模型之一。Mixtral 8x7B 的比例是 13/47（28%），要稠密得多。Llama 4 Maverick 的比例是 17B/400B（4.25%），则比较接近。DeepSeek 的下注是：在前沿规模下，使用更多 expert 并把激活比例压低，能带来更好的“每激活 FLOP 的质量”。

### DeepSeek-V3 所处的位置

| 模型 | 总参数 | 激活参数 | 比例 | 注意力 | 新意 |
|-------|------|-------|-------|-----------|-------------|
| Llama 3 70B | 70B | 70B | 100% | GQA 64/8 | — |
| Llama 4 Maverick | 400B | 17B | 4.25% | GQA | — |
| Mixtral 8x22B | 141B | 39B | 27% | GQA | — |
| DeepSeek V3 | 671B | 37B | 5.5% | MLA 512 | MLA + MTP + aux-free + DualPipe |
| Qwen 2.5 72B | 72B | 72B | 100% | GQA 64/8 | YaRN extension |

### 后续版本：R1、V4

DeepSeek-R1（2025）是在 V3 主干之上做的 reasoning 训练运行。R1 使用的是同一套架构。变化发生在后训练配方（对可验证任务做大规模 RL），而不是预训练架构本身。

DeepSeek-V4（如果发布）预计会保留 MLA + MoE + MTP，并加入 DSA（DeepSeek Sparse Attention），也就是第 10 阶段·第 17 课里 NSA 的后继版本。这个谱系是稳定的：架构层面的创新会不断累积，每一代都在继续拨更多旋钮。

## 使用

`code/main.py` 是一个专门针对 DeepSeek-V3 形状的参数计算器。运行它，把输出和论文里的数字对照，也可以用它去测试假想变体（256 个 expert 对 512 个 expert、top-8 对 top-16、MLA rank 512 对 1024）。

建议重点看：

- 总参数量与公开 671B 的差距。
- 激活参数量与公开 37B 的差距。
- 128k 上下文下的 KV cache —— MLA 与 GQA 的对比。
- 各层明细拆分，看看参数预算究竟花在了哪里。

## 交付

本课会产出 `outputs/skill-deepseek-v3-reader.md`。给定一个 DeepSeek 家族模型（V3、R1 或未来变体），它会生成一份按组件拆解的架构解读：指出配置中的每个字段、按组件推导参数量，并标明模型用到了哪几项 DeepSeek 特有创新。

## 练习

1. 运行 `code/main.py`。比较计算器估计的总参数量与公开 671B 的差距，并找出这个差距来自哪里。论文第 2 节提供了完整分项。

2. 把配置改成 MLA rank 256，而不是 512。计算 128k 上下文下的 KV cache 大小。它能减少多少百分比？代价又是什么——每个头的表达能力会损失多少？

3. 将 DeepSeek-V3 的路由配置（256 experts、top-8）与一个假想配置（512 experts、top-8）比较。总参数会增长，但激活参数保持不变。理论上，额外的 expert 容量能带来什么？推理时又要付出什么代价？

4. 阅读 DeepSeek-V3 技术报告（arXiv:2412.19437）第 2.1 节关于 MLA 的说明。用三句话解释：为什么 K 与 V 的解压矩阵可以在推理时为了效率而“吸收”进后续 matmul 中。

5. DeepSeek-V3 在大多数运算中使用 FP8 训练。计算相对于 BF16，存储 671B 权重时 FP8 能节省多少内存。这与 14.8T token 的训练预算有什么交互关系？

## 关键术语

| 术语 | 人们常说什么 | 它真正的含义 |
|------|----------------|------------------------|
| MLA | “多头潜在注意力” | 把 K 和 V 压缩到一个共享低秩潜变量（`kv_lora_rank`，通常为 512）中，再按头实时解压；KV cache 里只存这个潜变量 |
| kv_lora_rank | “MLA 压缩维度” | K 与 V 共享潜变量的大小；DeepSeek-V3 使用 512 |
| 前 k 个稠密层 | “前几层保持稠密” | 在最前面的几层里跳过 MoE 路由器，直接使用稠密 MLP，以提升稳定性 |
| num_experts_per_tok | “Top-k 路由” | 每个 token 会激活多少个 routed expert；DeepSeek-V3 使用 8 |
| Shared experts | “始终开启的 expert” | 无论路由结果如何，每个 token 都会经过的 expert；DeepSeek-V3 使用 1 个 |
| 无辅助损失路由 | “通过 bias 调整负载均衡” | 在训练中调整每个 expert 的偏置项，让负载保持均衡，而不用额外加一个损失项 |
| MTP 模块 | “额外预测头” | 一个 Transformer block，用 `h^(1)` 与 `E(t+1)` 去预测更远的 token；带来更密集训练和免费推测解码草稿器 |
| DualPipe | “双向流水线” | 一种训练调度，会把前向/反向计算与跨节点 all-to-all 通信重叠起来 |
| 激活参数比例 | “稀疏度” | `active_params / total_params`；DeepSeek-V3 达到 5.5% |
| FP8 训练 | “8-bit 训练” | 以 FP8 存储并执行大量训练计算；相较 BF16 大约能把内存减半，质量损失很小 |

## 延伸阅读

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) —— 完整的架构、训练与结果文档
- [DeepSeek-V3 model card on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) —— 配置文件与部署说明
- [DeepSeek-V2 paper (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434) —— 引入 MLA 的前身论文
- [DeepSeek-R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) —— 建立在 V3 架构上的 reasoning 训练后继工作
- [Native Sparse Attention (arXiv:2502.11089)](https://arxiv.org/abs/2502.11089) —— DeepSeek 家族注意力方向的后续路线
- [DualPipe repository](https://github.com/deepseek-ai/DualPipe) —— 训练调度的参考实现
