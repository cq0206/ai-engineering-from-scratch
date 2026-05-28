# LLaVA 与视觉指令微调（Visual Instruction Tuning）

> LLaVA（2023 年 4 月）是地球上被复制最多的多模态架构。它用一个 2 层 MLP 替代了 BLIP-2 的 Q-Former，用朴素的 token 拼接替代了 Flamingo 的门控交叉注意力（gated cross-attention），并基于由 GPT-4 通过纯文本标题生成的 15.8 万条视觉指令对话进行训练。凡是在 2023 到 2026 年间构建过 VLM 的实践者，几乎都做过某种 LLaVA 变体。LLaVA-1.5 引入了 AnyRes。LLaVA-NeXT 提高了分辨率。LLaVA-OneVision 则用一套统一配方同时处理单图、多图和视频。本课会拆解这套配方、实现 projector，并解释为什么“更简单的方案赢了”。

**类型：** 构建
**语言：** Python（stdlib，projector + instruction-template builder）
**先修要求：** 第 12 阶段 · 02（CLIP），第 11 阶段（LLM Engineering — instruction tuning）
**时间：** ~180 分钟

## 学习目标

- 构建一个 2 层 MLP projector，把 ViT patch embeddings（维度 1024）映射到 LLM 的 embedding 维度（维度 4096）。
- 走通 LLaVA 的两阶段配方： (1) 在 55.8 万组标题对上做 projector 对齐，(2) 在 15.8 万条由 GPT-4 生成的对话上做视觉指令微调。
- 构造一个符合 LLaVA 格式的 prompt，包含图像 token 占位符、system prompt，以及 user/assistant 多轮对话。
- 解释为什么社区最终从 Q-Former 转向 MLP，尽管 Q-Former 在 token 预算上更占优。

## 问题

BLIP-2 的 Q-Former（课程 12.03）可以把一张图像压缩为 32 个 token。整洁、高效、基准表现不错。但它有两个问题。

首先，Q-Former 是可训练的，但它的损失并不是最终任务本身。第 1 阶段训练的是 ITC+ITM+ITG。第 2 阶段训练的是 LM loss。query 学到的是某种中间表示，而 LLM 之后还要再去解码它。信息会在这个瓶颈里丢失。

其次，Q-Former 有 1.88 亿参数，而在 LLaVA 2023 年的规模下，你必须把它和目标 LLM 一起协同设计。换一个 LLM，就得重训 Q-Former。换一个视觉编码器，也得重训。每一种组合都是一个单独的研发项目。

LLaVA 的答案简单得近乎尴尬：取 ViT 的 576 个 patch token，让每个 token 经过一个 2 层 MLP（`1024 → 4096 → 4096`），然后把这 576 个 token 全部塞进 LLM 的输入序列。没有瓶颈。没有围绕奇怪目标的第 1 阶段预训练。只需要用直接的 LM loss 来训练 MLP。

数据从哪里来？LLaVA 的第二个洞见是：用 GPT-4（纯文本）来生成指令数据。把一张图像的 COCO 标题和边界框数据喂给 GPT-4，请它产出对话、描述和复杂推理问题。15.8 万条 instruction-response 对话几乎白拿。无需人工标注。

结果是：一个只用 8 张 A100 跑一天的 VLM，在 MMMU 上击败了 Flamingo，并发布了一个社区可扩展的开源 checkpoint。到 2023 年底，它已经衍生出 50+ 个分支。

## 概念

### 架构

LLaVA-1.5 13B 版本：
- Vision encoder：CLIP ViT-L/14 @ 336（第 1 阶段冻结，第 2 阶段可选解冻）。
- Projector：2 层 MLP，带 GELU 激活，`1024 → 4096 → 4096`。
- LLM：Vicuna-13B（后续也用了 Llama-3.1-8B）。

对“图像 + 文本 prompt”的前向过程：

```
img -> ViT -> 576 patches of dim 1024
patches -> MLP -> 576 tokens of dim 4096
prompt: system + "<image>" placeholder + user question
replace <image> token with the 576 projected tokens
feed the full sequence to the LLM
decode response
```

图像会占用 LLM 上下文中的 576 个 token。在 2048 上下文里，这意味着文本还剩 1472 个 token。在 32k 上下文里，这几乎只是舍入误差。

### 第 1 阶段：projector 对齐

冻结 ViT。冻结 LLM。只训练这个 2 层 MLP。数据集：55.8 万组图像-标题对（LAION-CC-SBU）。损失：在投影后的图像 token 条件下，对标题做 language modeling。

如果 batch 为 128，只需一个 epoch，几个小时就能完成。projector 学会把 ViT 空间映射到 LLM 空间。没有任务特定监督。

### 第 2 阶段：视觉指令微调

解冻 projector（它仍然可训练）。解冻 LLM（通常全部解冻，有时用 LoRA）。在 15.8 万条视觉指令对话上训练。

指令数据才是真正的诀窍。Liu 等人生成它的方式是：
1. 取一张 COCO 图像。
2. 提取文本描述（5 条人工标题 + 边界框列表）。
3. 发送给 GPT-4，并使用三个 prompt 模板：
   - Conversation：“Generate a back-and-forth dialogue between a user and assistant about this image.”
   - Detailed description：“Give a rich, detailed description of the image.”
   - Complex reasoning：“Ask a question that requires reasoning about the image, then answer it.”
4. 将 GPT-4 的输出解析成 (instruction, response) 对。

这里完全没有直接接触图像——只使用文本描述。GPT-4 会幻觉出一些看似合理的图像内容。确实存在噪声，但它奏效了：15.8 万条对话足以解锁对话能力。

### 为什么社区都复制了这套方案

- 不需要为第 1 阶段特意调各种损失。全程都用 LM loss。
- Projector 训练耗时按小时计，而不是按天计。
- LLM 可替换（LLaVA-Llama2、LLaVA-Mistral、LLaVA-Llama3），只需重训 projector。
- 视觉指令数据流水线使用 GPT-4，而且可以低成本为新领域重新生成。

### LLaVA-1.5 与 LLaVA-NeXT

LLaVA-1.5（2023 年 10 月）新增：
- 将学术任务数据（VQA、OKVQA、RefCOCO）混入 instruction tuning。
- 更好的 system prompt。
- 上下文从 2048 扩展到 32k。

LLaVA-NeXT（2024 年 1 月）新增：
- AnyRes：把高分辨率图像切成 2x2 或 1x3 的 336x336 crop 网格，再加上一张全局低分辨率缩略图。每个 crop 变成 576 个 token；每张图像总计大约 2880 个视觉 token。OCR 和图表任务大幅提升。
- 更好的指令数据混合方案，加入 ShareGPT4V（高质量 GPT-4V 标题）。
- 更强的基础 LLM（Mistral-7B、Yi-34B）。

### LLaVA-OneVision

课程 12.08 会深入讲解 OneVision。简短版本是：使用同样的 projector，但通过一个覆盖单图、多图和视频的课程式训练（curriculum），在共享视觉 token 预算下用一个模型统一处理三者。

### 与 Q-Former 的对比

| | Q-Former（BLIP-2） | MLP（LLaVA） |
|---|---|---|
| 每张图像的视觉 token 数 | 32 | 576（基础）或 2880（AnyRes） |
| 可训练参数 | 188M + LM | 40M + LM |
| 第 1 阶段损失 | ITC+ITM+ITG | 仅 LM |
| LLM 即插即用 | 需要重训 | 仅需少量重训即可替换 |
| 多图 | 别扭 | 自然（concat） |
| 视频 | 别扭 | 自然（逐帧 concat） |
| Token 预算 | 小 | 大 |

MLP 赢在简单性和 token 灵活性。Q-Former 赢在 token 预算。到了 2023 年底，token 预算已经不再是主要瓶颈（LLM 上下文增长到 32k-128k+），简单性占据了上风。

### Prompt 格式

```
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image> Describe this image in detail. ASSISTANT: The image shows ...
```

`&lt;image>` 是一个占位 token。在分词之前，它会被替换成 576 个视觉 token（使用 AnyRes 时为 2880 个）。Tokenizer 看到的序列会比它训练时稍长，但由于第 1 阶段已经教会了 LLM 处理这种新输入，因此它可以应对。

### 参数经济性

LLaVA-1.5-7B 的组成：
- CLIP ViT-L/14 @ 336：303M（第 1 阶段冻结，第 2 阶段通常解冻）。
- Projector（2x linear）：约 22M 可训练参数。
- Llama-7B：7B。
- 总计：7.3B 参数。第 2 阶段可训练部分：完整 7B + 22M projector。

第 2 阶段训练成本：在 8xA100 上约 20 小时。这是最关键的数字——一天、一个节点、可复现。这正是 LLaVA 能快速传播的原因。

## 使用它

`code/main.py` 实现了：

1. 纯 Python 版 2 层 MLP projector（玩具规模为 dim 16 → 32 → 32）。
2. Prompt 构建流水线：system prompt + `&lt;image>` 替换成 N 个投影 token + user 轮次 + assistant 生成占位符。
3. 一个可视化工具，用于展示 576-token 视觉块在 LLM 上下文中的占比（消耗 2k / 32k / 128k 上下文的百分比）。

## 交付它

本课会生成 `outputs/skill-llava-vibes-eval.md`。给定一个 LLaVA 家族 checkpoint，它会运行一个包含 10 个 prompt 的 vibes-eval 套件（3 个 captioning、3 个 VQA、2 个 reasoning、2 个 refusal），并输出一份人类可读的评分卡。这不是正式基准；而是一个冒烟测试，用于确认 projector 与 LLM 是否连接良好。

## 练习

1. 计算 `1024 → 4096 → 4096` 这个 2 层 MLP projector 的可训练参数量。带上 GELU 和 bias 后，它占 LLaVA-13B 的比例是多少？

2. 为一个“拒绝（refusal）”场景构造 LLaVA prompt——图像中包含一位私人个体。写出你期望 assistant 给出的回应。为什么 LLaVA 应该在零样本条件下拒绝这一请求？要强化这种拒绝，还需要什么训练数据？

3. 阅读 LLaVA-NeXT 博客中的 AnyRes 部分。计算一张 1344x672 图像在 AnyRes 下的视觉 token 数量。再与 336x336 基础模式下的 576 个 token 对比。

4. LLaVA 第 1 阶段的 projector 是在标题上用 LM loss 训练的。如果你跳过第 1 阶段，直接进入第 2 阶段（视觉指令微调），会发生什么？请引用 Prismatic VLMs 的消融实验（arXiv:2402.07865）回答。

5. LLaVA-Instruct-150k 使用 GPT-4 和 COCO 标题来生成指令。针对一个新领域（医学 X 光、卫星图像），描述生成领域指令的四步数据流水线。每一步可能出什么问题？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| Projector | “MLP bridge” | 用 GELU 把 ViT 维度映射到 LLM 维度的 2 层 MLP |
| Image token | “&lt;image> placeholder” | 推理前在 prompt 中被替换为 N 个投影视觉 token 的标记 |
| Visual instruction tuning | “LLaVA stage 2” | 在 GPT-4 生成的 (image, instruction, response) 三元组上训练 |
| Stage 1 alignment | “Projector pretraining” | 冻结 ViT 和 LLM，用标题上的 LM loss 训练 projector |
| AnyRes | “Multi-crop tiling” | 把高分辨率图像拆成 tile 网格，并拼接每个 tile 的视觉 token |
| LLaVA-Instruct | “GPT-4-generated” | 由 COCO 标题 + GPT-4 合成的 15.8 万组 instruction-response 对 |
| Vision encoder freeze | “Backbone locked” | CLIP 权重在第 1 阶段不更新，有时在第 2 阶段也不更新 |
| ShareGPT4V | “Better captions” | 由 GPT-4V 生成的 100 万条密集标题，用于更高质量的对齐 |
| VQA | “Visual question answering” | 回答关于图像的自由形式问题的任务 |
| Prismatic VLMs | “Design-space paper” | Karamcheti 2024 的消融论文，系统测试 projector 与数据选择 |

## 延伸阅读

- [Liu et al. — Visual Instruction Tuning (arXiv:2304.08485)](https://arxiv.org/abs/2304.08485) — LLaVA 论文。
- [Liu et al. — Improved Baselines with Visual Instruction Tuning (arXiv:2310.03744)](https://arxiv.org/abs/2310.03744) — LLaVA-1.5。
- [Chen et al. — ShareGPT4V (arXiv:2311.12793)](https://arxiv.org/abs/2311.12793) — 密集标题数据集。
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865) — 设计空间消融实验。
- [Li et al. — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326) — 统一单图、多图与视频。
