# 开放权重（Open-Weight）VLM 配方：真正重要的是什么

> 2024-2026 年的开放权重 VLM 文献像一片消融表（ablation table）森林。Apple 的 MM1 测试了 13 种图像编码器、连接器和数据混合的组合。Allen AI 的 Molmo 证明，细粒度人工标题优于 GPT-4V 蒸馏。Cambrian-1 跑了 20+ 组编码器对比。Idefics2 将设计空间正式归纳为五个轴。Prismatic VLMs 在受控基准上比较了 27 套训练配方。在所有这些噪声之中，跨论文都成立的结果其实只有少数几条：图像编码器比连接器架构更重要，数据混合比前两者都更重要，而细粒度人工标题优于蒸馏得到的合成数据。本课会替你把这些表读完。

**类型：** 学习 + 实验
**语言：** Python（stdlib，ablation table parser + recipe picker）
**先修要求：** 第 12 阶段 · 05（LLaVA baseline）
**时间：** ~180 分钟

## 学习目标

- 说出 VLM 设计空间的五个轴：image encoder、connector、LLM、data mix、resolution schedule。
- 阅读 MM1 / Idefics2 / Cambrian-1 的消融表，并预测某个 benchmark 会被哪个旋钮推动。
- 在给定算力预算与任务组合的情况下，为新的 VLM 选出一套配方（encoder、connector、data、resolution）。
- 解释为什么在相同 token 数下，细粒度人工标题优于 GPT-4V 蒸馏。

## 问题

开放权重 VLM 已经有数百种。“不错”和“最先进”之间的大部分差距并不来自架构，而是来自数据、分辨率调度（resolution schedule）以及编码器选择。知道当模型表现不佳时该先调哪个旋钮，能帮你避免一次耗费 500 万 GPU 小时的错误。

2023 年这一波（LLaVA-1.5、InstructBLIP、MiniGPT-4）采用的是标题对预训练 + LLaVA-Instruct-150k。作为基线已经不错，但 MMMU 大约止步于 35%。

2024 年这一波（MM1、Idefics2、Molmo、Cambrian-1、Prismatic VLMs）做了彻底的消融实验。结果既出人意料，又极具实用价值。

## 概念

### 五轴设计空间

Idefics2（Laurençon 等，2024）给这些轴命了名：

1. 图像编码器（image encoder）。CLIP ViT-L/14、SigLIP SO400m/14、DINOv2 ViT-g/14、InternViT-6B。编码器在 patch size、resolution 和预训练目标上各不相同。
2. 连接器（connector）。MLP（2-4 层）、Q-Former（32 个 query + cross-attn）、Perceiver Resampler（64 个 query）、C-Abstractor（convolutional + bilinear pooling）。
3. 语言模型（language model）。Llama-3 8B / 70B、Mistral 7B、Phi-3、Gemma-2、Qwen2.5。LLM 大小是参数成本的主导项。
4. 训练数据（training data）。标题对（CC3M、LAION）、交错数据（interleaved，如 OBELICS、MMC4）、指令数据（LLaVA-Instruct、ShareGPT4V、PixMo、Cauldron）。
5. 分辨率调度（resolution schedule）。固定 224/336/448、AnyRes、原生动态分辨率。可以在训练中逐步提升，也可以保持恒定。

每一个生产级 VLM 都会在每个轴上做出选择。MMMU 分数的大多数方差，都可以由轴 1、4 和 5 解释——而不是你选了哪种 connector。

### 轴 1：encoder > connector

MM1 第 3.2 节显示：把 CLIP ViT-L/14 换成 SigLIP SO400m/14，MMMU 会提升 3+ 分。把 connector 从 MLP 换成 Perceiver Resampler，提升不到 1 分。Idefics2 也复现了同样结论：SigLIP > CLIP，并且在相同 token 数下，Q-Former ≈ MLP ≈ Perceiver。

Cambrian-1 的 “Cambrian Vision Encoders Match-Up”（Tong 等，2024）在一个偏视觉中心（vision-centric）的 benchmark（CV-Bench）上比较了 20+ 个编码器。榜单顶部由 DINOv2 与 SigLIP 混合占据；CLIP 位于中游；ImageBind 与 ViT-MAE 更靠后。从 CLIP ViT-L 到 DINOv2 ViT-g/14，在 CV-Bench 上差距约为 5-7 分。

到 2026 年，开源 VLM 的默认编码器是 SigLIP 2 SO400m/14，兼顾语义特征与密集特征；有时还会与 DINOv2 ViT-g/14 特征拼接（Cambrian 的 “Spatial Vision Aggregator” 就是这么做的）。

### 轴 2：connector 设计基本不分胜负

MM1、Idefics2、Prismatic 和 MM-Interleaved 都得出了同一个结论：在视觉 token 数固定时，connector 架构几乎不重要。对平均池化后的 patch 使用一个 2 层 MLP，在相同 token 预算下，其表现与一个 32-query Q-Former 的差距不到 1 分。

真正重要的是 token 数。视觉 token 越多 = LLM 计算越多 = 在一定范围内性能越好，但之后会进入收益递减。每张图像 64 个 token 对 OCR 来说太少。576-1024 个 token 是大多数开源 VLM 的甜蜜点。2048+ 只在文档和图表任务上才明显有帮助。

Q-Former 与 MLP 的差异本质上是成本问题，而不是质量问题：Q-Former 不管图像分辨率多高，都把 token 限制在 32-64；MLP 则会输出全部 patch token。面对高分辨率输入时，Q-Former 能节省 LLM 上下文；对于低分辨率输入，这种差异几乎可以忽略。

### 轴 3：LLM 大小决定上限

把 LLM 从 7B 翻倍到 13B，在几乎每篇 VLM 论文中都能稳定地为 MMMU 增加 2-4 分。到 70B 时，大多数 benchmark 都接近饱和。VLM 的多模态推理上限，就是 LLM 的文本推理上限——视觉编码器只能给它喂信息，不能替它思考。

这就是为什么 Qwen2.5-VL-72B 和 Claude Opus 4.7 能横扫 MMMU-Pro 与 ScreenSpot-Pro：语言大脑足够大。一个 7B VLM 不可能靠巧妙的 connector 设计替代一个 70B VLM。

### 轴 4：数据——细粒度人工标题优于蒸馏

Molmo + PixMo（Deitke 等，2024）是 2024 年每个人都应该读的结果。Allen AI 让人工标注员用 1-3 分钟的密集语音转文本方式描述图像，得到 71.2 万张带密集标题的图像。整个训练数据中没有任何 GPT-4V 蒸馏。

Molmo-72B 在 11/11 个 benchmark 上击败了 Llama-3.2-90B-Vision。差异不在架构，而在标题质量。细粒度人工标题每张图像携带的信息量，比简短网页标题高 5-10 倍；而且它们保持事实扎根，不像 GPT-4V 蒸馏那样容易产生幻觉。

ShareGPT4V（Chen 等，2023）和 Cauldron（Idefics2）也采用了类似路线：混合人工标题与 GPT-4V 标题。趋势很明确：对 2026 年前沿模型而言，标题密度 > 标题数量 > 蒸馏便利性。

### 轴 5：分辨率与其调度

Idefics2 的消融显示：384 -> 448 可增加 1-2 分。448 -> 980 再配合图像切分（AnyRes），在 OCR benchmark 上还能再增 3-5 分。固定分辨率训练会在中等准确率处平台化；分辨率递增训练（从 224 开始，最终到 448 或原生分辨率）训练更快，最终精度也更高。

Cambrian-1 做了“分辨率 vs token”权衡实验：在固定算力下，你可以选择“更低分辨率但更多 token”，也可以选择“更高分辨率但更少 token”。对于 OCR，更高分辨率更占优；对于一般场景理解，较低分辨率配更多 token 更好。

2026 年的生产配方是：第 1 阶段在固定 384 分辨率上训练，第 2 阶段对 OCR 密集任务使用最高到 1280 的动态分辨率。

### Prismatic 的受控对比

Prismatic VLMs（Karamcheti 等，2024）是那篇真正控制了所有轴的论文。相同的 13B LLM、相同的指令数据、相同的评估——每次只改变一个轴。结果如下：

- 每图视觉 token 数解释了约 60% 的方差。
- 编码器选择解释了约 20%。
- Connector 架构解释了约 5%。
- 其余一切（数据混合、scheduler、LR）解释剩下约 15%。

这只是一个粗略分解，但它是文献中对“我该先做哪个消融”最干净的回答。

### 2026 年的选择器

基于这些证据，2026 年新项目的默认 open-VLM 配方是：

- Encoder：SigLIP 2 SO400m/14，使用 NaFlex 原生分辨率；如果你需要分割/grounding 等密集特征，则再拼接 DINOv2 ViT-g/14。
- Connector：对 patch token 使用 2 层 MLP。除非你极度受限于 token，否则跳过 Q-Former。
- LLM：Qwen2.5 / Llama-3.1 / Gemma 2，7B 用于控制成本，70B 用于追求质量，具体按目标延迟选择。
- Data：PixMo + ShareGPT4V + Cauldron，再补充任务特定的指令数据。
- Resolution：动态分辨率（长边最小 256、最大 1280 像素）。
- Schedule：第 1 阶段对齐（只训练 projector），第 2 阶段完整微调，第 3 阶段任务特定微调。

这些默认值中的每一项，都可以追溯到本课末尾所引用论文中的实测消融结果。

## 使用它

`code/main.py` 是一个消融表解析器和配方选择器。它编码了 MM1 与 Idefics2 的消融表（浓缩版），并允许你查询：

- “给定预算 X 和任务 Y，哪套配方最优？”
- “如果我在 7B Llama 上把 SigLIP 换成 CLIP，预期 MMMU 差值是多少？”
- “如果我只想得到一个 80% 可信度的答案，应该先消融哪个轴？”

输出会是一个按排名排序的配方列表，包含预期 benchmark 差值，以及“优先消融项”的建议。

## 交付它

本课会生成 `outputs/skill-vlm-recipe-picker.md`。给定目标任务组合、算力预算和延迟目标，它会输出一整套配方（encoder、connector、LLM、data mix、resolution schedule），并为每个选择附上相应消融结果的引用依据。这样工程师就不必在每个新 VLM 项目启动时都重新发明一次 Idefics2 消融表。

## 练习

1. 阅读 MM1 第 3.2 节。在固定 2B LLM、预算为 5000 万张图像时，哪个 encoder 最优？如果换成 13B LLM，答案会反转吗？为什么？

2. Cambrian-1 发现，在偏视觉 benchmark 上，将 DINOv2 + SigLIP 拼接优于单独使用任一编码器，但在 MMMU 上不会带来额外信号。预测哪些 benchmark 会受益，哪些会保持不变。

3. 你的目标是构建一个基于 2B LLM 的移动端 UI agent。请选定 encoder、connector、resolution 和 data mix。并用具体的消融表为每个选择给出依据。

4. Molmo 提供了 4B 和 72B 模型。4B 已经能与封闭式 7B VLM 竞争；72B 在 11/11 个 benchmark 上击败了 Llama-3.2-90B-Vision。这说明了什么关于“LLM 尺寸平台期”这一假设？

5. 设计一个消融表，用于在 7B VLM 上把 data mix 质量与 encoder 质量区分开。最少需要多少次训练？请提出这四个轴设置。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|----------|
| Ablation | “一次只拧一个旋钮” | 训练多个仅在设计空间某一个轴上不同的实验，其余全部保持不变 |
| Connector | “Bridge” / “projector” | 把视觉编码器输出映射到 LLM token 空间的可训练模块（MLP、Q-Former、Perceiver） |
| Detailed human caption | “Dense caption” | 由人工撰写的多句描述（通常 80-300 token），比网页 alt text 丰富得多 |
| Distillation | “GPT-4V captions” | 由更强的专有 VLM 生成的训练数据；很方便，但容易继承幻觉 |
| AnyRes / dynamic res | “高分辨率路径” | 通过切块或 M-RoPE，把大于编码器原生分辨率的图像送入模型的策略 |
| Resolution ramp | “Curriculum” | 从低分辨率开始、逐步升高的训练调度，用于加快对齐学习 |
| Vision-centric bench | “CV-Bench / BLINK” | 更强调细粒度视觉感知，而非重语言推理的评测 |
| PixMo | “Molmo 的数据” | Allen AI 的 71.2 万张密集标题图像数据集；由人工语音转写成 dense captions |

## 延伸阅读

- [McKinzie et al. — MM1 (arXiv:2403.09611)](https://arxiv.org/abs/2403.09611)
- [Laurençon et al. — Idefics2 / What matters building VLMs (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Deitke et al. — Molmo and PixMo (arXiv:2409.17146)](https://arxiv.org/abs/2409.17146)
- [Tong et al. — Cambrian-1 (arXiv:2406.16860)](https://arxiv.org/abs/2406.16860)
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865)
