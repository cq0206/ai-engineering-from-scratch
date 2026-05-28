# CLIP 与对比式（Contrastive）视觉-语言预训练

> OpenAI 的 CLIP（2021）证明了一个足以驱动接下来五年的核心想法：仅使用带噪声的网页图像-标题配对和对比损失（contrastive loss），就能让图像编码器与文本编码器在同一个向量空间中对齐。零监督标签。4 亿对样本。最终得到的嵌入空间（embedding space）可以执行零样本（zero-shot）分类、图文检索，并作为 2026 年几乎所有视觉语言模型（VLM）的视觉塔。SigLIP 2（2025）用 sigmoid 替代 softmax，并以更低成本将规模推进到超越 CLIP。 本课将带你从 InfoNCE 一路推导到 sigmoid 成对损失（pairwise loss）的数学原理，并用标准库 Python 实现训练步骤。

**类型：** 构建
**语言：** Python（stdlib，InfoNCE + sigmoid loss 实现）
**先修要求：** 第 12 阶段 · 01（ViT patches），第 7 阶段（Transformers）
**时间：** ~180 分钟

## 学习目标

- 从互信息推导 InfoNCE 损失，并实现一个数值稳定、向量化的版本。
- 解释为什么 sigmoid 成对损失（SigLIP）可以在不承担 softmax 所要求 all-gather 开销的情况下扩展到 32768+ 的 batch。
- 通过构造文本模板（`a photo of a {class}`）并对余弦相似度取 argmax，运行零样本 ImageNet 分类。
- 说出 CLIP / SigLIP 预训练带来的四个关键调节杆：batch size、temperature、prompt template、data quality。

## 问题

在 CLIP 之前，视觉任务主要依赖监督学习。收集带标签的数据集（ImageNet：120 万张图像、1000 个类别），训练一个 CNN，然后部署。标签昂贵；标签会偏向标注者能够达成一致的内容；而且如果不做微调（finetuning），标签通常无法迁移到新任务。

网页上的图像-标题数据天然提供了超过十亿对、带有宽松标签的样本，而且几乎免费。一张金毛寻回犬的图片，配上 alt 文本 “my dog Max in the park”，本身就携带监督信号——文本描述了图像。问题在于：你能否把这种信号转化成有用的训练方式？

CLIP 的回答是：把图像-标题对视为一个匹配任务。给定一个包含 N 张图像和 N 条标题的 batch，学习让每张图像匹配自己的标题，同时把其余 N-1 个干扰项排除掉。监督信号就是“这两个属于一起；那 N-1 个不属于”。没有类别标签。没有人工标注。只有对比损失。

最终得到的嵌入空间用途远不止 CLIP 的原始训练目标。ImageNet 零样本之所以有效，是因为 “a photo of a cat” 的嵌入会靠近那些从未被显式标成 cat 的猫图片。这正是催生 2026 年所有 VLM 的关键赌注。

## 概念

### 双编码器（dual encoder）

CLIP 有两个塔：

- 图像编码器 `f`：ViT 或 ResNet，为每张图像输出一个 D 维向量。
- 文本编码器 `g`：小型 transformer，为每条标题输出一个 D 维向量。

两个塔都会把输出归一化为单位长度。由于二者都是单位范数，相似度就是 `cos(f(x), g(y)) = f(x)^T g(y)`。

对于一个包含 N 个（图像，标题）配对的 batch，构造形状为 `(N, N)` 的相似度矩阵 `S`：

```
S[i, j] = cos(f(x_i), g(y_j)) / tau
```

其中 `tau` 是可学习的温度（temperature）参数（CLIP 初始化为 0.07；在对数空间中学习）。

### InfoNCE 损失

CLIP 在行和列上都使用对称的交叉熵（cross-entropy）：

```
loss_i2t = CE(S, labels=identity)     # each image's positive is its own caption
loss_t2i = CE(S^T, labels=identity)   # each caption's positive is its own image
loss = (loss_i2t + loss_t2i) / 2
```

这就是 InfoNCE。CE 中的 softmax 会迫使每张图像对自己的标题匹配得比 batch 中任何其他标题都更高。“负样本（negatives）”就是 batch 中的其他所有项。batch 越大 = 负样本越多 = 信号越强。CLIP 在 32k batch 上训练；规模确实重要。

### 温度（temperature）

`tau` 控制 softmax 的尖锐程度。tau 低 → 分布尖锐，产生类似困难负样本挖掘（hard negative mining）的效果。tau 高 → 分布更平缓，所有样本都会贡献。CLIP 学习的是 log(1/tau)，并通过裁剪避免塌缩。SigLIP 2 则固定初始 tau，并改用一个可学习的 bias。

### 为什么 sigmoid 扩展性更好（SigLIP）

softmax 需要整个相似度矩阵保持同步。在分布式训练中，你必须把所有嵌入 all-gather 到每个副本上，然后再做 softmax。这会让通信成本相对于 world size 呈二次增长。

SigLIP 用逐元素 sigmoid 替代 softmax：对每一个配对 `(i, j)`，损失都是一个二元分类问题——“这是不是匹配对？”正类标签位于对角线，其余全是负类。损失为：

```
L = -1/N sum over (i, j) [ y_ij log sigmoid(S[i,j]) + (1-y_ij) log sigmoid(-S[i,j]) ]
```

当 `i == j` 时，`y_ij = 1`，否则为 0。每一对的损失彼此独立。不需要 all-gather。每个 GPU 只需计算自己的局部块并求和。SigLIP 2 可以以较低成本扩展到 32k-512k batch，而 CLIP 则需要按比例增加更多通信。

### 零样本分类

给定 N 个类别名，为每个类别构造一个文本模板：

```
"a photo of a {class}"
```

用文本编码器嵌入每个模板，再用图像编码器嵌入你的图像。对余弦相似度取 argmax，就是预测类别。无需在目标类别上训练。

Prompt template 很重要。CLIP 原论文对每个类别使用了 80 个模板（普通、艺术、照片、绘画等），并对这些嵌入求平均。ImageNet 可提升 3 个点。现代实践通常只选一到两个模板。

### 线性探测（linear probe）与微调

零样本只是一个基线。线性探测（即在冻结的 CLIP 特征之上，仅为目标类别训练一个线性层）在领域内任务上通常优于零样本。完整微调在领域内通常又优于线性探测，但可能损害零样本迁移能力。三种范式，对应三种权衡。

### SigLIP 2：NaFlex 与密集特征

SigLIP 2（2025）新增了：
- NaFlex：单个模型可处理可变长宽比和分辨率。
- 更好的密集特征（dense features），用于分割和深度估计，目标是作为 VLM 中冻结的骨干网络（backbone）使用。
- 多语言：支持 100+ 种语言，而 CLIP 仅支持英语。
- 扩展到 10 亿参数规模，而 CLIP 最高约为 4 亿。

在 2026 年的开源 VLM 中，SigLIP 2 SO400m/14 是默认视觉塔。对于纯图文检索任务，如果特定的 LAION-2B 训练分布与查询模式匹配，CLIP 仍然是默认选择。

### ALIGN、BASIC、OpenCLIP、EVA-CLIP

ALIGN（Google，2021）：与 CLIP 思路相同，规模为 18 亿对，90% 为噪声数据。它证明了带噪数据同样可以扩展。OpenCLIP（LAION）：在 LAION-400M / 2B 上对 CLIP 的开源复现，支持多个规模，是最常用的开源 checkpoint。EVA-CLIP：从 masked image modeling 初始化；是 VLM 的强大骨干网络。BASIC：Google 的 CLIP+ALIGN 混合方案。它们都属于同一家族，只是在数据和调参上不同。

### 零样本天花板

CLIP 类模型在 ImageNet 零样本上的上限大约是 76%（CLIP-G、OpenCLIP-G）。再往上，要么需要更大规模的数据（SigLIP 2 达到 80%+），要么需要架构变化（监督头、更多参数）。这个基准正在趋于饱和；真正的价值在于下游 VLM 所消费的嵌入空间。

## 使用它

`code/main.py` 实现了：

1. 一个玩具版双编码器（基于哈希的图像特征、基于字符的文本特征），让你在没有 numpy 的情况下看清 InfoNCE 的形状。
2. 纯 Python 实现的 InfoNCE 损失（通过 log-sum-exp 保证数值稳定性）。
3. 作为对比的 sigmoid 成对损失。
4. 一个零样本分类流程：针对一组文本 prompts 计算余弦相似度，并用 argmax 做预测。

运行它，观察损失曲线。绝对数值只是玩具级别；但曲线形状与真实 CLIP 训练器的输出是一致的。

## 交付它

本课会生成 `outputs/skill-clip-zero-shot.md`。给定一组图像（通过路径）和一个目标类别列表，它会使用 CLIP 模板构造文本 prompts，用指定 checkpoint（例如 `openai/clip-vit-large-patch14`）对图像和文本两侧进行嵌入，并返回带相似度分数的 top-1 / top-5 预测。该技能会拒绝对提示列表之外的类别作出判断。

## 练习

1. 手工为一个包含 4 对样本的 batch 实现 InfoNCE。构造 4x4 相似度矩阵，运行 softmax，取出对角线，计算交叉熵。再用这个手工结果验证你的 Python 实现。

2. 除了温度参数外，SigLIP 还使用偏置参数 `b`：`S'[i,j] = S[i,j]/tau + b`。当 batch 存在严重类别不平衡（每行负样本远多于正样本）时，`b` 起什么作用？阅读 SigLIP 第 3 节（arXiv:2303.15343）。

3. 为 cats vs dogs 构建一个零样本分类器。尝试两个 prompt 模板：`a photo of a {class}` 和 `a picture of a {class}`。在 100 张测试图像上测量准确率。模板集成是否优于单模板？

4. 计算在 512-GPU、batch 32k 运行中，softmax InfoNCE 与 sigmoid 成对损失的通信成本。哪个按 O(N) 扩展，哪个按 O(N^2) 扩展？引用 SigLIP 第 4 节。

5. 阅读 OpenCLIP 的 scaling-laws 论文（arXiv:2212.07143，Cherti 等）。根据图表复现他们关于数据扩展的结论：在模型大小固定时，ImageNet 零样本准确率与训练数据规模之间的对数线性关系是什么？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| InfoNCE | “对比损失” | 在一个 batch 的相似度矩阵上做交叉熵；每个样本的正样本是与它配对的项，负样本是其他所有项 |
| Sigmoid loss | “SigLIP loss” | 对每一对做二元交叉熵；没有 softmax、没有 all-gather，因此在分布式训练中可以低成本扩展 |
| Temperature | “tau” | 在 softmax/sigmoid 之前缩放 logits 的标量；控制分布尖锐程度 |
| Zero-shot | “无需微调的分类” | 使用文本 prompts 构造类别嵌入，并通过余弦相似度分类；不在目标类别上训练 |
| Prompt template | “a photo of a ...” | 围绕类别名构造的文本框架；会使零样本准确率波动 1-5 个点 |
| Dual encoder | “双塔” | 一个图像编码器 + 一个文本编码器，在共享的 D 维空间中输出 |
| Hard negative | “困难干扰项” | 与正样本足够相似、迫使模型必须努力区分的负样本 |
| Linear probe | “冻结 + 一层” | 仅在冻结特征上训练一个线性分类器；用于衡量特征质量 |
| NaFlex | “原生弹性分辨率” | SigLIP 2 在不缩放的情况下摄取任意长宽比与分辨率图像的能力 |
| Temperature scaling | “对数参数化 tau” | CLIP 将 `log(1/tau)` 参数化，以获得更稳定的梯度；并通过裁剪防止 tau 塌缩到接近 0 |

## 延伸阅读

- [Radford et al. — Learning Transferable Visual Models From Natural Language Supervision (arXiv:2103.00020)](https://arxiv.org/abs/2103.00020) — CLIP 论文。
- [Zhai et al. — Sigmoid Loss for Language Image Pre-Training (arXiv:2303.15343)](https://arxiv.org/abs/2303.15343) — SigLIP。
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — 多语言 + NaFlex。
- [Jia et al. — ALIGN (arXiv:2102.05918)](https://arxiv.org/abs/2102.05918) — 使用带噪网页数据进行大规模训练。
- [Cherti et al. — Reproducible scaling laws for contrastive language-image learning (arXiv:2212.07143)](https://arxiv.org/abs/2212.07143) — OpenCLIP scaling laws。
