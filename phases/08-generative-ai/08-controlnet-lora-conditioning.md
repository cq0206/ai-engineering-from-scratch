# ControlNet、LoRA 与 Conditioning

> 仅靠文本（text）本身是一种很笨拙的控制信号。ControlNet 允许你复制一个预训练 diffusion 模型，并用深度图、姿态骨架、涂鸦或边缘图来引导它。LoRA 则让你通过只训练 1000 万个参数，就能微调一个拥有 20 亿参数的模型。两者结合，把 Stable Diffusion 从玩具变成了 2026 年几乎所有代理机构都在交付的图像流水线。

**类型：** Build
**语言：** Python
**先修要求：** Phase 8 · 07（Latent Diffusion）, Phase 10（LLMs from Scratch — for LoRA foundation）
**时长：** ~75 分钟

## 问题

像 “a woman in a red dress walking a dog on a busy street” 这样的提示词（prompt），并没有告诉模型狗在*哪里*、女人是什么*姿态*，或者街道的*透视关系*是什么。文本最多只能锁定你构造一张图像所需信息中的大约 10%。剩下的都是视觉信息，无法高效地用文字描述。

为每一种信号（姿态、深度、canny、分割）都从头训练一个新的条件模型（conditional model），代价高得难以接受。你希望保留 26 亿参数的 SDXL 主干（backbone）并冻结它，再接一个读取条件输入的小型侧网络（side-network），让它去轻推主干的中间特征。这就是 ControlNet。

你还希望让模型学会新概念（你的脸、你的产品、你的风格），而不必重训整个模型。你想要的是一个小 100 倍的增量（delta）。这就是 LoRA——可以插入现有 attention 权重中的低秩适配器（low-rank adapters）。

ControlNet + LoRA + text = 2026 年从业者的标准工具箱。大多数生产图像流水线都会在 SDXL / SD3 / Flux 基座之上，叠加 2–5 个 LoRA、1–3 个 ControlNet，再加一个 IP-Adapter。

## 概念

*ControlNet 复制编码器（encoder）；LoRA 添加低秩增量（low-rank deltas）*

### ControlNet（Zhang et al., 2023）

取一个预训练的 Stable Diffusion。*复制（clone）* U-Net 的编码器半部分。冻结原始编码器。训练复制出的编码器，让它接收额外的条件输入（边缘、深度、姿态）。再通过 *零卷积（zero-convolution）* 跳连，把这个复制编码器接回原始网络的解码器半部分（初始化为零的 1×1 卷积——一开始什么都不做，只学习一个增量）。

```
SD U-Net decoder:   ... ← orig_enc_features + zero_conv(controlnet_enc(condition))
```

零卷积初始化意味着 ControlNet 一开始等同于恒等映射——即使在训练前也不会造成破坏。用标准 diffusion loss 在 100 万组 `(prompt, condition, image)` 三元组上训练即可。

按模态（modality）区分的 ControlNet 会以小型侧模型的形式发布（SDXL 约 360M，SD 1.5 约 70M）。你可以在推理时组合它们：

```
features += weight_a * control_a(depth) + weight_b * control_b(pose)
```

### LoRA（Hu et al., 2021）

对于模型中的任意线性层 `W ∈ R^{d×d}`，冻结 `W`，再加上一个低秩增量：

```
W' = W + ΔW,  ΔW = B @ A,  A ∈ R^{r×d},  B ∈ R^{d×r}
```

其中 `r &lt;&lt; d`。对 attention 而言，rank 4–16 是常见设置；对较重的微调，rank 64–128 更常见。新增参数数量是 `2 · d · r`，而不是 `d²`。以 SDXL attention 的 `d=640`、`r=16` 为例：每个 adapter 只需 2 万参数，而不是 41 万——减少了 20 倍。放到整个模型上看：LoRA 通常只有 20–200MB，而基座模型则有 5GB。

在推理时，你还可以缩放 LoRA：`W' = W + α · B @ A`。`α = 0.5-1.5` 很常见。多个 LoRA 可以按加法方式叠加（当然，前提是你接受它们会以非线性方式相互影响）。

### IP-Adapter（Ye et al., 2023）

这是一个很小的 adapter，可以把*图像*作为条件输入（与文本一起）。它使用 CLIP 图像编码器（image encoder）生成图像 token，并把这些 token 与文本 token 一起注入 cross-attention。每个基座模型大约只需 ~20MB。它让你无需 LoRA，也能做到“按照这张参考图的风格生成图像”。

## 可组合性矩阵

| 工具 | 它控制什么 | 大小 | 何时使用 |
|------|------------------|------|-------------|
| ControlNet | 空间结构（姿态、深度、边缘） | 70-360MB | 需要精确布局与构图 |
| LoRA | 风格、主体、概念 | 20-200MB | 个性化、风格迁移 |
| IP-Adapter | 从参考图像提取风格或主体 | 20MB | 文本无法描述外观 |
| Textual Inversion | 以新 token 表示单一概念 | 10KB | 旧方案，大多已被 LoRA 取代 |
| DreamBooth | 针对某个主体进行完整微调 | 2-5GB | 强身份一致性、高算力 |
| T2I-Adapter | 更轻量的 ControlNet 替代方案 | 70MB | 边缘设备、推理预算有限 |

ControlNet ≈ 空间控制。LoRA ≈ 语义控制。两者一起用。

## 动手构建

`code/main.py` 在 1-D 上模拟了这两种机制：

1. **LoRA。** 一个预训练的线性层 `W`。冻结它。训练一个低秩 `B @ A`，使得 `W + BA` 能匹配目标线性层。展示 `r = 1` 就足以精确学到一个 rank-1 修正。

2. **ControlNet-lite。** 一个“冻结的基座（frozen base）”预测器，加上一个读取额外信号的“侧网络（side network）”。侧网络输出会乘上一个可学习标量门控（gate），该门控初始化为零（这是我们版本的 zero-conv）。训练时观察 gate 如何逐步抬升。

### 第 1 步：LoRA 数学

```python
def lora(W, A, B, x, alpha=1.0):
    # W is frozen; A, B are the trainable low-rank factors.
    return [W[i][j] * x[j] for i, j in ...] + alpha * (B @ (A @ x))
```

### 第 2 步：零初始化侧网络

```python
side_out = control_net(x, condition)
gated = gate * side_out  # gate initialized to 0
h = base(x) + gated
```

在第 0 步，输出与 base 完全一致。训练早期会缓慢更新 `gate`——不会发生灾难性漂移（catastrophic drift）。

## 常见陷阱

- **LoRA 缩放过大。** `α = 2` 或 `α = 3` 是常见的“让它更强”黑客式做法，但往往会得到过度风格化或损坏的输出。保持 `α ≤ 1.5`。
- **ControlNet 权重冲突。** 同时使用权重 1.0 的 Pose ControlNet 和权重 1.0 的 Depth ControlNet，通常会过冲（overshoot）。权重总和 ≈ 1.0 是更安全的默认值。
- **LoRA 挂在错误基座上。** SDXL LoRA 挂到 SD 1.5 上时通常会静默失效，因为 attention 维度不匹配。Diffusers 在 0.30+ 会给出警告。
- **Textual Inversion 漂移。** 在某个 checkpoint 上训练出的 token，换到另一个 checkpoint 往往会严重漂移。LoRA 的可迁移性更好。
- **LoRA 权重合并与存储。** 你可以把 LoRA 烘焙进基座权重，以获得更快的推理速度（无需运行时相加），但会失去运行时缩放 `α` 的能力。两种版本都保留。

## 如何使用

| 目标 | 2026 年流水线 |
|------|---------------|
| 复现某个品牌的艺术风格 | 用约 30 张精挑细选图片训练 rank 32 的 LoRA |
| 把我的脸放进生成图像里 | DreamBooth 或 LoRA + IP-Adapter-FaceID |
| 指定姿态 + prompt | ControlNet-Openpose + SDXL + text |
| 感知深度的构图 | ControlNet-Depth + SD3 |
| 参考图 + prompt | IP-Adapter + text |
| 精确布局 | ControlNet-Scribble 或 ControlNet-Canny |
| 替换背景 | ControlNet-Seg + Inpainting（Lesson 09） |
| 快速一步风格化 | SDXL-Turbo 上的 LCM-LoRA |

## 交付

保存 `outputs/skill-sd-toolkit-composer.md`。这个 Skill 接收一个任务（输入资源包括：prompt、可选参考图、可选姿态、可选深度、可选涂鸦），并输出推荐的工具栈、对应权重，以及可复现的 seed 协议。

## 练习

1. **简单。** 在 `code/main.py` 中，把 LoRA 的 rank `r` 从 1 调到 4。LoRA 在哪个 rank 时可以精确匹配一个 rank-2 的目标增量？
2. **中等。** 对两个目标变换分别训练两个独立的 LoRA。把它们一起加载，展示它们的加法交互。什么时候这种交互会破坏线性？
3. **困难。** 用 diffusers 叠加：SDXL-base + Canny-ControlNet（权重 0.8）+ 一个风格 LoRA（α 0.8）+ IP-Adapter（权重 0.6）。测量随着工具栈权重变化，FID 与 prompt adherence 之间的权衡。

## 关键术语

| 术语 | 人们常说什么 | 它真正的含义 |
|------|-----------------|-----------------------|
| ControlNet | “空间控制（spatial control）” | 复制编码器 + zero-conv 跳连；读取一张条件图像。 |
| Zero convolution | “一开始就是 identity” | 初始化为零的 1×1 卷积；ControlNet 起步时等于 no-op。 |
| LoRA | “低秩适配器（low-rank adapter）” | `W + B @ A`，`r &lt;&lt; d`；参数量比完整微调少 100 倍。 |
| rank r | “那个旋钮” | LoRA 的压缩率；通常 4–16，重个性化时 64+。 |
| α | “LoRA 强度” | 对 LoRA 增量做运行时缩放。 |
| IP-Adapter | “参考图像” | 通过 CLIP image token 做图像条件控制的小型适配器。 |
| DreamBooth | “完整主体微调” | 用约 30 张主体图像训练整个模型。 |
| Textual Inversion | “新 token” | 只学习一个新的词向量；旧方案，现多被替代。 |

## 生产说明：LoRA 热切换、ControlNet 通道与多租户服务

一个真正的文生图 SaaS，会在同一个基座 checkpoint 上同时服务数百个 LoRA 和十几个 ControlNet。这个服务问题看起来很像 LLM 多租户（生产文献通常会在 continuous batching 与 LoRAX / S-LoRA 语境下讨论 LLM 场景）：

- **热切换 LoRA，不要合并。** 把 `W' = W + α·B·A` 合并进基座，虽然每步推理能快约 ~3–5%，但也会把 `α` 和基座一起固定死。应该把 LoRA 作为 rank-r 增量常驻在 VRAM 中；diffusers 提供了 `pipe.load_lora_weights()` + `pipe.set_adapters([...], adapter_weights=[...])`，支持按请求激活。切换成本只与 `2 · d · r · num_layers` 这些权重有关——MB 级、亚秒级。
- **把 ControlNet 看成第二条 attention 通道。** 复制出来的编码器与基座并行运行。两个权重都为 1.0 的 ControlNet，意味着每一步要多跑两次前向传播，而不是一次合并后的前向。batch size 的余量会近似按平方下降。每启用一个 ControlNet，预算上应按约 ~1.5× 的 step cost 来估计。
- **LoRA 也可以量化。** 如果你已经把基座量化了（见 Lesson 07，Flux on 8GB），LoRA 增量同样可以很好地量化到 8-bit 或 4-bit。QLoRA 风格的加载方式，让你可以在一个 4-bit 的 Flux 基座之上叠加 5–10 个 LoRA，而不会把内存打爆。

Flux 特有说明：Niels 的 Flux-on-8GB notebook 会把基座量化到 4-bit；在那个量化基座上继续叠加一个风格 LoRA（`pipe.load_lora_weights("user/style-lora")`），并设置 `weight_name="pytorch_lora_weights.safetensors"`，依然能工作。这正是 2026 年多数 SaaS 代理机构在交付的配方。

## 延伸阅读

- [Zhang, Rao, Agrawala (2023). Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) — ControlNet.
- [Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — LoRA（最初用于 LLM；后移植到 diffusion）。
- [Ye et al. (2023). IP-Adapter: Text Compatible Image Prompt Adapter](https://arxiv.org/abs/2308.06721) — IP-Adapter.
- [Mou et al. (2023). T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability](https://arxiv.org/abs/2302.08453) — ControlNet 的更轻替代方案。
- [Ruiz et al. (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) — DreamBooth.
- [HuggingFace Diffusers — ControlNet / LoRA / IP-Adapter docs](https://huggingface.co/docs/diffusers/training/controlnet) — 参考流水线。
