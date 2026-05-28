# 条件 GAN（Conditional GANs）与 Pix2Pix

> 2014–2017 年间第一个重要突破，是让 GAN 学会“受控生成”。你可以给它一个标签、一张图像，或者一句话。Pix2Pix 做的是图像条件版本，直到今天它在狭窄的图像到图像任务上，依然优于所有通用型文本到图像模型。

**类型：** Build
**语言：** Python
**先修要求：** Phase 8 · 03（GANs）, Phase 4 · 06（U-Net）, Phase 3 · 07（CNNs）
**时长：** ~75 分钟

## 问题

一个无条件（unconditional）GAN 会随机采样出任意人脸。做演示可以，在生产中没用。你真正想要的是：*把草图映射成照片*、*把地图映射成航拍图*、*把白天场景映射成夜晚*、*给灰度图上色*。在所有这些任务中，你都会得到一个输入图像 `x`，并且必须输出与之在语义上对应的 `y`。对同一个 `x`，可能有很多合理的 `y`。均方误差（mean-squared error）会把这些可能性抹平成一团糊；对抗损失（adversarial loss）不会，因为“看起来真实”这件事本身就是尖锐的。

条件 GAN（Mirza & Osindero, 2014）会把一个条件 `c` 同时作为 `G` 和 `D` 的输入。Pix2Pix（Isola et al., 2017）则把这件事进一步专门化：条件是完整的输入图像，生成器（generator）是 U-Net，判别器（discriminator）是一个**基于 patch 的**分类器（PatchGAN），损失是对抗项 + L1。即使到了 2026 年，这套配方在狭窄图像到图像领域里仍然能压过从零开始训练的文本到图像模型，因为它训练在**成对数据（paired data）**上——你拥有任务真正需要的精确信号。

## 概念

*Pix2Pix：U-Net 生成器，PatchGAN 判别器*

**条件 G。** `G(x, z) → y`。在 Pix2Pix 中，`z` 体现为 G 内部的 dropout（而不是输入噪声——Isola 发现显式噪声会被模型忽略）。

**条件 D。** `D(x, y) → [0, 1]`。输入是这一对数据——（条件、输出）。这是关键差异：D 必须判断 `y` 是否与 `x` 一致，而不只是判断 `y` 看起来是不是真实。

**U-Net 生成器。** 这是一个带跳跃连接（skip connections）的编码器-解码器结构，跨越瓶颈层（bottleneck）。对于输入和输出共享底层结构（边缘、轮廓）的任务，它至关重要。没有这些跳连，高频细节就会消失。

**PatchGAN 判别器。** D 不再输出单个真假分数，而是输出一个 `N×N` 网格，其中每个单元负责判断一个约 70×70 像素感受野（receptive field）内的真实感，最后再求平均。这等价于一种马尔可夫随机场（Markov random field）假设：真实感是局部的。这样训练更快、参数更少、输出也更锐利。

**损失。**

```
loss_G = -log D(x, G(x)) + λ · ||y - G(x)||_1
loss_D = -log D(x, y) - log (1 - D(x, G(x)))
```

L1 项会稳定训练，并把 G 推向已知目标。相比 L2，L1 会保留更锐利的边缘（对应中位数而不是均值）。Pix2Pix 的默认值是 `λ = 100`。

## CycleGAN——当你没有配对数据时

Pix2Pix 需要配对的 `(x, y)` 数据。CycleGAN（Zhu et al., 2017）通过增加一项额外损失来移除这个要求：**循环一致性（cycle consistency）**损失。它有两个生成器 `G: X → Y` 和 `F: Y → X`。训练目标是让 `F(G(x)) ≈ x` 且 `G(F(y)) ≈ y`。这样，你就可以在没有配对样本的情况下，完成“马 ↔ 斑马”“夏天 ↔ 冬天”之类的转换。

到 2026 年，无配对图像到图像任务大多已经通过 diffusion（ControlNet、IP-Adapter）来完成，而不是 CycleGAN；但循环一致性的思想几乎仍然活在每一篇无配对领域自适应论文里。

## 动手构建

`code/main.py` 在 1-D 数据上实现了一个微型条件 GAN。条件 `c` 是一个类别标签（0 或 1）。任务是：针对给定类别，从对应的条件分布中生成一个样本。

### 第 1 步：把条件拼接到 G 和 D 的输入中

```python
def G(z, c, params):
    return mlp(concat([z, one_hot(c)]), params)

def D(x, c, params):
    return mlp(concat([x, one_hot(c)]), params)
```

最简单的方式就是 one-hot 编码。更大的模型会使用可学习 embedding、FiLM 调制，或者 cross-attention。

### 第 2 步：按条件训练

```python
for step in range(steps):
    x, c = sample_real_conditional()
    noise = sample_noise()
    update_D(x_real=x, x_fake=G(noise, c), c=c)
    update_G(noise, c)
```

生成器必须匹配**给定条件下**的真实分布，而不是整体边缘分布（marginal）。

### 第 3 步：验证每个类别的输出

```python
for c in [0, 1]:
    samples = [G(noise, c) for noise in batch]
    mean_c = mean(samples)
    assert_near(mean_c, real_mean_for_class_c)
```

## 常见陷阱

- **条件被忽略。** G 学会了边缘化处理，D 因为条件信号太弱而从不惩罚。修复方法：更强地对 D 施加条件（在早期层注入，而不只是后期层），或者使用 projection discriminator（Miyato & Koyama 2018）。
- **L1 权重过低。** G 会漂向任意“看起来真实”的输出，而不是忠实输出。对于 Pix2Pix 风格任务，可以从 λ≈100 开始。
- **L1 权重过高。** 由于 L1 仍然是一个 L_p 范数，G 会产生模糊输出。训练稳定后可以逐步降低。
- **D 中出现真实标签泄漏（ground-truth leakage）。** 要把 `(x, y)` 拼接后作为 D 的输入，而不只是 `y`。否则 D 无法检查一致性。
- **按类别发生 mode collapse。** 每个类别都可能独立塌缩。要做按类别的多样性检查。

## 使用

2026 年图像到图像任务的现状：

| 任务 | 最佳方法 |
|------|---------------|
| 草图 → 照片，同域，配对数据 | Pix2Pix / Pix2PixHD（依然快，依然锐利） |
| 草图 → 照片，无配对 | 使用 Scribble 条件模型的 ControlNet |
| 语义分割 → 照片 | SPADE / GauGAN2，或 SD + ControlNet-Seg |
| 风格迁移 | 使用 IP-Adapter 或 LoRA 的 diffusion；GAN 方法已成旧方案 |
| 深度图 → 照片 | Stable Diffusion 上的 ControlNet-Depth |
| 超分辨率 | Real-ESRGAN（GAN）、ESRGAN-Plus，或 SD-Upscale（diffusion） |
| 图像上色 | ColTran、基于 diffusion 的上色器，或 Pix2Pix-color |
| 白天 → 夜晚、季节、天气 | CycleGAN 或基于 ControlNet 的方案 |

当满足以下条件时，Pix2Pix 仍然是正确工具：（a）你有成千上万条配对样本；（b）任务狭窄且可重复；（c）你需要快速推理。对于开放域的通用任务，diffusion 会胜出。

## 交付

保存 `outputs/skill-img2img-chooser.md`。这个 skill 会接收任务描述、数据可用性（配对还是无配对、样本数 N）以及延迟/质量预算，然后输出：推荐方案（Pix2Pix、CycleGAN、ControlNet 变体、SDXL + IP-Adapter）、训练数据要求、推理成本，以及评估协议（LPIPS、FID、任务特定指标）。

## 练习

1. **简单。** 修改 `code/main.py`，加入第三个类别。确认 G 仍然会把每个类别的噪声映射到正确的模式。
2. **中等。** 在这个 1-D 场景中，把 L1 替换成感知风格（perceptual-style）损失（例如冻结一个小型 D 作为特征提取器）。它会改变条件分布的锐利程度吗？
3. **困难。** 在 1-D 场景中草拟一个 CycleGAN：两个分布、两个生成器、循环损失。证明它在没有配对数据的情况下，也能学会在两者之间做映射。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Conditional GAN | “带标签的 GAN” | G(z, c), D(x, c)。两个网络都能看到条件。 |
| Pix2Pix | “图像到图像 GAN” | 带配对数据的 cGAN，使用 U-Net 生成器和 PatchGAN 判别器 + L1 损失。 |
| U-Net | “带跳连的编码器-解码器” | 对称卷积网络；跳连能保留高频细节。 |
| PatchGAN | “局部真实感分类器” | D 输出的是每个 patch 的分数，而不是全局分数。 |
| CycleGAN | “无配对图像翻译” | 两个 G + 循环一致性损失；不需要配对数据。 |
| SPADE | “GauGAN” | 用语义图去归一化中间激活；用于分割到图像。 |
| FiLM | “逐特征线性调制” | 由条件产生逐特征仿射变换；一种低成本条件注入方式。 |

## 生产说明：Pix2Pix 作为受延迟约束的基线

当你拥有配对数据且任务狭窄（草图 → 渲染、语义图 → 照片、白天 → 夜晚）时，Pix2Pix 的单次前向推理在延迟上通常比 diffusion 快一个数量级。生产中的比较一般如下：

| 路径 | 步数 | 单张 L4 上 512² 的典型延迟 |
|------|-------|----------------------------------------|
| Pix2Pix（U-Net 前向） | 1 | ~30 ms |
| SD-Inpaint 或 SD-Img2Img | 20 | ~1.2 s |
| SDXL-Turbo Img2Img | 1-4 | ~0.15-0.35 s |
| ControlNet + SDXL base | 20-30 | ~3-5 s |

Pix2Pix 在静态批处理中的吞吐量更高（每个请求的 FLOPs 都一样）。diffusion 在质量和泛化能力上更强。现代做法通常是：为狭窄任务上线一个 Pix2Pix 风格的蒸馏模型，再为长尾输入准备一个 diffusion 兜底方案。

## 延伸阅读

- [Mirza & Osindero (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) —— cGAN 论文。
- [Isola et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) —— Pix2Pix。
- [Zhu et al. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) —— CycleGAN。
- [Wang et al. (2018). High-Resolution Image Synthesis with Conditional GANs](https://arxiv.org/abs/1711.11585) —— Pix2PixHD。
- [Park et al. (2019). Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291) —— SPADE / GauGAN。
- [Miyato & Koyama (2018). cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637) —— projection discriminator。
