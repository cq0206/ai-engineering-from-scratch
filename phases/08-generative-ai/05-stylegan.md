# StyleGAN

> 大多数生成器（generator）会同时把 `z` 混入每一层。StyleGAN 把这个过程拆开了：先把 `z` 映射到一个中间变量 `w`，再通过 AdaIN 在每个分辨率层级上把 `w` *注入（inject）* 进去。正是这一个改动，解耦了潜在空间（latent space），并让照片级逼真人脸在接下来七年里都几乎成了已解决问题。

**类型：** Build
**语言：** Python
**先修要求：** Phase 8 · 03（GANs）, Phase 4 · 08（Normalization）, Phase 3 · 07（CNNs）
**时长：** ~45 分钟

## 问题

DCGAN 通过一串转置卷积（transposed convolution），把 `z` 映射成图像。问题在于：`z` 控制了所有东西——姿态、光照、身份、背景——而且这些因素是纠缠在一起的。沿着 `z` 的某一个轴移动，这四者都会变化。你没法对模型说“同一个人，不同姿态”，因为它的表征（representation）并不是按这种方式分解的。

Karras 等人（2019，NVIDIA）提出：不要再把 `z` 直接喂给卷积层。改为把一个固定的 `4×4×512` 张量作为网络输入。再学习一个 8 层的多层感知机（MLP），把 `z ∈ Z → w ∈ W`。随后通过 *自适应实例归一化（adaptive instance normalization, AdaIN）* 在每个分辨率上注入 `w`：先对每个卷积特征图做归一化，再用 `w` 的仿射投影对其进行缩放和平移。再为每一层加入噪声，以表示随机细节（皮肤毛孔、发丝）。

结果是：`W` 大致拥有彼此正交的轴，分别对应“高层风格（style）”（姿态、身份）和“细粒度风格”（光照、颜色）。你可以在两张图之间交换风格：在低分辨率层使用图像 A 的 `w`，在高分辨率层使用图像 B 的 `w`。这开启了图像编辑、跨域风格化，以及整个“StyleGAN 反演（StyleGAN inversion）”研究路线。

## 概念

*StyleGAN：映射网络（mapping network）+ AdaIN + 逐层噪声（per-layer noise）*

**映射网络（mapping network）。** `f: Z → W`，一个 8 层 MLP。`Z = N(0, I)^512`。`W` 不必被强制为高斯分布——它会学习出适配数据的形状。

**合成网络（synthesis network）。** 从一个可学习的常量 `4×4×512` 开始。每个分辨率块：`upsample → conv → AdaIN(w_i) → noise → conv → AdaIN(w_i) → noise`。分辨率按 2 倍增长：4、8、16、32、64、128、256、512、1024。

**AdaIN。**

```
AdaIN(x, y) = y_scale · (x - mean(x)) / std(x) + y_bias
```

其中 `y_scale` 和 `y_bias` 来自 `w` 的仿射投影。先按特征图归一化，再重新施加风格。这里的“风格”指的是特征图的一阶与二阶统计量。

**逐层噪声（per-layer noise）。** 将单通道高斯噪声加到每个特征图上，并乘以一个按通道学习的缩放因子。它控制随机细节，而不影响全局结构。

**截断技巧（truncation trick）。** 在推理时，采样 `z`，计算 `w = mapping(z)`，然后令 `w' = ŵ + ψ·(w - ŵ)`，其中 `ŵ` 是大量样本对应 `w` 的均值。`ψ &lt; 1` 会用多样性换取质量。几乎所有 StyleGAN 演示都使用 `ψ ≈ 0.7`。

## StyleGAN 1 → 2 → 3

| 版本 | 年份 | 创新 |
|---------|------|------------|
| StyleGAN | 2019 | 映射网络 + AdaIN + 噪声 + 渐进式增长（progressive growing）。 |
| StyleGAN2 | 2020 | 用权重去调制（weight demodulation）替代 AdaIN（修复液滴伪影）；skip/residual 架构；路径长度正则化（path-length regularization）。 |
| StyleGAN3 | 2021 | 无混叠卷积（alias-free convolution）+ 等变卷积核（equivariant kernels）；消除纹理粘在像素网格上的问题。 |
| StyleGAN-XL | 2022 | 类条件生成（class-conditional），1024²，ImageNet。 |
| R3GAN | 2024 | 以更强的正则化重新包装；在 FFHQ-1024 上用少 20 倍参数把与 diffusion 的差距追平。 |

到 2026 年，StyleGAN3 依然是以下场景的默认选择：（a）高 FPS 的窄域照片级逼真生成，（b）小样本域适配（用 100 张图片训练新数据集，冻结映射网络），（c）基于反演的编辑（找到能重建真实照片的 `w`，再去编辑这个 `w`）。对于开放域文生图，它并不是合适工具——diffusion 才是。

## 动手构建

`code/main.py` 实现了一个 1-D 的玩具版“style-GAN lite”：包括映射 MLP、一个以可学习常量向量为输入并用 `w` 派生出的 scale/bias 进行调制的合成函数，以及逐层噪声。它展示了：通过仿射调制（affine modulation）注入 `w`，与把 `z` 拼接到生成器输入相比，效果相当甚至更好。

### 第 1 步：映射网络

```python
def mapping(z, M):
    h = z
    for i in range(num_layers):
        h = leaky_relu(add(matmul(M[f"W{i}"], h), M[f"b{i}"]))
    return h
```

### 第 2 步：自适应实例归一化

```python
def adain(x, w_scale, w_bias):
    mu = mean(x)
    sd = std(x)
    x_norm = [(xi - mu) / (sd + 1e-8) for xi in x]
    return [w_scale * xi + w_bias for xi in x_norm]
```

按特征图的 scale 和 bias 由 `w` 通过线性投影得到。

### 第 3 步：逐层噪声

```python
def add_noise(x, sigma, rng):
    return [xi + sigma * rng.gauss(0, 1) for xi in x]
```

每个通道的 sigma 都是可学习的。

## 常见陷阱

- **液滴伪影（droplet artifacts）。** StyleGAN 1 会在特征图里产生团块状液滴，因为 AdaIN 把均值清零了。StyleGAN 2 通过改为缩放卷积权重的权重去调制修复了这个问题。
- **纹理粘连（texture sticking）。** StyleGAN 1 和 2 的纹理跟随的是像素坐标，而不是物体坐标（在插值时很明显）。StyleGAN 3 用带窗 sinc 滤波器的无混叠卷积修复了这一点。
- **模式覆盖（mode coverage）。** 截断 `ψ &lt; 0.7` 看起来很干净，但样本只来自一个狭窄锥体；如果你需要多样性，请使用 `ψ = 1.0`。
- **反演有损（inversion is lossy）。** 把真实照片反演到 `W`，通常通过优化或编码器（e4e、ReStyle、HyperStyle）完成。迭代次数一多，结果就会漂移。

## 如何使用

| 用例 | 方法 |
|----------|----------|
| 照片级逼真人脸（动漫、商品、窄域） | StyleGAN3 FFHQ / 自定义微调 |
| 从照片进行人脸编辑 | e4e 反演 + StyleSpace / InterFaceGAN 方向 |
| 换脸 / 重演 | StyleGAN + 编码器 + blending |
| Avatar 流水线 | StyleGAN3 + ADA，用于小数据微调 |
| 基于少量图像做域适配 | 冻结映射网络，微调合成网络 |
| 多模态或文本条件生成 | 不要用——请用 diffusion |

对于“输出是一张人的脸部照片”这类产品级演示，StyleGAN 在推理成本（单次前向传播，在 4090 上 &lt;10ms）和同等质量门槛下的清晰度方面，优于 diffusion。

## 交付

保存 `outputs/skill-stylegan-inversion.md`。这个 Skill 读取一张真实照片，并输出：反演方法（e4e / ReStyle / HyperStyle）、预期潜变量损失（latent loss）、编辑预算（在出现伪影之前你能在 `W` 里移动多远），以及一组已验证有效的编辑方向（年龄、表情、姿态）。

## 练习

1. **简单。** 运行 `code/main.py`，分别设置 `adain_on=True` 和 `adain_on=False`。比较固定 latent 与扰动 latent 下输出的分布范围。
2. **中等。** 实现 mixing regularization：对一个训练 batch，计算 `w_a`、`w_b`，并在合成的前半部分应用 `w_a`，后半部分应用 `w_b`。解码器是否学到了可解耦的风格？
3. **困难。** 取一个预训练的 StyleGAN3 FFHQ 模型（ffhq-1024.pkl）。通过在带标签样本上训练 SVM，找出控制“微笑”的 `w` 方向；报告在身份开始漂移之前，你最多能推动多远。

## 关键术语

| 术语 | 人们常说什么 | 它真正的含义 |
|------|-----------------|-----------------------|
| Mapping network | “那个 MLP” | `f: Z → W`，8 层，把潜变量几何与数据统计解耦。 |
| W space | “风格空间（style space）” | 映射网络的输出；大致可解耦。 |
| AdaIN | “Adaptive instance norm” | 先归一化特征图，再用 `w` 投影做缩放与平移。 |
| Truncation trick | “Psi” | `w = mean + ψ·(w - mean)`，ψ&lt;1 用多样性换质量。 |
| Path-length regularization | “PL reg” | 惩罚 `w` 的单位变化导致图像变化过大；让 `W` 更平滑。 |
| Weight demodulation | “StyleGAN2 的修复” | 归一化卷积权重而不是激活值；消除液滴伪影。 |
| Alias-free | “StyleGAN3 的技巧” | 带窗 sinc 滤波器；消除纹理粘在像素网格上的问题。 |
| Inversion | “给真实图像找 w” | 通过优化或编码，把 `x → w`，使 `G(w) ≈ x`。 |

## 生产说明：为什么到 2026 年 StyleGAN 仍在上线

在 4090 上，StyleGAN3 生成一张 1024² 的 FFHQ 人脸耗时不到 10 ms——`num_steps = 1`，没有 VAE decode，也没有 cross-attention pass。从生产角度看，这就是任何图像生成器的最低延迟下限。同分辨率下，一个 50 步的 SDXL + VAE-decode 流水线大约要 **3 秒**。这是 **300× 的差距**，而对于窄域产品（avatar 服务、身份证件流水线、库存人脸生成），它在总体拥有成本（TCO）上更占优。

这会带来两个运维层面的结果：

- **不需要 scheduler，也不需要 batcher。** 以目标占用率运行静态 batch 就是最优解。连续批处理（对 LLM 和 diffusion 至关重要）在这里没有任何收益，因为每个请求消耗的 FLOPs 都相同。
- **截断 `ψ` 是安全旋钮（safety knob）。** `ψ &lt; 0.7` 会从映射网络输出范围的一个狭窄锥体中采样。这是服务层能够控制样本方差的唯一杠杆。高峰负载时降低 `ψ`，给付费用户再调高它。

## 延伸阅读

- [Karras et al. (2019). A Style-Based Generator Architecture for GANs](https://arxiv.org/abs/1812.04948) — StyleGAN.
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2.
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3.
- [Tov et al. (2021). Designing an Encoder for StyleGAN Image Manipulation](https://arxiv.org/abs/2102.02766) — e4e inversion.
- [Sauer et al. (2022). StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets](https://arxiv.org/abs/2202.00273) — StyleGAN-XL.
- [Huang et al. (2024). R3GAN: The GAN is dead; long live the GAN!](https://arxiv.org/abs/2501.05441) — modern minimal GAN recipe.
