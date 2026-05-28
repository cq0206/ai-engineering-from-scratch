# Flow Matching 与 Rectified Flows

> 扩散模型之所以需要 20-50 个采样步，是因为它们沿着一条从噪声走向数据的弯曲路径前进。Flow matching（Lipman 等，2023）和 rectified flow（Liu 等，2022）训练的是更直的路径。路径越直，所需步数越少，推理也就越快。Stable Diffusion 3、Flux.1 和 AudioCraft 2 都在 2024 年切换到了 flow matching。

**类型：** 构建
**语言：** Python
**前置要求：** 第 8 阶段 · 06（DDPM），第 1 阶段 · 微积分
**耗时：** ~45 分钟

## 问题

DDPM 的逆过程，是一个从 `N(0, I)` 返回数据分布的 1000 步随机游走。DDIM 把它压缩到了 20-50 个确定性步骤。你还想要更少的步数——最好一步就够。阻碍在于：求解逆过程的 ODE 是刚性的（stiff）；这条路径是弯的。

如果你能把模型训练成让噪声到数据的路径变成一条*直线*，那么从 `t=1` 到 `t=0` 的单个 Euler 步就能工作。Flow matching 直接构造了这一点：定义从 `x_1 ∼ N(0, I)` 到 `x_0 ∼ data` 的直线插值（straight-line interpolation），训练一个向量场（vector field）`v_θ(x, t)` 去匹配它的时间导数，并在推理时对其积分。

Rectified flow（Liu，2022）更进一步：通过 reflow 过程迭代地把路径拉直，得到越来越接近线性的 ODE。经过两轮 reflow 迭代后，2 步采样器就能达到 50 步 DDPM 的质量。

## 核心概念

*Flow matching：在噪声与数据之间做直线插值*

### 直线流

定义：

```
x_t = t · x_1 + (1 - t) · x_0,   t ∈ [0, 1]
```

其中 `x_0 ~ data` 且 `x_1 ~ N(0, I)`。沿着这条直线的时间导数是常数：

```
dx_t / dt = x_1 - x_0
```

定义一个神经向量场 `v_θ(x_t, t)`，并将其训练为匹配这个导数：

```
L = E_{x_0, x_1, t} || v_θ(x_t, t) - (x_1 - x_0) ||²
```

这就是**条件 Flow Matching（conditional flow matching）**损失（Lipman，2023）。训练时不需要模拟（simulation-free）：你永远不需要展开 ODE。只要采样 `(x_0, x_1, t)` 并做回归即可。

### 采样

在推理时，对学到的向量场沿时间*反向*积分：

```
x_{t-Δt} = x_t - Δt · v_θ(x_t, t)
```

从 `x_1 ~ N(0, I)` 开始，用 Euler 步一路走到 `t=0`。

### Rectified flow（校正流，Liu 2022）

直线流是有效的，但学到的路径*实际上并不直*——因为很多 `x_0` 都可能映射到同一个 `x_1`。Rectified flow 的 reflow 步骤如下：

1. 用随机配对训练 flow 模型 v_1。
2. 通过把 v_1 从 `x_1` 积分到其落点 `x_0`，采样 N 对 `(x_1, x_0)`。
3. 在这些配对样本上训练 v_2。因为这些对现在已经“与 ODE 匹配（ODE-matched）”，它们之间的直线插值会真正更平。
4. 重复。

在实践中，2 轮 reflow 迭代通常就足以接近线性，从而支持 2-4 步推理。SDXL-Turbo、SD3-Turbo、LCM 都属于从 flow matching 蒸馏而来的模型。

### 为什么它在 2024 年赢下了图像生成

原因有三：

1. **训练无需模拟**——训练期间不需要 ODE 展开，实现非常简单。
2. **损失几何更好**——直线路径拥有一致的信噪比，而 DDPM 的 ε-loss 在调度两端的 SNR 很差。
3. **推理更快**——在 SDXL-Turbo 级别质量下只需 4-8 步；使用 consistency distillation 时可降到 1 步。

## Flow matching vs DDPM —— 精确联系

带高斯条件路径（Gaussian-conditional path）的 flow matching，本质上就是*采用特定噪声调度的扩散*。选择 `x_t = α(t) x_0 + σ(t) x_1` 这样的调度后，flow matching 会恢复为以 Stratonovich 形式重写的扩散，其中 `v = α'·x_0 - σ'·x_1`。对于高斯路径，两者在代数上是等价的。

Flow matching 带来的新增价值：更清晰的目标（一个直接的 velocity）、更干净的损失，以及尝试非高斯插值（non-Gaussian interpolants）的自由。

## 动手实现

`code/main.py` 在一个双峰高斯混合分布上实现了 1 维 flow matching。向量场 `v_θ(x, t)` 是一个用直线目标训练的小型 MLP。在推理时，分别积分 1、2、4 和 20 个 Euler 步，并比较样本质量。

### 第 1 步：训练损失

```python
def train_step(x0, net, rng, lr):
    x1 = rng.gauss(0, 1)
    t = rng.random()
    x_t = t * x1 + (1 - t) * x0
    target = x1 - x0
    pred = net_forward(x_t, t)
    loss = (pred - target) ** 2
    # backprop + update
```

### 第 2 步：多步推理

```python
def sample(net, num_steps):
    x = rng.gauss(0, 1)
    for i in range(num_steps):
        t = 1.0 - i / num_steps
        dt = 1.0 / num_steps
        x -= dt * net_forward(x, t)
    return x
```

### 第 3 步：比较步数

预期 4 步采样器就已经能匹配 20 步的质量——这对延迟来说意义重大。

## 常见陷阱

- **时间参数化。** Flow matching 使用 `t ∈ [0, 1]`，其中 `t=0` 对应数据，`t=1` 对应噪声。DDPM 使用 `t ∈ [0, T]`，其中 `t=0` 对应数据，`t=T` 对应噪声。方向相同，尺度不同。论文经常把这点写错。
- **调度选择。** Rectified flow 的直线是“那个”标准的 flow-matching 调度，但你也可以使用 cosine 或 logit-normal 的 t 采样（SD3 就这么做）来获得更好的尺度覆盖。
- **Reflow 成本。** 为 reflow 生成配对数据集，需要对每个样本完整跑一次推理。只有在你确实需要 1-2 步推理时才值得做 reflow。
- **Classifier-free guidance 仍然适用。** 只需在线性组合里把 ε 换成 v：`v_cfg = (1+w) v_cond - w v_uncond`。

## 如何使用

| 用例 | 2026 年技术栈 |
|----------|-----------|
| 文本到图像，最高质量 | Flow matching: SD3, Flux.1-dev |
| 文本到图像，1-4 步 | Distilled flow matching: Flux.1-schnell, SD3-Turbo, SDXL-Turbo |
| 实时推理 | 基于 flow-matched base 的 consistency distillation（LCM, PCM） |
| 音频生成 | Flow matching: Stable Audio 2.5, AudioCraft 2 |
| 视频生成 | Flow matching 与 diffusion 混合（Sora, Veo, Stable Video） |
| 科学 / 物理（粒子轨迹、分子） | Flow matching + 等变向量场（equivariant vector field） |

每当一篇论文在 2025-2026 年声称“比 diffusion 更快”，它几乎总是在说 flow matching + distillation。

## 交付

保存 `outputs/skill-fm-tuner.md`。这个 skill 接收一个 diffusion 风格的模型规格，并将其转换为 flow-matching 训练配置：调度选择、时间采样分布（uniform / logit-normal）、优化器、reflow 计划、目标步数、评估协议。

## 练习

1. **简单。** 运行 `code/main.py`，比较 1 步与 20 步相对于真实数据分布的 MSE。
2. **中等。** 将均匀 `t` 采样切换为 logit-normal（会把采样集中在中间 t 区间）。模型质量有提升吗？
3. **困难。** 实现一轮 reflow 迭代：通过积分第一个模型生成配对的 `(x_0, x_1)`，在这些配对上训练第二个模型，并比较 1 步采样质量。

## 关键术语

| 术语 | 人们常说什么 | 实际含义 |
|------|-----------------|-----------------------|
| Flow matching | “直线版 diffusion” | 沿某个插值路径训练 `v_θ(x, t)` 去匹配 `x_1 - x_0`。 |
| Rectified flow | “Reflow” | 逐步把已学习流拉直的迭代过程。 |
| Velocity field | “v_θ” | 模型的输出——也就是推动 `x_t` 移动的方向。 |
| 直线插值器 | “那条路径” | `x_t = (1-t)·x_0 + t·x_1`；目标导数非常简单。 |
| Euler 采样器 | “一阶 ODE 求解器” | 最简单的积分器；当路径足够直时效果很好。 |
| Logit-normal t | “SD3 sampling” | 把 `t` 采样集中到梯度最强的中间区间。 |
| Consistency distillation | “1-step sampler” | 训练 student 直接把任意 `x_t` 映射到 `x_0`。 |
| 使用 velocity 的 CFG | “v-CFG” | `v_cfg = (1+w) v_cond - w v_uncond`；同样的技巧，换了个变量。 |

## 生产说明：Flux.1-schnell 是 flow matching 的极速形态

Flow matching 在生产上的代表性胜利是 Flux.1-schnell——一个 flow-matched DiT，被蒸馏到只需 1-4 个推理步，同时仍保留 Flux-dev 级别的质量。Niels 的“Run Flux on an 8GB machine”notebook 是参考部署方案：T5 + CLIP 编码，量化后的 MMDiT 去噪（schnell 用 4 步，dev 用 50 步），再由 VAE 解码。成本核算如下：

| 变体 | 步数 | L4 上 1024² 延迟 | 总 FLOPs（相对值） |
|---------|-------|------------------------|------------------------|
| Flux.1-dev（原始） | 50 | ~15 s | 1.0× |
| Flux.1-schnell | 4 | ~1.2 s | 0.08×（快 12×） |
| SDXL-base | 30 | ~4 s | 0.25× |
| SDXL-Lightning 2-step | 2 | ~0.3 s | 0.03× |

生产规则是：**flow-matched base + distillation = 2026 年快速 text-to-image 的默认方案。** 每一家主要厂商都在交付这个组合：SD3-Turbo（SD3 + flow + distillation）、Flux-schnell（Flux-dev + rectified-flow straightening）、CogView-4-Flash。纯 diffusion 基座只存在于遗留 checkpoint 中。

## 延伸阅读

- [Liu、Gong、Liu（2022）。Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) —— rectified flow。
- [Lipman 等（2023）。Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) —— flow matching。
- [Esser 等（2024）。Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) —— SD3，在大规模上应用 rectified flow。
- [Albergo、Vanden-Eijnden（2023）。Stochastic Interpolants](https://arxiv.org/abs/2303.08797) —— 覆盖 FM + diffusion 的通用框架。
- [Song 等（2023）。Consistency Models](https://arxiv.org/abs/2303.01469) —— diffusion / flow 的 1 步蒸馏。
- [Sauer 等（2023）。Adversarial Diffusion Distillation (SDXL-Turbo)](https://arxiv.org/abs/2311.17042) —— turbo 变体。
- [Black Forest Labs（2024）。Flux.1 models](https://blackforestlabs.ai/announcing-black-forest-labs/) —— 生产中的 flow matching。
