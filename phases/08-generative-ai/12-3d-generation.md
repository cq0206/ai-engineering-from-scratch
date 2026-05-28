# 3D 生成

> 3D 是 2D→3D 杠杆效应最强的模态。2023 年的突破是 3D Gaussian Splatting。2024-2026 年的生成式推进，则是在其之上叠加多视图扩散（multi-view diffusion）+ 3D 重建（3D reconstruction），从单个提示词或照片生成物体与场景。

**类型：** 学习
**语言：** Python
**前置要求：** 第 4 阶段（视觉），第 8 阶段 · 07（潜空间扩散）
**耗时：** ~45 分钟

## 问题

3D 内容很痛苦：

- **表示形式。** 网格（meshes）、点云（point clouds）、体素网格（voxel grids）、符号距离场（signed distance fields, SDFs）、神经辐射场（neural radiance fields, NeRFs）、3D 高斯（3D Gaussians）。每种都有取舍。
- **数据稀缺。** ImageNet 有 1400 万张图像。最大的高质量 3D 数据集（Objaverse-XL，2023）约有 1000 万个对象，但大多数质量不高。
- **内存。** 一个 512³ 体素网格有 1.28 亿个体素；一个可用的场景 NeRF 每条射线需要 100 万个样本。生成比重建更难。
- **监督信号。** 对 2D 图像来说你有像素本身。对 3D 来说，你通常只有少量 2D 视图，还得把它们提升到 3D。

2026 年的技术栈把问题拆成两个部分。第一步，用扩散模型生成*2D 多视图图像*。第二步，将这些图像拟合成*3D 表示*（通常是 Gaussian splatting）。

## 核心概念

*3D 生成：多视图扩散 + 3D 重建*

### 表示：3D Gaussian Splatting（Kerbl 等，2023）

把场景表示为约 100 万个 3D 高斯组成的云。每个高斯有 59 个参数：位置（3）、协方差（6，或四元数 4 + 缩放 3）、不透明度（1）、球谐颜色（spherical-harmonics color，3 阶时为 48，0 阶时为 3）。

渲染 = 投影 + alpha 合成。速度快（在 4090 上 1080p 约 100 fps）。可微。通过对真实照片做梯度下降来拟合。一个场景在消费级 GPU 上可在 5-30 分钟内完成拟合。

叠加在其上的两个 2023-2024 创新：
- **生成式 Gaussian splats。** 像 LGM、LRM、InstantMesh 这样的模型，可以直接从一张或几张图像预测高斯云。
- **4D Gaussian Splatting。** 为动态场景中的高斯添加逐帧偏移。

### 多视图扩散

对预训练图像扩散模型进行微调，使其能从文本提示词或单张图像生成同一物体的多个一致视图。典型模型有 Zero123（Liu 等，2023）、MVDream（Shi 等，2023）、SV3D（Stability，2024）、CAT3D（Google，2024）。通常输出物体周围 4-16 个视图，再通过 Gaussian splatting 或 NeRF 提升为 3D。

### 文本到 3D（Text-to-3D）流水线

| 模型 | 输入 | 输出 | 时间 |
|-------|-------|--------|------|
| DreamFusion (2022) | 文本 | 通过 SDS 的 NeRF | 单个资产约 1 小时 |
| Magic3D | 文本 | 网格 + 纹理 | ~40 分钟 |
| Shap-E (OpenAI, 2023) | 文本 | 隐式 3D | ~1 分钟 |
| SJC / ProlificDreamer | 文本 | NeRF / 网格 | ~30 分钟 |
| LRM (Meta, 2023) | 图像 | triplane | ~5 秒 |
| InstantMesh (2024) | 图像 | 网格 | ~10 秒 |
| SV3D (Stability, 2024) | 图像 | 新视角 | ~2 分钟 |
| CAT3D (Google, 2024) | 1-64 张图像 | 3D NeRF | ~1 分钟 |
| TripoSR (2024) | 图像 | 网格 | ~1 秒 |
| Meshy 4 (2025) | 文本 + 图像 | PBR 网格 | ~30 秒 |
| Rodin Gen-1.5 (2025) | 文本 + 图像 | PBR 网格 | ~60 秒 |
| Tencent Hunyuan3D 2.0 (2025) | 图像 | 网格 | ~30 秒 |

2025-2026 年的方向：直接生成适用于游戏引擎、带有 PBR 材质的 text-to-mesh 模型。对于通用物体，多视图扩散作为中间步骤仍然是效果最好的配方。

### NeRF（背景知识）

Neural Radiance Field（Mildenhall 等，2020）。一个小型 MLP 接收 `(x, y, z, view direction)` 并输出 `(color, density)`。通过沿射线积分来渲染。它在质量上优于基于网格的新视角合成，但渲染速度慢 100-1000 倍。对于大多数实时场景，它已被 Gaussian splatting 取代，但在研究中仍占主导地位。

## 动手实现

`code/main.py` 实现了一个玩具版 2D “Gaussian splatting” 拟合：把一个合成目标图像（平滑渐变）表示为若干 2D 高斯 splat 的和。通过梯度下降优化位置、颜色和协方差，使其匹配目标图像。你会看到两个核心操作：前向渲染（splat + alpha 合成）和通过梯度下降拟合。

### 第 1 步：2D Gaussian splat

```python
def gaussian_at(x, y, gaussian):
    px, py = gaussian["pos"]
    sigma = gaussian["sigma"]
    d2 = (x - px) ** 2 + (y - py) ** 2
    return math.exp(-d2 / (2 * sigma * sigma))
```

### 第 2 步：通过累加 splat 渲染

```python
def render(image_size, gaussians):
    img = [[0.0] * image_size for _ in range(image_size)]
    for g in gaussians:
        for y in range(image_size):
            for x in range(image_size):
                img[y][x] += g["color"] * gaussian_at(x, y, g)
    return img
```

真实的 3D Gaussian splatting 会按深度对高斯排序，并按顺序做 alpha 合成。我们的 2D 玩具示例只是简单求和。

### 第 3 步：通过梯度下降拟合

```python
for step in range(steps):
    pred = render(size, gaussians)
    loss = mse(pred, target)
    gradients = compute_grads(pred, target, gaussians)
    update(gaussians, gradients, lr)
```

## 常见陷阱

- **视图不一致。** 如果你独立生成 4 个视图，而它们对物体结构的理解彼此矛盾，3D 拟合就会发糊。解决办法：使用带共享注意力（shared attention）的多视图扩散。
- **背面幻觉。** 单图 → 3D 必须凭空补出不可见的一面。质量波动很大。
- **Gaussian splat 爆炸。** 无约束训练会增长到 1000 万个 splat 并过拟合。密化（densification）+ 剪枝（pruning）启发式（来自 3D-GS 原论文）是必需的。
- **拓扑问题。** 从隐式场（implicit fields，如 SDF）提取的网格经常有孔洞或自相交。交付前先跑一次重网格化器（remesher，例如 blender 的 voxel remesh）。
- **训练数据许可证。** Objaverse 的许可证混杂；商业用途因模型而异。

## 如何使用

| 任务 | 2026 年选择 |
|------|-----------|
| 从照片重建场景 | Gaussian splatting（3DGS、Gsplat、Scaniverse） |
| 为游戏生成 text-to-3D 物体 | Meshy 4 或 Rodin Gen-1.5（PBR 输出） |
| 图像到 3D | Hunyuan3D 2.0、TripoSR、InstantMesh |
| 从少量图像做新视角合成 | CAT3D, SV3D |
| 动态场景重建 | 4D Gaussian Splatting |
| 虚拟形象 / 穿衣人体 | Gaussian Avatar, HUGS |
| 研究 / SOTA | 上周刚发布的那个 |

如果你要在游戏或电商流水线中交付生产级 3D：Meshy 4 和 Rodin Gen-1.5 会输出可直接进入 Unity / Unreal 的 PBR 网格。

## 交付

保存 `outputs/skill-3d-pipeline.md`。这个 skill 接收一个 3D 简述（输入：文本 / 单张图像 / 少量图像；输出：网格 / splat / NeRF；用途：渲染 / 游戏 / VR），并输出：流水线（multi-view diffusion + fit，或 direct mesh model）、基础模型、迭代预算、拓扑后处理、所需材质通道。

## 练习

1. **简单。** 用 4、16、64 个 Gaussians 运行 `code/main.py`。报告相对于目标的最终 MSE。
2. **中等。** 扩展为彩色高斯（RGB）。确认重建结果匹配目标颜色模式。
3. **困难。** 使用 gsplat 或 Nerfstudio，从 50 张照片重建一个真实物体。报告拟合时间以及保留视图上的最终 SSIM。

## 关键术语

| 术语 | 人们常说什么 | 实际含义 |
|------|-----------------|-----------------------|
| 3D Gaussian Splatting | “3DGS” | 将场景表示为 3D 高斯云；使用可微 alpha 合成渲染。 |
| NeRF | “Neural radiance field” | 在 3D 点上输出颜色 + 密度的 MLP；通过射线积分渲染。 |
| Triplane | “Three 2-D planes” | 将 3D 分解为三个二维轴对齐特征网格；比体积表示更便宜。 |
| SDS | “Score distillation sampling” | 用 2D 扩散的 score 作为伪梯度来训练 3D 模型。 |
| 多视图扩散 | “Many views at once” | 一次输出一批一致相机视图的扩散模型。 |
| PBR | “Physically-based rendering” | 包含 albedo、roughness、metallic、normal 通道的材质。 |
| 密化 | “Grow splats” | 3DGS 的训练启发式：在高梯度区域拆分 / 克隆 splat。 |

## 生产说明：3D 还没有共享底座

与图像（latent diffusion + DiT）和视频（spatiotemporal DiT）不同，3D 在 2026 年还没有单一主导运行时。生产上的决策树取决于表示形式：

- **NeRF / triplane。** 推理是射线步进（ray-marching）+ 每个样本做一次 MLP 前向。一次 512² 渲染需要数百万次 MLP 前向。要激进地对射线样本做批处理；SDPA/xformers 适用。
- **多视图扩散 + LRM 重建。** 两阶段流水线。第 1 阶段（multi-view DiT）就是像第 07 课那样的扩散服务。第 2 阶段（LRM transformer）则是对视图做一次性前向。整体延迟画像是“diffusion + one-shot”——因此要按阶段分别选择服务原语。
- **SDS / DreamFusion。** 这是逐资产优化，不是推理。构建的是作业系统，而不是请求处理器。

对于大多数 2026 年产品，正确答案是：“按请求运行一个多视图扩散模型，异步重建为 3DGS，再提供 3DGS 供实时查看。” 这样可以把工作负载清晰地拆分到 GPU 推理服务器（快）和离线优化器（慢）之间。

## 延伸阅读

- [Mildenhall 等（2020）。NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) —— NeRF。
- [Kerbl 等（2023）。3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) —— 3DGS。
- [Poole 等（2022）。DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988) —— SDS。
- [Liu 等（2023）。Zero-1-to-3: Zero-shot One Image to 3D Object](https://arxiv.org/abs/2303.11328) —— Zero123。
- [Shi 等（2023）。MVDream](https://arxiv.org/abs/2308.16512) —— 多视图扩散。
- [Hong 等（2023）。LRM: Large Reconstruction Model for Single Image to 3D](https://arxiv.org/abs/2311.04400) —— LRM。
- [Gao 等（2024）。CAT3D: Create Anything in 3D with Multi-View Diffusion Models](https://arxiv.org/abs/2405.10314) —— CAT3D。
- [Stability AI（2024）。Stable Video 3D (SV3D)](https://stability.ai/research/sv3d) —— SV3D。
