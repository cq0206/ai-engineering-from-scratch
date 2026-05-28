# 视觉 Transformer (Vision Transformers, ViT)

> 图像是一个 patch 网格。句子是一个 token 网格。同一个 transformer 两者都能吃下去。

**类型：** 构建
**语言：** Python
**前置要求：** 第 7 阶段 · 05（完整 Transformer）、第 4 阶段 · 03（CNN）、第 4 阶段 · 14（Vision Transformers 入门）
**时长：** ~45 分钟

## 问题

在 2020 年之前，计算机视觉几乎就等于卷积。ImageNet、COCO 以及各种检测基准上的所有 SOTA，使用的都是 CNN backbone。Transformer 属于语言领域。

Dosovitskiy 等人在 2020 年发表的《An Image is Worth 16x16 Words》证明：你可以把卷积完全去掉。把一张图像切成固定大小的 patch，把每个 patch 线性投影成一个 embedding，再把这个序列送进标准的 transformer encoder。在足够大的规模下（ImageNet-21k 预训练或更大），ViT 可以追平甚至超过基于 ResNet 的模型。

ViT 开启了 2026 年一个更大的模式：一种架构，多种模态。Whisper 把音频 token 化。ViT 把图像 token 化。机器人领域有 action tokens。视频领域有 pixel tokens。Transformer 并不在乎——只要你喂给它一个序列，它就能学。

到 2026 年，ViT 及其后代（DeiT、Swin、DINOv2、ViT-22B、SAM 3）已经占据了大多数视觉场景。CNN 在边缘设备和延迟敏感任务上依然更强。除此之外，几乎所有系统栈里某个地方都会有一个 ViT。

## 概念

*图像 → patches → tokens → transformer*

### 第 1 步 —— patchify

把一张 `H × W × C` 的图像切成一个形状为 `N × (P·P·C)` 的扁平 patch 序列。典型配置是：`224 × 224` 图像，`16 × 16` patch → 196 个 patch，每个 patch 含 768 个值。

```
image (224, 224, 3) → 14 × 14 grid of 16x16x3 patches → 196 vectors of length 768
```

Patch 大小是关键控制杆。Patch 越小 = token 越多，分辨率越高，但注意力成本是二次增长。Patch 越大 = 更粗糙，但更便宜。

### 第 2 步 —— 线性嵌入

用一个可学习矩阵把每个扁平 patch 投影到 `d_model`。这等价于一个 kernel size 为 `P`、stride 为 `P` 的卷积。在 PyTorch 里，这甚至就是 `nn.Conv2d(C, d_model, kernel_size=P, stride=P)`——只需两行实现。

### 第 3 步 —— 在前面加上 `[CLS]` token，并加入位置嵌入

- 在序列开头添加一个可学习的 `[CLS]` token。它最终的隐藏状态就是用于分类的图像表示。
- 加入可学习的位置嵌入（ViT 原版），或使用二维正弦位置编码（后续变体）。
- 到 2024+，RoPE 也被扩展到了二维位置场景，有时甚至不再需要显式嵌入。

### 第 4 步 —— 标准 transformer encoder

堆叠 L 个 `LayerNorm → Self-Attention → + → LayerNorm → MLP → +` 模块。和 BERT 完全一样。没有任何视觉专属层。这也是这篇论文最有教育意义的 punchline。

### 第 5 步 —— 头部

对于分类任务：取 `[CLS]` 的隐藏状态 → 线性层 → softmax。对于 DINOv2 或 SAM，则丢弃 `[CLS]`，直接使用 patch embeddings。

### 真正重要的变体

| 模型 | 年份 | 改动 |
|-------|------|--------|
| ViT | 2020 | 原始版本。固定 patch 大小，全局完全注意力。 |
| DeiT | 2021 | 蒸馏；只用 ImageNet-1k 也能训练。 |
| Swin | 2021 | 分层结构 + shifted windows。把成本降到固定的次二次级别。 |
| DINOv2 | 2023 | 自监督（不需要标签）。通用视觉特征最强。 |
| ViT-22B | 2023 | 220 亿参数；同样遵循 scaling laws。 |
| SigLIP | 2023 | ViT + language 配对，使用 sigmoid contrastive loss。 |
| SAM 3 | 2025 | Segment Anything；ViT-Large + 可提示的 mask decoder。 |

### 为什么它花了这么久才真正起飞

ViT 需要*大量*数据才能追平 CNN，因为它不具备 CNN 的那些归纳偏置 (inductive bias)（平移不变性、局部性）。如果没有超过 1 亿张带标签图像，或足够强的自监督预训练，那么在相同算力下 CNN 仍然更强。DeiT 在 2021 年用蒸馏技巧缓解了这个问题；DINOv2 则在 2023 年通过自监督几乎永久性地解决了它。

## 动手实现

参见 `code/main.py`。其中包含纯标准库实现的 patchify、线性嵌入以及若干 sanity checks。这里不涉及训练——任何有现实意义规模的 ViT 都需要 PyTorch 和数小时 GPU 时间。

### 第 1 步：构造假图像

构造一张 24 × 24 的 RGB 图像，表示为由 `(R, G, B)` 元组组成的行列表。我们使用 6×6 patch，因此会得到 16 个 patch，每个 patch 对应 108 维嵌入向量。

### 第 2 步：patchify

```python
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
```

采用光栅顺序：按行优先遍历整个网格。每个 ViT 都使用这种顺序。

### 第 3 步：线性嵌入

把每个扁平 patch 乘上一个随机的 `(patch_flat_size, d_model)` 矩阵。确认在前面加上 `[CLS]` 之后，输出形状为 `(N_patches + 1, d_model)`。

### 第 4 步：统计一个真实 ViT 的参数量

打印 ViT-Base 的参数量：12 层、12 个头、d=768、patch=16。把它和 ResNet-50（约 2500 万）进行比较。ViT-Base 大约是 8600 万参数。ViT-Large 约 3.07 亿。ViT-Huge 约 6.32 亿。

## 使用它

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768): [CLS] + 196 patches
cls_emb = out[:, 0]                       # image representation
```

**DINOv2 embeddings 是 2026 年图像特征的默认选择。** 冻结 backbone，只训练一个小头。它适用于分类、检索、检测、captioning。Meta 的 DINOv2 checkpoints 在所有非文本视觉任务上都优于 CLIP。

**如何选择 patch 大小。** 小模型通常使用 16×16（ViT-B/16）。稠密预测任务（分割）使用 8×8 或 14×14（SAM、DINOv2）。超大模型则常用 14×14。

## 交付物

参见 `outputs/skill-vit-configurator.md`。这个 skill 会根据数据集大小、分辨率和算力预算，为新的视觉任务选择合适的 ViT 变体与 patch 大小。

## 练习

1. **简单。** 运行 `code/main.py`。确认 patch 数量等于 `(H/P) * (W/P)`，并且扁平 patch 维度等于 `P*P*C`。
2. **中等。** 实现二维正弦位置嵌入：为每个 patch 的 `row` 和 `col` 分别生成独立的正弦编码，然后拼接。把它送入一个微型 PyTorch ViT，并与 CIFAR-10 上的可学习位置嵌入进行精度对比。
3. **困难。** 用 PyTorch 构建一个 3 层 ViT，在 1,000 张 MNIST 图像上、使用 4×4 patch 进行训练。测量测试准确率。然后在同样的 1,000 张图像上加入 DINOv2 风格的预训练（简化版：只训练 encoder，让它根据被 mask 的 patch 预测 patch embeddings）。准确率有提升吗？

## 关键术语

| 术语 | 人们常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Patch | “视觉 transformer 的 token” | 图像中一个 `P × P × C` 区域的像素值扁平向量。 |
| Patchify | “切块 + 展平” | 把图像切成互不重叠的 patch，并把每个 patch 展平成向量。 |
| `[CLS]` token | “图像摘要” | 预先加在序列前面的可学习 token；它最终的 embedding 就是图像表示。 |
| 归纳偏置 (Inductive bias) | “模型默认假设了什么” | ViT 的先验比 CNN 更少；因此需要更多数据来弥补差距。 |
| DINOv2 | “自监督 ViT” | 不依赖标签训练，通过图像增强 + momentum teacher 学习。是 2026 年最强的通用图像特征之一。 |
| SigLIP | “CLIP 的继任者” | 使用 sigmoid contrastive loss 训练的 ViT + 文本编码器；在相同算力下优于 CLIP。 |
| Swin | “窗口化 ViT” | 带局部注意力 + shifted windows 的分层 ViT；复杂度是次二次的。 |
| Register tokens | “2023 年的小技巧” | 少量额外的可学习 token，用来吸收 attention sinks；能提升 DINOv2 特征质量。 |

## 延伸阅读

- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — ViT 论文。
- [Touvron et al. (2021). Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) — DeiT。
- [Liu et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) — Swin。
- [Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — DINOv2。
- [Darcet et al. (2023). Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) — 针对 DINOv2 的 register-token 修复。
