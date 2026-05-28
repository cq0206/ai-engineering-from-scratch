# 视觉变换器 (Vision Transformers, ViT)

> 把图像切成补丁，把每个补丁当作一个词，运行标准 Transformer。不要回头看。

**类型：** 构建
**语言：** Python
**先修要求：** 第 7 阶段第 02 课（自注意力 / Self-Attention），第 4 阶段第 04 课（图像分类 / Image Classification）
**时间：** ~45 分钟

## 学习目标

- 从零实现补丁嵌入、学习式位置嵌入、类别 token 和 Transformer 编码器块，构建一个最小化 ViT
- 解释为什么人们曾认为 ViT 需要海量预训练数据，直到 DeiT 和 MAE 证明事实并非如此
- 从架构先验（无先验、局部窗口注意力、卷积骨干网络）比较 ViT、Swin 和 ConvNeXt
- 使用 `timm` 和标准的线性探测 / 微调（linear-probe / fine-tune）配方，在小数据集上微调一个预训练 ViT

## 问题

十年来，卷积 (convolution) 几乎就是计算机视觉的代名词。卷积神经网络 (Convolutional Neural Networks, CNNs) 具有很强的归纳偏置——局部性、平移等变性——几乎没人认为这些特性可以被替代。随后 Dosovitskiy 等人（2020）表明，一个普通 Transformer 只要作用在展平后的图像补丁上，完全不依赖任何卷积机制，就能够在大规模场景下匹敌甚至超过最好的 CNN。

问题在于“在大规模场景下”。在 ImageNet-1k 上，ViT 不如 ResNet。先在 ImageNet-21k 或 JFT-300M 上预训练，再在 ImageNet-1k 上微调的 ViT，则能胜过它。由此得到的结论是：Transformer 缺少有用的先验，但只要数据足够多，它也能把这些先验学出来。后续工作（DeiT、MAE、DINO）则表明，只要训练配方合适——强增强、自监督预训练、蒸馏——ViT 在小数据上同样能训练得很好。

到 2026 年，纯 CNN 在边缘设备上仍然很有竞争力（ConvNeXt 最强），但 Transformer 已经主导了其他几乎所有方向：分割 (segmentation)（Mask2Former、SegFormer）、检测 (detection)（DETR、RT-DETR）、多模态 (multimodal)（CLIP、SigLIP）、视频 (video)（VideoMAE、VJEPA）。ViT 的模块结构是你必须掌握的那一种。

## 概念

### 流水线

```mermaid
flowchart LR
    IMG["图像<br/>(3, 224, 224)"] --> PATCH["补丁嵌入<br/>conv 16x16 s=16<br/>-> (768, 14, 14)"]
    PATCH --> FLAT["展平为<br/>(196, 768) token"]
    FLAT --> CAT["前置<br/>[CLS] token"]
    CAT --> POS["加入学习式<br/>位置嵌入"]
    POS --> ENC["N 个 Transformer<br/>编码器块"]
    ENC --> CLS["取 [CLS]<br/>token 输出"]
    CLS --> HEAD["MLP 分类器"]

    style PATCH fill:#dbeafe,stroke:#2563eb
    style ENC fill:#fef3c7,stroke:#d97706
    style HEAD fill:#dcfce7,stroke:#16a34a
```

七个步骤。补丁 -> token -> 注意力 -> 分类器。每个变体（DeiT、Swin、ConvNeXt、MAE 预训练）都只会改动这七步中的一两步，其余部分保持不变。

### 补丁嵌入

第一个卷积层就是秘诀。卷积核大小为 16，步幅也为 16，因此一张 224x224 图像会变成一个 14x14 的 16x16 补丁网格，每个补丁都会被投影成一个 768 维嵌入。这个单独的卷积层同时完成了补丁化和线性投影。

```
Input:  (3, 224, 224)
Conv (3 -> 768, k=16, s=16, no padding):
Output: (768, 14, 14)
Flatten spatial: (196, 768)
```

196 个补丁 = 196 个 token。每个 token 的特征维度可以是 768（ViT-B）、1024（ViT-L）或 1280（ViT-H）。

### 类别 token

在序列前面添加一个可学习向量：

```
tokens = [CLS; patch_1; patch_2; ...; patch_196]   shape (197, 768)
```

经过 N 个 Transformer 块之后，`[CLS]` 的输出就是全局图像表示。分类头只读取这一个向量。

### 位置嵌入

Transformer 天生没有空间位置的内置概念。做法是给每个 token 加上一个可学习向量：

```
tokens = tokens + learned_pos_embedding   (also shape (197, 768))
```

这个嵌入是模型参数的一部分；基于梯度的训练会让它适应二维图像结构。二维正弦替代方案也存在，但实践里很少使用。

### Transformer 编码器块

标准配置。多头自注意力 (multi-head self-attention)、MLP、残差连接、pre-LayerNorm。

```
x = x + MSA(LN(x))
x = x + MLP(LN(x))

MLP is two-layer with GELU: Linear(d -> 4d) -> GELU -> Linear(4d -> d)
```

ViT-B/16 堆叠了 12 个这样的块，每个块有 12 个注意力头，总参数量为 8600 万。

### 为什么使用 pre-LN

早期 Transformer 使用 post-LN（`x = LN(x + sublayer(x))`），如果没有预热（warmup），训练深度超过 6–8 层就会变得困难。pre-LN（`x = x + sublayer(LN(x))`）则可以在没有预热的情况下稳定训练更深的网络。所有 ViT 和所有现代 LLM 都使用 pre-LN。

### 补丁大小的权衡

- 16x16 补丁 -> 196 个 token，标准配置。
- 32x32 补丁 -> 49 个 token，更快但分辨率更低。
- 8x8 补丁 -> 784 个 token，更细但 O(n^2) 注意力成本扩展很差。

补丁越大 = token 越少 = 速度越快，但空间细节越少。SwinV2 在分层窗口中使用 4x4 补丁。

### DeiT 在 ImageNet-1k 上训练 ViT 的配方

原始 ViT 需要 JFT-300M 才能超过 CNN。DeiT（Touvron 等，2020）只靠 ImageNet-1k 就把 ViT-B 训练到了 81.8% 的 top-1 准确率，具体有四个改动：

1. 重度增强：RandAugment、Mixup、CutMix、Random Erasing。
2. 随机深度 (stochastic depth)（训练时随机丢弃整个块）。
3. 重复增强（同一张图在每个 batch 中采样 3 次）。
4. 来自 CNN 教师模型的蒸馏（可选，但会进一步提升精度）。

所有现代 ViT 训练配方都源自 DeiT。

### Swin 与 ConvNeXt

- **Swin**（Liu 等，2021）——基于窗口的注意力。每个块只在局部窗口内做注意力；相邻块交替平移窗口，以便在窗口之间混合信息。它在保留注意力算子的同时，把类似 CNN 的局部性先验重新带了回来。
- **ConvNeXt**（Liu 等，2022）——重新设计的 CNN，对齐了 Swin 的架构选择（逐通道卷积 / depthwise conv、LayerNorm、GELU、倒置瓶颈 / inverted bottleneck）。它说明真正的差距并不是“注意力 vs 卷积”，而是“现代训练配方 + 架构设计”。

到 2026 年，ConvNeXt-V2 和 Swin-V2 都已经是生产可用级别；正确选择取决于你的推理栈（ConvNeXt 在边缘端编译效果更好）以及预训练语料。

### MAE 预训练

掩码自编码器 (Masked Autoencoder, MAE)（He 等，2022）：随机遮住 75% 的补丁，让编码器只处理可见的 25%，再训练一个小解码器，根据编码器输出重建被遮住的补丁。预训练结束后，丢弃解码器，对编码器做微调。

MAE 让 ViT 仅依赖 ImageNet-1k 也能稳定训练、达到 SOTA，并成为当前默认的自监督配方。

## 动手构建

### 第 1 步：补丁嵌入

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, dim=192, image_size=64):
        super().__init__()
        assert image_size % patch_size == 0
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.num_patches = num_patches

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)
```

一个卷积、一次展平、一次转置。这就是完整的“图像到 token”步骤。

### 第 2 步：Transformer 块

pre-LN、多头自注意力、带 GELU 的 MLP、残差连接。

```python
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x
```

`nn.MultiheadAttention` 会处理拆分注意力头、缩放点积（scaled dot-product）以及输出投影。`batch_first=True` 表示张量形状是 `(N, seq, dim)`。

### 第 3 步：ViT

```python
class ViT(nn.Module):
    def __init__(self, image_size=64, patch_size=16, in_channels=3,
                 num_classes=10, dim=192, depth=6, num_heads=3, mlp_ratio=4):
        super().__init__()
        self.patch = PatchEmbedding(in_channels, patch_size, dim, image_size)
        num_patches = self.patch.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x[:, 0])
        return self.head(x)

vit = ViT(image_size=64, patch_size=16, num_classes=10, dim=192, depth=6, num_heads=3)
x = torch.randn(2, 3, 64, 64)
print(f"output: {vit(x).shape}")
print(f"params: {sum(p.numel() for p in vit.parameters()):,}")
```

大约 280 万参数——这是一个可以在 CPU 上处理的微型 ViT。真实的 ViT-B 有 8600 万参数；只需把同一个类定义改成 `dim=768, depth=12, num_heads=12`。

### 第 4 步：合理性检查——单张图像推理

```python
logits = vit(torch.randn(1, 3, 64, 64))
print(f"logits: {logits}")
print(f"probs:  {logits.softmax(-1)}")
```

应该能够无报错运行。概率之和应为 1。

## 使用它

`timm` 内置了所有 ViT 变体及其 ImageNet 预训练权重。一行就够：

```python
import timm

model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=10)
```

到 2026 年，`timm` 已经是视觉 Transformer 的生产默认库。它在同一套 API 下支持 ViT、DeiT、Swin、Swin-V2、ConvNeXt、ConvNeXt-V2、MaxViT、MViT、EfficientFormer 以及几十种其他模型。

对于多模态工作（图像 + 文本），`transformers` 提供了 CLIP、SigLIP、BLIP-2、LLaVA。这些模型里的图像编码器全部都是某种 ViT 变体。

## 交付它

本课会产出：

- `outputs/prompt-vit-vs-cnn-picker.md` —— 一个提示词，可根据数据集规模、算力和推理栈，在 ViT、ConvNeXt 和 Swin 之间做选择。
- `outputs/skill-vit-patch-and-pos-embed-inspector.md` —— 一个技能，用于验证 ViT 的补丁嵌入和位置嵌入形状是否与模型期望的序列长度一致，从而捕获最常见的移植错误。

## 练习

1. **（简单）** 打印上面这个微型 ViT 在一次前向传播中每个中间张量的形状。确认：输入 `(N, 3, 64, 64)` -> 补丁 `(N, 16, 192)` -> 加上 CLS 后 `(N, 17, 192)` -> 分类器输入 `(N, 192)` -> 输出 `(N, num_classes)`。
2. **（中等）** 在第 4 课的 synthetic-CIFAR 数据集上微调一个预训练 `timm` ViT-S/16。与在同一数据上微调 ResNet-18 做比较。报告训练时间和最终精度。
3. **（困难）** 为这个微型 ViT 实现 MAE 预训练：遮住 75% 的补丁，训练编码器 + 一个小解码器来重建被遮住的补丁。评估预训练前后在合成数据上的线性探测（linear-probe）精度。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| 补丁嵌入 | “第一个卷积层” | 一个卷积层，其 kernel size = stride = patch size；把图像变成 token 嵌入网格 |
| 类别 token | “[CLS]” | 一个加在 token 序列最前面的可学习向量；其最终输出就是全局图像表示 |
| 位置嵌入 | “学习式位置编码” | 一个加到每个 token 上的可学习向量，让 Transformer 知道每个补丁来自哪里 |
| Pre-LN | “子层之前的 LayerNorm” | 稳定版 Transformer：`x + sublayer(LN(x))`，而不是 `LN(x + sublayer(x))` |
| 多头注意力 | “并行注意力” | 标准 Transformer 注意力，被拆分到 `num_heads` 个独立子空间后再拼接 |
| ViT-B/16 | “Base，patch 16” | 经典配置：dim=768、depth=12、heads=12、patch_size=16、image=224；约 8600 万参数 |
| DeiT | “数据高效 ViT” | 仅用 ImageNet-1k 和强增强训练出的 ViT；证明大规模预训练数据集并非绝对必要 |
| MAE | “掩码自编码器” | 自监督预训练：遮住 75% 的补丁，再重建；是主流的 ViT 预训练配方 |

## 延伸阅读

- [An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929) —— ViT 论文
- [DeiT: Data-efficient Image Transformers (Touvron et al., 2020)](https://arxiv.org/abs/2012.12877) —— 如何只用 ImageNet-1k 训练 ViT
- [Masked Autoencoders are Scalable Vision Learners (He et al., 2022)](https://arxiv.org/abs/2111.06377) —— MAE 预训练
- [timm documentation](https://huggingface.co/docs/timm) —— 你在生产环境中会用到的所有视觉 Transformer 的参考文档
