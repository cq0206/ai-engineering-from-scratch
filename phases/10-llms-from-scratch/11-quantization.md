# 量化（Quantization）：让模型装得下

> 一个 70B 模型用 FP16 需要 140GB。仅权重就要两张 A100。量化到 FP8：一张 80GB GPU。INT4：一台 MacBook。

**类型：** 构建
**语言：** Python（使用 numpy）
**前置要求：** 第 10 阶段，第 01-10 课（从零开始构建 LLM）
**时间：** 约 120 分钟

## 学习目标

- 实现从 FP16 到 INT8 和 INT4 的对称量化（symmetric quantization）与非对称量化（asymmetric quantization），包括按张量（per-tensor）和按通道（per-channel）缩放
- 计算量化带来的内存节省，并判断某种精度是否能装进给定 GPU 的显存（VRAM）
- 解释训练后量化（post-training quantization, PTQ）和量化感知训练（quantization-aware training, QAT）之间的区别
- 应用 GPTQ 或 AWQ 对真实模型进行量化，并在基准测试上衡量准确率与内存之间的权衡

## 问题

Llama 3 70B 有 700 亿个参数。每个参数都是一个 16 位浮点数（floating-point number）。这就是 1400 亿字节，也就是 140GB。一张 A100 只有 80GB 显存。你甚至无法把权重加载进去，更别说在单张 GPU 上做推理（inference）了。你需要两张 A100，而且每张每小时 2 美元，才能提供一个模型的服务。

但每个参数用 16 位其实很浪费。神经网络（neural network）中的大多数权重都聚集在零附近。FP16 的完整动态范围（dynamic range，从 0.000000059 到 65,504）几乎完全没有被用到。如果你去测量 Llama 3 70B 的真实权重分布，会发现其中 95% 都落在 -0.1 到 +0.1 之间。你正在用 16 位表示那些其实 4 位就能装下的值。

量化会用低精度数字替换高精度数字。FP16 到 FP8 会把内存减半。FP16 到 INT4 会把内存压到四分之一。那个 140GB 的模型会变成 35GB，可以装进一张消费级 GPU。再激进一点做 2 位量化（有损，但对某些任务可用），同一个模型甚至能在 16GB 笔记本上运行。

代价是准确率。你每去掉一位，都会丢失一部分信息。问题在于：你会损失多少准确率，以及损失发生在哪里。一个量化得当的 INT4 模型，在大多数基准上仍能保留原模型 95-99% 的质量。一个天真的 INT4 量化则可能把模型彻底毁掉。差别就在方法。

社区用 GPTQ 把 Llama 3 量化到 INT4 的结果显示，在 WikiText 上大约只损失 1-2 个困惑度（perplexity）点。Mistral 发布的 Mixtral 8x22B FP8 检查点（checkpoint）在 MMLU 上几乎没有可测的质量损失。GGUF 格式支撑着 llama.cpp，让 70B 模型可以在配备 M 系列芯片的 MacBook 上运行。量化不是权宜之计，而是所有大于 7B 模型的标准部署路径。

## 概念

### 数字格式：每一位在做什么

每个浮点数都有三个部分：符号（sign）、指数（exponent）和尾数（mantissa，也叫 significand）。符号占 1 位。指数决定范围（这个数能有多大或多小）。尾数决定精度（你能保留多少小数位）。

```
FP32:  [1 sign] [8 exponent] [23 mantissa]  = 32 bits
FP16:  [1 sign] [5 exponent] [10 mantissa]  = 16 bits
BF16:  [1 sign] [8 exponent] [7  mantissa]  = 16 bits
FP8:   [1 sign] [4 exponent] [3  mantissa]  = 8  bits (E4M3)
FP8:   [1 sign] [5 exponent] [2  mantissa]  = 8  bits (E5M2)
INT8:  [1 sign] [7 value]                   = 8  bits (uniform steps)
INT4:  [1 sign] [3 value]                   = 4  bits (16 levels total)
```

**FP32** 是全精度。23 位尾数能给你大约 7 位十进制有效数字。范围大致是 1.2 x 10^-38 到 3.4 x 10^38。训练过去几乎完全在 FP32 中进行。现在它仍然用于累加（比如矩阵乘法中的运行和）。

**FP16** 把位数减半。10 位尾数能提供大约 3.3 位十进制有效数字。指数缩小到 5 位，范围也大幅缩小（最大值约 65,504）。这对权重（weights，通常聚集在零附近）来说没问题，但对激活值（activations）和梯度（gradients）来说就危险了，因为它们在训练中可能会突然飙升。FP16 训练需要做损失缩放（loss scaling）来防止下溢。

**BF16**（Brain Float 16）保留了 FP32 的 8 位指数，但把尾数缩到 7 位。它的范围和 FP32 相同，但精度比 FP16 更低。Google 专门为深度学习设计了它。直觉是：对神经网络来说，范围比精度更重要。一个在 FP16 中会下溢成 0 的 10^-20 梯度，在 BF16 中仍然能保留下来。一个在 BF16 中从 0.07342 舍入成 0.0734 的权重，已经足够接近。现代训练基本都在使用 BF16 或 BF16/FP32 混合。

**FP8** 有两种变体。E4M3（4 位指数，3 位尾数）用于推理时的权重和激活值。E5M2（5 位指数，2 位尾数）用于训练时的梯度，因为此时范围比精度更重要。H100 GPU 上的 FP8 推理相较 FP16 能获得 30-50% 的加速，而质量损失几乎可以忽略。

**INT8** 是整数格式。没有指数，也没有尾数。只有从 -128 到 127 的 256 个均匀间隔值。你需要一个缩放因子（scale factor）把浮点权重映射到这个范围。好处是：整数算术比浮点算术更快，也更省电。A100 上的 INT8 矩阵乘法可达到 624 TOPS，而 FP16 是 312 TFLOPS。

**INT4** 更进一步。它只有 16 个可能取值。缩放因子承担了大量工作。质量完全取决于你如何选择缩放，以及你量化的是哪些权重。最先进的 INT4 方法（GPTQ、AWQ）仍能保留原模型 95% 以上的质量。

```mermaid
graph LR
    subgraph Formats["数字格式全景"]
        direction TB
        FP32["FP32\n32 位\n4 字节/参数\n训练黄金标准"]
        BF16["BF16\n16 位\n2 字节/参数\n训练默认格式"]
        FP16["FP16\n16 位\n2 字节/参数\n推理基线"]
        FP8["FP8\n8 位\n1 字节/参数\n快 30-50%"]
        INT8["INT8\n8 位\n1 字节/参数\n2x 吞吐量"]
        INT4["INT4\n4 位\n0.5 字节/参数\n4x 压缩"]
    end

    FP32 -->|"训练"| BF16
    BF16 -->|"推理"| FP16
    FP16 -->|"H100 原生"| FP8
    FP16 -->|"服务器部署"| INT8
    FP16 -->|"边缘/笔记本"| INT4

    style FP32 fill:#1a1a2e,stroke:#0f3460,color:#fff
    style BF16 fill:#1a1a2e,stroke:#0f3460,color:#fff
    style FP16 fill:#1a1a2e,stroke:#ffa500,color:#fff
    style FP8 fill:#1a1a2e,stroke:#51cf66,color:#fff
    style INT8 fill:#1a1a2e,stroke:#51cf66,color:#fff
    style INT4 fill:#1a1a2e,stroke:#e94560,color:#fff
```

### 量化如何工作

核心操作很简单：拿到一个由浮点值组成的张量（tensor），找到一个缩放因子，做乘法，四舍五入到最近的整数，然后把这些整数连同缩放因子一起存下来。

**Quantize:**
```
scale = max(abs(tensor)) / max_int_value
quantized = round(tensor / scale)
```

**Dequantize:**
```
reconstructed = quantized * scale
```

对于使用对称范围（-127 到 127）的 INT8：
```
scale = max(abs(tensor)) / 127
quantized = clamp(round(tensor / scale), -128, 127)
```

误差来自舍入误差。每个值的偏差最多是 `scale / 2`。一整层的总误差取决于你有多少权重，以及模型对这些权重扰动有多敏感。

**按张量量化（per-tensor quantization）与按通道量化（per-channel quantization）。** 按张量量化对整个权重矩阵只使用一个缩放因子。它简单，但有损：如果某一列值很大而另一列值很小，小值就会失去大部分精度。按通道量化会对每个输出通道（也就是权重矩阵的每一行或每一列）使用一个缩放因子。开销更高（你要存 N 个缩放因子而不是 1 个），但质量会显著更好。所有生产级量化方法都使用按通道或更细粒度的方案。

**非对称量化** 会加入一个零点偏移（zero-point offset）：`quantized = round(tensor / scale) + zero_point`。这适用于分布不以零为中心的情况。例如 ReLU 激活值始终非负。对称量化会把一半的整数范围浪费在永远不会出现的负值上。非对称量化会把真实范围 [min, max] 映射到完整整数范围。

### 敏感度层级

模型里的不同部分，对量化的容忍度并不一样，而且层级很清晰。

**权重（最稳健）。** 模型权重在训练中变化缓慢，并且通常服从以零为中心的近似高斯分布。它们很适合量化。带按通道缩放的 INT8 权重几乎是无损的。INT4 需要更复杂的方法，但同样可行。

**激活值（中等敏感）。** 激活值是在推理时沿着网络流动的中间值。它们的动态范围比权重大，而且常常有离群值（outliers）。单个注意力头（attention head）可能产生比均值大 100 倍的激活值。这些离群值对模型质量至关重要。天真地量化它们会摧毁信息。解决方案包括：把离群通道保留为更高精度（LLM.int8()），或使用按词元（token）/按通道的激活缩放。

**KV 缓存（高敏感）。** 键值缓存（key-value cache）会保存所有先前词元的注意力状态。上下文长度一长，KV 缓存就会主导内存占用。对于一个 70B 模型，32K 上下文时仅 KV 缓存用 FP16 就要 40GB。把 KV 缓存量化到 FP8 或 INT8 可以节省大量内存，但任何误差都会在后续所有注意力计算中不断累积。质量影响会随着序列长度增长而放大。

**注意力 logits（最敏感）。** 注意力中的 softmax 函数（softmax）对输入的微小变化极其敏感。在 softmax 之前，某个 logit（未归一化分数）出现 0.01 的量化误差，就足以明显改变注意力分布。大多数量化方案都会把注意力计算保留在更高精度（FP16 或 BF16）中，即便其他部分都已经量化。

```mermaid
graph TD
    subgraph Sensitivity["量化敏感度（低到高）"]
        direction LR
        W["权重\n高斯分布，接近零\nINT4 效果好"]
        A["激活值\n范围更宽，有离群值\nINT8 需谨慎"]
        KV["KV 缓存\n误差会累积\nFP8 或 INT8"]
        ATT["注意力 Logits\nSoftmax 放大误差\n保持 FP16"]
    end

    W -->|"安全"| A
    A -->|"谨慎"| KV
    KV -->|"危险"| ATT

    style W fill:#1a1a2e,stroke:#51cf66,color:#fff
    style A fill:#1a1a2e,stroke:#ffa500,color:#fff
    style KV fill:#1a1a2e,stroke:#e94560,color:#fff
    style ATT fill:#1a1a2e,stroke:#ff0000,color:#fff
```

### PTQ 与 QAT

**训练后量化（PTQ）** 会对一个已经训练完成的模型做量化。不需要重新训练。你拿到 FP16 权重，计算缩放因子，做舍入，然后部署。它速度快（几分钟到几小时）且成本低。对 INT8 和 FP8 效果很好。对于 INT4，天真的 PTQ 往往会失败得很惨，因为舍入误差会不断累积。高级 PTQ 方法（GPTQ、AWQ）会使用校准数据（calibration data）来尽量减小量化误差。

**量化感知训练（QAT）** 会在训练时的前向传播（forward pass）中插入伪量化（fake quantization）操作。模型会学着把权重放在那些舍入误差较小的位置上。梯度通过直通估计器（straight-through estimator, STE）穿过这些伪量化操作：假装舍入操作的梯度是 1。QAT 产生的 INT4 和 INT2 模型通常比 PTQ 更好，但它要求完整跑一遍训练。Google 用 QAT 支撑 Gemini 的高效服务。Meta 也在一些 Llama 部署目标上使用了 QAT。

| 方面 | PTQ | QAT |
|--------|-----|-----|
| 成本 | 几分钟到几小时 | 完整训练过程 |
| INT8 下的质量 | 极好（&lt; 0.1% 损失） | 极好 |
| INT4 下的质量 | 配合 GPTQ/AWQ 时较好（1-3% 损失） | 更好（&lt; 1% 损失） |
| INT2 下的质量 | 较差 | 某些任务可用 |
| 校准数据 | 128-1024 个样本 | 完整训练数据集 |
| 适用时机 | 部署、快速迭代 | 追求低比特宽度下的最高质量 |

### GPTQ、AWQ、GGUF

**GPTQ（GPT Quantization）** 是一种一次性的 PTQ 方法。它一次量化一层中的权重，使用一个很小的校准数据集（通常 128 个样本）去估计海森矩阵（Hessian，描述输出对每个权重有多敏感的二阶信息）。被海森矩阵判断为重要的权重，会被更仔细地量化。GPTQ 是第一个让 LLM 的 INT4 量化真正变得实用的方法。Hugging Face 上的 TheBloke 通过发布数百个模型的量化版本，普及了 GPTQ。

**AWQ（Activation-Aware Weight Quantization）** 观察到，只有一小部分权重（约 1%）格外重要，因为它们会和较大的激活值相乘。AWQ 会利用校准数据识别这些显著权重（salient weights），并在量化前把它们放大（再把对应激活值缩小）。这样就能让这些重要权重落在 INT4 量化更准确的范围里。AWQ 的质量通常与 GPTQ 持平或略优，同时应用速度快 1.5-2 倍。

**GGUF（GPT-Generated Unified Format）** 是 llama.cpp 及其生态所使用的文件格式。它支持混合量化（mixed quantization）：不同层使用不同位宽。第一层和最后一层（嵌入层与输出头）通常保留更高精度，中间层使用 INT4 或 INT3。GGUF 文件是自包含的：权重、分词器（tokenizer）、元数据都在同一个文件里。这个格式是为 CPU 推理和 Apple Silicon 设计的，因为在这些环境里，标准路径就是把整个模型载入内存，并在 CPU 或 Metal GPU 上运行矩阵乘法。Q4_K_M 是最流行的 GGUF 量化变体，在质量与体积之间做了很好的平衡。

```mermaid
graph TD
    subgraph Methods["量化方法"]
        direction TB
        GPTQ_["GPTQ\n海森矩阵引导\n逐层优化\n在 Hugging Face 上很流行"]
        AWQ_["AWQ\n感知激活值\n显著权重缩放\n比 GPTQ 快 1.5-2 倍"]
        GGUF_["GGUF\n混合精度\n针对 CPU + Metal 优化\nllama.cpp 生态"]
    end

    subgraph Use["最适合"]
        GPU["GPU 推理\n(CUDA, ROCm)"]
        EDGE["边缘 / 笔记本\n(CPU, Metal)"]
    end

    GPTQ_ --> GPU
    AWQ_ --> GPU
    GGUF_ --> EDGE

    style GPTQ_ fill:#1a1a2e,stroke:#ffa500,color:#fff
    style AWQ_ fill:#1a1a2e,stroke:#51cf66,color:#fff
    style GGUF_ fill:#1a1a2e,stroke:#0f3460,color:#fff
```

### 质量评估

你怎么知道量化后的模型是否依然够好？

**困惑度。** 这是最常见的指标，越低越好。对原始模型和量化模型都在一个留出数据集（WikiText-2 是标准选择）上计算困惑度。它们的差值会告诉你量化毁掉了多少信息。经验法则：差值 &lt; 0.5 非常好，0.5-1.0 较好，1.0-2.0 对大多数任务可以接受，> 2.0 基本说明出了问题。

**任务特定基准。** 在 MMLU、HumanEval、GSM8K 或你自己的评测集上运行量化模型，并与原模型比较。量化对不同能力的影响并不均匀。数学和代码任务比通识类任务更容易受到精度损失影响。

**输出对比。** 对相同提示词分别用两个模型生成回答，然后比较。LLM-as-judge（第 10 课）在这里很好用。你可以计算一个胜率（win rate）：量化模型在多大比例的提示词上能与原模型打平或胜出？

**延迟与吞吐量。** 量化存在的意义就是让模型更快、更便宜。要测量每秒 token 数、首 token 时间以及内存占用。一个量化后比原模型还慢的模型，几乎毫无价值。

| 模型 | 格式 | 大小 | 困惑度（WikiText-2） | MMLU | Tokens/秒（A100） |
|-------|--------|------|------------------------|------|-------------------|
| Llama 3 70B | FP16 | 140GB | 3.12 | 79.5% | 38 |
| Llama 3 70B | FP8 | 70GB | 3.14 | 79.3% | 55 |
| Llama 3 70B | GPTQ INT4 | 35GB | 4.32 | 77.8% | 72 |
| Llama 3 70B | AWQ INT4 | 35GB | 4.18 | 78.1% | 75 |
| Llama 3 70B | GGUF Q4_K_M | 40GB | 4.25 | 77.9% | 28 (CPU) |

规律很明显：FP8 几乎是白捡的提升。INT4 会损失 1-2 个 MMLU 点，但吞吐量翻倍、内存变成四分之一。对几乎所有部署场景来说，这个权衡都值得。

### 实际数字

H100 上从 FP16 到 FP8：推理速度提升 30-50%，质量损失 &lt; 0.1%。这是最无需犹豫的量化方式。所有 H100 部署都应该使用它。

从 FP16 到 INT8（LLM.int8()）：内存减少 2 倍，质量损失 &lt; 0.5%。这种混合精度方法会把离群特征保留在 FP16 中，而把其他所有部分量化到 INT8。

从 FP16 到 INT4（GPTQ/AWQ）：内存减少 4 倍，质量损失 1-3%，具体取决于模型和方法。它让 70B 模型可以跑在单张 48GB GPU 上。

从 FP16 到 INT4（GGUF Q4_K_M）：内存减少 3.5 倍，质量损失 1-2%。这是针对 CPU 推理优化的。一个 Q4_K_M 的 70B 模型大约是 40GB，在配备 64GB 内存的 M3 Max 上可达到 10-15 tokens/秒。

从 FP16 到 INT2：内存减少 8 倍，质量损失 5-15%。它只适用于那些你能接受明显退化的窄任务场景。仍然属于研究前沿，不适合通用生产环境。

## 动手构建

### 第 1 步：数字格式表示

构建每种格式的位级表示，这样你就能精确看到符号、指数和尾数分别在做什么。

```python
import numpy as np


def float_to_fp32_bits(value):
    bits = np.float32(value).view(np.uint32)
    sign = (bits >> 31) & 1
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    return {"sign": int(sign), "exponent": int(exponent), "mantissa": int(mantissa),
            "exponent_bits": format(int(exponent), '08b'),
            "mantissa_bits": format(int(mantissa), '023b'),
            "value": float(value),
            "actual_exponent": int(exponent) - 127}


def float_to_fp16_bits(value):
    fp16 = np.float16(value)
    bits = fp16.view(np.uint16)
    sign = (bits >> 15) & 1
    exponent = (bits >> 10) & 0x1F
    mantissa = bits & 0x3FF
    return {"sign": int(sign), "exponent": int(exponent), "mantissa": int(mantissa),
            "exponent_bits": format(int(exponent), '05b'),
            "mantissa_bits": format(int(mantissa), '010b'),
            "value": float(fp16),
            "actual_exponent": int(exponent) - 15}


def float_to_bf16_bits(value):
    fp32_bits = np.float32(value).view(np.uint32)
    bf16_bits = (fp32_bits >> 16).astype(np.uint16)
    sign = (bf16_bits >> 15) & 1
    exponent = (bf16_bits >> 7) & 0xFF
    mantissa = bf16_bits & 0x7F
    reconstructed = np.uint32(bf16_bits.astype(np.uint32) << 16).view(np.float32)
    return {"sign": int(sign), "exponent": int(exponent), "mantissa": int(mantissa),
            "exponent_bits": format(int(exponent), '08b'),
            "mantissa_bits": format(int(mantissa), '07b'),
            "value": float(reconstructed),
            "actual_exponent": int(exponent) - 127}


def simulate_fp8_e4m3(value):
    sign = 1 if value < 0 else 0
    abs_val = abs(value)
    max_val = 448.0
    abs_val = min(abs_val, max_val)
    if abs_val == 0:
        return {"sign": sign, "exponent": 0, "mantissa": 0, "value": 0.0,
                "exponent_bits": "0000", "mantissa_bits": "000"}
    exp = int(np.floor(np.log2(abs_val)))
    exp = max(-6, min(8, exp))
    mantissa_val = abs_val / (2.0 ** exp) - 1.0
    mantissa_quant = round(mantissa_val * 8) / 8
    mantissa_quant = max(0, min(0.875, mantissa_quant))
    reconstructed = (1.0 + mantissa_quant) * (2.0 ** exp)
    if sign:
        reconstructed = -reconstructed
    mantissa_int = int(round(mantissa_quant * 8))
    return {"sign": sign, "exponent": exp + 7, "mantissa": mantissa_int,
            "exponent_bits": format(exp + 7, '04b'),
            "mantissa_bits": format(mantissa_int, '03b'),
            "value": float(reconstructed),
            "actual_exponent": exp}


def display_format_comparison(value):
    fp32 = float_to_fp32_bits(value)
    fp16 = float_to_fp16_bits(value)
    bf16 = float_to_bf16_bits(value)
    fp8 = simulate_fp8_e4m3(value)

    print(f"\n  Value: {value}")
    print(f"  {'Format':<8} {'Stored Value':>14} {'Error':>12} {'Sign':>5} {'Exp Bits':>10} {'Man Bits':>25}")
    print(f"  {'-'*76}")
    print(f"  {'FP32':<8} {fp32['value']:>14.6f} {abs(fp32['value'] - value):>12.8f} {fp32['sign']:>5} {fp32['exponent_bits']:>10} {fp32['mantissa_bits']:>25}")
    print(f"  {'FP16':<8} {fp16['value']:>14.6f} {abs(fp16['value'] - value):>12.8f} {fp16['sign']:>5} {fp16['exponent_bits']:>10} {fp16['mantissa_bits']:>25}")
    print(f"  {'BF16':<8} {bf16['value']:>14.6f} {abs(bf16['value'] - value):>12.8f} {bf16['sign']:>5} {bf16['exponent_bits']:>10} {bf16['mantissa_bits']:>25}")
    print(f"  {'FP8e4m3':<8} {fp8['value']:>14.6f} {abs(fp8['value'] - value):>12.8f} {fp8['sign']:>5} {fp8['exponent_bits']:>10} {fp8['mantissa_bits']:>25}")
```

### 第 2 步：对称量化（按张量与按通道）

最基础的量化操作。按张量对整个矩阵使用一个缩放因子；按通道则对每一行或每一列使用一个缩放因子。

```python
def quantize_symmetric(tensor, num_bits=8):
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1
    abs_max = np.max(np.abs(tensor))
    if abs_max == 0:
        return np.zeros_like(tensor, dtype=np.int32), 1.0
    scale = abs_max / qmax
    quantized = np.clip(np.round(tensor / scale), qmin, qmax).astype(np.int32)
    return quantized, float(scale)


def dequantize_symmetric(quantized, scale):
    return quantized.astype(np.float64) * scale


def quantize_per_channel(tensor, num_bits=8, axis=0):
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1

    if axis == 0:
        abs_max = np.max(np.abs(tensor), axis=1, keepdims=True)
    else:
        abs_max = np.max(np.abs(tensor), axis=0, keepdims=True)

    abs_max = np.where(abs_max == 0, 1.0, abs_max)
    scales = abs_max / qmax
    quantized = np.clip(np.round(tensor / scales), qmin, qmax).astype(np.int32)
    return quantized, scales.squeeze()


def dequantize_per_channel(quantized, scales, axis=0):
    if axis == 0:
        return quantized.astype(np.float64) * scales.reshape(-1, 1)
    else:
        return quantized.astype(np.float64) * scales.reshape(1, -1)


def quantize_asymmetric(tensor, num_bits=8):
    qmin = 0
    qmax = 2 ** num_bits - 1
    t_min = np.min(tensor)
    t_max = np.max(tensor)
    if t_max == t_min:
        return np.zeros_like(tensor, dtype=np.int32), 1.0, 0
    scale = (t_max - t_min) / (qmax - qmin)
    zero_point = int(np.round(qmin - t_min / scale))
    zero_point = max(qmin, min(qmax, zero_point))
    quantized = np.clip(np.round(tensor / scale + zero_point), qmin, qmax).astype(np.int32)
    return quantized, float(scale), int(zero_point)


def dequantize_asymmetric(quantized, scale, zero_point):
    return (quantized.astype(np.float64) - zero_point) * scale
```

### 第 3 步：质量度量

衡量量化到底破坏了多少信息。这里会计算原始张量与重建张量之间的均方误差、信噪比和余弦相似度。

```python
def quantization_error(original, reconstructed):
    diff = original - reconstructed
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    max_error = float(np.max(np.abs(diff)))
    signal_power = float(np.mean(original ** 2))
    snr_db = 10 * np.log10(signal_power / max(mse, 1e-20))

    orig_flat = original.flatten()
    recon_flat = reconstructed.flatten()
    norm_orig = np.linalg.norm(orig_flat)
    norm_recon = np.linalg.norm(recon_flat)
    if norm_orig == 0 or norm_recon == 0:
        cosine_sim = 0.0
    else:
        cosine_sim = float(np.dot(orig_flat, recon_flat) / (norm_orig * norm_recon))

    return {"mse": mse, "rmse": rmse, "max_error": max_error,
            "snr_db": float(snr_db), "cosine_similarity": cosine_sim}


def compare_quantization_methods(tensor, num_bits=8):
    q_pt, s_pt = quantize_symmetric(tensor, num_bits)
    recon_pt = dequantize_symmetric(q_pt, s_pt)
    err_pt = quantization_error(tensor, recon_pt)

    q_pc, s_pc = quantize_per_channel(tensor, num_bits, axis=0)
    recon_pc = dequantize_per_channel(q_pc, s_pc, axis=0)
    err_pc = quantization_error(tensor, recon_pc)

    q_asym, s_asym, zp = quantize_asymmetric(tensor, num_bits)
    recon_asym = dequantize_asymmetric(q_asym, s_asym, zp)
    err_asym = quantization_error(tensor, recon_asym)

    print(f"\n  Quantization Comparison ({num_bits}-bit, tensor shape {tensor.shape}):")
    print(f"  {'Method':<20} {'MSE':>12} {'SNR (dB)':>10} {'Cosine Sim':>12} {'Max Error':>12}")
    print(f"  {'-'*68}")
    print(f"  {'Per-tensor sym':<20} {err_pt['mse']:>12.8f} {err_pt['snr_db']:>10.2f} {err_pt['cosine_similarity']:>12.8f} {err_pt['max_error']:>12.8f}")
    print(f"  {'Per-channel sym':<20} {err_pc['mse']:>12.8f} {err_pc['snr_db']:>10.2f} {err_pc['cosine_similarity']:>12.8f} {err_pc['max_error']:>12.8f}")
    print(f"  {'Asymmetric':<20} {err_asym['mse']:>12.8f} {err_asym['snr_db']:>10.2f} {err_asym['cosine_similarity']:>12.8f} {err_asym['max_error']:>12.8f}")

    return {"per_tensor": err_pt, "per_channel": err_pc, "asymmetric": err_asym}
```

### 第 4 步：位宽扫描

以不同位宽（2、3、4、8、16）量化同一个张量，并在每个级别测量质量。这能让你非常直观地看到质量悬崖从哪里开始出现。

```python
def bit_width_sweep(tensor):
    print(f"\n  Bit-Width Sweep (tensor shape {tensor.shape}):")
    print(f"  {'Bits':>6} {'Levels':>8} {'MSE':>14} {'SNR (dB)':>10} {'Cosine Sim':>12} {'Compression':>12}")
    print(f"  {'-'*64}")

    results = []
    for bits in [2, 3, 4, 8, 16]:
        q, s = quantize_per_channel(tensor, bits, axis=0)
        recon = dequantize_per_channel(q, s, axis=0)
        err = quantization_error(tensor, recon)
        levels = 2 ** bits
        compression = 32.0 / bits

        print(f"  {bits:>6} {levels:>8} {err['mse']:>14.8f} {err['snr_db']:>10.2f} {err['cosine_similarity']:>12.8f} {compression:>11.1f}x")
        results.append({"bits": bits, "levels": levels, "error": err, "compression": compression})

    return results
```

### 第 5 步：敏感度实验

模拟对 Transformer 不同部分进行量化，并测量哪些组件最敏感。这能直观展示敏感度层级：权重 &lt; 激活值 &lt; KV 缓存 &lt; 注意力。

```python
def simulate_transformer_layer(input_data, weights, kv_scale=1.0):
    hidden = input_data @ weights["qkv"]
    seq_len = hidden.shape[1]
    d_model = weights["qkv"].shape[1] // 3
    q, k, v = hidden[:, :, :d_model], hidden[:, :, d_model:2*d_model], hidden[:, :, 2*d_model:]

    attn_scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(d_model) * kv_scale
    attn_max = np.max(attn_scores, axis=-1, keepdims=True)
    attn_exp = np.exp(attn_scores - attn_max)
    attn_weights = attn_exp / np.sum(attn_exp, axis=-1, keepdims=True)

    attn_output = attn_weights @ v
    output = attn_output @ weights["out"]
    return output, {"q": q, "k": k, "v": v, "attn_scores": attn_scores,
                    "attn_weights": attn_weights, "attn_output": attn_output}


def sensitivity_experiment(batch_size=2, seq_len=16, d_model=64, num_bits=8):
    np.random.seed(42)
    input_data = np.random.randn(batch_size, seq_len, d_model) * 0.1

    weights = {
        "qkv": np.random.randn(d_model, 3 * d_model) * (2.0 / d_model) ** 0.5,
        "out": np.random.randn(d_model, d_model) * (2.0 / d_model) ** 0.5,
    }

    baseline_output, baseline_internals = simulate_transformer_layer(input_data, weights)

    experiments = {}

    q_qkv, s_qkv = quantize_per_channel(weights["qkv"], num_bits, axis=0)
    q_out, s_out = quantize_per_channel(weights["out"], num_bits, axis=0)
    quantized_weights = {
        "qkv": dequantize_per_channel(q_qkv, s_qkv, axis=0),
        "out": dequantize_per_channel(q_out, s_out, axis=0),
    }
    weight_quant_output, _ = simulate_transformer_layer(input_data, quantized_weights)
    experiments["Weights only"] = quantization_error(baseline_output, weight_quant_output)

    _, fresh_internals = simulate_transformer_layer(input_data, weights)
    q_act, s_act = quantize_per_channel(
        fresh_internals["attn_output"].reshape(-1, d_model), num_bits, axis=0
    )
    quant_attn_out = dequantize_per_channel(q_act, s_act, axis=0).reshape(batch_size, seq_len, d_model)
    act_quant_output = quant_attn_out @ weights["out"]
    experiments["Activations only"] = quantization_error(baseline_output, act_quant_output)

    q_k, s_k = quantize_per_channel(fresh_internals["k"].reshape(-1, d_model), num_bits, axis=0)
    q_v, s_v = quantize_per_channel(fresh_internals["v"].reshape(-1, d_model), num_bits, axis=0)
    quant_k = dequantize_per_channel(q_k, s_k, axis=0).reshape(batch_size, seq_len, d_model)
    quant_v = dequantize_per_channel(q_v, s_v, axis=0).reshape(batch_size, seq_len, d_model)
    attn_scores_kv = (fresh_internals["q"] @ quant_k.transpose(0, 2, 1)) / np.sqrt(d_model)
    attn_max_kv = np.max(attn_scores_kv, axis=-1, keepdims=True)
    attn_exp_kv = np.exp(attn_scores_kv - attn_max_kv)
    attn_weights_kv = attn_exp_kv / np.sum(attn_exp_kv, axis=-1, keepdims=True)
    kv_quant_output = (attn_weights_kv @ quant_v) @ weights["out"]
    experiments["KV cache only"] = quantization_error(baseline_output, kv_quant_output)

    noise_scale = np.std(fresh_internals["attn_scores"]) * 0.05
    noisy_scores = fresh_internals["attn_scores"] + np.random.randn(*fresh_internals["attn_scores"].shape) * noise_scale
    noisy_max = np.max(noisy_scores, axis=-1, keepdims=True)
    noisy_exp = np.exp(noisy_scores - noisy_max)
    noisy_weights = noisy_exp / np.sum(noisy_exp, axis=-1, keepdims=True)
    attn_quant_output = (noisy_weights @ fresh_internals["v"]) @ weights["out"]
    experiments["Attention logits (5% noise)"] = quantization_error(baseline_output, attn_quant_output)

    print(f"\n  Sensitivity Experiment ({num_bits}-bit quantization):")
    print(f"  {'Component':<30} {'MSE':>14} {'SNR (dB)':>10} {'Cosine Sim':>12}")
    print(f"  {'-'*68}")
    for name, err in sorted(experiments.items(), key=lambda x: x[1]["mse"]):
        print(f"  {name:<30} {err['mse']:>14.8f} {err['snr_db']:>10.2f} {err['cosine_similarity']:>12.8f}")

    return experiments
```

### 第 6 步：模拟 GPTQ

GPTQ 会一次量化一列，并用 Hessian 来决定如何分配舍入误差。这里给出一个简化版本，用来捕捉其核心思想：用校准数据衡量权重的重要性，然后对不那么重要的权重更激进地量化。

```python
def simulated_gptq(weight_matrix, calibration_inputs, num_bits=4):
    n_in, n_out = weight_matrix.shape
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1

    H = np.zeros((n_in, n_in))
    for x in calibration_inputs:
        x = x.reshape(-1, 1) if x.ndim == 1 else x
        for row in range(x.shape[0]):
            xi = x[row].reshape(-1, 1)
            H += xi @ xi.T
    H /= len(calibration_inputs)
    H += np.eye(n_in) * 1e-4

    weight_importance = np.diag(H)

    quantized = np.zeros_like(weight_matrix, dtype=np.int32)
    scales = np.zeros(n_out)
    errors = np.zeros(n_out)

    W = weight_matrix.copy()

    for col in range(n_out):
        w_col = W[:, col]
        abs_max = np.max(np.abs(w_col))
        if abs_max == 0:
            scales[col] = 1.0
            continue
        scale = abs_max / qmax
        scales[col] = scale

        q_col = np.clip(np.round(w_col / scale), qmin, qmax).astype(np.int32)
        quantized[:, col] = q_col

        quant_error = w_col - q_col * scale
        errors[col] = np.sqrt(np.mean(quant_error ** 2))

        if col < n_out - 1:
            importance_weights = weight_importance / (np.max(weight_importance) + 1e-10)
            for next_col in range(col + 1, min(col + 4, n_out)):
                compensation = quant_error * importance_weights * 0.1
                W[:, next_col] += compensation

    return quantized, scales, {"column_errors": errors,
                               "mean_error": float(np.mean(errors)),
                               "max_error": float(np.max(errors))}


def dequantize_gptq(quantized, scales):
    result = np.zeros_like(quantized, dtype=np.float64)
    for col in range(quantized.shape[1]):
        result[:, col] = quantized[:, col] * scales[col]
    return result
```

### 第 7 步：AWQ 模拟

AWQ 会识别显著权重（也就是会与大激活值相乘的那些权重），并在量化前通过缩放来保护它们。

```python
def simulated_awq(weight_matrix, calibration_inputs, num_bits=4, salient_fraction=0.01):
    n_in, n_out = weight_matrix.shape
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1

    activation_magnitudes = np.zeros(n_in)
    for x in calibration_inputs:
        if x.ndim == 1:
            activation_magnitudes += np.abs(x)
        else:
            activation_magnitudes += np.mean(np.abs(x), axis=0)
    activation_magnitudes /= len(calibration_inputs)

    n_salient = max(1, int(n_in * salient_fraction))
    salient_indices = np.argsort(activation_magnitudes)[-n_salient:]

    scale_factors = np.ones(n_in)
    for idx in salient_indices:
        col_max = np.max(np.abs(weight_matrix[idx, :]))
        if col_max > 0:
            scale_factors[idx] = min(4.0, 1.0 / (col_max + 1e-8) * np.mean(np.abs(weight_matrix)))

    scaled_weights = weight_matrix * scale_factors.reshape(-1, 1)

    quantized, scales = quantize_per_channel(scaled_weights, num_bits, axis=0)
    dequantized = dequantize_per_channel(quantized, scales, axis=0)

    result = dequantized / scale_factors.reshape(-1, 1)

    err = quantization_error(weight_matrix, result)

    return result, {"salient_indices": salient_indices,
                    "scale_factors": scale_factors[salient_indices],
                    "error": err,
                    "n_salient": n_salient}
```

### 第 8 步：完整流水线

把所有部分接起来。在同一个权重矩阵上比较朴素量化、按通道量化、GPTQ 和 AWQ。

```python
def full_quantization_comparison(d_in=256, d_out=512, num_bits=4, n_calibration=32):
    np.random.seed(42)

    weight = np.random.randn(d_in, d_out) * 0.02
    outlier_rows = np.random.choice(d_in, size=5, replace=False)
    weight[outlier_rows] *= 10

    calibration = [np.random.randn(8, d_in) * 0.1 for _ in range(n_calibration)]

    q_naive, s_naive = quantize_symmetric(weight, num_bits)
    recon_naive = dequantize_symmetric(q_naive, s_naive)
    err_naive = quantization_error(weight, recon_naive)

    q_pc, s_pc = quantize_per_channel(weight, num_bits, axis=0)
    recon_pc = dequantize_per_channel(q_pc, s_pc, axis=0)
    err_pc = quantization_error(weight, recon_pc)

    q_gptq, s_gptq, gptq_info = simulated_gptq(weight, calibration, num_bits)
    recon_gptq = dequantize_gptq(q_gptq, s_gptq)
    err_gptq = quantization_error(weight, recon_gptq)

    recon_awq, awq_info = simulated_awq(weight, calibration, num_bits)
    err_awq = awq_info["error"]

    print(f"\n  Full Quantization Comparison ({num_bits}-bit, {d_in}x{d_out} matrix)")
    print(f"  Matrix has {len(outlier_rows)} outlier rows (10x scale)")
    print()
    print(f"  {'Method':<20} {'MSE':>14} {'SNR (dB)':>10} {'Cosine Sim':>12}")
    print(f"  {'-'*58}")
    print(f"  {'Naive per-tensor':<20} {err_naive['mse']:>14.8f} {err_naive['snr_db']:>10.2f} {err_naive['cosine_similarity']:>12.8f}")
    print(f"  {'Per-channel':<20} {err_pc['mse']:>14.8f} {err_pc['snr_db']:>10.2f} {err_pc['cosine_similarity']:>12.8f}")
    print(f"  {'Simulated GPTQ':<20} {err_gptq['mse']:>14.8f} {err_gptq['snr_db']:>10.2f} {err_gptq['cosine_similarity']:>12.8f}")
    print(f"  {'Simulated AWQ':<20} {err_awq['mse']:>14.8f} {err_awq['snr_db']:>10.2f} {err_awq['cosine_similarity']:>12.8f}")

    test_input = np.random.randn(4, d_in) * 0.1
    baseline = test_input @ weight
    output_naive = test_input @ recon_naive
    output_pc = test_input @ recon_pc
    output_gptq = test_input @ recon_gptq
    output_awq = test_input @ recon_awq

    print(f"\n  End-to-End Output Error (matmul with test input):")
    print(f"  {'Method':<20} {'Output MSE':>14} {'Output Cosine':>14}")
    print(f"  {'-'*50}")
    for name, output in [("Naive", output_naive), ("Per-channel", output_pc),
                          ("GPTQ", output_gptq), ("AWQ", output_awq)]:
        out_err = quantization_error(baseline, output)
        print(f"  {name:<20} {out_err['mse']:>14.8f} {out_err['cosine_similarity']:>14.8f}")

    return {"naive": err_naive, "per_channel": err_pc, "gptq": err_gptq, "awq": err_awq}


def memory_calculator(num_params_billions, bits_per_param):
    bytes_per_param = bits_per_param / 8
    total_bytes = num_params_billions * 1e9 * bytes_per_param
    total_gb = total_bytes / (1024 ** 3)
    return total_gb


def print_memory_table():
    print("\n  Memory Requirements by Model and Precision:")
    print(f"  {'Model':<15} {'FP32':>8} {'FP16':>8} {'FP8':>8} {'INT8':>8} {'INT4':>8} {'INT2':>8}")
    print(f"  {'-'*64}")
    for name, params in [("7B", 7), ("13B", 13), ("34B", 34), ("70B", 70), ("405B", 405)]:
        fp32 = memory_calculator(params, 32)
        fp16 = memory_calculator(params, 16)
        fp8 = memory_calculator(params, 8)
        int8 = memory_calculator(params, 8)
        int4 = memory_calculator(params, 4)
        int2 = memory_calculator(params, 2)
        print(f"  {name:<15} {fp32:>7.1f}G {fp16:>7.1f}G {fp8:>7.1f}G {int8:>7.1f}G {int4:>7.1f}G {int2:>7.1f}G")


if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 70)
    print("QUANTIZATION: MAKING MODELS FIT")
    print("=" * 70)

    print("\nSTEP 1: Number Format Comparison")
    print("-" * 50)
    for val in [0.1, 3.14159, -0.00073, 42.5, 0.0000012]:
        display_format_comparison(val)

    print("\n\nSTEP 2: Memory Requirements")
    print("-" * 50)
    print_memory_table()

    print("\n\nSTEP 3: Quantization Methods Comparison")
    print("-" * 50)
    weight_matrix = np.random.randn(128, 256) * 0.02
    weight_matrix[0] *= 15
    weight_matrix[42] *= 8
    compare_quantization_methods(weight_matrix, num_bits=8)
    compare_quantization_methods(weight_matrix, num_bits=4)

    print("\n\nSTEP 4: Bit-Width Sweep")
    print("-" * 50)
    sweep_tensor = np.random.randn(64, 128) * 0.05
    bit_width_sweep(sweep_tensor)

    print("\n\nSTEP 5: Sensitivity Experiment")
    print("-" * 50)
    print("\n  INT8:")
    sensitivity_experiment(num_bits=8)
    print("\n  INT4:")
    sensitivity_experiment(num_bits=4)

    print("\n\nSTEP 6: GPTQ vs AWQ vs Naive (INT4)")
    print("-" * 50)
    full_quantization_comparison(d_in=256, d_out=512, num_bits=4)

    print("\n\nSTEP 7: Distribution Analysis")
    print("-" * 50)
    np.random.seed(0)
    simulated_weights = np.random.randn(1000) * 0.02
    abs_vals = np.abs(simulated_weights)
    pct_in_range = np.mean(abs_vals < 0.1) * 100
    print(f"\n  Simulated weight distribution (1000 params, std=0.02):")
    print(f"  Weights in [-0.1, 0.1]: {pct_in_range:.1f}%")
    print(f"  Weights in [-0.05, 0.05]: {np.mean(abs_vals < 0.05) * 100:.1f}%")
    print(f"  Weights in [-0.01, 0.01]: {np.mean(abs_vals < 0.01) * 100:.1f}%")
    print(f"  Max absolute value: {np.max(abs_vals):.6f}")
    print(f"  Mean absolute value: {np.mean(abs_vals):.6f}")

    histogram = np.histogram(simulated_weights, bins=20)
    print(f"\n  Weight histogram:")
    max_count = max(histogram[0])
    for i in range(len(histogram[0])):
        bar_len = int(histogram[0][i] / max_count * 40)
        lo = histogram[1][i]
        hi = histogram[1][i + 1]
        print(f"  [{lo:>7.4f}, {hi:>7.4f}] {'#' * bar_len} ({histogram[0][i]})")

    print("\n\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
```

## 开始使用

### 用 AutoGPTQ 进行量化

```python
# pip install auto-gptq transformers
# from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
# from transformers import AutoTokenizer
#
# model_id = "meta-llama/Llama-3.1-8B"
# quantize_config = BaseQuantizeConfig(
#     bits=4,
#     group_size=128,
#     desc_act=False,
# )
#
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoGPTQForCausalLM.from_pretrained(model_id, quantize_config)
#
# calibration = [tokenizer(t, return_tensors="pt") for t in calibration_texts[:128]]
# model.quantize(calibration)
# model.save_quantized("llama-8b-gptq-int4")
```

### 用 AutoAWQ 进行量化

```python
# pip install autoawq
# from awq import AutoAWQForCausalLM
# from transformers import AutoTokenizer
#
# model_id = "meta-llama/Llama-3.1-8B"
# model = AutoAWQForCausalLM.from_pretrained(model_id)
# tokenizer = AutoTokenizer.from_pretrained(model_id)
#
# model.quantize(tokenizer, quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4})
# model.save_quantized("llama-8b-awq-int4")
```

### 转换为 GGUF

```bash
# pip install llama-cpp-python
# python convert_hf_to_gguf.py meta-llama/Llama-3.1-8B --outtype q4_k_m --outfile llama-8b-q4km.gguf
# llama-server -m llama-8b-q4km.gguf -c 4096 -ngl 99
```

### 用 vLLM 提供服务

```python
# pip install vllm
# vllm serve model-awq --quantization awq --dtype half --max-model-len 8192
```

vLLM 原生支持 AWQ 和 GPTQ 模型。它会在矩阵乘法期间处理反量化（dequantization），并为 KV 缓存使用分页注意力（paged attention）。对于 H100 上的 FP8，请加上 `--dtype float8_e4m3fn`。

## 交付

本课会产出 `outputs/skill-quantization.md`，这是一份帮助你选择合适量化策略的决策框架。给定模型大小、目标硬件和质量要求，它会告诉你应该使用哪种格式、哪种方法，以及需要做哪些验证步骤。它包含内存预算计算、按组件给出的精度建议，以及面向 vLLM、llama.cpp 和 TensorRT-LLM 的部署配方。

## 练习

1. 实现分组量化（group quantization）。不要对每个通道只用一个缩放因子，而是在通道内部每 128 个权重使用一个缩放因子。这才是 GPTQ 和 AWQ 实际采用的做法。在同一个权重矩阵上比较 32、64、128 和 256 的组大小（group size）。组越小，质量越好，但存储缩放因子的开销也越大。

2. 构建一个混合精度量化器。把一个多层网络的第一层和最后一层量化为 INT8，同时把中间层量化为 INT4。把端到端输出质量与统一 INT4、统一 INT8 进行比较。并测量相对于全 INT8 的内存节省。

3. 为量化感知训练实现直通估计器（STE）。在一个简单的两层网络前向传播中插入伪量化/反量化操作，并在一个回归任务上训练。比较两种做法的最终损失：一种是正常训练后再 PTQ 到 INT4，另一种是从一开始就使用 QAT 训练。

4. 构建一个受 LLM.int8() 启发的离群值感知量化器。检测激活值幅度超过均值 6 倍的通道。把这些通道保留为 FP16，把其他所有部分量化到 INT8。在第 5 步的 Transformer 层上，用不同的离群阈值（3x、6x、10x）测量端到端质量。

5. 实现一个量化质量仪表盘。给定一个权重矩阵，计算并展示：权重分布直方图、量化误差分布、按通道缩放因子、量化最差的通道（重建误差最高），以及在 100 个随机输入上原始输出与量化输出之间的余弦相似度。识别哪些通道应该保留更高精度。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| FP16 | “半精度” | 16 位浮点数，5 位指数、10 位尾数，最大值 65,504，标准推理格式 |
| BF16 | “Brain float” | 16 位浮点数，8 位指数（与 FP32 范围相同）和 7 位尾数，由 Google 为训练设计 |
| FP8 | “8 位浮点” | 有两种变体：E4M3（推理，精度更高）和 E5M2（训练，范围更大），H100 原生支持 |
| INT8 | “8 位整数” | 从 -128 到 127 的 256 个均匀间隔值，需要缩放因子把浮点数映射进来 |
| INT4 | “4 位整数” | 总共只有 16 个级别，需要 GPTQ、AWQ 之类的复杂方法才能维持质量 |
| 按通道量化 | “每行一个缩放” | 为每个输出通道使用独立缩放因子，而不是整个张量共用一个，能显著降低误差 |
| GPTQ | “海森矩阵方法” | 一种训练后量化方法，利用二阶信息最小化输出误差，一次处理一层 |
| AWQ | “感知激活值” | 在量化前缩放显著权重（即与大激活值相乘的权重）以保护它们 |
| GGUF | “llama.cpp 格式” | 自包含模型文件，支持混合精度层，针对 CPU 和 Apple Silicon 推理优化 |
| PTQ | “训练后再量化” | 把已训练模型的权重转换为更低精度而不重新训练，速度快，但在极端压缩下受限 |
| QAT | “训练时量化” | 在前向传播中插入伪量化，让模型学会容忍舍入，在 INT4/INT2 下效果更好 |
| 校准数据 | “那 128 个样本” | 一小批送入模型的数据，用来计算激活统计量并设置缩放因子 |
| 缩放因子 | “那个乘数” | 在浮点范围和整数范围之间做转换：`float_val = int_val * scale` |
| 困惑度差值 | “差了多少” | 原始模型与量化模型之间的困惑度差异，&lt; 0.5 很优秀，> 2.0 就有问题 |

## 延伸阅读

- [Frantar et al., 2022 -- "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"](https://arxiv.org/abs/2210.17323) -- 这篇论文用 Hessian 引导的权重舍入，让 LLM 的 INT4 量化真正变得可行
- [Lin et al., 2023 -- "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"](https://arxiv.org/abs/2306.00978) -- 通过在量化前缩放来保护显著权重，效果可与 GPTQ 持平甚至更好
- [Dettmers et al., 2022 -- "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale"](https://arxiv.org/abs/2208.07339) -- 混合精度 INT8，把离群特征保留在 FP16 中，从而在几乎不损失质量的情况下实现 INT8 推理
- [Xiao et al., 2023 -- "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models"](https://arxiv.org/abs/2211.10438) -- 把量化难度从激活值迁移到权重上，以支持 W8A8 部署
- [Micikevicius et al., 2022 -- "FP8 Formats for Deep Learning"](https://arxiv.org/abs/2209.05433) -- NVIDIA/ARM/Intel 关于 E4M3 和 E5M2 格式的论文，这两种格式现已成为 H100 原生支持的 FP8 格式
