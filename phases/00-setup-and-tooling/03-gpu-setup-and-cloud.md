# GPU 设置与云端

> 在 CPU 上训练很适合学习。真正要训练时，你需要 GPU。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段 0，第 01 课
**时间：** ~45 分钟

## 学习目标

- 使用 `nvidia-smi` 和 PyTorch 的 CUDA API 验证本地 GPU 是否可用
- 将 Google Colab 配置为 T4 GPU，用于免费的云端实验
- 在 CPU 与 GPU 上对矩阵乘法做基准测试，并测量加速比
- 使用 fp16 经验法则估算你的 VRAM 能容纳的最大模型

## 问题

阶段 1-3 的大多数课程在 CPU 上都可以顺利运行。但一旦你开始训练 CNN、transformers 或 LLMs（阶段 4+），你就需要 GPU 加速。一次在 CPU 上需要 8 小时的训练，在 GPU 上只要 10 分钟。

你有三个选择：本地 GPU、云 GPU，或者 Google Colab（免费）。

## 概念

```
Your options:

1. Local NVIDIA GPU
   Cost: $0 (you already have it)
   Setup: Install CUDA + cuDNN
   Best for: Regular use, large datasets

2. Google Colab (free tier)
   Cost: $0
   Setup: None
   Best for: Quick experiments, no GPU at home

3. Cloud GPU (Lambda, RunPod, Vast.ai)
   Cost: $0.20-2.00/hr
   Setup: SSH + install
   Best for: Serious training, large models
```

## 动手构建

### 选项 1：本地 NVIDIA GPU

检查你是否有可用的 GPU：

```bash
nvidia-smi
```

安装带 CUDA 的 PyTorch：

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### 选项 2：Google Colab

1. 打开 [colab.research.google.com](https://colab.research.google.com)
2. 依次点击 Runtime > Change runtime type > T4 GPU
3. 运行 `!nvidia-smi` 进行验证

把本课程的 notebooks 直接上传到 Colab。

### 选项 3：云 GPU

对于 Lambda Labs、RunPod 或 Vast.ai：

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### 没有 GPU？也没关系。

大多数课程在 CPU 上也能运行。需要 GPU 的课程会明确说明，并附上 Colab 链接。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## 动手构建：GPU vs CPU 基准测试

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
```

## 练习

1. 运行上面的基准测试，并比较 CPU 与 GPU 的时间
2. 如果你没有 GPU，就在 Google Colab 上运行它并进行比较
3. 检查你有多少 GPU 内存，并估算你能容纳的最大模型（经验法则：fp16 下每个参数占 2 字节）

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| CUDA | “GPU 编程” | NVIDIA 的并行计算平台，让你可以在 GPU 上运行代码 |
| VRAM | “GPU 内存” | GPU 上的视频内存，与系统 RAM 分离，会限制模型大小。 |
| fp16 | “半精度” | 16 位浮点数，在几乎不损失精度的情况下使用一半于 fp32 的内存 |
| Tensor Core | “高速矩阵硬件” | 用于矩阵乘法的专用 GPU 核心，比普通核心快 4-8 倍 |
