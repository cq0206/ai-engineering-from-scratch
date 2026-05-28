# 开发环境

> 你的工具会塑造你的思维。一次配置好，就要正确配置。

**类型：** 构建
**语言：** Python, Node.js, Rust
**前置条件：** 无
**时间：** ~45 分钟

## 学习目标

- 从零开始搭建 Python 3.11+、Node.js 20+ 和 Rust 工具链 (toolchain)
- 配置虚拟环境 (virtual environment) 和包管理器 (package manager)，实现可复现构建
- 使用 CUDA/MPS 验证 GPU 访问，并运行一个测试张量操作
- 理解四层栈 (four-layer stack)：系统、包、运行时、AI 库

## 问题

你将通过 200+ 节课程学习 AI 工程，涉及 Python、TypeScript、Rust 和 Julia。如果你的环境有问题，每一节课都会变成和工具链对抗，而不是学习。

大多数人会跳过环境配置。然后他们会花上数小时排查导入错误、版本冲突以及缺失的 CUDA 驱动。我们这次要一次性、正确地把它做好。

## 概念

AI 工程环境有四层：

```mermaid
graph TD
    A["4. AI/ML 库\nPyTorch, JAX, transformers, etc."] --> B["3. 语言运行时\nPython 3.11+, Node 20+, Rust, Julia"]
    B --> C["2. 包管理器\nuv, pnpm, cargo, juliaup"]
    C --> D["1. 系统基础\nOS, shell, git, editor, GPU 驱动"]
```

我们按自下而上的顺序安装。每一层都依赖它下面的那一层。

## 动手构建

### 第 1 步：系统基础

检查你的系统并安装基础工具。

```bash
# macOS
xcode-select --install
brew install git curl wget

# Ubuntu/Debian
sudo apt update && sudo apt install -y build-essential git curl wget

# Windows (use WSL2)
wsl --install -d Ubuntu-24.04
```

### 第 2 步：使用 uv 安装 Python

我们使用 `uv`——它比 pip 快 10-100 倍，并且会自动处理虚拟环境。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

uv python install 3.12

uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

uv pip install numpy matplotlib jupyter
```

验证：

```python
import sys
print(f"Python {sys.version}")

import numpy as np
print(f"NumPy {np.__version__}")
a = np.array([1, 2, 3])
print(f"Vector: {a}, dot product with itself: {np.dot(a, a)}")
```

### 第 3 步：使用 pnpm 安装 Node.js

用于 TypeScript 课程（智能体、MCP 服务器、Web 应用）。

```bash
curl -fsSL https://fnm.vercel.app/install | bash
fnm install 22
fnm use 22

npm install -g pnpm

node -e "console.log('Node', process.version)"
```

### 第 4 步：Rust

用于性能关键的课程（推理、系统）。

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

rustc --version
cargo --version
```

### 第 5 步：Julia（可选）

用于 Julia 擅长的数学密集型课程。

```bash
curl -fsSL https://install.julialang.org | sh

julia -e 'println("Julia ", VERSION)'
```

### 第 6 步：GPU 设置（如果你有）

```bash
# NVIDIA
nvidia-smi

# Install PyTorch with CUDA
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

没有 GPU？也没关系。大多数课程在 CPU 上也能运行。对于训练负载较重的课程，请使用 Google Colab 或云 GPU。

### 第 7 步：验证一切

运行验证脚本：

```bash
python phases/00-setup-and-tooling/01-dev-environment/code/verify.py
```

## 使用它

你的环境现在已经为本课程中的每一节课准备就绪。下面是各语言的使用场景：

| 语言 | 用于 | 包管理器 |
|----------|---------|-----------------|
| Python | 阶段 1-12（ML, DL, NLP, Vision, Audio, LLMs） | uv |
| TypeScript | 阶段 13-17（工具、智能体、群体、基础设施） | pnpm |
| Rust | 阶段 12、15-17（性能关键系统） | cargo |
| Julia | 阶段 1（数学基础） | Pkg |

## 交付

本课会产出一个任何人都可以运行的验证脚本，用来检查他们的环境配置。

请参见 `outputs/prompt-env-check.md`，其中包含一个可帮助 AI 助手诊断环境问题的提示词。

## 练习

1. 运行验证脚本并修复所有失败项
2. 为本课程创建一个 Python 虚拟环境并安装 PyTorch
3. 用这四种语言分别写一个 “hello world” 并运行它们
