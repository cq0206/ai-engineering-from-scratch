# 数据管理

> 数据是燃料。你如何管理它，决定了你前进得有多快。

**类型：** 构建
**语言：** Python
**前置要求：** 第 0 阶段，第 01 课
**时间：** ~45 分钟

## 学习目标

- 使用 Hugging Face `datasets` 库加载、流式处理和缓存数据集
- 在 CSV、JSON、Parquet 和 Arrow 格式之间转换，并解释它们的权衡
- 使用固定随机种子创建可复现的 train/validation/test 切分
- 使用 `.gitignore`、Git LFS 或 DVC 管理大型模型和数据集文件

## 问题

每个 AI 项目都从数据开始。你需要找到数据集、下载它们、在不同格式之间转换、为训练和评估切分数据，并对它们做版本管理，以便实验可复现。每次都手动完成这些事既慢又容易出错。你需要一套可重复的工作流。

## 概念

```mermaid
graph TD
    A["Hugging Face Hub"] --> B["datasets 库"]
    B --> C["加载 / 流式处理"]
    C --> D["本地缓存<br/>~/.cache/huggingface/"]
    B --> E["格式转换<br/>CSV, JSON, Parquet, Arrow"]
    E --> F["数据切分<br/>train / val / test"]
    F --> G["你的训练流水线"]
```

Hugging Face 的 `datasets` 库是 AI 工作中加载数据的标准方式。它开箱即用地处理下载、缓存(cache)、格式转换和流式处理(streaming)。

## 构建它

### 第 1 步：安装 datasets 库

```bash
pip install datasets huggingface_hub
```

### 第 2 步：加载一个数据集

```python
from datasets import load_dataset

dataset = load_dataset("imdb")
print(dataset)
print(dataset["train"][0])
```

这会下载 IMDB 电影评论数据集。第一次下载后，它会从 `~/.cache/huggingface/datasets/` 中的缓存加载。

### 第 3 步：流式处理大型数据集

有些数据集太大，无法放进磁盘。流式处理会逐行加载它们，而不需要下载完整数据集。

```python
dataset = load_dataset("wikimedia/wikipedia", "20220301.en", split="train", streaming=True)

for i, example in enumerate(dataset):
    print(example["title"])
    if i >= 4:
        break
```

流式处理会给你一个 `IterableDataset`。你在数据到达时逐行处理它。无论数据集有多大，内存占用都保持不变。

### 第 4 步：数据集格式

`datasets` 库底层使用 Apache Arrow。你可以根据流水线的需要把它转换成其他格式。

```python
dataset = load_dataset("imdb", split="train")

dataset.to_csv("imdb_train.csv")
dataset.to_json("imdb_train.json")
dataset.to_parquet("imdb_train.parquet")
```

格式对比：

| 格式 | 大小 | 读取速度 | 最适合 |
|--------|------|-----------|----------|
| CSV | 大 | 慢 | 人类可读性、电子表格 |
| JSON | 大 | 慢 | API、嵌套数据 |
| Parquet | 小 | 快 | 分析、列式查询 |
| Arrow | 小 | 最快 | 内存内处理（`datasets` 内部使用的格式） |

对于 AI 工作，Parquet 是最好的存储格式。Arrow 是你在内存中操作的格式。CSV 和 JSON 更适合交换数据。

### 第 5 步：数据切分

每个 ML 项目都需要三种切分：

- **Train**：模型从这里学习（通常 80%）
- **Validation**：训练过程中用来检查进度（通常 10%）
- **Test**：训练完成后的最终评估（通常 10%）

有些数据集已经预先切分好了。如果没有，就自己切：

```python
dataset = load_dataset("imdb", split="train")

split = dataset.train_test_split(test_size=0.2, seed=42)
train_val = split["train"].train_test_split(test_size=0.125, seed=42)

train_ds = train_val["train"]
val_ds = train_val["test"]
test_ds = split["test"]

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
```

一定要设置随机种子以保证可复现性。同一个种子每次都会产生相同的切分。

### 第 6 步：下载并缓存模型

模型是大文件。`huggingface_hub` 库负责下载和缓存它们。

```python
from huggingface_hub import hf_hub_download, snapshot_download

model_path = hf_hub_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    filename="config.json"
)
print(f"Cached at: {model_path}")

model_dir = snapshot_download("sentence-transformers/all-MiniLM-L6-v2")
print(f"Full model at: {model_dir}")
```

模型会缓存到 `~/.cache/huggingface/hub/`。一旦下载完成，后续运行就能立即加载。

### 第 7 步：处理大文件

模型权重和大型数据集不应该进入 git。有三种选择：

**选项 A：.gitignore（最简单）**

```
*.bin
*.safetensors
*.pt
*.onnx
data/*.parquet
data/*.csv
models/
```

**选项 B：Git LFS（在 git 中跟踪大文件）**

```bash
git lfs install
git lfs track "*.bin"
git lfs track "*.safetensors"
git add .gitattributes
```

Git LFS 会在仓库里存放指针，把实际文件放在独立服务器上。GitHub 提供 1 GB 免费额度。

**选项 C：DVC（数据版本控制）**

```bash
pip install dvc
dvc init
dvc add data/training_set.parquet
git add data/training_set.parquet.dvc data/.gitignore
git commit -m "Track training data with DVC"
```

DVC 会创建很小的 `.dvc` 文件来指向你的数据。数据本体则存放在 S3、GCS 或其他远程存储后端中。

| 方案 | 复杂度 | 最适合 |
|----------|-----------|----------|
| .gitignore | 低 | 个人项目、可重新拉取的下载数据 |
| Git LFS | 中 | 通过 git 共享模型权重的团队 |
| DVC | 高 | 可复现实验、大型数据集、团队协作 |

对这门课来说，`.gitignore` 就足够了。当你需要在不同机器之间复现精确实验时，再使用 DVC。

### 第 8 步：存储模式

**本地存储** 适用于 ~10 GB 以下的数据集。HF 缓存会自动处理这一点。

**云存储** 适用于更大的数据，或需要在多台机器之间共享的场景：

```python
import os

local_path = os.path.expanduser("~/.cache/huggingface/datasets/")

# s3_path = "s3://my-bucket/datasets/"
# gcs_path = "gs://my-bucket/datasets/"
```

DVC 可以直接与 S3 和 GCS 集成：

```bash
dvc remote add -d myremote s3://my-bucket/dvc-store
dvc push
```

对这门课来说，本地存储已经足够。当你在远程 GPU 实例上做微调时，云存储才会变得重要。

## 本课程使用的数据集

| 数据集 | 课程 | 大小 | 它教会你什么 |
|---------|---------|------|----------------|
| IMDB | 分词、分类 | 84 MB | 文本分类基础 |
| WikiText | 语言建模 | 181 MB | 下一个 token 预测 |
| SQuAD | 问答系统 | 35 MB | 问答、跨度抽取 |
| Common Crawl (subset) | 嵌入 | 不定 | 大规模文本处理 |
| MNIST | 视觉基础 | 21 MB | 图像分类基础 |
| COCO (subset) | 多模态 | 不定 | 图文配对 |

你现在不需要把这些全部下载下来。每节课都会说明自己需要什么。

## 使用它

运行这个工具脚本，验证一切是否正常：

```bash
python code/data_utils.py
```

它会下载一个小数据集、完成格式转换、切分数据，并打印摘要。

## 交付它

本课会产出：
- `code/data_utils.py` - 可复用的数据加载与缓存工具
- `outputs/prompt-data-helper.md` - 用于为任务找到合适数据集的 prompt

## 练习

1. 用 `mrpc` 配置加载 `glue` 数据集，并检查前 5 个样本
2. 流式处理 `c4` 数据集，并统计你在 10 秒内能处理多少个样本
3. 把一个数据集转换为 Parquet，并将文件大小与 CSV 做比较
4. 使用固定种子创建 70/15/15 的 train/val/test 切分，并验证各自大小

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| 数据集切分(Dataset split) | “训练数据” | ML 生命周期中在不同阶段使用的命名子集（train/val/test） |
| 流式处理(Streaming) | “惰性加载” | 不下载完整数据集，而是从远程源逐行处理数据 |
| Parquet | “压缩版 CSV” | 一种针对分析查询和存储效率优化的列式文件格式 |
| Arrow | “快速 dataframe” | `datasets` 库内部使用的内存列式格式，支持零拷贝读取 |
| Git LFS | “大文件版 Git” | 一种扩展，把大文件存放在 git 仓库之外，同时在版本控制中保留指针 |
| DVC | “数据版 Git” | 一个用于数据集和模型的版本控制系统，并可与云存储集成 |
| 缓存(Cache) | “已经下载过了” | 之前获取过的数据的本地副本，默认存放在 ~/.cache/huggingface/ |
