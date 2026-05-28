# 用于文本的 CNN 与 RNN

> 卷积会学习 n-gram，循环会记忆。两者都已被注意力机制超越；两者在受限硬件上依然重要。

**类型：** 构建
**语言：** Python
**前置要求：** 第 3 阶段 · 11（PyTorch 入门），第 5 阶段 · 03（词嵌入），第 4 阶段 · 02（从零实现卷积）
**时间：** ~75 分钟

## 问题

TF-IDF 和 Word2Vec 产生的都是扁平向量，忽略了词序。基于它们构建的分类器无法区分 `dog bites man` 和 `man bites dog`。而词序有时恰恰就是信号所在。

在 transformer 出现之前，有两大类架构填补了这个空白。

**用于文本的卷积神经网络（Convolutional Neural Networks, CNNs，常见形式为 TextCNN）。** 在词嵌入序列上施加一维卷积。宽度为 3 的卷积核，本质上就是一个可学习的 trigram 探测器：它跨越三个词并输出一个分数。把不同宽度（2、3、4、5）的卷积核堆起来，就能检测多尺度模式。再做最大池化，得到一个定长表示。结构平坦、可并行、速度快。

**循环神经网络（Recurrent Neural Networks, RNNs，包括 LSTM、GRU）。** 一次处理一个词元，并维护一个向前传递信息的隐藏状态。它是顺序式的、有记忆的，能处理可变长度输入。2014 到 2017 年间，序列建模几乎被它统治；随后注意力机制出现了。

本课会把这两类都搭出来，然后指出那个最终促使人们发明注意力的失败点。

## 概念

**TextCNN**（Kim, 2014）：先给词元做嵌入。然后用宽度为 `k` 的一维卷积，在连续 `k` 个嵌入上滑动，产出一个特征图。对这个特征图做全局最大池化，就能取出最强激活。把多个卷积宽度的最大池化结果拼起来，再送入分类头。

它为什么有效？因为每个卷积核都是一个可学习的 n-gram。最大池化具有位置不变性，所以无论 “not good” 出现在评论开头还是中间，都会激活同一个特征。假设你有三个卷积宽度、每个宽度 100 个卷积核，那就相当于拥有 300 个学出来的 n-gram 探测器。训练是并行的，不存在时间步之间的顺序依赖。

**RNN**：在每个时间步 `t`，隐藏状态满足 `h_t = f(W * x_t + U * h_{t-1} + b)`。`W`、`U` 和 `b` 在时间维上共享。时间 `T` 的隐藏状态，概括了整个前缀。做分类时，你可以对 `h_1 ... h_T` 做池化（最大、平均或取最后一个）。

普通 RNN 会遭遇梯度消失。长短期记忆网络（Long Short-Term Memory, LSTM）通过门控机制决定遗忘什么、存储什么、输出什么，从而稳定长序列上的梯度。门控循环单元（Gated Recurrent Unit, GRU）则把 LSTM 简化成两个门；参数更少，但效果相近。

**双向 RNN（Bidirectional RNNs）**会同时跑一个正向 RNN 和一个反向 RNN，再把隐藏状态拼接起来。这样，每个词元的表示都能同时看到左上下文和右上下文。对标注任务来说，这几乎是必需的。

## 动手构建

### 第 1 步：用 PyTorch 实现 TextCNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

`transpose(1, 2)` 会把 `[batch, seq_len, embed_dim]` 变成 `[batch, embed_dim, seq_len]`，因为 `nn.Conv1d` 把中间那个维度视为通道。池化后的输出大小固定，与输入长度无关。

### 第 2 步：LSTM 分类器

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

做分类时，池化整个序列，而不是只拿最后一个状态。因为在长序列中，末尾信息往往会主导最后隐藏状态，所以最大池化通常比“只取最后状态”更好。

### 第 3 步：梯度消失演示（直觉版）

没有门控的普通 RNN 无法学习长距离依赖。设想一个玩具任务：判断序列里是否曾出现过 token `A`。如果 `A` 出现在位置 1，而整个序列长达 100，那么损失的梯度必须穿过 99 次循环权重相乘才能回传到开头。如果这个权重小于 1，梯度就会消失；如果大于 1，梯度就会爆炸。

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# At weight=0.9 over 100 steps:
#   0.9 ^ 100 ≈ 2.7e-5
# The gradient from step 100 to step 1 is effectively zero.
```

LSTM 通过**细胞状态（cell state）**修复了这个问题：它在网络中沿着一条几乎只包含加法交互的通路前进（遗忘门虽然会做乘法缩放，但梯度依然能沿着这条“高速公路”传播）。GRU 用更少的参数实现了类似效果。两者都能让你在 100+ 步序列上稳定训练。

### 第 4 步：为什么这仍然不够

即使有了 LSTM，仍然还有三个问题。

1. **顺序瓶颈。** 在长度为 1000 的序列上训练 RNN，需要 1000 次串行前向/反向步骤。无法跨时间并行。
2. **编码器-解码器中的定长上下文向量。** 解码器只能看到编码器最后一个隐藏状态，也就是整个输入被压缩后的结果。长输入会丢细节。第 09 课会直接讲这个问题。
3. **远距离依赖的准确率上限。** LSTM 比普通 RNN 强得多，但在跨越 200+ 步传播某条具体信息时，依然会吃力。

注意力机制一次性解决了这三个问题。transformer 彻底抛弃了循环。第 10 课就是这个转折点。

## 使用它

PyTorch 的 `nn.LSTM`、`nn.GRU` 和 `nn.Conv1d` 都是生产可用组件。训练代码也很标准。

Hugging Face 提供了可直接接入输入层的预训练表示：

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

判断“什么时候它符合约束”的清单：

- **边缘端 / 端侧推理。** 带 GloVe 嵌入的 TextCNN 比 transformer 小 10–100 倍。如果你的部署目标是手机，这就是该用的栈。
- **流式 / 在线分类。** RNN 一次处理一个词元；transformer 需要完整序列。对实时流入的文本，LSTM 仍然会赢。
- **做小模型基线。** 新任务上快速迭代。你可以在 CPU 上 5 分钟内训练一个 TextCNN。
- **有限数据下的序列标注。** BiLSTM-CRF（第 06 课）在 1k–10k 条标注句子的范围内，依然是生产级 NER 架构。

除此之外，其他一切基本都应该交给 transformer。

## 交付它

保存为 `outputs/prompt-text-encoder-picker.md`：

```markdown
---
name: text-encoder-picker
description: Pick a text encoder architecture for a given constraint set.
phase: 5
lesson: 08
---

Given constraints (task, data volume, latency budget, deploy target, compute budget), output:

1. Encoder architecture: TextCNN, BiLSTM, BiLSTM-CRF, transformer fine-tune, or "use a pretrained transformer as a frozen encoder + small head".
2. Embedding input: random init, GloVe / fastText frozen, or contextualized transformer embeddings.
3. Training recipe in 5 lines: optimizer, learning rate, batch size, epochs, regularization.
4. One monitoring signal. For RNN/CNN models: attention mechanism absence means they miss long-range deps; check per-length accuracy. For transformers: fine-tuning collapse if LR too high; check train loss.

Refuse to recommend fine-tuning a transformer when data is under ~500 labeled examples without showing that a TextCNN / BiLSTM baseline has plateaued. Flag edge deployment as needing architecture-before-everything.
```

## 练习

1. **简单。** 在一个 3 类玩具数据集上训练 TextCNN（数据自己造即可）。验证卷积宽度（2、3、4）的组合，在平均 F1 上通常优于单一宽度（3）。
2. **中等。** 为 LSTM 分类器实现最大池化、平均池化和最后状态池化。比较它们在一个小数据集上的表现；记录哪个池化最好，并猜测原因。
3. **困难。** 构建一个 BiLSTM-CRF NER 标注器（把第 06 课和本课结合起来）。在 CoNLL-2003 上训练。分别与第 06 课中的纯 CRF 基线和 BERT 微调比较。报告训练时间、内存占用和 F1。

## 关键术语

| 术语 | 人们常说什么 | 它真正表示什么 |
|------|--------------|----------------|
| TextCNN | 用于文本的 CNN | 在词嵌入上堆叠一维卷积并做全局最大池化。Kim（2014）。 |
| RNN | 循环网络 | 每个时间步更新隐藏状态：`h_t = f(W x_t + U h_{t-1})`。 |
| LSTM | 带门控的 RNN | 增加输入门、遗忘门、输出门和细胞状态。可稳定训练长序列。 |
| GRU | 更简单的 LSTM | 两个门而不是三个。精度相近，参数更少。 |
| 双向 | 两个方向 | 正向 + 反向 RNN 拼接。每个词元都能看到上下文两侧。 |
| 梯度消失 | 训练信号死掉了 | 在普通 RNN 中，反复乘以小于 1 的权重，会让早期时间步的梯度几乎变成零。 |

## 延伸阅读

- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) —— TextCNN 论文。只有八页，很好读。
- [Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) —— LSTM 论文，意外地清晰。
- [Olah, C. (2015). Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) —— 那些让所有人都看懂 LSTM 的图。
