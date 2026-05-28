# 序列到序列模型（Sequence-to-Sequence Models）

> 两个 RNN 假装自己是翻译器。它们撞上的瓶颈，正是注意力（attention）存在的原因。

**类型：** 构建
**语言：** Python
**前置条件：** 第 5 阶段 · 08（用于文本的 CNN + RNN），第 3 阶段 · 11（PyTorch 入门）
**时间：** ~75 分钟

## 问题

分类把一个变长序列映射为单个标签。翻译把一个变长序列映射为另一个变长序列。输入和输出位于不同词表中，可能还是不同语言，并且长度没有任何保证会一致。

序列到序列（seq2seq）架构（Sutskever、Vinyals、Le，2014）用一个刻意保持简单的配方解决了这个问题：两个 RNN。一个读取源句子并产出一个定长上下文向量（context vector）。另一个读取这个向量，并逐个 token 生成目标句子。和你在第 08 课写的代码是同一套东西，只是拼接方式不同。

这值得学习有两个原因。第一，上下文向量瓶颈是 NLP 中最有教学价值的失败案例。它解释了注意力（attention）和 Transformer 擅长解决什么问题。第二，这套训练配方——教师强制（teacher forcing）、计划采样（scheduled sampling）、以及推理时的束搜索（beam search）——直到今天仍适用于所有现代生成系统，包括 LLM。

## 概念

**编码器（Encoder）。** 一个读取源句子的 RNN。它最后的隐藏状态就是**上下文向量（context vector）**——对整个输入的定长摘要。理论上什么都不能丢，除了源句本身。

**解码器（Decoder）。** 另一个从上下文向量初始化的 RNN。在每一步，它把先前生成的 token 作为输入，并在目标词表上产生一个分布。通过采样或 argmax 选出下一个 token。再把它喂回去。重复，直到生成 `&lt;EOS>` token 或达到最大长度。

**训练：** 在解码器的每一步计算交叉熵损失（cross-entropy loss），再对整个序列求和。对两个网络一起执行标准的时间反向传播（backprop through time）。

**教师强制（Teacher forcing）。** 在训练时，解码器在步骤 `t` 的输入是位置 `t-1` 的*真实* token，而不是解码器自己上一步的预测。这会稳定训练；没有它，早期错误会不断连锁放大，模型就学不会。在推理时，你必须使用模型自己的预测，因此训练分布和推理分布之间始终存在差距。这个差距叫作**暴露偏差（exposure bias）**。

**瓶颈（bottleneck）。** 编码器从源句中学到的一切，都必须被压缩进那一个上下文向量里。长句会丢细节。罕见词会被模糊。重排序（如 *chat noir* vs. *black cat*）必须靠记忆，而不是动态计算。

注意力（attention，第 10 课）通过让解码器查看*每一个*编码器隐藏状态，而不仅仅是最后一个，修复了这个问题。这就是它全部的核心卖点。

## 动手构建

### 第 1 步：一个编码器

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs` 的形状是 `[batch, seq_len, hidden_dim]` —— 每个输入位置一个隐藏状态。`hidden` 的形状是 `[1, batch, hidden_dim]` —— 最后一步的状态。第 08 课里我们说“为分类对 outputs 做池化”。在这里，我们保留最后的隐藏状态作为上下文向量，并忽略逐步输出。

### 第 2 步：一个解码器

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

解码器一次只调用一步。输入：一批单个 token，以及当前隐藏状态。输出：下一个 token 的词表 logits，以及更新后的隐藏状态。

### 第 3 步：带教师强制的训练循环

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

这里有两个值得点名的旋钮。`ignore_index=0` 会跳过 padding token 的损失。`teacher_forcing_ratio` 是每一步使用真实 token 而非模型预测的概率。从 1.0（完全教师强制）开始，并在训练过程中逐步退火到 ~0.5，以缩小暴露偏差。

### 第 4 步：推理循环（贪心）

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

贪心解码（greedy decoding）在每一步都选概率最高的 token。它可能会走偏：一旦你选了一个 token，就无法反悔。**束搜索（Beam search）**会同时保留 top-`k` 条部分序列，并在最后选出得分最高的完整序列。标准束宽一般是 3-5。

### 第 5 步：演示瓶颈

在一个玩具复制任务上训练模型：源序列 `[a, b, c, d, e]`，目标序列 `[a, b, c, d, e]`。逐步增加序列长度，观察准确率。

```
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
```

单个 GRU 隐藏状态无法无损记住一个 40-token 的输入。信息其实分布在编码器的每一步里，但解码器只能看到最后一个状态。注意力直接解决了这一点。

## 使用它

PyTorch 提供了 `nn.Transformer` 和基于 `nn.LSTM` 的 seq2seq 模板。Hugging Face 的 `transformers` 库则直接提供了完整的编码器-解码器（encoder-decoder）模型（BART、T5、mBART、NLLB），训练数据规模达到数十亿 token。

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

现代编码器-解码器已经用 Transformer 取代了 RNN。高层结构（编码器、解码器、逐 token 生成）与 2014 年的 seq2seq 论文完全一致。变化的是每个模块内部的机制。

### 什么时候仍然会用基于 RNN 的 seq2seq

对于新项目，几乎不会。少数例外：

- 流式翻译（streaming translation）：输入按一个 token 一个 token 到来，且内存有上界。
- 设备端文本生成：Transformer 的内存成本高得无法接受。
- 教学。理解编码器-解码器瓶颈，是理解 Transformer 为什么获胜的最快路径。

### 暴露偏差及其缓解方法

- **计划采样（Scheduled sampling）。** 在训练时逐步降低教师强制比例，让模型学会从自己的错误中恢复。
- **最小风险训练（Minimum risk training）。** 按句级 BLEU 分数训练，而不是按 token 级交叉熵训练。更接近你真正想优化的目标。
- **强化学习微调（Reinforcement learning fine-tuning）。** 用某个指标作为奖励来训练序列生成器。现代 LLM 的 RLHF 就在这样做。

这三种方法对基于 Transformer 的生成仍然适用。

## 交付它

保存为 `outputs/prompt-seq2seq-design.md`：

```markdown
---
name: seq2seq-design
description: Design a sequence-to-sequence pipeline for a given task.
phase: 5
lesson: 09
---

Given a task (translation, summarization, paraphrase, question rewrite), output:

1. Architecture. Pretrained transformer encoder-decoder (BART, T5, mBART, NLLB) is the default. RNN-based seq2seq only for specific constraints.
2. Starting checkpoint. Name it (`facebook/bart-base`, `google/flan-t5-base`, `facebook/nllb-200-distilled-600M`). Match the checkpoint to task and language coverage.
3. Decoding strategy. Greedy for deterministic output, beam search (width 4-5) for quality, sampling with temperature for diversity. One sentence justification.
4. One failure mode to verify before shipping. Exposure bias manifests as generation drift on longer outputs; sample 20 outputs at the 90th-percentile length and eyeball.

Refuse to recommend training a seq2seq from scratch for under a million parallel examples. Flag any pipeline that uses greedy decoding for user-facing content as fragile (greedy repeats and loops).
```

## 练习

1. **简单。** 实现这个玩具复制任务。训练一个 GRU seq2seq，让输入输出对中的目标等于源序列。测量长度为 5、10、20 时的准确率。复现这个瓶颈。
2. **中等。** 加入束宽为 3 的束搜索解码。在一个小型平行语料（parallel corpus）上，对比贪心解码与束搜索的 BLEU。记录束搜索在哪些地方更好（通常是最后几个 token），以及在哪些地方没有区别。
3. **困难。** 在一个 1 万对样本的释义数据集（paraphrase dataset）上微调 `facebook/bart-base`。把微调后模型的 beam-4 输出，与基础模型在留出集输入上的输出做比较。报告 BLEU，并挑选 10 个定性样例。

## 关键术语

| 术语 | 大家常说 | 实际含义 |
|------|----------|----------|
| 编码器（Encoder） | 输入 RNN | 读取源序列。产生逐步隐藏状态和最终上下文向量。 |
| 解码器（Decoder） | 输出 RNN | 从上下文向量初始化。一次生成一个目标 token。 |
| 上下文向量（Context vector） | 那个摘要 | 编码器最终的隐藏状态。定长。也是注意力要解决的瓶颈。 |
| 教师强制（Teacher forcing） | 用真实 token | 训练时喂入真实的前一个 token。能稳定学习。 |
| 暴露偏差（Exposure bias） | 训练/测试差距 | 模型训练时只见过真实 token，因此从没练过如何从自己的错误中恢复。 |
| 束搜索（Beam search） | 更好的解码 | 每一步保留 top-k 条部分序列，而不是贪心地立刻提交一个选择。 |

## 延伸阅读

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) —— 原始 seq2seq 论文。只有四页。
- [Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) —— 提出了 GRU 和编码器-解码器范式。
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) —— 注意力论文。学完这课立刻读。
- [PyTorch NLP from Scratch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) —— 可直接运行的 seq2seq + attention 代码。
