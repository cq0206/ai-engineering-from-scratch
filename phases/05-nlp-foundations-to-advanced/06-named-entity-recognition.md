# 命名实体识别

> 把名字抽出来。听起来很简单，直到你开始处理边界歧义、嵌套实体和领域黑话。

**类型：** 构建
**语言：** Python
**前置要求：** 第 5 阶段 · 02（BoW + TF-IDF），第 5 阶段 · 03（词嵌入）
**时间：** ~75 分钟

## 问题

“Apple sued Google over its iPhone search deal in the US.” 里面至少有五个实体：Apple（ORG）、Google（ORG）、iPhone（PRODUCT）、search deal（也许算）、US（GPE）。一个好的命名实体识别（Named Entity Recognition, NER）系统会把它们全部抽出来，并给出正确类型。一个差的系统会漏掉 iPhone、把水果 Apple 和公司 Apple 混淆，还会把 “US” 标成 PERSON。

NER 是一切结构化抽取流水线下面的主力。简历解析、合规日志扫描、医疗记录匿名化、搜索查询理解、聊天机器人回答的 grounding、法律合同抽取。你很少直接看到它，但你总是在依赖它。

本课会带你从经典路线（规则系统、隐马尔可夫模型（Hidden Markov Model, HMM）、条件随机场（Conditional Random Field, CRF））一路走到现代路线（BiLSTM-CRF，再到 transformer）。每一步都在解决前一步的某个明确局限。这种演进模式本身就是课程内容。

## 概念

**BIO 标注（BIO tagging）**（或 BILOU）把实体抽取转成一个序列标注问题。给每个词元打上 `B-TYPE`（实体开始）、`I-TYPE`（实体内部）或 `O`（不属于任何实体）。

```
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

多词实体会连起来标：`New B-GPE`、`York I-GPE`、`City I-GPE`。一个真正理解 BIO 的模型，就能抽出任意长度的跨度。

架构演进如下：

- **规则法。** 正则表达式 + 词典（gazetteer）查找。对已知实体精度很高，对新实体覆盖为零。
- **HMM。** 隐状态是标签，观测是词元。学习“给定标签时词出现的发射概率”，以及“标签到标签的转移概率”。再用 Viterbi 解码。依赖标注数据训练。
- **CRF。** 条件随机场。和 HMM 类似，但它是判别式模型，因此你可以混入任意特征（词形、大小写、相邻词）。即便到了 2026 年，在低资源部署里它依然是传统生产系统的主力。
- **BiLSTM-CRF。** 用神经特征替代手工特征。LSTM 双向读取句子，顶部的 CRF 层负责约束标签序列一致性。
- **基于 Transformer 的方法。** 在 BERT 上加一个 token-classification 头并微调。准确率最好，算力需求也最高。

## 动手构建

### 第 1 步：BIO 标注辅助函数

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

```python
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
```

### 第 2 步：手工特征

对于经典（非神经）NER，特征就是一切。常用特征包括：

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
```

`word_shape("iPhone")` 会返回 `xXxxxx`。`word_shape("USA-2024")` 会返回 `XXX-dddd`。大小写模式对专有名词来说是高信号特征。

### 第 3 步：一个简单的规则 + 词典基线

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

生产级 gazetteer 往往会从 Wikipedia 和 DBpedia 抓来数百万个条目，因此覆盖率不错。但消歧（`Apple` 是公司还是水果）很糟。这就是统计模型最终胜出的原因。

### 第 4 步：CRF 这一步（示意，不做完整实现）

如果没有概率论基础，从零用 50 行写出完整 CRF 并不会让人更明白。直接用 `sklearn-crfsuite`：

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
```

`c1` 和 `c2` 分别是 L1 与 L2 正则。`all_possible_transitions=True` 会让模型学会“非法序列（例如 `O` 后面直接接 `I-ORG`）不太可能”，这也是 CRF 无需你手写约束、却仍能保持 BIO 一致性的方式。

### 第 5 步：BiLSTM-CRF 额外带来了什么

特征不再手工编写，而是学习得到。输入是词嵌入（GloVe 或 fastText）。LSTM 从左到右和从右到左各读一遍句子。拼接后的隐藏状态送入一个 CRF 输出层。CRF 仍负责约束标签序列一致性；LSTM 则用学习到的特征替换手工特征。

```python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

CRF 层可以直接用 `torchcrf.CRF`（`pip install pytorch-crf`）。相对手工特征 CRF，它的提升是可测量的，但通常没有你想象中那么大——除非你拥有成千上万条标注句子。

## 使用它

spaCy 开箱就带有生产级 NER。

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

```
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
```

注意，`iPhone` 被标成了 `ORG` 而不是 `PRODUCT`——spaCy 的小模型在产品实体上的覆盖较弱。大模型（`en_core_web_lg`）会更好，transformer 模型（`en_core_web_trf`）还会更好。

使用 Hugging Face 做 BERT 风格的 NER：

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

```
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
```

`aggregation_strategy="simple"` 会把连续的 B-X、I-X token 合并成一个 span。如果不用它，你拿到的只是 token 级标签，还得自己再做一次合并。

### 基于 LLM 的 NER（2026 年的选项）

零样本和少样本的 LLM NER，如今在很多领域里已经能与微调模型竞争；而在标注数据稀缺时，它往往会显著更强。

- **零样本提示。** 给 LLM 一组实体类型和一个示例 schema，要求输出 JSON。开箱即可用；在新领域上的准确率中等。
- **ZeroTuneBio 风格提示。** 把任务拆成“候选抽取 → 含义解释 → 判断 → 复查”。这种多阶段提示（不是 one-shot）会显著提升生物医学 NER 的准确率。同样的模式也适用于法律、金融和科学领域。
- **带 RAG 的动态提示。** 对每次推理，都从一个小型已标注种子集中检索最相似的例子，并动态构建 few-shot 提示。2026 年基准显示，这能让 GPT-4 在生物医学 NER 上的 F1 比静态提示高 11–12%。
- **按实体类型拆分。** 对长文档来说，单次调用同时抽所有实体类型，随着长度增加会明显掉召回。改成每种实体类型各跑一遍抽取。推理成本更高，但准确率也明显更高。这是临床记录和法律合同里的标准模式。

截至 2026 年的生产建议：在你开始收集训练数据之前，先做一个 LLM 零样本基线。很多时候 F1 已经足够好，以至于你根本不需要微调。

### 经典 NER 仍然会赢的地方

即使 LLM 已经可用，下面这些场景里，经典 NER 仍然有优势：

- 延迟预算低于 50ms。
- 你已经有数千条标注样本，并且需要 98% 以上的 F1。
- 领域本体稳定，预训练好的 CRF 或 BiLSTM 迁移效果很好。
- 监管要求必须使用本地部署、非生成式模型。

### 它会在哪些地方崩掉

- **领域漂移。** 在 CoNLL 上训练的 NER，用到法律合同上时，效果可能还不如一个 gazetteer。要在你的领域上微调。
- **嵌套实体。** “Bank of America Tower” 同时可以被视为 ORG 和 FACILITY。标准 BIO 无法表示重叠跨度。你需要嵌套 NER（多轮或基于 span 的模型）。
- **长实体。** “United States Federal Deposit Insurance Corporation.” 这种实体，token 级模型有时会拆裂。要用 `aggregation_strategy` 或后处理。
- **稀疏类型。** 医疗 NER 中的 DRUG_BRAND、ADVERSE_EVENT、DOSE 等标签，通用模型往往完全不知道。ScispaCy 和 BioBERT 才是那里的起点。

## 交付它

保存为 `outputs/skill-ner-picker.md`：

```markdown
---
name: ner-picker
description: Pick the right NER approach for a given extraction task.
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

Given a task description (domain, label set, language, latency, data volume), output:

1. Approach. Rule-based + gazetteer, CRF, BiLSTM-CRF, or transformer fine-tune.
2. Starting model. Name it (spaCy model ID, Hugging Face checkpoint ID, or "custom, trained from scratch").
3. Labeling strategy. BIO, BILOU, or span-based. Justify in one sentence.
4. Evaluation. Use `seqeval`. Always report entity-level F1 (not token-level).

Refuse to recommend fine-tuning a transformer for under 500 labeled examples unless the user already has a pretrained domain model. Flag nested entities as needing span-based or multi-pass models. Require a gazetteer audit if the user mentions "production scale" and labels are unchanged from CoNLL-2003.
```

## 练习

1. **简单。** 实现 `bio_to_spans`（`spans_to_bio` 的逆操作），并在 10 个句子上验证往返一致性。
2. **中等。** 在 CoNLL-2003 英文 NER 数据集上训练上面的 sklearn-crfsuite CRF。使用 `seqeval` 报告实体级 F1。典型结果约为 ~84 F1。
3. **困难。** 在一个领域特定的 NER 数据集（医疗、法律或金融）上微调 `distilbert-base-cased`。与 spaCy 小模型对比。记录数据泄漏检查，并写下让你意外的地方。

## 关键术语

| 术语 | 人们常说什么 | 它真正表示什么 |
|------|--------------|----------------|
| NER | 抽名字 | 给词元跨度打上类型标签（PERSON、ORG、GPE、DATE 等）。 |
| BIO | 标注方案 | `B-X` 表示开始，`I-X` 表示继续，`O` 表示外部。 |
| BILOU | 更好的 BIO | 额外加入 `L-X`（最后一个）、`U-X`（单独成实体），边界更清晰。 |
| CRF | 结构化分类器 | 建模标签之间的转移，而不只是发射概率。能强制生成合法序列。 |
| 嵌套 NER | 重叠实体 | 某个 span 与它的子 span 可能是不同实体。BIO 无法表达。 |
| 实体级 F1 | 正确的 NER 指标 | 预测的 span 必须与真实 span 完全一致。token 级 F1 会高估准确率。 |

## 延伸阅读

- [Lample et al. (2016). Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) —— BiLSTM-CRF 论文，经典之作。
- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) —— 引入后来成为标准做法的 token-classification 模式。
- [spaCy linguistic features — named entities](https://spacy.io/usage/linguistic-features#named-entities) —— 关于 `Doc.ents` 与 `Span` 上各种属性的实用参考。
- [seqeval](https://github.com/chakki-works/seqeval) —— 正确的评估库。始终使用它。
