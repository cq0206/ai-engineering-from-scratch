# 多语言自然语言处理（Multilingual NLP）

> 一个模型，100+ 种语言，其中大多数几乎没有训练数据。跨语言迁移（cross-lingual transfer）是 2020 年代最实用的奇迹之一。

**类型：** 学习
**语言：** Python
**前置条件：** 第 5 阶段 · 04（GloVe、FastText、Subword），第 5 阶段 · 11（机器翻译）
**时间：** ~45 分钟

## 问题

英语有数十亿条带标注样本。乌尔都语只有几千条。迈蒂利语几乎没有。任何面向全球用户的实用 NLP 系统，都必须在那些根本不存在任务专用训练数据的长尾语言上工作。

多语言模型通过在多种语言上同时训练一个模型来解决这个问题。共享表示让模型能把高资源语言中学到的技能迁移到低资源语言上。把模型在英文情感分析上微调后，它往往就能直接对乌尔都语给出出人意料地不错的情感预测。这就是零样本跨语言迁移（zero-shot cross-lingual transfer），它已经彻底改变了 NLP 向世界交付的方式。

本课会讲清楚其中的权衡、经典模型，以及一个最容易绊倒多语言新手团队的决策：为迁移选择哪种源语言。

## 概念

*通过共享的多语言嵌入空间进行跨语言迁移*

**共享词表（Shared vocabulary）。** 多语言模型使用在所有目标语言文本上训练出来的 SentencePiece 或 WordPiece 分词器。词表是共享的：相关语言里表示相同语素的子词单元会对应到同一个 token。英语和意大利语里的 `anti-` 会用同一个 token。

**共享表示（Shared representation）。** 在多种语言上做掩码语言建模预训练的 Transformer，会学到：不同语言中语义相近的句子会产生相近的隐藏状态。mBERT、XLM-R 和 NLLB 都体现了这一点。英语里的 “cat” 与法语里的 “chat”、西班牙语里的 “gato” 的嵌入会聚在一起，整句嵌入也一样。

**零样本迁移（Zero-shot transfer）。** 在一种语言（通常是英语）上用带标注数据微调模型。推理时，把它直接跑到模型支持的任意其他语言上。不需要目标语言标签。对于类型学相近的语言，这个效果很强；对于差异较大的语言，会弱不少。

**少样本微调（Few-shot fine-tuning）。** 在目标语言里额外加入 100-500 条带标注样本。对分类任务来说，准确率通常能跳到英语基线的 95-98%。这是多语言 NLP 里性价比最高的杠杆。

## 模型

| 模型 | 年份 | 覆盖范围 | 备注 |
|-------|------|----------|-------|
| mBERT | 2018 | 104 种语言 | 在 Wikipedia 上训练。第一个实用的多语言 LM。对低资源语言较弱。 |
| XLM-R | 2019 | 100 种语言 | 在 CommonCrawl 上训练（规模远大于 Wikipedia）。设定了跨语言基线。Base 270M，Large 550M。 |
| XLM-V | 2023 | 100 种语言 | XLM-R 的 1M-token 词表版本（原来是 250k）。对低资源语言更好。 |
| mT5 | 2020 | 101 种语言 | 面向多语言生成的 T5 架构。 |
| NLLB-200 | 2022 | 200 种语言 | Meta 的翻译模型；包含 55 种低资源语言。 |
| BLOOM | 2022 | 46 种语言 + 13 种编程语言 | 开放的 176B 多语言 LLM。 |
| Aya-23 | 2024 | 23 种语言 | Cohere 的多语言 LLM。对阿拉伯语、印地语、斯瓦希里语很强。 |

按用例来选。分类任务里，XLM-R-base 是最稳妥的默认。生成任务则根据“翻译”还是“开放生成”选择 mT5 或 NLLB。类 LLM 工作则更适合 Aya-23，或使用显式多语言提示的 Claude。

## 源语言决策（2026 年研究）

大多数团队默认把英语作为微调源语言。最新研究（2026）表明，这个默认经常是错的。

语言相似性比原始语料规模更能预测迁移质量。对斯拉夫语目标来说，德语或俄语往往比英语更好。对印度语系目标来说，印地语常常比英语更好。**qWALS** 相似度指标（2026，基于 World Atlas of Language Structures 特征）对此做了量化。**LANGRANK**（Lin 等，ACL 2019）则是更早的一种方法，会综合语言相似性、语料规模与谱系亲缘性，对候选源语言进行排序。

实用规则：如果目标语言有一个类型学上接近且资源丰富的“亲戚”，先试着在那个语言上微调，再与英语微调做比较。

## 动手构建

### 第 1 步：零样本跨语言分类

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")


def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))
```

一个模型，三种语言，同一个 API。XLM-R 在 NLI 数据上训练后，通过 entailment 这套技巧，能很好地迁移到分类任务。

### 第 2 步：多语言嵌入空间

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
```

翻译对在嵌入空间里会靠得很近。一个不同的英文句子则会更远。这就是跨语言检索、聚类和相似度能成立的基础。

### 第 3 步：少样本微调策略

```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)

    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
```

对于 100-500 条目标语言样本，`num_train_epochs=5` 和 `learning_rate=2e-5` 是安全默认。更高的学习率会让多语言对齐崩掉，最后你得到的就成了一个只会英语的模型。

## 真正有效的评估

- **按语言分别看留出集准确率。** 不要只看汇总。总平均会把长尾问题遮住。
- **对比单语基线（monolingual baseline）。** 对于有足够数据的语言，从零训练的单语模型有时会超过多语言模型。一定要测。
- **实体级测试。** 重点看目标语言里的命名实体。对远离拉丁字母的书写系统，多语言模型的分词往往比较弱。
- **跨语言一致性。** 两种语言表达同一含义时，模型应该给出同样的预测。测量这个差距。

## 使用它

2026 年的组合：

| 任务 | 推荐 |
|-----|-------------|
| 分类，100 种语言 | 微调后的 XLM-R-base（约 270M） |
| 零样本文本分类 | `joeddav/xlm-roberta-large-xnli` |
| 多语言句向量 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| 翻译，200 种语言 | `facebook/nllb-200-distilled-600M`（见第 11 课） |
| 多语言生成 | Claude、GPT-4、Aya-23、mT5-XXL |
| 低资源语言 NLP | XLM-V，或在相关高资源语言上做领域微调 |

如果性能真的重要，就必须为目标语言的微调预留预算。零样本只是起点，不是最终答案。

### 分词税（低资源语言为什么会出问题）

多语言模型在所有语言之间共享一个分词器。这个词表是用以英语、法语、西班牙语、中文、德语为主的大语料训练出来的。对任何不在主导集合里的语言来说，有三种“税”会悄悄叠加：

- **词元膨胀税（Fertility tax）。** 低资源语言文本在分词后，每个词对应的 token 数会远多于英语。一个印地语句子可能需要同义英语句子的 3-5 倍 token。这个 3-5 倍会直接吞掉你的上下文窗口、训练效率和推理延迟预算。
- **变体恢复税（Variant recovery tax）。** 每个拼写错误、变音符变体、Unicode 归一化不一致，或大小写变化，都会在嵌入空间中变成一段冷启动、互不相干的序列。模型学不到母语者看来理所当然的正字法对应关系。
- **容量溢出税（Capacity spillover tax）。** 前两种税会消耗上下文位置、层深和嵌入维度。留给实际推理的容量，会系统性地小于高资源语言在同一模型上得到的容量。

实际症状是：模型在印地语上训练看起来一切正常，loss 曲线没问题，eval perplexity 也还不错，但生产输出会出现微妙错误。词形变化会在句中崩掉。稀有屈折形式始终学不会。**你无法靠堆更多数据来摆脱一个坏掉的 tokenizer。**

缓解办法：选择对目标语言覆盖更好的 tokenizer（XLM-V 的 1M-token 词表就是直接修复）；在训练前先在留出的目标语言文本上验证分词膨胀率；对于真正长尾的书写系统，使用字节级回退（SentencePiece `byte_fallback=True`、GPT-2 风格 byte-level BPE），这样任何内容都不会 OOV。

## 交付它

保存为 `outputs/skill-multilingual-picker.md`：

```markdown
---
name: multilingual-picker
description: Pick source language, target model, and evaluation plan for a multilingual NLP task.
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

Given requirements (target languages, task type, available labeled data per language), output:

1. Source language for fine-tuning. Default English; check LANGRANK or qWALS if target language has a typologically close high-resource language.
2. Base model. XLM-R (classification), mT5 (generation), NLLB (translation), Aya-23 (generative LLM).
3. Few-shot budget. Start with 100-500 target-language examples if available. Zero-shot only if labeling is infeasible.
4. Evaluation plan. Per-language accuracy (not aggregate), cross-lingual consistency, entity-level F1 on non-Latin scripts.

Refuse to ship a multilingual model without per-language evaluation — aggregate metrics hide long-tail failures. Flag scripts with low tokenization coverage (Amharic, Tigrinya, many African languages) as needing a model with byte-fallback (SentencePiece with byte_fallback=True, or byte-level tokenizer like GPT-2).
```

## 练习

1. **简单。** 在英语、法语、印地语和阿拉伯语四种语言中，每种各跑 10 句话的零样本分类流水线。分别报告准确率。你应该会看到法语很强，印地语不错，阿拉伯语波动较大。
2. **中等。** 用 `paraphrase-multilingual-MiniLM-L12-v2` 在一个小型混合语言语料上构建跨语言检索器。用英语提问，检索任意语言文档。测量 recall@5。
3. **困难。** 针对一个印地语分类任务，比较“英语作为源语言微调”和“印地语作为源语言微调”。在两种方案下都使用 500 条目标语言样本做少样本微调。报告哪一种得到更高的印地语准确率，以及高出多少。这就是 LANGRANK 论文观点的迷你版复现。

## 关键术语

| 术语 | 大家常说 | 实际含义 |
|------|----------|----------|
| 多语言模型（Multilingual model） | 一个模型，多种语言 | 在多种语言之间共享词表和参数。 |
| 跨语言迁移（Cross-lingual transfer） | 在一种语言上训练，在另一种语言上运行 | 在源语言上微调，在没有目标语言标签的情况下评估目标语言。 |
| 零样本（Zero-shot） | 没有目标语言标签 | 不在目标语言上微调，直接做迁移。 |
| 少样本（Few-shot） | 少量目标标签 | 使用 100-500 条目标语言样本进行微调。 |
| mBERT | 第一个多语言 LM | 在 Wikipedia 上预训练的 104 语言 BERT。 |
| XLM-R | 标准跨语言基线 | 在 CommonCrawl 上预训练的 100 语言 RoBERTa。 |
| NLLB | Meta 的 200 语种 MT | No Language Left Behind。包含 55 种低资源语言。 |

## 延伸阅读

- [Conneau et al. (2019). Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) —— XLM-R 论文。
- [Pires, Schlinger, Garrette (2019). How Multilingual is Multilingual BERT?](https://arxiv.org/abs/1906.01502) —— 开启跨语言迁移研究路线的分析论文。
- [Costa-jussà et al. (2022). No Language Left Behind](https://arxiv.org/abs/2207.04672) —— NLLB-200 论文。
- [Üstün et al. (2024). Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model](https://arxiv.org/abs/2402.07827) —— Aya，Cohere 的多语言 LLM。
- [Language Similarity Predicts Cross-Lingual Transfer Learning Performance (2026)](https://www.mdpi.com/2504-4990/8/3/65) —— qWALS / LANGRANK 源语言选择论文。
