# 文本摘要（Text Summarization）

> 抽取式系统（extractive systems）告诉你文档说了什么。生成式系统（abstractive systems）告诉你作者想表达什么。任务不同，坑也不同。

**类型：** 构建
**语言：** Python
**前置条件：** 第 5 阶段 · 02（BoW + TF-IDF），第 5 阶段 · 11（机器翻译）
**时间：** ~75 分钟

## 问题

一篇 2,000 词的新闻文章出现在你的信息流里。你需要 120 个词来概括它。你可以从文章中挑出最重要的三句话（抽取式摘要，extractive summarization），也可以用自己的话重写内容（生成式摘要，abstractive summarization）。这两者都叫摘要，但其实是完全不同的问题。

抽取式摘要是一个排序问题。给每个句子打分，返回 top-`k`。因为输出是原文逐字摘取的，所以语法一定没问题。风险在于，文章里分散分布的信息可能被漏掉。

生成式摘要是一个生成问题。Transformer 根据输入生成新的文本。输出流畅、压缩率高，但也可能幻觉出源文中根本没有的事实。它的风险，是自信地胡编。

本课把这两种都搭起来，同时点出它们各自最典型的失败模式。

## 概念

*抽取式 TextRank vs 生成式 Transformer*

**抽取式（Extractive）。** 把文章看作一张图：节点是句子，边是句子之间的相似度。在这张图上运行 PageRank（或类似算法），按句子与其他句子的连接程度来打分。分数最高的句子组成摘要。经典实现是 **TextRank**（Mihalcea 和 Tarau，2004）。

**生成式（Abstractive）。** 在文档-摘要对上微调一个 Transformer 编码器-解码器（BART、T5、Pegasus）。推理时，模型读取文档，并通过交叉注意力（cross-attention）逐 token 生成摘要。尤其是 Pegasus，它使用缺失句预训练目标（gap-sentence pretraining objective），因此即使不做太多微调也非常擅长摘要。

用 **ROUGE**（Recall-Oriented Understudy for Gisting Evaluation）做评估。ROUGE-1 和 ROUGE-2 分别衡量 unigram 与 bigram 重叠。ROUGE-L 衡量最长公共子序列。分数越高越好，但 40 的 ROUGE-L 已经算“不错”，50 算“非常出色”。论文通常三个都会报。使用 `rouge-score` 包。

## 动手构建

### 第 1 步：TextRank（抽取式）

```python
import math
import re
from collections import Counter


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]
```

这里有两点值得点名。相似度函数使用的是经过对数归一化的词重叠，这就是最初的 TextRank 变体。改用 TF-IDF 向量的余弦相似度也可以。阻尼系数 0.85 和迭代次数则沿用了 PageRank 的默认设置。

### 第 2 步：用 BART 做生成式摘要

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(long news article text)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

BART-large-CNN 在 CNN/DailyMail 语料上做过微调。它开箱即用就能生成新闻风格摘要。对于其他领域（科学论文、对话、法律），使用对应的 Pegasus checkpoint，或在目标数据上微调。

### 第 3 步：ROUGE 评估

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

一定要启用 stemming。不然 “running” 和 “run” 会被当成不同词，ROUGE 就会少算匹配。

### 超越 ROUGE（2026 年摘要评估）

ROUGE 已经主导摘要评估二十年了，但到 2026 年，单靠它远远不够。一项针对 NLG 论文的大规模元分析显示：

- **BERTScore**（上下文化嵌入相似度）在 2023 年前后快速普及，现在大多数摘要论文都会和 ROUGE 一起报告。
- **BARTScore** 把评估视为生成任务：看一个预训练 BART 在给定源文时，认为摘要有多大概率出现。
- **MoverScore**（基于上下文化嵌入上的 Earth Mover's Distance）在 2025 年摘要基准中登顶，因为它比 ROUGE 更能捕捉语义重叠。
- **FactCC** 和 **基于 QA 的忠实度（QA-based faithfulness）** 在 2021-2023 年很常见，如今常被 **G-Eval** 取代：它通过一条 GPT-4 提示链，用链式思维推理给连贯性、一致性、流畅度和相关性打分。
- **G-Eval** 以及类似的 LLM 评委方法，在 rubric 设计合理时，与人工判断的一致率约为 80%。

生产建议：用 ROUGE-L 做历史对比，用 BERTScore 看语义重叠，用 G-Eval 看连贯性和事实性。先拿 50-100 个人工标注摘要做校准。

### 第 4 步：事实性问题

生成式摘要很容易出现幻觉。抽取式摘要的幻觉风险低得多，因为输出是从源文逐字摘取的；当然，如果原句脱离上下文、已经过时，或被打乱引用顺序，它仍可能误导。这也是为什么在接近合规场景的生产系统里，人们仍偏爱抽取式方法。

需要点名的幻觉类型：

- **实体替换（Entity swap）。** 源文写的是 “John Smith.”，摘要写成 “John Brown.”
- **数字漂移（Number drift）。** 源文写的是 “25,000.”，摘要写成 “25 million.”
- **极性翻转（Polarity flip）。** 源文写的是 “rejected the offer.”，摘要却写成 “accepted the offer.”
- **事实捏造（Fact invention）。** 源文没有提到 CEO，摘要却说 CEO 批准了。

有效的评估方法：

- **FactCC。** 一个在源句与摘要句之间做蕴含判断的二分类器。预测是否符合事实。
- **基于 QA 的事实性检查（QA-based factuality）。** 让 QA 模型回答那些答案明确存在于源文中的问题。如果摘要支持的是不同答案，就标红。
- **实体级 F1（Entity-level F1）。** 比较源文和摘要中的命名实体。只出现在摘要中的实体都值得怀疑。

对于任何面向用户、且事实性重要的场景（新闻、医疗、法律、金融），抽取式都是更安全的默认选项。生成式摘要必须加事实性检查。

## 使用它

2026 年的组合：

| 用例 | 推荐 |
|---------|-------------|
| 新闻，3-5 句摘要，英文 | `facebook/bart-large-cnn` |
| 科学论文 | `google/pegasus-pubmed` 或调优后的 T5 |
| 多文档、长篇摘要 | 任意支持 32k+ 上下文并经提示设计的 LLM |
| 对话摘要 | `philschmid/bart-large-cnn-samsum` |
| 低幻觉风险、结构上更安全的抽取式 | TextRank 或 `sumy` 的 LSA / LexRank |

在计算不是约束时，长上下文 LLM 到 2026 年往往会超过专用摘要模型。代价是成本与可复现性；专用模型的输出通常更稳定。

## 交付它

保存为 `outputs/skill-summary-picker.md`：

```markdown
---
name: summary-picker
description: Pick extractive or abstractive, named library, factuality check.
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

Given a task (document type, compliance requirement, length, compute budget), output:

1. Approach. Extractive or abstractive. Explain in one sentence why.
2. Starting model / library. Name it. `sumy.TextRankSummarizer`, `facebook/bart-large-cnn`, `google/pegasus-pubmed`, or an LLM prompt.
3. Evaluation plan. ROUGE-1, ROUGE-2, ROUGE-L (use rouge-score with stemming). Plus factuality check if abstractive.
4. One failure mode to probe. Entity swap is the most common in abstractive news summarization; flag samples where source entities do not appear in summary.

Refuse abstractive summarization for medical, legal, financial, or regulated content without a factuality gate. Flag input over the model's context window as needing chunked map-reduce summarization (not just truncation).
```

## 练习

1. **简单。** 在 5 篇新闻文章上运行 TextRank。把 top-3 句和参考摘要比较，测量 ROUGE-L。对于 CNN/DailyMail 风格文章，你应该会看到 30-45 的 ROUGE-L。
2. **中等。** 实现实体级事实性检查：从源文和摘要中提取命名实体（spaCy），计算摘要中覆盖源文实体的召回率，以及摘要实体相对于源文的精确率。高精确率、低召回率意味着安全但简略；低精确率则意味着有幻觉实体。
3. **困难。** 在 50 篇 CNN/DailyMail 文章上比较 BART-large-CNN 和一个 LLM（Claude 或 GPT-4）。报告 ROUGE-L、事实性（用实体 F1 评估），以及每条摘要的成本。记录各自胜出的场景。

## 关键术语

| 术语 | 大家常说 | 实际含义 |
|------|----------|----------|
| 抽取式（Extractive） | 挑句子 | 直接返回源文中的句子。永远不会幻觉。 |
| 生成式（Abstractive） | 改写 | 基于源文生成新文本。可能会幻觉。 |
| ROUGE | 摘要指标 | 系统输出与参考摘要之间的 n-gram / LCS 重叠。 |
| TextRank | 基于图的抽取式 | 在句子相似度图上跑 PageRank。 |
| 事实性（Factuality） | 对不对 | 摘要中的陈述是否得到源文支持。 |
| 幻觉（Hallucination） | 编出来的内容 | 摘要里出现了源文不支持的内容。 |

## 延伸阅读

- [Mihalcea and Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) —— 抽取式摘要的经典论文。
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) —— BART 论文。
- [Zhang et al. (2019). PEGASUS: Pre-training with Extracted Gap-sentences](https://arxiv.org/abs/1912.08777) —— Pegasus 与 gap-sentence 目标。
- [Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) —— ROUGE 论文。
- [Maynez et al. (2020). On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/abs/2005.00661) —— 关于忠实度与事实性的综述论文。
