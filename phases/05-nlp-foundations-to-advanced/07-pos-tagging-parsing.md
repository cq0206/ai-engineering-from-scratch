# 词性标注与句法解析

> 语法学曾经一度不时髦。后来每条 LLM 流水线都需要验证结构化抽取，它又回来了。

**类型：** 构建
**语言：** Python
**前置要求：** 第 5 阶段 · 01（文本处理），第 2 阶段 · 14（朴素贝叶斯）
**时间：** ~45 分钟

## 问题

第 01 课曾承诺：词形还原需要词性标签。不知道 `running` 是动词，词形还原器就无法把它还原成 `run`；不知道 `better` 是形容词，它就无法把它还原成 `good`。

这个承诺背后其实藏着整整一个子领域。词性标注（Part-of-Speech tagging, POS tagging）负责为词分配语法类别。句法解析（syntactic parsing）则要恢复句子的树状结构：哪个词修饰哪个词、哪个动词支配哪些论元。传统 NLP 花了二十年持续打磨这两件事。后来，深度学习把它们压缩成“预训练 transformer 顶上的 token 分类任务”，研究社区也随之转身离开。

但应用界没有离开。每条结构化抽取流水线，在底层都还依赖 POS 和依存树。LLM 生成的 JSON 要用语法约束去验证。问答系统会借助依存解析拆解查询。机器翻译质量评估器会检查句法树的对齐。

值得了解。本课会介绍常见标注集、基础基线，以及你该在什么时候停止手写实现，直接调用 spaCy。

## 概念

**词性标注（POS tagging）**为每个词元分配语法类别。**宾州树库（Penn Treebank, PTB）**标注集是英语默认方案。它有 36 个标签，区分细得让普通读者觉得有些吹毛求疵：`NN` 单数名词、`NNS` 复数名词、`NNP` 单数专有名词、`VBD` 动词过去式、`VBZ` 第三人称单数现在时，等等。**通用依存（Universal Dependencies, UD）**标注集更粗（17 个标签），且与语言无关，因此成了跨语言工作的默认选择。

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**句法解析**会产生一棵树，主要有两种风格：

- **成分句法解析。** 名词短语、动词短语、介词短语相互嵌套。输出是一棵由非终结符类别（NP、VP、PP）组成、叶子是词的树。
- **依存句法解析。** 每个词都依附于一个唯一的中心词，并带有语法关系标签。输出是一棵树，其中每条边都是一个（中心词、依赖词、关系）三元组。

依存句法解析在 2010 年代胜出，因为它能自然泛化到不同语言，尤其是自由语序语言。

```
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

## 动手构建

### 第 1 步：最频繁标签基线

这是“最笨但能用”的 POS 标注器。对每个词，预测它在训练中最常见的标签。

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

在 Brown 语料上，这个基线大约能达到 ~85% 准确率。不算好，但它是所有严肃模型都不该跌破的地板线。

### 第 2 步：二元 HMM 标注器

对序列的联合概率建模：

```
P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

这里有两张表：转移概率（给定上一个标签，当前标签的概率）和发射概率（给定标签，当前词的概率）。两者都可以用计数加拉普拉斯平滑估计。解码时用 Viterbi（在标签格上做动态规划）。

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

在 Brown 上，二元 HMM 大约能达到 ~93% 准确率。从 85% 跳到 93%，大部分提升来自转移概率——模型学会了 `DET NOUN` 常见，而 `NOUN DET` 很少见。

### 第 3 步：为什么现代标注器能超过它

转移概率与发射概率都是局部的。它们无法捕捉 `saw` 在 “I bought a saw” 里是名词，而在 “I saw the movie” 里是动词。允许任意特征（后缀、词形、前后词、词本身）的 CRF，大约能做到 ~97%。BiLSTM-CRF 或 transformer 则能达到 ~98%+。

这个任务的上限其实受标注员分歧限制。人在 Penn Treebank 上的一致率大约只有 97%。超过 98% 的模型，往往已经开始对测试集过拟合。

### 第 4 步：依存句法解析示意

从零完整实现依存句法解析超出了本课范围；教科书级讲解见 Jurafsky 和 Martin。这里你只需要知道两大经典流派：

- **基于转移的解析器。**（arc-eager、arc-standard）它像 shift-reduce 解析器一样工作：读取词元，把它们压栈，再通过 reduce 动作创建依存弧。贪心解码很快。经典实现是 MaltParser；现代神经版本是 Chen 和 Manning 的 transition-based parser。
- **基于图的解析器。**（Eisner 算法、Dozat-Manning biaffine）它为每一条可能的中心词-依赖词边打分，再选出最大生成树。更慢，但通常更准。

对大多数应用工作来说，直接调用 spaCy：

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

把 `dep` 这一列自下而上读，你就能把句子的语法结构读出来。

## 使用它

每个生产级 NLP 库都会把 POS 与依存解析作为标准流水线的一部分提供出来。

- **spaCy**（`en_core_web_sm` / `md` / `lg` / `trf`）。快、准，并且与分词、NER、词形还原整合在一起。`token.tag_`（Penn）、`token.pos_`（UD）、`token.dep_`（依存关系）。
- **Stanford NLP（stanza）**。Stanford 对 CoreNLP 的继任者。60 多种语言上都是先进水平。
- **trankit**。基于 transformer，UD 准确率很高。
- **NLTK**。`pos_tag`。能用，但慢，也更老。适合教学。

### 它在 2026 年仍然重要的地方

- **词形还原。** 第 01 课需要 POS 才能正确做词形还原。始终如此。
- **验证 LLM 输出的结构化抽取。** 检查生成句子是否满足语法约束（例如主谓一致、必要修饰语是否存在）。
- **基于方面的情感分析。** 依存解析能告诉你某个形容词到底修饰哪个名词。
- **查询理解。** “movies directed by Wes Anderson starring Bill Murray” 可以通过解析拆成结构化约束。
- **跨语言迁移。** UD 标签和依存关系与语言无关，因此可以实现对新语言的零样本结构化分析。
- **低算力流水线。** 如果你没法上 transformer，POS + 依存解析 + gazetteer 其实已经能做很多事。

## 交付它

保存为 `outputs/skill-grammar-pipeline.md`：

```markdown
---
name: grammar-pipeline
description: Design a classical POS + dependency pipeline for a downstream NLP task.
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

Given a downstream task (information extraction, rewrite validation, query decomposition, lemmatization), you output:

1. Tagset to use. Penn Treebank for English-only legacy pipelines, Universal Dependencies for multilingual or cross-lingual.
2. Library. spaCy for most production, stanza for academic-grade multilingual, trankit for highest UD accuracy. Name the specific model ID.
3. Integration pattern. Show the 3-5 lines that call the library and consume the needed attributes (`.pos_`, `.dep_`, `.head`).
4. Failure mode to test. Noun-verb ambiguity (`saw`, `book`, `can`) and PP-attachment ambiguity are the classical traps. Sample 20 outputs and eyeball.

Refuse to recommend rolling your own parser. Building parsers from scratch is a research project, not an application task. Flag any pipeline that consumes POS tags without handling lowercase/uppercase variants as fragile.
```

## 练习

1. **简单。** 在一个小型已标注语料（例如 NLTK 的 Brown 子集）上，使用最频繁标签基线测量留出集准确率。验证大约 ~85% 的结果。
2. **中等。** 训练上面的二元 HMM，并报告逐标签的精确率/召回率。HMM 最容易混淆哪些标签？
3. **困难。** 使用 spaCy 的依存解析，从 1000 句样本中抽取主语-动词-宾语三元组。在 50 条人工标注三元组上评估。记录抽取失败的情况（通常是被动句、并列结构和省略主语）。

## 关键术语

| 术语 | 人们常说什么 | 它真正表示什么 |
|------|--------------|----------------|
| POS 标签 | 词的类型 | 语法类别。PTB 有 36 类；UD 有 17 类。 |
| Penn Treebank | 标准标注集 | 英语专用。对动词时态和名词数有细粒度区分。 |
| Universal Dependencies | 多语标注集 | 比 PTB 更粗；语言中立；跨语言工作默认使用。 |
| 依存句法解析 | 句子树 | 每个词只有一个中心词，每条边都有语法关系。 |
| Viterbi | 动态规划 | 在给定发射与转移的情况下，找到概率最高的标签序列。 |

## 延伸阅读

- [Jurafsky and Martin — Speech and Language Processing, chapters 8 and 18](https://web.stanford.edu/~jurafsky/slp3/) —— 关于 POS 和解析的经典教科书讲解。
- [Universal Dependencies project](https://universaldependencies.org/) —— 所有多语解析器都会使用的跨语言标注集与树库集合。
- [spaCy linguistic features guide](https://spacy.io/usage/linguistic-features) —— `Token` 上各种属性的实用参考。
- [Chen and Manning (2014). A Fast and Accurate Dependency Parser using Neural Networks](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) —— 把神经解析器带入主流的论文。
