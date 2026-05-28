# 指代消解（Coreference Resolution）

> “她给他打了电话。他没有接。那位医生在吃午饭。” 三个提及，对应两个人，但没人被明确点名。指代消解要做的，就是弄清谁指的是谁。

**类型：** 学习
**语言：** Python
**前置要求：** 第 5 阶段 · 06（NER），第 5 阶段 · 07（词性标注与句法分析）
**时长：** ~60 分钟

## 问题

从一篇 300 词的文章里抽取 Apple Inc. 的所有提及。如果文章直接写“Apple”，这很容易；如果写的是“这家公司”“他们”“这家库比蒂诺科技巨头”或“乔布斯的公司”，事情就难了。如果不能把这些提及解析到同一个实体，你的 NER 流水线会漏掉 60-80% 的提及。

指代消解会把所有指向同一现实世界实体的表达，链接成一个簇。它是表层 NLP（NER、句法分析）与下游语义任务（信息抽取、问答、摘要、知识图谱）之间的胶水。

为什么它在 2026 年仍然重要：

- 摘要：“CEO 宣布了……” 和 “Tim Cook 宣布了……” 不一样——摘要应该把 CEO 的名字说出来。
- 问答：“她给谁打了电话？” 需要先解析出“她”指的是谁。
- 信息抽取：如果知识图谱里同时出现 “PER1 创立了 Apple” 和 “Jobs 创立了 Apple” 两条独立事实，那就是错误的。
- 多文档信息抽取：合并多篇文章中关于同一事件的提及，就是跨文档指代消解。

## 概念

*指代聚类：提及 → 实体*

**任务定义。** 输入：一篇文档。输出：对提及（span）的聚类，其中每个簇都指向一个实体。

**提及类型。**

- **命名实体。** “Tim Cook”
- **名词性提及。** “这位 CEO”“这家公司”
- **代词性提及。** “他”“她”“他们”“它”
- **同位语。** “Tim Cook，Apple 的 CEO，”

**常见架构。**

1. **基于规则（Hobbs, 1978）。** 使用语法规则、基于句法树来解析代词。是很好的基线，而且在代词场景里意外地很难被超越。
2. **提及对分类器（Mention-pair classifier）。** 对每一对提及 `(m_i, m_j)` 预测它们是否共指，再通过传递闭包聚类。2016 年前的标准方法。
3. **提及排序（Mention-ranking）。** 对每个提及给候选先行词排序（包括“没有先行词”），选择分数最高者。
4. **基于 span 的端到端模型（Lee et al., 2017）。** Transformer 编码器。枚举所有不超过长度上限的候选 span，预测提及分数，再为每个 span 预测先行词概率，最后贪心聚类。这是现代默认方案。
5. **生成式方法（2024+）。** 提示 LLM：“列出文本中的每个代词及其先行词。” 在简单场景效果不错，但在长文档和罕见指称上仍然吃力。

**评估指标。** 标准上要看五种指标（MUC、B³、CEAF、BLANC、LEA），因为没有任何单一指标能完整反映聚类质量。通常报告前三者的平均值，也就是 CoNLL F1。到 2026 年，CoNLL-2012 上的 SOTA 大约在 ~83 F1。

**已知难例。**

- 指向若干页前引入实体的定指描述。
- 桥接照应（bridging anaphora）：“车轮”指向前面提到的一辆车。
- 中文、日语等语言中的零照应（zero anaphora）。
- 回指前置（cataphora，代词先于所指对象出现）：“当**她**走进来时，Mary 笑了。”

## 动手构建

### 步骤 1：预训练神经指代模型（AllenNLP / spaCy-experimental）

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")   # experimental model
doc = nlp("Apple announced new products. The company said they would ship soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

在更长的文档上，你会得到类似这样的结果：
- 簇 1：[Apple, The company, they]
- 簇 2：[new products]

### 步骤 2：基于规则的代词消解器（教学版）

请看 `code/main.py` 中仅依赖标准库的实现：

1. 提取提及：命名实体（大写 span）、代词（字典查表）、定指描述（“the X”）。
2. 对每个代词，查看前面 K 个提及，并按以下因素打分：
   - 性别/数的一致性（启发式）
   - 近邻性（越近越好）
   - 句法角色（优先主语）
3. 链接到得分最高的先行词。

它无法与神经模型竞争，但它能清楚展示搜索空间，以及端到端模型必须做出的那些决策。

### 步骤 3：用 LLM 做指代消解

```python
prompt = f"""Text: {text}

List every pronoun and noun phrase that refers to a person or company.
Cluster them by what they refer to. Output JSON:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

需要警惕两种失败模式。第一，LLM 会过度合并（把分别指向两个人的“他”和“她”合并起来）。第二，LLM 会在长文档里悄悄漏掉提及。一定要用字符偏移（span-offset）检查做验证。

### 步骤 4：评估

标准的 conll-2012 脚本会计算 MUC、B³、CEAF-φ4，并报告它们的平均值。如果做内部评估，先从你的标注测试集上的 span 级 precision / recall 开始，再加入 mention-linking F1。

## 常见陷阱

- **单例爆炸。** 有些系统会把每个提及都报告成独立簇。B³ 对此较宽松，MUC 则会严厉惩罚。务必同时看这三个指标。
- **长上下文中的代词。** 文档长度超过 2,000 token 时，性能可能下降约 15 F1。切块要谨慎。
- **性别假设。** 硬编码的性别规则在非二元指称、组织、动物上都会失效。应使用学习式模型或中性的评分方式。
- **LLM 在长文档中的漂移。** 单次 API 调用无法可靠地对 50 多段文本中的提及做聚类。要使用滑动窗口 + 合并。

## 如何使用

2026 年的技术栈：

| 场景 | 选择 |
|-----------|------|
| 英文、单文档 | `en_coreference_web_trf`（spaCy-experimental）或 AllenNLP neural coref |
| 多语言 | 在 OntoNotes 或 Multilingual CoNLL 上训练的 SpanBERT / XLM-R |
| 跨文档事件共指 | 专门的端到端模型（2025–26 SOTA） |
| 快速 LLM 基线 | GPT-4o / Claude + 结构化输出指代提示 |
| 生产级对话系统 | 规则回退 + 神经主模型 + 对关键槽位做人工复核 |

2026 年真正上线的集成模式：先跑 NER，再跑 coref，把共指簇合并进 NER 实体。下游任务看到的是“每簇一个实体”，而不是“每个提及一个实体”。

## 交付

保存为 `outputs/skill-coref-picker.md`：

```markdown
---
name: coref-picker
description: Pick a coreference approach, evaluation plan, and integration strategy.
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

Given a use case (single-doc / multi-doc, domain, language), output:

1. Approach. Rule-based / neural span-based / LLM-prompted / hybrid. One-sentence reason.
2. Model. Named checkpoint if neural.
3. Integration. Order of operations: tokenize → NER → coref → downstream task.
4. Evaluation. CoNLL F1 (MUC + B³ + CEAF-φ4 average) on held-out set + manual cluster review on 20 documents.

Refuse LLM-only coref for documents over 2,000 tokens without sliding-window merge. Refuse any pipeline that runs coref without a mention-level precision-recall report. Flag gender-heuristic systems deployed in demographically diverse text.
```

## 练习

1. **简单。** 在 `code/main.py` 中的规则消解器上运行 5 段手工编写的段落。对照真值标注，测量 mention-link 准确率。
2. **中等。** 在一篇新闻文章上使用预训练神经 coref 模型。把聚类结果与你自己的人工标注比较。它失败在什么地方？
3. **困难。** 构建一个加入 coref 的 NER 流水线：先做 NER，再通过 coref 簇合并。与仅用 NER 相比，在 100 篇文章上测量实体覆盖率提升。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 提及（Mention） | 一个指代 | 指向某个实体的一段文本（名字、代词、名词短语）。 |
| 先行词（Antecedent） | “it” 指的是什么 | 后续提及所共指的、更早出现的提及。 |
| 簇（Cluster） | 某个实体的所有提及 | 所有都指向同一现实实体的一组提及。 |
| 照应（Anaphora） | 向后回指 | 后面的提及指向前文（“he” → “John”）。 |
| 回指前置（Cataphora） | 向前回指 | 前面的提及指向后文（“When he arrived, John...”）。 |
| 桥接（Bridging） | 隐式指代 | “I bought a car. The wheels were bad.”（是那辆车的轮子。） |
| CoNLL F1 | 排行榜上的那个数字 | MUC、B³、CEAF-φ4 三个 F1 分数的平均值。 |

## 延伸阅读

- [Jurafsky & Martin, SLP3 Ch. 26 — Coreference Resolution and Entity Linking](https://web.stanford.edu/~jurafsky/slp3/26.pdf) —— 经典教材章节。
- [Lee et al. (2017). End-to-end Neural Coreference Resolution](https://arxiv.org/abs/1707.07045) —— 基于 span 的端到端方法。
- [Joshi et al. (2020). SpanBERT](https://arxiv.org/abs/1907.10529) —— 能提升 coref 的预训练方法。
- [Pradhan et al. (2012). CoNLL-2012 Shared Task](https://aclanthology.org/W12-4501/) —— 基准任务。
- [Hobbs (1978). Resolving Pronoun References](https://www.sciencedirect.com/science/article/pii/0024384178900064) —— 经典规则方法。
