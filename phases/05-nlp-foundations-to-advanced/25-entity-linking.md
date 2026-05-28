# 实体链接（Entity Linking）与消歧（Disambiguation）

> NER 找到了 “Paris”。实体链接要决定它究竟是法国巴黎、Paris Hilton、德州 Paris，还是特洛伊王子 Paris。没有链接，知识图谱就始终处于歧义状态。

**类型：** 构建
**语言：** Python
**前置要求：** 第 5 阶段 · 06（NER），第 5 阶段 · 24（指代消解）
**时长：** ~60 分钟

## 问题

有一句话写着：“Jordan 顶住了压力。” 你的 NER 把 “Jordan” 标成了 PERSON。很好。但*到底是哪个* Jordan？

- Michael Jordan（篮球运动员）？
- Michael B. Jordan（演员）？
- Michael I. Jordan（伯克利机器学习教授——没错，这种混淆在 ML 论文里真实存在）？
- Jordan（这个国家）？
- Jordan（希伯来语名字）？

实体链接（EL）会把每个提及解析到知识库中的唯一条目：Wikidata、Wikipedia、DBpedia，或你自己的领域知识库。它包含两个子任务：

1. **候选生成（Candidate generation）。** 给定 “Jordan”，哪些知识库条目是合理候选？
2. **消歧（Disambiguation）。** 给定上下文，哪个候选才是正确的？

这两个步骤都可以学习，也都有基准评测。这个组合式流水线十年来都很稳定，变化的主要是消歧器的质量。

## 概念

*实体链接流水线：提及 → 候选 → 消歧后的实体*

**候选生成。** 给定提及表面形式（如 “Jordan”），在别名字典里查找候选。Wikipedia 的 alias 字典覆盖了大多数命名实体："JFK" → John F. Kennedy、Jacqueline Kennedy、JFK airport、JFK（电影）。典型索引会为一个提及返回 10-30 个候选。

**消歧：三种方法。**

1. **先验 + 上下文（Milne & Witten, 2008）。** `P(entity | mention) × context-similarity(entity, text)`。效果好、速度快、无需训练。
2. **基于嵌入（ESS / REL / BLINK）。** 编码提及 + 上下文，再编码每个候选实体的描述，取余弦相似度最大的一个。是 2020-2024 年的默认方案。
3. **生成式（GENRE, 2021；以及 2023+ 的 LLM 方法）。** 按 token 逐步解码实体的规范名称，并约束在有效实体名称构成的 trie 上，因此输出一定是合法的 KB id。

**端到端 vs 流水线。** 现代模型（ELQ、BLINK、ExtEnD、GENRE）可以在一次前向中同时完成 NER、候选生成和消歧。生产环境仍然以流水线系统为主，因为你可以随时替换其中任何一个组件。

### 两个必须看的指标

- **提及召回率（candidate gen）。** 在所有金标准提及中，正确 KB 条目出现在候选列表里的比例。这是整个流水线的下限。
- **消歧准确率 / F1。** 在候选集合正确的前提下，top-1 命中的频率。

一定要同时报告这两个指标。一个系统如果消歧准确率有 99%，但候选召回只有 80%，那整条流水线也只有 80%。

## 动手构建

### 步骤 1：从 Wikipedia 重定向构建 alias 索引

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

Wikipedia 的 alias 数据大约有 1800 万个 `(alias, entity)` 对。可以从 Wikidata dump 下载，并以倒排索引方式存储。

### 步骤 2：基于上下文的消歧

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

这里的 Jaccard 重叠只是玩具版本。请用嵌入余弦相似度替换它（见 `code/main.py` 第 2 步里的 transformer 版本）。

### 步骤 3：基于嵌入的方法（BLINK 风格）

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

建索引时，把每个 KB 实体嵌入一次；查询时，把提及 + 上下文嵌入一次，再与候选池做点积，取最大值即可。

### 步骤 4：生成式实体链接（概念）

GENRE 会逐字符解码实体的 Wikipedia 标题。约束解码（见第 20 课）保证只能输出有效标题。它与基于 KB 的 trie 深度集成。它的现代后继者包括 REL-GEN，以及使用结构化输出的 LLM 提示式 EL。

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

结合白名单约束（Outlines `choice`），这是 2026 年最容易落地的 EL 流水线。

### 步骤 5：在 AIDA-CoNLL 上评估

AIDA-CoNLL 是标准 EL 基准：1,393 篇 Reuters 文章、3.4 万个提及、Wikipedia 实体。通常报告库内准确率（`P@1`）和库外实体的 NIL 检测率。

## 常见陷阱

- **NIL 处理。** 有些提及不在知识库里（新兴实体、冷门人物）。系统必须预测 NIL，而不是随便猜一个错误实体。这个指标需要单独度量。
- **提及边界错误。** 上游 NER 漏掉了部分 span（例如把 “Bank of America” 只标成 “Bank”），EL 召回就会下降。
- **流行度偏置。** 训练出来的系统会过度预测高频实体。在机器学习论文里提到 “Michael I. Jordan”，系统常常会连到篮球运动员。
- **跨语言 EL。** 把中文文本中的提及映射到英文 Wikipedia 实体，需要多语言编码器或翻译步骤。
- **知识库陈旧。** 新公司、新事件、新人物不在去年的 Wikipedia dump 里。生产流水线需要刷新机制。

## 如何使用

2026 年的技术栈：

| 场景 | 选择 |
|-----------|------|
| 通用英文 + Wikipedia | BLINK 或 REL |
| 跨语言，KB = Wikipedia | mGENRE |
| LLM 友好、每天提及量很少 | Claude/GPT-4 + 候选列表 + 约束 JSON |
| 领域专用 KB（医疗、法律） | 自定义 BERT + KB 感知检索，并在领域版 AIDA 风格数据上微调 |
| 极低延迟 | 仅用精确匹配先验（Milne-Witten 基线） |
| 研究级 SOTA | GENRE / ExtEnD / 生成式 LLM-EL |

2026 年实际落地的生产模式：NER → coref → 对每个提及做 EL → 把簇折叠成每个簇一个规范实体。输出的是“文档中每个实体一个 KB id”，而不是“每个提及一个 KB id”。

## 交付

保存为 `outputs/skill-entity-linker.md`：

```markdown
---
name: entity-linker
description: Design an entity linking pipeline — KB, candidate generator, disambiguator, evaluation.
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

Given a use case (domain KB, language, volume, latency budget), output:

1. Knowledge base. Wikidata / Wikipedia / custom KB. Version date. Refresh cadence.
2. Candidate generator. Alias-index, embedding, or hybrid. Target mention recall @ K.
3. Disambiguator. Prior + context, embedding-based, generative, or LLM-prompted.
4. NIL strategy. Threshold on top score, classifier, or explicit NIL candidate.
5. Evaluation. Mention recall @ 30, top-1 accuracy, NIL-detection F1 on held-out set.

Refuse any EL pipeline without a mention-recall baseline (you cannot evaluate a disambiguator without knowing candidate gen surfaced the right entity). Refuse any pipeline using LLM-prompted EL without constrained output to valid KB ids. Flag systems where popularity bias affects minority entities (e.g. name-clashes) without domain fine-tuning.
```

## 练习

1. **简单。** 在 `code/main.py` 上的 prior+context 消歧器中测试 10 个歧义提及（Paris、Jordan、Apple）。手工标注正确实体，并测量准确率。
2. **中等。** 用 sentence transformer 编码 50 个歧义提及，再嵌入每个候选的描述。将基于嵌入的消歧与 Jaccard 上下文重叠进行比较。
3. **困难。** 构建一个 1000 实体规模的领域 KB（例如你公司的员工 + 产品）。实现端到端的 NER + EL，并在 100 条留出句子上测量 precision 和 recall。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 实体链接（EL） | 链到 Wikipedia | 将一个提及映射到唯一的 KB 条目。 |
| 候选生成 | 它可能是谁？ | 为一个提及返回一小组合理的 KB 候选。 |
| 消歧 | 选出正确那个 | 利用上下文给候选打分，并选出赢家。 |
| Alias 索引 | 查找表 | 从表面形式映射到候选实体。 |
| NIL | 不在 KB 中 | 显式预测“没有任何 KB 条目匹配”。 |
| KB | 知识库 | Wikidata、Wikipedia、DBpedia 或你的领域 KB。 |
| AIDA-CoNLL | 那个基准 | 1,393 篇 Reuters 文章，带金标准实体链接。 |

## 延伸阅读

- [Milne, Witten (2008). Learning to Link with Wikipedia](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf) —— 奠基性的先验 + 上下文方法。
- [Wu et al. (2020). Zero-shot Entity Linking with Dense Entity Retrieval (BLINK)](https://arxiv.org/abs/1911.03814) —— 基于嵌入的主力方法。
- [De Cao et al. (2021). Autoregressive Entity Retrieval (GENRE)](https://arxiv.org/abs/2010.00904) —— 使用约束解码的生成式 EL。
- [Hoffart et al. (2011). Robust Disambiguation of Named Entities in Text (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) —— 基准论文。
- [REL: An Entity Linker Standing on the Shoulders of Giants (2020)](https://arxiv.org/abs/2006.01969) —— 开源生产栈。
