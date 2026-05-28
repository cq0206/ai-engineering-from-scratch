# 关系抽取（Relation Extraction）与知识图谱构建（Knowledge Graph Construction）

> NER 找到了实体，实体链接把它们锚定下来，关系抽取则找出它们之间的边。知识图谱就是节点、边以及其来源信息的总和。

**类型：** 构建
**语言：** Python
**前置要求：** 第 5 阶段 · 06（NER），第 5 阶段 · 25（实体链接）
**时长：** ~60 分钟

## 问题

一位分析师读到：“Tim Cook 在 2011 年成为 Apple 的 CEO。” 这里至少有四个事实：

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

关系抽取（RE）会把自由文本转换为结构化三元组 `(subject, relation, object)`。在整个语料库上聚合后，你就得到了知识图谱。再进一步聚合与查询，它就会成为 RAG、分析系统或合规审计的推理底座。

2026 年的问题是：LLM 抽关系非常积极——积极过头了。它们会幻觉出文本并不支持的三元组。没有来源信息（provenance），你就分不清哪些三元组是真的，哪些只是看起来合理的虚构。2026 年的答案是 AEVS 风格的“锚定并验证”流水线。

## 概念

*文本 → 三元组 → 知识图谱*

**三元组形式。** `(subject_entity, relation_type, object_entity)`。关系可以来自封闭本体（Wikidata properties、FIBO、UMLS），也可以来自开放集合（OpenIE 风格，任何说法都行）。

**三种抽取方式。**

1. **规则 / 模式法。** Hearst patterns："X such as Y" → `(Y, isA, X)`，再加一些手写正则。脆弱，但精确、可解释。
2. **监督分类器。** 给定一句话中的两个实体提及，从固定关系集合中预测它们之间的关系。训练数据通常来自 TACRED、ACE、KBP。是 2015–2022 年的标准方法。
3. **生成式 LLM。** 提示模型直接输出三元组。开箱即用，但如果没有 provenance，就会产生看似合理的幻觉垃圾。

**AEVS（Anchor-Extraction-Verification-Supplement，2026）。** 当前主流的幻觉缓解框架：

- **Anchor。** 找出每个实体 span 和关系短语 span 的精确位置。
- **Extract。** 生成与这些锚点 span 相连的三元组。
- **Verify。** 把每个三元组元素回对到源文本；凡是不被文本支持的都拒绝。
- **Supplement。** 再做一轮覆盖率检查，确保没有已锚定的 span 被漏掉。

幻觉会显著下降。代价是更多计算，但可审计。

**开放 vs 封闭的权衡。**

- **封闭本体。** 固定属性列表（例如 Wikidata 的 11,000+ 属性）。可预测、可查询，但很难扩展发明新关系。
- **开放信息抽取（Open IE）。** 任何动词短语都可以成为关系。召回高，精度低，查询起来很乱。

生产级知识图谱通常两者混用：先用 Open IE 做发现，再把关系规范化到封闭本体上，最后合并进主图。

## 动手构建

### 步骤 1：基于模式的抽取

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

完整的玩具抽取器见 `code/main.py`。Hearst patterns 之所以在特定领域流水线中仍被使用，是因为它们非常好调试。

### 步骤 2：监督式关系分类

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL 是一个 seq2seq 关系抽取器：文本输入，三元组输出，而且已经使用 Wikidata property id。它基于远程监督数据微调，是开放权重里的标准基线。

### 步骤 3：使用锚定的 LLM 提示抽取

```python
prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text. No inference beyond what is stated.
"""
```

一定要把返回的每个 span 与源文本逐一核对。凡是 `text[start:end] != triple_entity` 的都要拒绝。这就是最小化版本的 AEVS “verify” 步骤。

### 步骤 4：映射到封闭本体上做规范化

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by" (inverted subject/object)
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # drop unmapped open relations or route to manual review
```

规范化通常占到 60-80% 的工程工作量。要提前为它留预算。

### 步骤 5：构建一个小图并查询

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

这就是每一个基于知识图谱的 RAG 系统的原子单元。要扩展时，可以使用 RDF 三元组存储（Blazegraph、Virtuoso）、属性图（Neo4j）或向量增强型图存储。

## 常见陷阱

- **RE 之前要先做指代消解。** “He founded Apple”——RE 需要知道 “he” 是谁。先做 coref（第 24 课）。
- **实体规范化。** “Apple Inc” 和 “Apple” 必须落到同一个节点上。先做实体链接（第 25 课）。
- **幻觉三元组。** LLM 会输出文本并不支持的三元组。一定要做 span 验证。
- **关系规范化漂移。** Open IE 的关系表述并不一致（“was born in”“came from”“is a native of”）。如果不收敛到规范 id，整张图就无法查询。
- **时间错误。** “Tim Cook is CEO of Apple”——现在是真的，2005 年却是假的。很多关系都受时间约束。要使用限定词（例如 Wikidata 里的 `P580` 开始时间、`P582` 结束时间）。
- **领域失配。** REBEL 训练于 Wikipedia。法律、医疗和科学文本往往需要做领域微调的 RE 模型。

## 如何使用

2026 年的技术栈：

| 场景 | 选择 |
|-----------|------|
| 快速生产、通用领域 | REBEL 或 LlamaPred + Wikidata 规范化 |
| 领域专用（生物医学、法律） | SciREX 风格领域微调 + 自定义本体 |
| LLM 提示式、需审计输出 | AEVS 流水线：anchor → extract → verify → supplement |
| 高吞吐新闻 IE | 模式法 + 监督法混合 |
| 从零构建知识图谱 | Open IE + 人工规范化流程 |
| 时序知识图谱 | 带限定词抽取（开始/结束时间、时间点） |

集成模式是：NER → coref → entity linking → relation extraction → ontology mapping → graph load。每个阶段都可以成为质量关卡。

## 交付

保存为 `outputs/skill-re-designer.md`：

```markdown
---
name: re-designer
description: Design a relation extraction pipeline with provenance and canonicalization.
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

Given a corpus (domain, language, volume) and downstream use (KG-RAG, analytics, compliance), output:

1. Extractor. Pattern-based / supervised / LLM / AEVS hybrid. Reason tied to precision vs recall target.
2. Ontology. Closed property list (Wikidata / domain) or open IE with canonicalization pass.
3. Provenance. Every triple carries source char-span + doc id. Non-negotiable for audit.
4. Merge strategy. Canonical entity id + relation id + temporal qualifiers; dedup policy.
5. Evaluation. Precision / recall on 200 hand-labelled triples + hallucination-rate on LLM-extracted sample.

Refuse any LLM-based RE pipeline without span verification (source provenance). Refuse open-IE output flowing into a production graph without canonicalization. Flag pipelines with no temporal qualifier on time-bounded relations (employer, spouse, position).
```

## 练习

1. **简单。** 在 `code/main.py` 中的模式抽取器上运行 5 条新闻句子。手工检查 precision。
2. **中等。** 在同样的句子上使用 REBEL（或一个小型 LLM）。比较抽出的三元组。哪个抽取器 precision 更高？哪个 recall 更高？
3. **困难。** 构建 AEVS 流水线：用 LLM 抽取，再把 span 与源文本核对。对 50 条 Wikipedia 风格句子，比较 verify 步骤前后的幻觉率。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 三元组（Triple） | 主体-关系-客体 | 知识图谱的原子单位，即 `(s, r, o)`。 |
| Open IE | 什么都抽 | 开放词表的关系短语；召回高，精度低。 |
| 封闭本体 | 固定 schema | 有边界的关系类型集合（Wikidata、UMLS、FIBO）。 |
| 规范化（Canonicalization） | 把一切统一起来 | 将表面名称 / 关系映射到规范 id。 |
| AEVS | 有依据的抽取 | Anchor-Extraction-Verification-Supplement 流水线（2026）。 |
| 来源信息（Provenance） | 真相链接 | 每个三元组都带有 doc id + char-span，指回其来源。 |
| 远程监督 | 便宜标签 | 将文本与现有知识图谱对齐来生成训练数据。 |

## 延伸阅读

- [Mintz et al. (2009). Distant supervision for relation extraction without labeled data](https://www.aclweb.org/anthology/P09-1113.pdf) —— 远程监督论文。
- [Huguet Cabot, Navigli (2021). REBEL: Relation Extraction By End-to-end Language generation](https://aclanthology.org/2021.findings-emnlp.204.pdf) —— seq2seq RE 主力模型。
- [Wadden et al. (2019). Entity, Relation, and Event Extraction with Contextualized Span Representations (DyGIE++)](https://arxiv.org/abs/1909.03546) —— 联合信息抽取。
- [AEVS — Anchor-Extraction-Verification-Supplement framework](https://www.mdpi.com/2073-431X/15/3/178) —— 2026 年的幻觉缓解设计。
- [Wikidata SPARQL tutorial](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) —— 规范的图查询方法。
