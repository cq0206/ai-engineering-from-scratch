# RAG 的分块策略（Chunking Strategies）

> 分块配置对检索质量的影响，与嵌入模型的选择同样大（Vectara NAACL 2025）。如果分块做错了，再多的重排序也救不回来。

**类型：** 构建
**语言：** Python
**前置要求：** 第 5 阶段 · 14（信息检索），第 5 阶段 · 22（嵌入模型）
**时长：** ~60 分钟

## 问题

你把一份 50 页的合同放进 RAG 系统。用户问：“终止条款是什么？” 检索器却返回了封面页。为什么？因为模型是在 512-token 分块上训练的，而终止条款位于 20 页之后，正好被分页符切开，本地关键词又不足以把它和查询关联起来。

解决办法不是“换一个更好的嵌入模型”，而是分块（chunking）。多大？要不要重叠？在哪切分？是否要带上周边上下文？

2026 年 2 月的基准结果给出了不少反直觉结论：

- Vectara 2026 年研究：递归 512-token 分块的准确率优于语义分块，69% → 54%。
- 在 Natural Questions 上使用 SPLADE + Mistral-8B：重叠没有带来任何可测收益。
- 上下文悬崖（context cliff）：上下文达到约 2,500 token 时，回答质量会明显下降。

“显而易见”的答案（语义分块、20% 重叠、1000 token）往往是错的。本课会帮你建立对六种策略的直觉，并告诉你分别该在什么情况下使用。

## 概念

*在同一段文本上可视化六种分块策略*

**固定分块（Fixed chunking）。** 每 N 个字符或 token 切一次。最简单的基线。会在句子中间截断。压缩效果好，但连贯性差。

**递归分块（Recursive）。** LangChain 的 `RecursiveCharacterTextSplitter`。先尝试按 `\n\n` 切，再按 `\n`、`.`、空格切。能够优雅地逐级回退。是 2026 年的默认选择。

**语义分块（Semantic）。** 先嵌入每个句子，再计算相邻句子的余弦相似度；当相似度低于阈值时切分。能保持主题连贯，但速度较慢，而且有时会产生只有 40 token 的碎片，反而伤害检索。

**句子分块（Sentence）。** 按句边界切分。可以一块只放一个句子，也可以用 N 句滑窗。在最多约 5k token 的范围内，它以远低于语义分块的成本达到近似效果。

**父文档分块（Parent-document）。** 检索时存小的子块，同时保留更大的父块作为上下文。用子块召回，再返回父块。它的退化更平滑：即使子块质量一般，也往往能返回还不错的父块。

**后置分块（Late chunking，2024）。** 先在 token 级别嵌入整个文档，再把 token 嵌入池化成块嵌入。能保留跨块上下文。适合长上下文嵌入模型（BGE-M3、Jina v3），但计算成本更高。

**上下文化检索（Contextual retrieval，Anthropic，2024）。** 给每个块前面加上一段由 LLM 生成的摘要，用来说明该块在文档中的位置（例如：“这一块位于终止条款的第 3.2 节……”）。在 Anthropic 自家的基准中，检索提升为 35-50%，但索引成本较高。

### 打败所有默认值的一条规则

让块大小匹配查询类型：

| 查询类型 | 块大小 |
|------------|-----------|
| 事实型（“CEO 叫什么？”） | 256-512 tokens |
| 分析型 / 多跳型 | 512-1024 tokens |
| 整段理解型 | 1024-2048 tokens |

这来自 NVIDIA 2026 年基准。块应当足够大，能容纳答案及其局部上下文；也要足够小，使检索器的 top-K 聚焦在答案上，而不是被上下文噪声稀释。

## 动手构建

### 步骤 1：固定分块与递归分块

```python
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
```

### 步骤 2：语义分块

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```

在你的领域数据上调 `threshold`。太高会碎成很多小片段；太低则会变成一个巨块。

### 步骤 3：父文档策略

```python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents
```

关键洞见：要对父块去重。多个子块可能映射到同一个父块；如果全部返回，只会浪费上下文窗口。

### 步骤 4：上下文化检索（Anthropic 模式）

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

将上下文化后的块建立索引。查询时，额外的周边信号会改善检索效果。

### 步骤 5：评估

```python
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)
```

一定要做基准评测。对你的语料来说，“最优”策略未必和任何博客文章一致。

## 常见陷阱

- **只用事实型查询评估分块。** 多跳查询会显露出完全不同的赢家。请使用按查询类型分层的评估集。
- **语义分块没有最小尺寸。** 会产生 40-token 的碎片，伤害检索。一定要设置 `min_tokens`。
- **把重叠当成教条。** 2026 年研究发现，重叠通常没有收益，却会让索引成本翻倍。先测量，不要想当然。
- **没有最小/最大限制。** 5 token 或 5000 token 的块都会破坏检索。必须做夹紧。
- **跨文档分块。** 绝不要让一个块跨越两个文档。务必逐文档分块，再做合并。

## 如何使用

2026 年的技术栈：

| 场景 | 策略 |
|-----------|----------|
| 首次构建、语料未知 | 递归分块，512 tokens，无重叠 |
| 事实型问答 | 递归分块，256-512 tokens |
| 分析型 / 多跳型 | 递归分块，512-1024 tokens + 父文档 |
| 强交叉引用（合同、论文） | 后置分块或上下文化检索 |
| 对话 / 会话语料 | 按轮次分块 + 说话人元数据 |
| 短文本（推文、评论） | 一个文档 = 一个块 |

从递归 512 开始。在 50 条查询的评估集上测量 recall@5，然后再往下调优。

## 交付

保存为 `outputs/skill-chunker.md`：

```markdown
---
name: chunker
description: Pick a chunking strategy, size, and overlap for a given corpus and query distribution.
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

Given a corpus (document types, avg length, domain) and query distribution (factoid / analytical / multi-hop), output:

1. Strategy. Recursive / sentence / semantic / parent-document / late / contextual. Reason.
2. Chunk size. Token count. Reason tied to query type.
3. Overlap. Default 0; justify if >0.
4. Min/max enforcement. `min_tokens`, `max_tokens` guards.
5. Evaluation plan. Recall@5 on 50-query stratified eval set (factoid, analytical, multi-hop).

Refuse any chunking strategy without min/max chunk size enforcement. Refuse overlap above 20% without an ablation showing it helps. Flag semantic chunking recommendations without a min-token floor.
```

## 练习

1. **简单。** 对同一份 20 页文档分别使用 fixed(512, 0)、recursive(512, 0) 和 recursive(512, 100) 分块。比较块数量和边界质量。
2. **中等。** 在 5 篇文档上建立一个 30 条查询的评估集。测量 recursive、semantic 和 parent-document 的 recall@5。谁赢了？和博客里的说法一致吗？
3. **困难。** 实现上下文化检索。测量相对于基线递归分块的 MRR 提升。报告索引成本（LLM 调用）与精度收益之间的权衡。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 块（Chunk） | 文档的一小段 | 会被嵌入、建立索引并参与检索的子文档单元。 |
| 重叠（Overlap） | 安全边界 | 相邻块共享的 N 个 token；在 2026 年基准中通常没什么用。 |
| 语义分块 | 更聪明的分块 | 在相邻句子嵌入相似度下降时切分。 |
| 父文档 | 两级检索 | 检索小子块，返回更大的父块。 |
| 后置分块 | 先嵌入再切块 | 先在 token 级嵌入整篇文档，再池化成块向量。 |
| 上下文化检索 | Anthropic 的技巧 | 建索引前给每个块前置一段 LLM 生成的摘要。 |
| 上下文悬崖 | 2500-token 墙 | 2026 年 1 月在 RAG 中观察到，约 2.5k 上下文 token 后质量明显下降。 |

## 延伸阅读

- [Yepes et al. / LangChain — Recursive Character Splitting docs](https://python.langchain.com/docs/how_to/recursive_text_splitter/) —— 生产环境中的默认方案。
- [Vectara (2024, NAACL 2025). Chunking configurations analysis](https://arxiv.org/abs/2410.13070) —— 分块的重要性与嵌入模型选择相当。
- [Jina AI — Late Chunking in Long-Context Embedding Models (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) —— 后置分块论文。
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) —— 通过 LLM 生成的上下文前缀带来 35-50% 的检索提升。
- [NVIDIA 2026 chunk-size benchmark — Premai summary](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) —— 按查询类型选块大小。
