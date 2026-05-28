# 信息检索与搜索（Information Retrieval and Search）

> BM25 很精准，但也很脆。稠密检索（dense）撒网更广，却会漏掉关键词。混合检索（hybrid）是 2026 年的默认方案。剩下的基本都是调参。

**类型：** 构建
**语言：** Python
**前置条件：** 第 5 阶段 · 02（BoW + TF-IDF），第 5 阶段 · 04（GloVe、FastText、Subword）
**时间：** ~75 分钟

## 问题

用户输入“如果有人靠撒谎骗钱会怎样”，却希望搜到真正对应的法条：“IPC 第 420 条”。关键词搜索会完全错过它（没有共享词汇）。如果嵌入没有在法律文本上训练，语义搜索也会错过。真实搜索必须同时处理这两种情况。

信息检索（Information Retrieval, IR）是每个 RAG 系统、每个搜索框、每个文档站模糊查找背后的流水线。到 2026 年，生产里真正有效的架构从来不是单一方法，而是一条由互补方法组成的链路：后一层专门补前一层的失败。

本课会把每个部件都搭出来，并点明各自补住了哪种失败。

## 概念

*混合检索：BM25 + 稠密检索 + RRF + cross-encoder 重排*

四层。按需选用。

1. **稀疏检索（Sparse retrieval，BM25）。** 很快；对精确匹配很准；对语义非常差。运行在倒排索引上。面对百万文档，单次查询低于 10ms。它擅长法条编号、产品编码、错误信息、命名实体这类精确匹配。
2. **稠密检索（Dense retrieval）。** 把 query 和 document 编码成向量，再做最近邻搜索。能抓释义和语义相似，但会漏掉只差一个字符的精确关键词。使用 FAISS 或向量数据库时，单次查询通常 50-200ms。
3. **融合（Fusion）。** 把稀疏和稠密的排序列表合并。倒数排名融合（Reciprocal Rank Fusion, RRF）是最简单的默认选择，因为它忽略原始分数（不同方法的分数本来就不在同一尺度上），只看排名位置。如果你知道某种信号在自己的领域明显更强，也可以用加权融合。
4. **交叉编码器重排（Cross-encoder rerank）。** 先取融合结果的 top-30。再用 cross-encoder（把 query 和 document 一起输入，对每一对打分）重新排序。Cross-encoder 单对更慢，但比 bi-encoder 准得多。之所以负担得起，是因为你只在 top-30 上跑它。

三路检索（BM25 + dense + learned-sparse，如 SPLADE）在 2026 基准上优于双路检索，但它需要 learned-sparse 索引基础设施。对大多数团队来说，双路加 cross-encoder 重排就是甜蜜点。

## 动手构建

### 第 1 步：从零实现 BM25

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

这里有两个参数值得认识。`k1=1.5` 控制词频饱和（term-frequency saturation）；更高意味着词重复次数权重更大。`b=0.75` 控制长度归一化；0 表示忽略文档长度，1 表示完全归一化。默认值来自 Robertson 在原始论文中的建议，几乎不需要调。

### 第 2 步：用 bi-encoder 做稠密检索

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

对嵌入做 L2 归一化，这样点积就等于余弦相似度。`all-MiniLM-L6-v2` 是 384 维，速度快，对大多数英文检索已经够强。多语言场景用 `paraphrase-multilingual-MiniLM-L12-v2`。追求最高精度则用 `bge-large-en-v1.5` 或 `e5-large-v2`。

### 第 3 步：倒数排名融合（Reciprocal Rank Fusion）

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

这里的 `k=60` 常数来自最初的 RRF 论文。更高的 `k` 会压平排名差异的贡献；更低的 `k` 会让前几名主导结果。60 是公开论文里的默认值，几乎不需要调。

### 第 4 步：混合搜索 + 重排

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

三阶段组合。BM25 找词面匹配。稠密检索找语义匹配。RRF 在无需分数校准的前提下融合两种排序。Cross-encoder 再用 query-document 成对输入，对 top-30 重新打分，从而抓住 bi-encoder 漏掉的细粒度相关性。最后保留 top-5。

### 第 5 步：评估

| 指标 | 含义 |
|--------|---------|
| Recall@k | 当正确文档存在时，有多大比例能进入 top-k？ |
| MRR（Mean Reciprocal Rank） | 第一个相关文档的 `1/rank` 的平均值。 |
| nDCG@k | 不只看是否相关，也考虑相关程度高低。 |

对 RAG 来说，最重要的数字是检索器（retriever）的 **Recall@k**。如果正确段落不在检索集合里，阅读器根本不可能回答对。

调试技巧：对于失败 query，把稀疏和稠密两套排序做 diff。如果一种能找到正确文档、另一种找不到，你遇到的就是词汇不匹配（修法：补上缺的那一半）或语义歧义（修法：更好的嵌入或更强的 reranker）。

## 使用它

2026 年的组合：

| 规模 | 组合 |
|-------|-------|
| 1k-100k 文档 | 内存内 BM25 + `all-MiniLM-L6-v2` 嵌入 + RRF。不需要独立数据库。 |
| 100k-10M 文档 | 稠密检索用 FAISS 或 pgvector，BM25 用 Elasticsearch / OpenSearch。并行运行。 |
| 10M+ 文档 | 带混合支持的 Qdrant / Weaviate / Vespa / Milvus。上层再加 cross-encoder 重排 top-30。 |
| 质量前沿 | 三路（BM25 + dense + SPLADE）+ ColBERT 晚交互重排 |

无论你选什么，都要给评估留预算。先 benchmark 检索召回，再 benchmark 端到端 RAG 准确率。阅读器修不好检索器漏掉的东西。

### 2026 年生产 RAG 的血泪经验

- **80% 的 RAG 失败都能追溯到摄取和切块，而不是模型。** 团队往往花几周换 LLM、调提示，而检索器却每三个 query 就静悄悄地返回一次错误上下文。先修切块。
- **切块策略比切块大小更重要。** 固定大小切分会打断表格、代码和嵌套标题。句子感知切分是默认；对技术文档和产品手册，语义切分或 LLM 切分更值得投入。
- **父文档模式（parent-doc pattern）。** 先检索较小的“子”块以获得精确度。当同一父章节下出现多个子块时，再换回父块以保留上下文。这个模式能稳定提升答案质量，而且不需要重新训练。
- **`k_rerank=3` 通常最优。** 再多加 chunk，只会增加 token 成本和生成延迟，却不提升答案质量。如果对你来说 k=8 仍然比 k=3 好，说明 reranker 表现不够好。
- **HyDE / query expansion。** 先根据 query 生成一个假想答案，再对它做嵌入并检索。它能弥合短问题和长文档之间的措辞差距。不训练也能白拿一些精度提升。
- **上下文预算控制在 8K token 内。** 如果你经常撞到这个上限，说明 reranker 阈值太松了。
- **所有东西都要版本化。** 提示、切块规则、嵌入模型、reranker。任何漂移都会悄悄破坏答案质量。对忠实度、上下文精确率和未回答问题率设置 CI 闸门，能在用户看到前拦住回归。
- **三路检索（BM25 + dense + learned-sparse，如 SPLADE）优于双路检索。** 这是 2026 基准的结论，尤其适用于同时混合专有名词和语义的 query。只要基础设施支持 SPLADE 索引，就该上。

根据 2026 年的行业测量，恰当的检索设计能把幻觉降低 70-90%。大多数 RAG 性能提升来自更好的检索，而不是模型微调。

## 交付它

保存为 `outputs/skill-retrieval-picker.md`：

```markdown
---
name: retrieval-picker
description: Pick a retrieval stack for a given corpus and query pattern.
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

Given requirements (corpus size, query pattern, latency budget, quality bar, infra constraints), output:

1. Stack. BM25 only, dense only, hybrid (BM25 + dense + RRF), hybrid + cross-encoder rerank, or three-way (BM25 + dense + learned-sparse).
2. Dense encoder. Name the specific model. Match to language(s), domain, and context length.
3. Reranker. Name the specific cross-encoder model if used. Flag that rerank adds 30-100ms latency on top-30.
4. Evaluation plan. Recall@10 is the primary retriever metric. MRR for multi-answer. Baseline first, incremental improvements measured against it.

Refuse to recommend dense-only for corpora with named entities, error codes, or product SKUs unless the user has evidence dense handles exact matches. Refuse to skip reranking for high-stakes retrieval (legal, medical) where the final top-5 decides the user's answer.
```

## 练习

1. **简单。** 在一个 500 文档语料上实现上面的 `hybrid_search`。测试 20 个 query。比较 BM25-only、dense-only 和 hybrid 在 recall@5 上的差异。
2. **中等。** 加入 MRR 计算。对于每个已知正确文档的测试 query，找出正确文档在 BM25、dense 和 hybrid 排序中的名次。分别报告它们的 MRR。
3. **困难。** 用 MultipleNegativesRankingLoss（Sentence Transformers）在你的领域上微调一个稠密编码器。用 500 个 query-document 对构建训练集。比较微调前后的召回率。

## 关键术语

| 术语 | 大家常说 | 实际含义 |
|------|----------|----------|
| BM25 | 关键词搜索 | Okapi BM25。按词频、IDF 和长度给文档打分。 |
| 稠密检索（Dense retrieval） | 向量搜索 | 把 query 和 doc 编码成向量，再找最近邻。 |
| 双编码器（Bi-encoder） | 嵌入模型 | 独立编码 query 和 doc。查询时很快。 |
| 交叉编码器（Cross-encoder） | 重排模型 | 把 query 和 doc 一起编码。慢，但准。 |
| RRF | 排名融合 | 通过累加 `1/(k + rank)` 来合并两份排序。 |
| Recall@k | 检索指标 | 有相关文档进入 top-k 的 query 比例。 |

## 延伸阅读

- [Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) —— BM25 的权威综述。
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) —— DPR，经典 bi-encoder。
- [Formal et al. (2021). SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720) —— learned-sparse 检索器，显著缩小了与 dense 的差距。
- [Cormack, Clarke, Büttcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) —— RRF 论文。
- [Khattab and Zaharia (2020). ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832) —— 晚交互检索。
