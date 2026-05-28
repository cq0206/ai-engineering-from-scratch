# 问答系统（Question Answering Systems）

> 有三类系统塑造了现代 QA。抽取式找到答案片段。检索增强式把答案锚定在文档里。生成式直接生成答案。每个现代 AI 助手，都是这三者的某种混合。

**类型：** 构建
**语言：** Python
**前置条件：** 第 5 阶段 · 11（机器翻译），第 5 阶段 · 10（注意力机制）
**时间：** ~75 分钟

## 问题

用户输入“第一代 iPhone 是什么时候发布的？”，期望得到“2007 年 6 月 29 日。”而不是“Apple 的历史悠久而复杂”。也不是孤零零一个“2007”。而是一个直接、有依据、正确的答案。

过去十年里，三种架构主导了问答（Question Answering, QA）。

- **抽取式问答（Extractive QA）。** 给定一个已知包含答案的问题和段落，在段落中找到答案片段（span）的起止位置。SQuAD 是经典基准。
- **开放域问答（Open-domain QA）。** 不给段落。先检索相关段落，再抽取或生成答案。它是今天每条 RAG 流水线的基石。
- **生成式 / 闭卷问答（Generative / Closed-book QA）。** 大语言模型从参数记忆里作答。不做检索。推理最快，但在事实问题上最不可靠。

到 2026 年，趋势是混合式：先检索出最优的若干段落，再提示一个生成模型，让它基于这些段落给出有依据的回答。这就是检索增强生成（Retrieval-Augmented Generation, RAG），第 14 课会深入讲检索部分。本课构建 QA 这一半。

## 概念

*QA 架构：抽取式、检索增强式、生成式*

**抽取式（Extractive）。** 用 Transformer（BERT 家族）把问题和段落一起编码。训练两个 head 来预测答案起始和结束 token 的索引。损失是对合法位置做交叉熵。输出是段落中的一个 span。它按构造不会幻觉，也按构造无法回答段落中没有答案的问题。

**检索增强式（Retrieval-augmented, RAG）。** 两阶段。第一，检索器从语料中找出 top-`k` 段落。第二，阅读器（extractive 或 generative）利用这些段落产出答案。检索器-阅读器拆分，使两者可以独立训练和评估。现代 RAG 常在中间再加一个 reranker。

**生成式（Generative）。** 解码器式 LLM（GPT、Claude、Llama）直接从已学权重作答。不做检索。对常见知识表现极好，对罕见或最新事实则可能灾难性失真。幻觉率与事实在预训练数据中的出现频率大致成反比。

## 动手构建

### 第 1 步：用预训练模型做抽取式 QA

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)
```

```python
{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}
```

`deepset/roberta-base-squad2` 在 SQuAD 2.0 上训练，而该数据集包含无法作答的问题。默认情况下，`question-answering` pipeline 即使在模型的 null score 胜出时，也会返回得分最高的 span——它*不会*自动返回空答案。若要得到显式的 “no answer” 行为，请在 pipeline 调用中传入 `handle_impossible_answer=True`：只有当 null score 超过所有 span score 时，它才会返回空答案。无论哪种方式，都要检查 `score` 字段。

### 第 2 步：一个检索增强流水线（草图）

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)


def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))
```

两阶段流水线。稠密检索器（dense retriever，Sentence-BERT）靠语义相似度找出相关段落。抽取式阅读器（RoBERTa-SQuAD）从合并后的 top 段落里抽取答案 span。对小语料很好用。若是百万文档级语料，请用 FAISS 或向量数据库。

### 第 3 步：带 RAG 的生成式 QA

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)
```

提示模式很重要。明确告诉模型必须基于上下文作答，并在上下文不足时返回 “I don't know.”，与天真提示相比，可以把幻觉率降低 40-60%。更复杂的模式还会加上引用、置信分数和结构化抽取。

### 第 4 步：反映真实世界的评估

SQuAD 使用 **精确匹配（Exact Match, EM）** 和 **token 级 F1（token-level F1）**。EM 是在标准化后（小写化、去标点、去冠词）的严格匹配——要么完全匹配，要么得 0。F1 则根据预测与参考之间的 token 重叠计算，允许部分得分。二者都会低估释义：例如 “June 29, 2007” 对比 “June 29th, 2007”，通常 EM 为 0（因为序数词破坏了标准化），但由于 token 重叠，F1 仍会给出相当高的分数。

对于生产 QA：

- **答案准确性（Answer accuracy）。** 由 LLM 或人工判定，因为常规指标抓不住语义等价。
- **引用准确性（Citation accuracy）。** 被引用的段落是否真的支持答案？这很容易自动检查：看生成的引用与检索段落之间是否存在字符串匹配。
- **拒答校准（Refusal calibration）。** 当答案不在检索段落里时，系统能否正确说 “I don't know”？测量错误自信率。
- **检索召回率（Retrieval recall）。** 在评估阅读器之前，先测检索器是否把正确段落放进 top-`k`。阅读器修不好没被检索到的段落。

### RAGAS：2026 年的生产评估框架

`RAGAS` 是专门为 RAG 系统设计的，并且在 2026 年已经是出货默认。它在不需要 gold reference 的情况下，对四个维度打分：

- **忠实度（Faithfulness）。** 答案里的每个 claim 是否都来自检索上下文？通过基于 NLI 的蕴含判断衡量。这是你最核心的幻觉指标。
- **答案相关性（Answer relevance）。** 答案是否真的回答了问题？做法是从答案反向生成假设问题，再与真实问题比较。
- **上下文精确率（Context precision）。** 检索到的 chunk 中，有多大比例真的相关？精确率低 = 提示里噪声太多。
- **上下文召回率（Context recall）。** 检索集合是否包含了所有必要信息？召回率低 = 阅读器不可能成功。

无参考打分让你能在真实生产流量上做评估，而不需要人工整理 gold answer。对于开放式问题，还可以在其上再叠一层 LLM-as-judge，因为这类问题的 exact-match 指标几乎没意义。

`pip install ragas`。接上你的 retriever + reader。每个 query 都会得到四个标量。对回归发警报。

## 使用它

2026 年的组合。

| 用例 | 推荐 |
|---------|-------------|
| 给定段落，找答案 span | `deepset/roberta-base-squad2` |
| 面向固定语料，不接受闭卷回答 | RAG：稠密检索器 + LLM 阅读器 |
| 面向文档存储的实时问答 | 使用混合检索器（BM25 + dense）+ reranker 的 RAG（见第 14 课） |
| 对话式问答（追问） | 带会话历史的 LLM + 每轮都做 RAG |
| 高事实要求、受监管领域 | 在权威语料上做抽取式；绝不单独使用生成式 |

到 2026 年，抽取式 QA 已经没那么时髦，因为带 LLM 的 RAG 能覆盖更多场景。但在要求逐字引用的上下文里，它依然在生产：法律检索、合规审查、审计工具。

## 交付它

保存为 `outputs/skill-qa-architect.md`：

```markdown
---
name: qa-architect
description: Choose QA architecture, retrieval strategy, and evaluation plan.
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

Given requirements (corpus size, question type, factuality constraint, latency budget), output:

1. Architecture. Extractive, RAG with extractive reader, RAG with generative reader, or closed-book LLM. One-sentence reason.
2. Retriever. None, BM25, dense (name the encoder), or hybrid.
3. Reader. SQuAD-tuned model, LLM by name, or "domain-fine-tuned DistilBERT."
4. Evaluation. EM + F1 for extractive benchmarks; answer accuracy + citation accuracy + refusal calibration for production. Name what you are measuring and how you are measuring it.

Refuse closed-book LLM answers for regulatory or compliance-sensitive questions. Refuse any QA system without a retrieval-recall baseline (you cannot evaluate the reader without knowing the retriever surfaced the right passage). Flag questions that require multi-hop reasoning as needing specialized multi-hop retrievers like HotpotQA-trained systems.
```

## 练习

1. **简单。** 在 10 段 Wikipedia 文本上搭好上面的 SQuAD 抽取式流水线。手工写 10 个问题。测量答案正确的次数。如果段落和问题都比较干净，你应该会答对 7-9 个。
2. **中等。** 加一个拒答分类器。当最高检索分数低于某个阈值（比如 0.3 余弦相似度）时，直接返回 “I don't know”，而不是调用阅读器。用留出集来调这个阈值。
3. **困难。** 在你选择的一个 10,000 文档语料上构建 RAG 流水线。实现混合检索（BM25 + dense）和 RRF 融合（见第 14 课）。比较加上混合检索前后的答案准确性。记录哪些问题类型提升最大。

## 关键术语

| 术语 | 大家常说 | 实际含义 |
|------|----------|----------|
| 抽取式问答（Extractive QA） | 找答案片段 | 在给定段落中预测答案的起止索引。 |
| 开放域问答（Open-domain QA） | 在语料上做 QA | 不提供段落；必须先检索再回答。 |
| RAG | 先检索再生成 | 检索增强生成。由检索器 + 阅读器组成的流水线。 |
| SQuAD | 经典基准 | Stanford Question Answering Dataset。指标是 EM + F1。 |
| 幻觉（Hallucination） | 编出来的答案 | 阅读器输出的内容没有得到检索上下文支持。 |
| 拒答校准（Refusal calibration） | 知道什么时候该闭嘴 | 系统在无法回答时，能正确说出 “I don't know”。 |

## 延伸阅读

- [Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) —— 基准论文。
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) —— DPR，QA 里经典的稠密检索器。
- [Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) —— 给 RAG 命名的论文。
- [Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) —— 全面的 RAG 综述。
