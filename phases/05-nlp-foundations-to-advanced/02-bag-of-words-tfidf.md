# 词袋模型、TF-IDF 与文本表示

> 先计数，后思考。到了 2026 年，在定义清晰的任务上，TF-IDF 依然常常胜过嵌入。

**类型：** 构建
**语言：** Python
**前置要求：** 第 5 阶段 · 01（文本处理），第 2 阶段 · 02（从零实现线性回归）
**时间：** ~75 分钟

## 问题

模型需要数字，而你手里是字符串。

每条自然语言处理（NLP）流水线都必须回答同一个问题：怎样把长度可变的词元序列，变成分类器可以消费的定长向量？这个领域最早稳定采用的答案，是那个“虽然笨但能用”的方法：数词。把它们做成向量。

这种向量支撑过的生产级 NLP，比任何一种嵌入模型都多。垃圾邮件过滤、主题分类、日志异常检测、搜索排序（BM25 之前）、第一波情感分析、学术 NLP 基准的前十年。到 2026 年，面对范围较窄的分类任务，实践者依然会先拿它试手。它快、可解释，而且在“只要知道词有没有出现”就足够的任务上，它的效果常常与一个 4 亿参数的嵌入模型难分高下。

本课会从零实现词袋模型（Bag of Words, BoW）和词频-逆文档频率（TF-IDF, Term Frequency-Inverse Document Frequency），再展示如何用 scikit-learn 三行完成同样的事。最后，我们会指出那个会迫使你转向嵌入的失败模式。

## 概念

**词袋模型（Bag of Words, BoW）**会丢弃顺序信息。对于每篇文档，统计词表中每个词出现了多少次。向量长度就是词表大小，第 `i` 个位置表示第 `i` 个词的计数。

**词频-逆文档频率（TF-IDF）**会对 BoW 重新加权。一个词如果出现在每篇文档里，就没什么信息量，因此要把它压低。一个词如果在整个语料里很少见、却在单篇文档里频繁出现，那就是信号，因此要把它抬高。

```
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

其中，`TF` 是词在文档中的词频，`df` 是文档频率（有多少篇文档包含这个词），`N` 是文档总数。`log` 用来把那些无处不在的词的权重控制在有限范围内。

关键性质：两者都会产生稀疏向量，而且每个轴都可解释。你可以查看一个训练好分类器的权重，直接读出哪些词会把文档推向某个类别。对一个 768 维的 BERT 嵌入，你做不到这一点。

## 动手构建

### 第 1 步：建立词表

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

输入：分词后的文档列表（任何词级分词器都可以；本课里的 `code/main.py` 用的是一个简化版的小写分词）。输出：`{word: index}` 字典。稳定的插入顺序意味着，词索引 0 是第一篇文档里第一个出现的词。具体约定并不统一；scikit-learn 会按字母顺序排序。

### 第 2 步：词袋模型

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

行表示文档，列表示词表索引。位置 `[i][j]` 的含义是：“词 `j` 在文档 `i` 中出现了多少次。”文档 1 里的 `cat` 出现两次，因为它确实出现了两次；文档 0 里的 `ran` 是 0，因为它没有出现。

### 第 3 步：词频与文档频率

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

这里有两个值得点名的平滑技巧。`(n+1)/(d+1)` 避免了 `log(x/0)`。结尾那个 `+1` 则保证“出现在每篇文档中的词”其 IDF 仍然是 1（而不是 0），这与 scikit-learn 的默认设置一致。其他实现会使用原始的 `log(N/df)`。两种都能用；平滑版本更友好。

### 第 4 步：TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

三篇文档，五个词表词（`the`、`cat`、`sat`、`dog`、`ran`）。`the` 在三篇里都出现，所以它的 IDF 很低；`dog` 只出现在一篇里，所以它的 IDF 很高。得到的向量是稀疏的（大多数位置都很小），而真正有区分力的词会凸显出来。

### 第 5 步：对每一行做 L2 归一化

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

如果不做归一化，更长的文档就会得到更大的向量，并在相似度计算中占据主导。L2 归一化会把每篇文档都放到单位超球面上。此时，行与行之间的余弦相似度就只是点积。

## 使用它

scikit-learn 自带生产可用版本。

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer` 一次调用就完成分词、构建词表和 BoW。`TfidfVectorizer` 则进一步加上 IDF 加权与 L2 归一化。两者都返回稀疏矩阵。对于 10 万篇文档，稠密版本根本放不进内存；在分类器明确要求稠密矩阵之前，都保持稀疏。

会彻底改变结果的几个旋钮：

| 参数 | 作用 |
|-----|------|
| `ngram_range=(1, 2)` | 包含二元语法。通常能提升分类效果。 |
| `min_df=2` | 丢弃只在少于 2 篇文档中出现的词。在噪声数据上可裁剪词表。 |
| `max_df=0.95` | 丢弃出现在超过 95% 文档中的词。无需硬编码停用词表，就能近似完成停用词移除。 |
| `stop_words="english"` | scikit-learn 内置的英文停用词表。是否使用取决于任务——情感分析**不应该**丢弃否定词。 |
| `sublinear_tf=True` | 用 `1 + log(tf)` 替代原始 `tf`。当某个词在单篇文档里重复很多次时会更稳。 |

### TF-IDF 仍然占优的场景（截至 2026 年）

- 垃圾邮件检测、主题标注、日志异常告警。真正重要的是词是否出现，而不是语义细腻度。
- 低数据场景（几百个带标签样本）。TF-IDF 加逻辑回归没有预训练成本。
- 任何对延迟敏感的地方。TF-IDF 加线性模型能在微秒级返回结果；把一篇文档送进 transformer 做嵌入要 10–100ms。
- 必须解释预测原因的系统。直接查看分类器系数即可；权重最高的正向词就是原因。

### TF-IDF 失效的时候

第一个失败模式是**语义盲**。看下面两篇文档：

- “这部电影一点也不好。”
- “这部电影非常棒。”

一篇是负面评论，一篇是正面评论。它们的 TF-IDF 重叠只有 `{the, movie, was}`。词袋分类器必须死记硬背：当 `not` 靠近 `good` 时，标签会翻转。只要数据够多，它能学会；但它永远不如一个真正理解句法的模型来得自然。

第二个失败模式是推理时出现词表外词。一个在 IMDb 评论上训练的 BoW 模型，面对从未在训练中出现过的 `Zoomer-approved`，根本不知道该怎么办。子词嵌入（第 04 课）可以处理这个问题，TF-IDF 不行。

### 混合方案：TF-IDF 加权嵌入

到了 2026 年，在中等数据规模分类任务上的务实默认方案是：把 TF-IDF 权重当成对词嵌入的注意力。

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

你会同时得到嵌入带来的语义能力，以及 TF-IDF 带来的稀有词强调。分类器在池化后的向量上训练。对于情感、主题和意图分类，在大约 5 万个带标签样本以下时，这种做法通常优于单独使用任意一种表示。

## 交付它

保存为 `outputs/prompt-vectorization-picker.md`：

```markdown
---
name: vectorization-picker
description: Given a text-classification task, recommend BoW, TF-IDF, embeddings, or a hybrid.
phase: 5
lesson: 02
---

You recommend a text-vectorization strategy. Given a task description, output:

1. Representation (BoW, TF-IDF, transformer embeddings, or a hybrid). Explain why in one sentence.
2. Specific vectorizer configuration. Name the library. Quote the arguments (`ngram_range`, `min_df`, `max_df`, `sublinear_tf`, `stop_words`).
3. One failure mode to test before shipping.

Refuse to recommend embeddings when the user has under 500 labeled examples unless they show evidence of semantic failure in a TF-IDF baseline. Refuse to remove stopwords for sentiment analysis (negations carry signal). Flag class imbalance as needing more than a vectorizer change.

Example input: "Classifying 30k customer support tickets into 12 categories. Most tickets are 2-3 sentences. English only. Need explainability for audit logs."

Example output:

- Representation: TF-IDF. 30k examples is not small; explainability requirement rules out dense embeddings.
- Config: `TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`. Keep stopwords because category keywords sometimes are stopwords ("not working" vs "working").
- Failure to test: verify `min_df=3` does not drop rare category keywords. Run `get_feature_names_out` filtered by class and eyeball.
```

## 练习

1. **简单。** 在经过 L2 归一化的 TF-IDF 输出上实现 `cosine_similarity(doc_vec_a, doc_vec_b)`。验证相同文档得分为 1.0，而词表完全不重叠的文档得分为 0.0。
2. **中等。** 给 `bag_of_words` 增加 `n-gram` 支持。参数 `n` 应产生 `n` 元语法计数。测试 `n=2` 时，`["the", "cat", "sat"]` 能生成 `["the cat", "cat sat"]` 的二元语法计数。
3. **困难。** 使用 GloVe 100d 向量（下载一次后缓存）实现上面的 TF-IDF 加权嵌入混合方案。把它在 20 Newsgroups 数据集上的分类准确率，与纯 TF-IDF 和纯平均池化嵌入进行比较。报告谁在什么场景下获胜。

## 关键术语

| 术语 | 人们常说什么 | 它真正表示什么 |
|------|--------------|----------------|
| 词袋模型（BoW） | 词频向量 | 单篇文档中词表词的计数。丢弃词序。 |
| TF | 词频 | 一个词在文档中的出现次数，也可以按文档长度归一化。 |
| DF | 文档频率 | 至少出现过一次该词的文档数。 |
| IDF | 逆文档频率 | 经过平滑的 `log(N / df)`。降低“到处都出现”的词的权重。 |
| 稀疏向量 | 大部分是零 | 词表通常有 1 万到 10 万词；其中大部分不会出现在任意一篇文档里。 |
| 余弦相似度 | 向量夹角 | 对 L2 归一化向量做点积。1 表示完全相同，0 表示正交。 |

## 延伸阅读

- [scikit-learn — feature extraction from text](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) —— 权威 API 参考，也解释了每个关键参数。
- [Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval](https://www.sciencedirect.com/science/article/pii/0306457388900210) —— 让 TF-IDF 成为十年默认方案的论文。
- ["Why TF-IDF Still Beats Embeddings" — Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) —— 2026 年视角下，为什么老方法仍会赢。
