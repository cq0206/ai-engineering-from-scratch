# GloVe、FastText 与子词嵌入

> Word2Vec 为每个词训练一个嵌入。全局词向量（GloVe）分解共现矩阵。子词词向量（FastText）嵌入词的组成部分。字节对编码（BPE）则把这条路接到了 transformer 上。

**类型：** 构建
**语言：** Python
**前置要求：** 第 5 阶段 · 03（从零实现 Word2Vec）
**时间：** ~45 分钟

## 问题

Word2Vec 留下了两个悬而未决的问题。

第一，存在一条并行的研究路线：直接分解共现矩阵（如 LSA、HAL），而不是做在线的 skip-gram 更新。Word2Vec 的迭代式方法真的从根本上更好吗，还是两者差异只是因为它们处理计数的方式不同？全局词向量（Global Vectors, GloVe）回答了这个问题：只要损失函数选得合理，矩阵分解不仅能达到甚至超过 Word2Vec 的效果，训练成本还更低。

第二，这两种方法都无法解释模型从未见过的词。`Zoomer-approved`、`dogecoin`、上周才造出来的专有名词、某个罕见词根的所有屈折形式。FastText 通过引入字符 n-gram 嵌入解决了这个问题：一个词是它各个部分之和，其中包括语素，因此即使是词表外词，也能得到一个合理的向量。

第三，transformer 出现之后，问题又变了。词级词表的规模上限大约在一百万左右，而真实语言远比这开放。字节对编码（Byte-Pair Encoding, BPE）及其变体通过学习一个高频子词单元词表解决了这个问题，从而覆盖一切内容。今天所有现代 LLM 的 tokenizer，本质上都是子词 tokenizer。

本课会依次讲清这三者，然后说明在不同场景下该用哪一个。

## 概念

**GloVe（Global Vectors）**：构建词-词共现矩阵 `X`，其中 `X[i][j]` 表示词 `j` 在词 `i` 的上下文中出现了多少次。训练向量，使得 `v_i · v_j + b_i + b_j ≈ log(X[i][j])`。再对损失加权，避免高频词对主导训练。就这么简单。

**FastText（子词词向量）**：一个词等于它的字符 n-gram 加上词本身的和。`where` 会变成 `&lt;wh, whe, her, ere, re>, &lt;where>`。词向量就是这些组成向量之和。训练方式与 Word2Vec 相同。好处是：未见过的词（如 `whereupon`）也可以由已知 n-gram 组合得到。

**BPE（Byte-Pair Encoding）**：从单个字节（或字符）构成的词表开始。统计语料中每一对相邻符号。把出现频率最高的一对合并成一个新词元。重复 `k` 次。最终得到一个大小为 `k + 256` 的词表：高频序列（`ing`、`tion`、`the`）会成为单个词元，而罕见词会被拆成熟悉的片段。任何句子都能被切成某些词元。

## 动手构建

### GloVe：分解共现矩阵

```python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
```

这里有两个值得点名的活动部件。加权函数 `f(x) = (x/x_max)^alpha` 会压低超高频词对（例如 `(the, and)`）的影响，避免它们主导损失。最终嵌入是 `W`（中心词表）与 `W_tilde`（上下文词表）之和。把两者相加是论文中提出的小技巧，通常比只用其中一个表更好。

### FastText：感知子词的嵌入

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
```

每个词都由一组 n-gram 表示（通常是 3 到 6 个字符）。词嵌入等于这些 n-gram 嵌入之和。在 skip-gram 训练里，把 Word2Vec 原来使用的单个词向量替换成这个组合即可。

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

对于未见词，只要它的一部分 n-gram 已知，你依然能得到一个向量。`whereupon` 与 `where` 共享 `&lt;wh`、`her`、`ere` 和 `&lt;where`，所以两者会落在彼此接近的位置上。

### BPE：学习得到的子词词表

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

```python
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
```

第一次迭代会合并最常见的相邻对。经过足够多轮后，高频子串（`low`、`est`、`tion`）会成为单个词元，而罕见词则会被自然拆开。

真正的 GPT / BERT / T5 tokenizer 会学习 3 万到 10 万次合并。结果是：任何文本都会被切成一个长度受控、由已知 ID 组成的序列，再也不存在 OOV。

## 使用它

在实践中，你几乎不会亲手训练这些模型，而是直接加载预训练检查点。

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

在 transformer 时代，如果你想用 BPE 风格的子词分词：

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

前缀 `Ġ` 用来标记词边界（这是 GPT-2 的约定）。今天所有现代 tokenizer，要么是 BPE 变体、要么是 WordPiece（BERT）、要么是 SentencePiece（T5、LLaMA）。

### 该选哪一个

| 场景 | 选择 |
|-----------|------|
| 需要预训练的通用词向量，且不要求 OOV 容忍 | GloVe 300d |
| 需要预训练的通用词向量，且必须处理拼写错误 / 新造词 / 形态丰富语言 | FastText |
| 任何要送进 transformer 的内容（训练或推理） | 使用模型随附的 tokenizer。绝不要替换。 |
| 从零训练自己的语言模型 | 先在你的语料上训练一个 BPE 或 SentencePiece tokenizer |
| 用线性模型做生产级文本分类 | 仍然是 TF-IDF。见第 02 课。 |

## 交付它

保存为 `outputs/skill-embeddings-picker.md`：

```markdown
---
name: tokenizer-picker
description: Pick a tokenization approach for a new language model or text pipeline.
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

Given a task and dataset description, you output:

1. Tokenization strategy (word-level, BPE, WordPiece, SentencePiece, byte-level). One-sentence reason.
2. Vocabulary size target (e.g., 32k for an English-only LM, 64k-100k for multilingual).
3. Library call with the exact training command. Name the library. Quote the arguments.
4. One reproducibility pitfall. Tokenizer-model mismatch is the single most common silent production bug; call out which pair must be used together.

Refuse to recommend training a custom tokenizer when the user is fine-tuning a pretrained LLM. Refuse to recommend word-level tokenization for any model targeting production inference. Flag non-English / multi-script corpora as needing SentencePiece with byte fallback.
```

## 练习

1. **简单。** 运行 `char_ngrams("playing")` 和 `char_ngrams("played")`。计算这两个 n-gram 集合的 Jaccard 重叠度。你应该会看到大量共享片段（`pla`、`lay`、`play`），这也是 FastText 能够很好迁移到词形变体上的原因。
2. **中等。** 扩展 `learn_bpe`，跟踪词表增长。把“每个词元对应多少语料字符”画成随合并次数变化的曲线。你应该会看到：一开始压缩很快，随后渐近到大约每个词元 2–3 个字符。
3. **困难。** 在莎士比亚全集上训练一个 1k-merge 的 BPE。比较常见词与罕见专有名词的分词结果。测量前后平均“每个词对应多少个词元”。把让你意外的现象写下来。

## 关键术语

| 术语 | 人们常说什么 | 它真正表示什么 |
|------|--------------|----------------|
| 共现矩阵 | 词-词频率表 | `X[i][j]` = 词 `j` 在词 `i` 周围窗口中出现的频次。 |
| 子词 | 词的一部分 | 可以是字符 n-gram（FastText），也可以是学习得到的词元（BPE/WordPiece/SentencePiece）。 |
| BPE | 字节对编码 | 反复合并频率最高的相邻对，直到词表达到目标大小。 |
| OOV | 词表外 | 模型从未见过的词。Word2Vec/GloVe 会失效；FastText 和 BPE 可以处理。 |
| 字节级 BPE | 在原始字节上做 BPE | GPT-2 采用的方案。词表从 256 个字节开始，因此不会出现 OOV。 |

## 延伸阅读

- [Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) —— GloVe 论文，只有七页，至今仍是关于该损失函数推导的最佳材料。
- [Bojanowski et al. (2017). Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) —— FastText。
- [Sennrich, Haddow, Birch (2016). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) —— 把 BPE 引入现代 NLP 的论文。
- [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary) —— 在实践中，BPE、WordPiece 与 SentencePiece 究竟有什么不同。
