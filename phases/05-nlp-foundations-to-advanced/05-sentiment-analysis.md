# 情感分析

> 这是最经典的 NLP 任务。你需要了解的传统文本分类知识，大多都会在这里出现。

**类型：** 构建
**语言：** Python
**前置要求：** 第 5 阶段 · 02（BoW + TF-IDF），第 2 阶段 · 14（朴素贝叶斯）
**时间：** ~75 分钟

## 问题

“The food was not great.” 是正面还是负面？

情感分析（Sentiment Analysis）听起来很简单。评论者说了自己喜欢还是不喜欢某样东西。给句子打标签就行。它之所以会成为经典 NLP 任务，是因为每个看起来简单的案例背后，都藏着一个难点。否定会翻转含义，反讽会把含义彻底倒过来。“Not bad at all” 虽然带着两个看似负向的词，实际却是正面的。表情符号携带的信号，往往比周围文本还强。领域词汇也非常重要（音乐评论里的 `tight`，和时尚评论里的 `tight`，含义不同）。

情感分析是传统 NLP 的工作实验室。如果你理解了为什么每个朴素基线都有一个明确的失败模式，你也就理解了为什么后来的每种更复杂模型会被发明出来。本课会从零搭一个朴素贝叶斯（Naive Bayes）基线，再加入逻辑回归（Logistic Regression），并指出那些会让生产级情感分析变成“合规级问题”的陷阱。

## 概念

传统情感分析可以归结为两个步骤。

1. **表示。** 把文本变成特征向量。可以是 BoW、TF-IDF 或 n-gram。
2. **分类。** 在带标签样本上训练一个线性模型（朴素贝叶斯、逻辑回归、支持向量机（SVM））。

朴素贝叶斯是“最笨但能用”的模型。假设在给定标签的条件下，每个特征彼此独立。根据计数估计 `P(word | positive)` 和 `P(word | negative)`。推理时，把这些概率乘起来。“朴素”的独立性假设显然不成立，但它的效果却好得惊人。原因在于：当文本特征稀疏、数据量适中时，分类器更在乎每个词整体偏向哪一边，而没那么在乎它们之间的精细组合关系。

逻辑回归修复了这个独立性假设。它会为每个特征学习一个权重，而且允许负权重。`not good` 作为一个二元语法特征，就会学到负权重。朴素贝叶斯则无法对那些尚未被标注过的二元语法做这种处理。

## 动手构建

### 第 1 步：一个真实的迷你数据集

```python
POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]
```

数据故意做得很小。真实工作会用成千上万条样本（IMDb、SST-2、Yelp polarity）。数学完全一样。

### 第 2 步：从零实现多项式朴素贝叶斯

```python
import math
from collections import Counter


def train_nb(docs_by_class, vocab, alpha=1.0):
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs


def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)
```

加性平滑（alpha=1.0）就是拉普拉斯平滑（Laplace smoothing）。没有它的话，某个类别里从未出现过的词概率就是零，取对数后会直接炸掉。实践里常见 `alpha=0.01`，而 `alpha=1.0` 是教学默认值。

### 第 3 步：从零实现逻辑回归

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)
```

这里 L2 正则化很重要。文本特征是稀疏的；没有 L2，模型会记住训练样本。先从 `0.01` 起步，然后调参。

### 第 4 步：处理否定（最典型的失败模式）

看看 “not good” 和 “not bad”。BoW 分类器看到的是 `{not, good}` 与 `{not, bad}`，然后根据训练中哪种更常见来学习。二元语法分类器看到的是 `not_good` 和 `not_bad`，于是把它们学成两个不同特征。通常这就够了。

如果你没有 bigram，也有一个更粗糙但有效的修补办法：**否定作用域（negation scoping）**。在否定词后直到下一个标点之前，给所有词加上 `NOT_` 前缀。

```python
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}


def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out
```

```python
>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']
```

现在，`good` 和 `NOT_good` 成了不同特征。分类器可以为它们赋相反权重。只要三行预处理，情感基准上的准确率就会显著提升。

### 第 5 步：真正重要的评估指标

如果类别不平衡，只看准确率会非常误导。真实情感语料通常是 70–80% 正面，或者 70–80% 负面；一个永远预测多数类的分类器就能拿到 80% 准确率，但毫无价值。下面这些指标，你都应该报告：

- **每类精确率与召回率。** 每个类别各一对，再做宏平均，得到一个尊重类别平衡的单一数字。
- **宏平均 F1（Macro-F1，处理不平衡数据的主指标）。** 各类别 F1 的等权平均。类别不平衡时，优先用它而不是准确率。
- **加权 F1（Weighted-F1，可选指标）。** 与宏平均类似，但按类别频次加权。当“不平衡本身”具有业务含义时，应与 Macro-F1 一起报告。
- **混淆矩阵。** 原始计数。任何标量指标都不能在你查看它之前被完全信任；它能直接暴露模型最容易混淆的类别对。
- **每类错误样本。** 每个类别抽 5 个预测错误的例子，亲自读一遍。没有什么能替代读真实错误。

对于极度不平衡的数据（> 95:5），请报告 **AUROC** 和 **AUPRC**，而不是准确率。AUPRC 对少数类更敏感，而少数类往往才是你真正关心的对象（垃圾信息、欺诈、罕见情感）。

**一个常见 bug。** 在不平衡数据上报告 micro-F1 而不是 macro-F1，会得到一个看起来很高的数字，因为它被多数类主导了。Macro-F1 会逼着你直视少数类表现。

```python
def evaluate(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
```

## 使用它

scikit-learn 只用六行就能正确实现。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

这里有三点需要注意：`stop_words=None` 会保留否定词；`ngram_range=(1, 2)` 会加入二元语法，因此 `not_good` 能成为特征；`sublinear_tf=True` 会压低重复词的影响。在 SST-2 上，这三个参数往往就是 75% 准确率基线与 85% 准确率基线之间的差距。

### 什么时候该上 transformer

- 反讽检测。传统模型在这里基本没戏。
- 长篇评论，且情感在文档中途发生转折。
- 基于方面的情感分析（aspect-based sentiment）。“相机很好，但电池很差。”你需要把情感归因到具体方面。这里只有 transformer 或结构化输出模型靠谱。
- 非英语、低资源语言。多语 BERT 能免费给你一个零样本基线。

如果你需要以上任何一种能力，直接跳到第 7 阶段（transformer 深入）。否则，TF-IDF + bigram + 否定处理上的朴素贝叶斯或逻辑回归，就是你在 2026 年的生产级基线。

### 可复现性陷阱（又来了）

重训练情感模型是常规操作，重新评估它们却不是。论文里报出的准确率，依赖的是特定数据切分、特定预处理、特定 tokenizer。如果你拿自己的新模型和一个没有用相同流水线重跑出来的基线相比，得到的增量会非常误导。永远在**你的**流水线上重建基线，而不是抄论文中的数字。

## 交付它

保存为 `outputs/prompt-sentiment-baseline.md`：

```markdown
---
name: sentiment-baseline
description: Design a sentiment analysis baseline for a new dataset.
phase: 5
lesson: 05
---

Given a dataset description (domain, language, size, label granularity, latency budget), you output:

1. Feature extraction recipe. Specify tokenizer, n-gram range, stopword policy (usually keep), negation handling (scoped prefix or bigrams).
2. Classifier. Naive Bayes for baseline, logistic regression for production, transformer only if the domain needs sarcasm / aspects / cross-lingual.
3. Evaluation plan. Report precision, recall, F1, confusion matrix, and per-class error samples (not just scalars).
4. One failure mode to monitor post-deployment. Domain drift and sarcasm are the top two.

Refuse to recommend dropping stopwords for sentiment tasks. Refuse to report accuracy as the sole metric when classes are imbalanced (e.g., 90% positive). Flag subword-rich languages as needing FastText or transformer embeddings over word-level TF-IDF.
```

## 练习

1. **简单。** 把 `apply_negation` 加入 scikit-learn 流水线的预处理步骤，并测量它在一个小型情感数据集上的 F1 变化。
2. **中等。** 实现类别加权的逻辑回归（在 scikit-learn 里传 `class_weight="balanced"`，或者自己推导梯度）。测量它在一个人为构造的 90:10 类别不平衡上的影响。
3. **困难。** 训练一个讽刺检测器：在情感模型的残差上再训练第二个分类器。记录你的实验设置。如果准确率低于随机水平，请明确警告读者（两类讽刺任务的随机水平约为 50%，而大多数第一次尝试都会落在那里）。

## 关键术语

| 术语 | 人们常说什么 | 它真正表示什么 |
|------|--------------|----------------|
| 极性 | 正面或负面 | 二元标签；有时也会扩展为中性或更细粒度（如 5 星）。 |
| 基于方面的情感分析 | 面向方面的极性 | 把情感归因到文本中提到的具体实体或属性。 |
| 否定作用域 | 反转附近词元 | 在 “not” 之后直到标点前，为词加上 `NOT_` 前缀。 |
| 拉普拉斯平滑 | 给计数加 1 | 防止朴素贝叶斯里出现零概率特征。 |
| L2 正则化 | 缩小权重 | 在损失里加入 `lambda * sum(w^2)`。对稀疏文本特征至关重要。 |

## 延伸阅读

- [Pang and Lee (2008). Opinion Mining and Sentiment Analysis](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html) —— 奠基性综述。很长，但前四节已经覆盖了所有传统方法。
- [Wang and Manning (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://aclanthology.org/P12-2018/) —— 证明 “bigram + 朴素贝叶斯” 在短文本上很难被击败的论文。
- [scikit-learn text feature extraction docs](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) —— `CountVectorizer`、`TfidfVectorizer` 以及所有关键参数的参考文档。
