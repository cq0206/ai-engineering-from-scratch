# 朴素贝叶斯 (Naive Bayes)

> “朴素”的假设是错的，但它照样有效。这正是它的魅力所在。

**类型：** 构建
**语言：** Python
**前置要求：** 第 2 阶段，第 01-07 课（分类、贝叶斯定理）
**时间：** ~75 分钟

## 学习目标

- 使用拉普拉斯平滑 (Laplace smoothing) 从零实现多项式朴素贝叶斯 (Multinomial Naive Bayes)，用于文本分类 (text classification)
- 解释为什么“朴素”的独立性假设在数学上是错的，但在实践中仍然能产生正确的类别排序
- 比较多项式朴素贝叶斯 (Multinomial Naive Bayes)、伯努利朴素贝叶斯 (Bernoulli Naive Bayes) 和高斯朴素贝叶斯 (Gaussian Naive Bayes) 三种变体，并为给定的特征类型选择合适版本
- 在高维稀疏数据 (sparse data) 上，将朴素贝叶斯与逻辑回归 (logistic regression) 对比，并解释其中起作用的偏差-方差权衡 (bias-variance tradeoff)

## 问题

你需要对文本进行分类。比如把邮件分成垃圾邮件或非垃圾邮件，把用户评论分成正面或负面，把支持工单分到不同类别。你有成千上万个特征（每个词一个特征），但训练数据有限。

大多数分类器都会在这里失灵。逻辑回归需要足够多的样本，才能可靠地估计成千上万的权重。决策树 (decision trees) 一次只按一个词进行分裂，并且会严重过拟合。KNN 在 10,000 维空间里几乎没有意义，因为每个点看起来都和其他点差不多远。

朴素贝叶斯能处理这种情况。它做了一个在数学上错误的假设（给定类别时，每个特征都与其他特征独立），却仍然常常在文本分类中胜过那些“更聪明”的模型，尤其是在训练集较小时。它只需遍历一次数据就能完成训练。它可以扩展到数百万个特征。它还能输出概率估计（尽管由于独立性假设，这些概率通常校准得不太好）。

理解为什么一个错误的假设仍能带来好的预测，会让你学到机器学习中的一个根本事实：最好的模型不是最“正确”的模型，而是对你的数据来说具有最佳偏差-方差权衡的模型。

## 概念

### 贝叶斯定理 (Bayes' Theorem)（快速回顾）

贝叶斯定理会把条件概率“翻转”过来：

```
P(class | features) = P(features | class) * P(class) / P(features)
```

我们想要的是 `P(class | features)`——给定文档中的词后，该文档属于某个类别的概率。我们可以通过以下几部分来计算它：
- `P(features | class)`——在这个类别的文档里看到这些词的似然 (likelihood)
- `P(class)`——该类别的先验概率 (prior probability)（比如，垃圾邮件通常有多常见？）
- `P(features)`——证据项 (evidence)，对所有类别都相同，所以在比较时可以忽略

`P(class | features)` 最大的类别获胜。

### 朴素独立性假设

要精确计算 `P(features | class)`，就必须估计所有特征联合出现的概率。假设词表有 10,000 个词，你就需要估计一个覆盖 2^10,000 种可能组合的分布。这是不可能的。

朴素假设是：给定类别后，每个特征都条件独立 (conditionally independent)。

```
P(w1, w2, ..., wn | class) = P(w1 | class) * P(w2 | class) * ... * P(wn | class)
```

这样一来，你不必估计一个不可能完成的联合分布，而是只需要估计 n 个简单的单特征分布。每个分布只需要一个计数。

这个假设显然是错的。在任何文档里，“machine”和“learning”都不可能彼此独立。但分类器并不需要正确的概率估计，它需要的是正确的排序——哪个类别的概率最高。独立性假设会引入系统性误差，但这些误差往往会以相似方式影响所有类别，所以排序仍然是对的。

### 为什么它依然有效

有三个原因：

1. **排序比概率校准 (calibration) 更重要。** 分类只需要排在最前面的类别是对的。即使 P(spam) = 0.99999，而真实概率其实是 0.7，分类器仍然会正确地选出 spam。我们不需要精确的概率，我们需要的是正确的赢家。

2. **高偏差、低方差。** 独立性假设是一个很强的先验。它对模型施加了很强的约束，从而阻止过拟合。在训练数据有限时，一个略有偏差但稳定的模型，会胜过一个理论上更正确却极不稳定的模型。这就是偏差-方差权衡在起作用。

3. **特征冗余会相互抵消。** 相关特征提供的是重复证据。分类器会把这份证据重复计算，但它也会为正确类别重复计算。如果 “machine” 和 “learning” 总是一起出现，那么它们都会为 “tech” 类提供证据。朴素贝叶斯会把它算两次，但它是为正确的类别算了两次。

第四个更实际的原因是：朴素贝叶斯极其快。训练就是遍历一次数据并统计频次。预测就是一次矩阵乘法。你可以在几秒内用一百万篇文档完成训练。这样的速度意味着你可以更快迭代、尝试更多特征集，并比使用更慢的模型做更多实验。

### 数学过程逐步拆解

让我们通过一个具体例子走一遍。假设我们有两个类别：spam 和 not-spam。词表中有三个词：“free”、“money”、“meeting”。

训练数据：
- 垃圾邮件中，“free” 出现 80 次，“money” 出现 60 次，“meeting” 出现 10 次（总词数 150）
- 非垃圾邮件中，“free” 出现 5 次，“money” 出现 10 次，“meeting” 出现 100 次（总词数 115）
- 40% 的邮件是垃圾邮件，60% 是非垃圾邮件

使用拉普拉斯平滑（alpha=1）后：

```
P(free | spam)    = (80 + 1) / (150 + 3) = 81/153 = 0.529
P(money | spam)   = (60 + 1) / (150 + 3) = 61/153 = 0.399
P(meeting | spam) = (10 + 1) / (150 + 3) = 11/153 = 0.072

P(free | not-spam)    = (5 + 1) / (115 + 3) = 6/118 = 0.051
P(money | not-spam)   = (10 + 1) / (115 + 3) = 11/118 = 0.093
P(meeting | not-spam) = (100 + 1) / (115 + 3) = 101/118 = 0.856
```

一封新邮件包含：“free”（2 次）、“money”（1 次）、“meeting”（0 次）。

```
log P(spam | email) = log(0.4) + 2*log(0.529) + 1*log(0.399) + 0*log(0.072)
                    = -0.916 + 2*(-0.637) + (-0.919) + 0
                    = -3.109

log P(not-spam | email) = log(0.6) + 2*log(0.051) + 1*log(0.093) + 0*log(0.856)
                        = -0.511 + 2*(-2.976) + (-2.375) + 0
                        = -8.838
```

spam 以明显优势胜出。词 “free” 出现两次，是支持 spam 的强证据。注意，“meeting” 没有出现，因此它对两个对数和的贡献都是零（0 * log(P)）。在多项式朴素贝叶斯中，未出现的词没有影响；显式建模“词没有出现”这件事的是伯努利朴素贝叶斯。

### 三种变体

朴素贝叶斯通常有三种形式。它们对 `P(feature | class)` 的建模方式各不相同。

#### 多项式朴素贝叶斯 (Multinomial Naive Bayes)

把每个特征建模为一个计数。最适合特征是词频或 TF-IDF 值的文本数据。

```
P(word_i | class) = (count of word_i in class + alpha) / (total words in class + alpha * vocab_size)
```

`alpha` 就是拉普拉斯平滑（下文解释）。这是文本分类中最常用的变体。

#### 高斯朴素贝叶斯 (Gaussian Naive Bayes)

把每个特征建模为正态分布。最适合连续特征。

```
P(x_i | class) = (1 / sqrt(2 * pi * var)) * exp(-(x_i - mean)^2 / (2 * var))
```

每个类别、每个特征都各自拥有一个均值和方差。当特征在每个类别内部确实近似服从钟形曲线时，这种方法效果很好。

#### 伯努利朴素贝叶斯 (Bernoulli Naive Bayes)

把每个特征建模为二元值（出现或未出现）。最适合短文本或二值特征向量。

```
P(word_i | class) = (docs in class containing word_i + alpha) / (total docs in class + 2 * alpha)
```

与多项式版本不同，伯努利版本会显式惩罚某个词的缺失。如果 “free” 通常出现在垃圾邮件中，但它没有出现在这封邮件里，伯努利模型会把这件事当作反对 spam 的证据。

### 何时使用每种变体

| 变体 | 特征类型 | 最适用场景 | 示例 |
|---------|-------------|----------|---------|
| 多项式 | 计数或频率 | 文本分类、词袋模型 (bag-of-words) | 邮件垃圾分类、主题分类 |
| 高斯 | 连续值 | 特征近似正态的表格数据 | Iris 分类、传感器数据 |
| 伯努利 | 二值（0/1） | 短文本、二值特征向量 | SMS 垃圾分类、出现/缺失特征 |

### 拉普拉斯平滑

如果测试数据里出现了某个词，而它在某个类别的训练数据里从未出现过，会怎样？

如果不做平滑：`P(word | class) = 0/N = 0`。整个乘积里只要出现一个零，就会让 `P(class | features) = 0`，无论其他证据多么支持这个类别。一个从未见过的词，就能毁掉整个预测。

拉普拉斯平滑会给每个特征计数都加上一个很小的值 `alpha`（通常是 1）：

```
P(word_i | class) = (count(word_i, class) + alpha) / (total_words_in_class + alpha * vocab_size)
```

当 alpha=1 时，每个词至少都会得到一个极小的概率。测试邮件里出现 “discombobulate” 这样的词，也不会直接把 spam 的概率打成零。平滑还有一个贝叶斯解释：它等价于给词分布加上一个均匀狄利克雷先验 (uniform Dirichlet prior)。

alpha 越高，平滑越强（分布越趋于均匀）。alpha 越低，模型越相信数据。alpha 是一个需要调优的超参数。

alpha 的影响：

| Alpha | 效果 | 适用场景 |
|-------|--------|-------------|
| 0.001 | 几乎不做平滑，完全相信数据 | 训练集非常大，预计不会出现未见特征 |
| 0.1 | 轻度平滑 | 大训练集 |
| 1.0 | 标准拉普拉斯平滑 | 默认起点 |
| 10.0 | 强平滑，会压平分布 | 训练集很小，预计会有很多未见特征 |

### 对数空间计算

把几百个概率（每个都小于 1）相乘，会导致浮点下溢 (floating-point underflow)。即使真实值是一个非常小的正数，在浮点表示里，这个乘积也会直接变成零。

解决办法是：在对数空间里工作。不要直接相乘概率，而是把它们的对数加起来：

```
log P(class | x1, x2, ..., xn) = log P(class) + sum_i log P(xi | class)
```

这会把预测过程变成一个点积：

```
log_scores = X @ log_feature_probs.T + log_class_priors
prediction = argmax(log_scores)
```

也就是矩阵乘法。这就是朴素贝叶斯预测如此之快的原因——它和单层线性模型执行的是同一种运算。

### 朴素贝叶斯 vs 逻辑回归

两者都是用于文本的线性分类器。区别在于它们建模的对象不同。

| 方面 | 朴素贝叶斯 | 逻辑回归 |
|--------|------------|-------------------|
| 类型 | 生成式 (generative)（建模 P(X\|Y)） | 判别式 (discriminative)（建模 P(Y\|X)） |
| 训练 | 统计频率 | 优化损失函数 |
| 小数据 | 更好（强先验有帮助） | 更差（样本不足以估计权重） |
| 大数据 | 更差（错误假设开始伤害性能） | 更好（边界更灵活） |
| 特征 | 假设独立 | 能处理相关性 |
| 速度 | 单次遍历，非常快 | 迭代优化 |
| 校准 | 概率校准较差 | 概率校准较好 |

经验法则：先从朴素贝叶斯开始。如果你的数据足够多，而且朴素贝叶斯的效果进入平台期，再切换到逻辑回归。

### 分类流水线

```mermaid
flowchart LR
    A[原始文本] --> B[分词]
    B --> C[构建词汇表]
    C --> D[统计词频]
    D --> E[应用平滑]
    E --> F[计算对数概率]
    F --> G[预测：选择概率最高的类别]

    style A fill:#f9f,stroke:#333
    style G fill:#9f9,stroke:#333
```

在实践中，我们会在对数空间里工作，以避免浮点下溢。我们不是把很多很小的概率相乘，而是把它们的对数相加：

```
log P(class | features) = log P(class) + sum_i log P(feature_i | class)
```

## 动手实现

`code/naive_bayes.py` 中的代码从零实现了 MultinomialNB 和 GaussianNB。

### MultinomialNB

从零实现的版本分三步：

1. **fit(X, y)**：对每个类别，统计每个特征的频率。加入拉普拉斯平滑。计算对数概率。存储类别先验（类别频率的对数）。

2. **predict_log_proba(X)**：对每个样本、每个类别，计算 log P(class) + 所有特征的 log P(feature_i | class) 之和。这本质上是一次矩阵乘法：X @ log_probs.T + log_priors。

3. **predict(X)**：返回对数概率最高的类别。

```python
class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        classes = np.unique(y)
        n_classes = len(classes)
        n_features = X.shape[1]

        self.classes_ = classes
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.class_log_prior_[i] = np.log(X_c.shape[0] / X.shape[0])
            counts = X_c.sum(axis=0) + self.alpha
            self.feature_log_prob_[i] = np.log(counts / counts.sum())

        return self
```

关键洞察是：拟合完成后，预测只剩下矩阵乘法再加一个偏置。这就是朴素贝叶斯如此快的原因。

### GaussianNB

对于连续特征，我们会为每个类别、每个特征估计均值和方差：

```python
class GaussianNB:
    def __init__(self):
        pass

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        self.means_ = np.zeros((len(classes), X.shape[1]))
        self.vars_ = np.zeros((len(classes), X.shape[1]))
        self.priors_ = np.zeros(len(classes))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.means_[i] = X_c.mean(axis=0)
            self.vars_[i] = X_c.var(axis=0) + 1e-9
            self.priors_[i] = X_c.shape[0] / X.shape[0]

        return self
```

预测时会对每个特征使用高斯 PDF，并在特征之间相乘（在对数空间里则变成相加）。

### 演示：文本分类

代码会生成模拟两类数据的合成词袋特征（科技文章 vs 体育文章）。每个类别都有不同的词频分布。MultinomialNB 使用词频来完成分类。

这份合成数据的构造方式如下：我们创建 200 个“词”（特征列）。0-39 号词在科技文章中频率高、在体育文章中频率低。80-119 号词在体育文章中频率高、在科技文章中频率低。40-79 号词在两类里频率都处于中等水平。这样就构造出了一个相对真实的场景：有些词是强类别指示器，另一些只是噪声。

### 演示：连续特征

代码会生成类似 Iris 的数据（3 个类别、4 个特征、高斯簇）。GaussianNB 使用每个类别自己的均值和方差来分类。每个类别都有不同的中心（均值向量）和不同的离散程度（方差），这模仿了真实世界中不同类别的测量值会系统性不同的情况。

代码还演示了：
- **平滑对比：** 使用不同 alpha 值训练 MultinomialNB，展示平滑强度对准确率的影响。
- **训练集规模实验：** 当训练数据从 20 个样本增长到 1600 个样本时，朴素贝叶斯的准确率如何变化。即使样本很少，它也能较快达到还不错的准确率——这是它最重要的优势。
- **混淆矩阵：** 逐类查看精确率、召回率和 F1 分数，观察朴素贝叶斯会在哪些地方犯错。

### 预测速度

朴素贝叶斯的预测本质上就是一次矩阵乘法。对于 n 个样本、d 个特征、k 个类别：
- MultinomialNB：一次矩阵乘法 `(n x d) @ (d x k)` = `O(n * d * k)`
- GaussianNB：n * k 次高斯 PDF 计算，每次覆盖 d 个特征 = `O(n * d * k)`

它们在每个维度上都是线性的。把它和 KNN（需要对所有训练点计算距离）或使用 RBF 核的 SVM（需要对所有支持向量计算核函数）相比，朴素贝叶斯在预测时要快几个数量级。

## 实际使用

在 sklearn 中，这两个变体都可以一行搞定：

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"GaussianNB accuracy: {gnb.score(X_test, y_test):.3f}")

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_counts, y_train)
print(f"MultinomialNB accuracy: {mnb.score(X_test_counts, y_test):.3f}")
```

如果要用 sklearn 做文本分类：

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB(alpha=1.0)),
])

text_clf.fit(train_texts, train_labels)
accuracy = text_clf.score(test_texts, test_labels)
```

`naive_bayes.py` 里的代码会在同一份数据上，把从零实现的版本和 sklearn 进行对比，以验证实现是否正确。

### 朴素贝叶斯与 TF-IDF

原始词频会让每个词的每次出现都拥有相同权重。但像 “the” 和 “is” 这样的常见词会在所有类别中高频出现——它们不携带信息。TF-IDF（词频-逆文档频率，Term Frequency - Inverse Document Frequency）会降低常见词的权重，提高稀有且有区分力的词的权重。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB(alpha=0.1)),
])
```

TF-IDF 值是非负的，因此可以和 MultinomialNB 配合使用。TF-IDF + MultinomialNB 这个组合是文本分类中最强的基线之一。在训练样本少于 10,000 的数据集上，它经常能击败更复杂的模型。

### 适用于短文本的 BernoulliNB

对于短文本（推文、SMS、聊天消息），BernoulliNB 可能比 MultinomialNB 表现更好。短文本的词数很少，因此多项式模型依赖的词频信息噪声较大。BernoulliNB 只关心某个词是否出现，这在短文本里往往更可靠。

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

text_clf = Pipeline([
    ("vectorizer", CountVectorizer(binary=True)),
    ("classifier", BernoulliNB(alpha=1.0)),
])
```

`CountVectorizer` 中的 `binary=True` 标志会把所有计数转换成 0/1。如果不加它，BernoulliNB 依然能运行，但它看到的将是自己并不是为之设计的计数特征。

### 校准朴素贝叶斯的概率

朴素贝叶斯输出的概率通常校准得不好。当它说 P(spam) = 0.95 时，真实概率可能只有 0.7。如果你需要可靠的概率估计（例如设置阈值，或者和其他模型组合），可以使用 sklearn 的 `CalibratedClassifierCV`：

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_nb = CalibratedClassifierCV(MultinomialNB(), cv=5, method="sigmoid")
calibrated_nb.fit(X_train, y_train)
proba = calibrated_nb.predict_proba(X_test)
```

这相当于在朴素贝叶斯的原始分数之上，再用交叉验证拟合一个逻辑回归。得到的概率会更接近真实类别频率。

### 常见陷阱

1. **特征值为负。** MultinomialNB 要求特征非负。如果你有负值（比如某些设置下的 TF-IDF，或者标准化后的特征），那就改用 GaussianNB，或者先把特征整体平移到正值区间。

2. **零方差特征。** GaussianNB 会除以方差。如果某个类别下某个特征的方差为零（所有值都相同），概率计算就会出问题。代码里给所有方差都加上了一个很小的平滑项（1e-9）来避免这种情况。

3. **类别不平衡。** 如果 99% 的邮件都是非垃圾邮件，那么先验 P(not-spam) = 0.99 会强到淹没似然证据。你可以手动设置类别先验，或者在 sklearn 中使用 `class_prior` 参数。

4. **特征缩放。** MultinomialNB 不需要缩放（它处理的是计数）。GaussianNB 也不需要缩放（它会估计每个特征自己的统计量）。这是它相较于逻辑回归和 SVM 的一个优势，因为后两者对特征尺度更敏感。

## 交付成果

本课会产出：
- `outputs/skill-naive-bayes-chooser.md` —— 一个用于选择合适 NB 变体的决策技能
- `code/naive_bayes.py` —— 从零实现的 MultinomialNB 和 GaussianNB，并附带 sklearn 对比

### 朴素贝叶斯何时会失效

当独立性假设不仅导致概率不准，而且连类别排序都错了时，朴素贝叶斯就会失效。这通常发生在以下情况：

1. **强特征交互。** 如果类别取决于两个特征的组合，而不取决于其中任何一个单独特征（类似 XOR 的模式），朴素贝叶斯就会完全错过。因为单独看每个特征都没有证据，而它又无法用非线性方式把它们组合起来。

2. **高度相关且证据方向相反的特征。** 如果特征 A 指向 “spam”，特征 B 指向 “not-spam”，但 A 和 B 实际上是完全相关的（现实里它们总是一致），朴素贝叶斯就会看到实际上并不存在的“冲突证据”。

3. **训练集非常大。** 当数据足够多时，像逻辑回归这样的判别式模型会学到更接近真实的决策边界，并超过朴素贝叶斯。此前在小数据上帮到它的独立性假设，此时反而成了限制。

在实践中，这些失效模式在文本分类里并不常见。文本特征数量多、单个特征信号弱，而独立性假设带来的误差往往会相互抵消。对于只有少数强相关特征的表格数据，通常应该先考虑逻辑回归或基于树的模型。

## 练习

1. **平滑实验。** 在文本数据上，用 alpha = 0.01、0.1、1.0、10.0 和 100.0 训练 MultinomialNB。绘制准确率随 alpha 变化的曲线。性能在哪个位置达到峰值？为什么 alpha 非常大时反而会伤害性能？

2. **特征独立性测试。** 选一个真实文本数据集。挑两个明显相关的词（比如 “machine” 和 “learning”）。计算 P(word1 | class) * P(word2 | class)，再与 P(word1 AND word2 | class) 对比。独立性假设到底错了多少？它会影响分类准确率吗？

3. **Bernoulli 实现。** 在代码中扩展一个 BernoulliNB 类。把词袋特征转换成二值（出现/未出现），并在文本数据上与 MultinomialNB 比较准确率。什么时候 Bernoulli 会赢？

4. **朴素贝叶斯 vs 逻辑回归。** 在文本数据上同时训练两者。先从 100 个训练样本开始，再增加到 10,000。绘制两者准确率随训练集大小变化的曲线。逻辑回归会在什么节点超过朴素贝叶斯？

5. **垃圾邮件过滤器。** 构建一个完整的垃圾邮件分类器：对原始邮件文本做分词、建立词表、创建词袋特征、训练 MultinomialNB，并用精确率和召回率来评估（不只是准确率——为什么？）。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| 朴素贝叶斯 | “简单的概率分类器” | 一个使用贝叶斯定理并假设给定类别后特征条件独立的分类器 |
| 条件独立 | “特征之间不会互相影响” | P(A, B \| C) = P(A \| C) * P(B \| C)——一旦已知 C，再知道 B 不会给 A 带来额外信息 |
| 拉普拉斯平滑 | “加一平滑” | 给每个特征都加上一个很小的计数，以避免零概率主导整个预测 |
| 先验 | “看到数据之前的判断” | P(class)——在观察到任何特征之前，每个类别本身的概率 |
| 似然 | “数据有多匹配” | P(features \| class)——如果类别已知，观测到这些特征的概率 |
| 后验 (Posterior) | “看到数据之后更新后的判断” | P(class \| features)——观察到特征后更新得到的类别概率 |
| 生成式模型 | “建模数据是如何生成的” | 学习 P(X \| Y) 和 P(Y) 的模型，然后用贝叶斯定理得到 P(Y \| X) |
| 判别式模型 | “建模决策边界” | 直接学习 P(Y \| X)，而不去建模 X 是如何生成的模型 |
| 对数概率 | “避免下溢” | 使用 log P 而不是 P 来计算，避免许多小数相乘后在浮点表示中变成零 |

## 延伸阅读

- [scikit-learn Naive Bayes docs](https://scikit-learn.org/stable/modules/naive_bayes.html) —— 三种变体的官方文档，含数学细节
- [McCallum and Nigam, A Comparison of Event Models for Naive Bayes Text Classification (1998)](https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf) —— 关于文本中 Multinomial 与 Bernoulli 的经典对比论文
- [Rennie et al., Tackling the Poor Assumptions of Naive Bayes Text Classifiers (2003)](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf) —— 针对文本朴素贝叶斯的改进方法
- [Ng and Jordan, On Discriminative vs. Generative Classifiers (2001)](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf) —— 证明在数据较少时，朴素贝叶斯比逻辑回归收敛更快
