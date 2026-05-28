# 模型评估（Model Evaluation）

> 一个模型只会和你衡量它的方式一样好。

**类型：** 构建
**语言：** Python
**前置要求：** 第 1 阶段（概率与分布、面向机器学习的统计学），第 2 阶段第 1-8 课
**时间：** ~90 分钟

## 学习目标

- 从零实现 K 折交叉验证（K-fold cross-validation）和分层 K 折交叉验证（stratified K-fold cross-validation），并解释为什么分层对不平衡数据很重要
- 从零计算准确率（accuracy）、精确率（precision）、召回率（recall）、F1、AUC-ROC，以及回归指标（regression metrics）（MSE、RMSE、MAE、R-squared）
- 解读学习曲线（learning curves），诊断模型是高偏差（high bias）还是高方差（high variance）
- 识别常见评估错误，包括数据泄漏（data leakage）、错误的指标选择，以及测试集污染（test set contamination）

## 问题

你训练了一个模型。它在你的数据上有 95% 的准确率（accuracy）。这就说明它很好吗？

也许是，也许不是。如果你有 95% 的数据都属于同一类，那么一个永远预测这一类的模型也能拿到 95% 的准确率，但它实际上毫无用处。如果你是在训练时用过的同一份数据上做评估，这个 95% 的数字就没有意义，因为模型只是把答案记住了。如果你的数据集带有时间维度，而你在切分前随机打乱了数据，那么模型可能实际上是在用未来的数据去预测过去。

模型评估是大多数机器学习项目出错的地方。错误的指标会让一个糟糕的模型看起来很好。错误的切分会让模型“作弊”。错误的比较会让你选中更差的模型。把评估做好不是可选项，而是必须项。这决定了你的模型是能在生产环境中工作，还是一见到真实数据就失败。

## 概念

### 训练集（Training Set）、验证集（Validation Set）、测试集（Test Set）

```mermaid
flowchart LR
    A[完整数据集] --> B[训练集 60-70%]
    A --> C[验证集 15-20%]
    A --> D[测试集 15-20%]
    B --> E[拟合模型]
    E --> C
    C --> F[调优超参数]
    F --> E
    F --> G[最终模型]
    G --> D
    D --> H[报告性能]
```

三种切分，三个用途：

- **训练集**：模型从这部分数据中学习。它会在训练过程中看到这些样本。
- **验证集**：用于调优超参数以及在模型之间做选择。模型不会在这部分数据上训练，但你的决策会受到它的影响。
- **测试集**：只在最后使用一次，用来报告最终性能。如果你看了测试性能后又回去改模型，那它就不再是测试集了，而是变成了第二个验证集。

测试集是你的留出保证（hold-out guarantee）：它保证你报告出来的性能，确实反映了模型在真正未见数据上的表现。

### K 折交叉验证（K-Fold Cross-Validation）

对于小数据集来说，只做一次训练/验证切分会浪费数据，而且得到的估计噪声很大。K 折交叉验证会让所有数据都同时参与训练和验证：

```mermaid
flowchart TB
    subgraph Fold1["第 1 折"]
        direction LR
        V1["验证"] --- T1a["训练"] --- T1b["训练"] --- T1c["训练"] --- T1d["训练"]
    end
    subgraph Fold2["第 2 折"]
        direction LR
        T2a["训练"] --- V2["验证"] --- T2b["训练"] --- T2c["训练"] --- T2d["训练"]
    end
    subgraph Fold3["第 3 折"]
        direction LR
        T3a["训练"] --- T3b["训练"] --- V3["验证"] --- T3c["训练"] --- T3d["训练"]
    end
    subgraph Fold4["第 4 折"]
        direction LR
        T4a["训练"] --- T4b["训练"] --- T4c["训练"] --- V4["验证"] --- T4d["训练"]
    end
    subgraph Fold5["第 5 折"]
        direction LR
        T5a["训练"] --- T5b["训练"] --- T5c["训练"] --- T5d["训练"] --- V5["验证"]
    end
    Fold1 --> R["平均得分"]
    Fold2 --> R
    Fold3 --> R
    Fold4 --> R
    Fold5 --> R
```

1. 把数据切成 K 个大小相等的折（fold）
2. 对每一折，用 K-1 折训练，用剩下那一折做验证
3. 对 K 次验证分数取平均值

K=5 或 K=10 是标准选择。每个数据点都会恰好被用作一次验证数据。平均分数比任何一次单独切分都更稳定。

**分层 K 折（Stratified K-Fold）**：在每一折中保留类别分布。如果你的数据集是 70% 的 A 类、30% 的 B 类，那么每一折里大致都会保持这个比例。对于不平衡数据集，这一点尤其重要，因为随机切分可能会把所有少数类样本都放进同一折里。

### 分类指标（Classification Metrics）

**混淆矩阵（confusion matrix）**：这是基础。对于二分类问题：

|  | 预测为正类 | 预测为负类 |
|--|---|---|
| 实际为正类 | 真正例（TP） | 假负例（FN） |
| 实际为负类 | 假正例（FP） | 真负例（TN） |

所有其他指标都由这个矩阵推导而来：

- **准确率（Accuracy）** = (TP + TN) / (TP + TN + FP + FN)。正确预测所占的比例。当类别不平衡时会产生误导。
- **精确率（Precision）** = TP / (TP + FP)。所有被预测为正的样本里，真正为正的有多少？适用于假正例代价很高的场景（例如垃圾邮件过滤器把真实邮件标成垃圾邮件）。
- **召回率（Recall）**（敏感度，sensitivity）= TP / (TP + FN)。所有真实正类中，我们抓住了多少？适用于假负例代价很高的场景（例如癌症筛查漏掉肿瘤）。
- **F1 分数（F1 score）** = 2 * precision * recall / (precision + recall)。精确率与召回率的调和平均数。在两者都重要且没有明显主次时使用。
- **AUC-ROC**：受试者工作特征曲线下面积（Area Under the Receiver Operating Characteristic curve）。它在不同分类阈值下绘制真正率与假正率。AUC = 0.5 表示随机猜测，AUC = 1.0 表示完美区分。它与阈值无关：衡量的是模型把正类排在负类前面的能力，而不依赖于你选定的截断点。

### 回归指标（Regression Metrics）

- **MSE**（均方误差，Mean Squared Error）= mean((y_true - y_pred)^2)。会以平方方式惩罚大误差，对离群点敏感。
- **RMSE**（均方根误差，Root Mean Squared Error）= sqrt(MSE)。与目标变量单位相同，比 MSE 更容易解释。
- **MAE**（平均绝对误差，Mean Absolute Error）= mean(|y_true - y_pred|)。对所有误差线性处理，比 MSE 更稳健。
- **R-squared**（决定系数）= 1 - SS_res / SS_tot，其中 SS_res = sum((y_true - y_pred)^2)，SS_tot = sum((y_true - y_mean)^2)。表示模型解释了多少方差。R^2 = 1.0 是完美结果；R^2 = 0.0 表示模型并不比始终预测均值更好；如果模型比均值还差，R^2 可以为负。

### 学习曲线（Learning Curves）

把训练分数和验证分数画成训练集大小的函数：

- **高偏差（underfitting，欠拟合）**：两条曲线都收敛到一个较低的分数。再加更多数据也没有帮助。你需要一个更复杂的模型。
- **高方差（overfitting，过拟合）**：训练分数很高，但验证分数明显更低，两者之间差距很大。增加更多数据通常会有帮助。

### 验证曲线（Validation Curves）

把训练分数和验证分数画成某个超参数的函数：

- 复杂度低时：两者都低（欠拟合）
- 复杂度合适时：两者都高，而且彼此接近
- 复杂度过高时：训练分数依然很高，但验证分数下降（过拟合）

最佳的超参数取值，就是验证分数达到峰值的位置。

### 常见评估错误

**数据泄漏（Data leakage）**：测试集中的信息泄漏进了训练过程。例子包括：在切分前就在完整数据集上拟合缩放器；在时间序列预测中包含未来数据；使用由目标变量派生出的特征。永远都是先切分，再预处理。

**类别不平衡（Class imbalance）**：99% 的交易是正常的，1% 是欺诈。一个始终预测“正常”的模型也能拿到 99% 的准确率。此时应改用精确率、召回率、F1 或 AUC-ROC。

**错误的指标（Wrong metric）**：比如你本该优化召回率（医疗诊断），却在优化准确率；或者你的数据有大量离群点，却在优化 RMSE（这时应改用 MAE）。

**没有使用分层切分（stratified splits）**：对于不平衡数据，随机切分可能会让验证折里几乎没有少数类样本，从而得到非常不稳定的估计。

**测试过于频繁**：每当你查看测试性能并据此调整模型时，你就在对测试集过拟合。测试集只能使用一次。

## 动手构建

### 步骤 1：训练/验证/测试集切分

```python
import random
import math


def train_val_test_split(X, y, train_ratio=0.6, val_ratio=0.2, seed=42):
    random.seed(seed)
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_val = [X[i] for i in val_idx]
    y_val = [y[i] for i in val_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test
```

### 步骤 2：K 折与分层 K 折交叉验证

```python
def kfold_split(n, k=5, seed=42):
    random.seed(seed)
    indices = list(range(n))
    random.shuffle(indices)

    fold_size = n // k
    folds = []

    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n
        val_idx = indices[start:end]
        train_idx = indices[:start] + indices[end:]
        folds.append((train_idx, val_idx))

    return folds


def stratified_kfold_split(y, k=5, seed=42):
    random.seed(seed)

    class_indices = {}
    for i, label in enumerate(y):
        class_indices.setdefault(label, []).append(i)

    for label in class_indices:
        random.shuffle(class_indices[label])

    folds = [{"train": [], "val": []} for _ in range(k)]

    for label, indices in class_indices.items():
        fold_size = len(indices) // k
        for i in range(k):
            start = i * fold_size
            end = start + fold_size if i < k - 1 else len(indices)
            val_part = indices[start:end]
            train_part = indices[:start] + indices[end:]
            folds[i]["val"].extend(val_part)
            folds[i]["train"].extend(train_part)

    return [(f["train"], f["val"]) for f in folds]


def cross_validate(X, y, model_fn, k=5, metric_fn=None, stratified=False):
    n = len(X)

    if stratified:
        folds = stratified_kfold_split(y, k)
    else:
        folds = kfold_split(n, k)

    scores = []
    for train_idx, val_idx in folds:
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_val = [X[i] for i in val_idx]
        y_val = [y[i] for i in val_idx]

        model = model_fn()
        model.fit(X_train, y_train)
        predictions = [model.predict(x) for x in X_val]

        if metric_fn:
            score = metric_fn(y_val, predictions)
        else:
            score = sum(1 for yt, yp in zip(y_val, predictions) if yt == yp) / len(y_val)
        scores.append(score)

    return scores
```

### 步骤 3：混淆矩阵与分类指标

```python
def confusion_matrix(y_true, y_pred):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    return tp, tn, fp, fn


def accuracy(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    total = tp + tn + fp + fn
    return (tp + tn) / total if total > 0 else 0.0


def precision(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def roc_curve(y_true, y_scores):
    thresholds = sorted(set(y_scores), reverse=True)
    tpr_list = []
    fpr_list = []

    total_positives = sum(y_true)
    total_negatives = len(y_true) - total_positives

    for threshold in thresholds:
        y_pred = [1 if s >= threshold else 0 for s in y_scores]
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)

        tpr = tp / total_positives if total_positives > 0 else 0.0
        fpr = fp / total_negatives if total_negatives > 0 else 0.0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return fpr_list, tpr_list, thresholds


def auc_roc(y_true, y_scores):
    fpr_list, tpr_list, _ = roc_curve(y_true, y_scores)

    pairs = sorted(zip(fpr_list, tpr_list))
    fpr_sorted = [p[0] for p in pairs]
    tpr_sorted = [p[1] for p in pairs]

    area = 0.0
    for i in range(1, len(fpr_sorted)):
        width = fpr_sorted[i] - fpr_sorted[i - 1]
        height = (tpr_sorted[i] + tpr_sorted[i - 1]) / 2
        area += width * height

    return area
```

### 步骤 4：回归指标

```python
def mse(y_true, y_pred):
    n = len(y_true)
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n


def rmse(y_true, y_pred):
    return math.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    n = len(y_true)
    return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / n


def r_squared(y_true, y_pred):
    mean_y = sum(y_true) / len(y_true)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    ss_tot = sum((yt - mean_y) ** 2 for yt in y_true)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot
```

### 步骤 5：学习曲线

```python
def learning_curve(X, y, model_fn, metric_fn, train_sizes=None, val_ratio=0.2, seed=42):
    random.seed(seed)
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)

    val_size = int(n * val_ratio)
    val_idx = indices[:val_size]
    pool_idx = indices[val_size:]

    X_val = [X[i] for i in val_idx]
    y_val = [y[i] for i in val_idx]

    if train_sizes is None:
        train_sizes = [int(len(pool_idx) * r) for r in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]]

    train_scores = []
    val_scores = []

    for size in train_sizes:
        subset = pool_idx[:size]
        X_train = [X[i] for i in subset]
        y_train = [y[i] for i in subset]

        model = model_fn()
        model.fit(X_train, y_train)

        train_pred = [model.predict(x) for x in X_train]
        val_pred = [model.predict(x) for x in X_val]

        train_scores.append(metric_fn(y_train, train_pred))
        val_scores.append(metric_fn(y_val, val_pred))

    return train_sizes, train_scores, val_scores
```

### 步骤 6：一个用于测试的简单分类器，以及完整演示

```python
class SimpleLogistic:
    def __init__(self, lr=0.1, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def sigmoid(self, z):
        z = max(-500, min(500, z))
        return 1.0 / (1.0 + math.exp(-z))

    def fit(self, X, y):
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0

        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                z = sum(w * x for w, x in zip(self.weights, xi)) + self.bias
                pred = self.sigmoid(z)
                error = yi - pred
                for j in range(n_features):
                    self.weights[j] += self.lr * error * xi[j]
                self.bias += self.lr * error

    def predict_proba(self, x):
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return self.sigmoid(z)

    def predict(self, x):
        return 1 if self.predict_proba(x) >= 0.5 else 0


class SimpleLinearRegression:
    def __init__(self, lr=0.001, epochs=200):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0
        n = len(X)

        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                pred = sum(w * x for w, x in zip(self.weights, xi)) + self.bias
                error = yi - pred
                for j in range(n_features):
                    self.weights[j] += self.lr * error * xi[j] / n
                self.bias += self.lr * error / n

    def predict(self, x):
        return sum(w * xi for w, xi in zip(self.weights, x)) + self.bias


def standardize(values):
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(var) if var > 0 else 1.0
    return [(v - mean) / std for v in values], mean, std


def make_classification_data(n=300, seed=42):
    random.seed(seed)
    X = []
    y = []
    for _ in range(n):
        x1 = random.gauss(0, 1)
        x2 = random.gauss(0, 1)
        label = 1 if (x1 + x2 + random.gauss(0, 0.5)) > 0 else 0
        X.append([x1, x2])
        y.append(label)
    return X, y


def make_regression_data(n=200, seed=42):
    random.seed(seed)
    X = []
    y = []
    for _ in range(n):
        x1 = random.uniform(0, 10)
        x2 = random.uniform(0, 5)
        target = 3 * x1 + 2 * x2 + random.gauss(0, 2)
        X.append([x1, x2])
        y.append(target)
    return X, y


def make_imbalanced_data(n=300, minority_ratio=0.05, seed=42):
    random.seed(seed)
    X = []
    y = []
    for _ in range(n):
        if random.random() < minority_ratio:
            x1 = random.gauss(3, 0.5)
            x2 = random.gauss(3, 0.5)
            label = 1
        else:
            x1 = random.gauss(0, 1)
            x2 = random.gauss(0, 1)
            label = 0
        X.append([x1, x2])
        y.append(label)
    return X, y


if __name__ == "__main__":
    X_clf, y_clf = make_classification_data(300)

    print("=== Train/Validation/Test Split ===")
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X_clf, y_clf)
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"  Train class distribution: {sum(y_train)}/{len(y_train)} positive")
    print(f"  Val class distribution: {sum(y_val)}/{len(y_val)} positive")

    model = SimpleLogistic(lr=0.1, epochs=200)
    model.fit(X_train, y_train)

    print("\n=== Classification Metrics ===")
    y_pred = [model.predict(x) for x in X_test]
    tp, tn, fp, fn = confusion_matrix(y_test, y_pred)
    print(f"  Confusion matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"  Accuracy:  {accuracy(y_test, y_pred):.4f}")
    print(f"  Precision: {precision(y_test, y_pred):.4f}")
    print(f"  Recall:    {recall(y_test, y_pred):.4f}")
    print(f"  F1 Score:  {f1_score(y_test, y_pred):.4f}")

    y_scores = [model.predict_proba(x) for x in X_test]
    auc = auc_roc(y_test, y_scores)
    print(f"  AUC-ROC:   {auc:.4f}")

    print("\n=== K-Fold Cross-Validation (K=5) ===")
    cv_scores = cross_validate(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=200),
        k=5,
        metric_fn=accuracy,
    )
    mean_cv = sum(cv_scores) / len(cv_scores)
    std_cv = math.sqrt(sum((s - mean_cv) ** 2 for s in cv_scores) / len(cv_scores))
    print(f"  Fold scores: {[round(s, 4) for s in cv_scores]}")
    print(f"  Mean: {mean_cv:.4f} (+/- {std_cv:.4f})")

    print("\n=== Stratified K-Fold Cross-Validation (K=5) ===")
    strat_scores = cross_validate(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=200),
        k=5,
        metric_fn=accuracy,
        stratified=True,
    )
    strat_mean = sum(strat_scores) / len(strat_scores)
    strat_std = math.sqrt(sum((s - strat_mean) ** 2 for s in strat_scores) / len(strat_scores))
    print(f"  Fold scores: {[round(s, 4) for s in strat_scores]}")
    print(f"  Mean: {strat_mean:.4f} (+/- {strat_std:.4f})")

    print("\n=== Imbalanced Data: Why Accuracy Lies ===")
    X_imb, y_imb = make_imbalanced_data(300, minority_ratio=0.05)
    positives = sum(y_imb)
    print(f"  Class distribution: {positives} positive, {len(y_imb) - positives} negative ({positives/len(y_imb)*100:.1f}% positive)")

    always_negative = [0] * len(y_imb)
    print(f"  Always-negative baseline:")
    print(f"    Accuracy:  {accuracy(y_imb, always_negative):.4f}")
    print(f"    Precision: {precision(y_imb, always_negative):.4f}")
    print(f"    Recall:    {recall(y_imb, always_negative):.4f}")
    print(f"    F1 Score:  {f1_score(y_imb, always_negative):.4f}")

    X_tr_i, y_tr_i, X_v_i, y_v_i, X_te_i, y_te_i = train_val_test_split(X_imb, y_imb)
    model_imb = SimpleLogistic(lr=0.5, epochs=500)
    model_imb.fit(X_tr_i, y_tr_i)
    y_pred_imb = [model_imb.predict(x) for x in X_te_i]
    print(f"\n  Trained model on imbalanced data:")
    print(f"    Accuracy:  {accuracy(y_te_i, y_pred_imb):.4f}")
    print(f"    Precision: {precision(y_te_i, y_pred_imb):.4f}")
    print(f"    Recall:    {recall(y_te_i, y_pred_imb):.4f}")
    print(f"    F1 Score:  {f1_score(y_te_i, y_pred_imb):.4f}")

    print("\n=== Regression Metrics ===")
    X_reg, y_reg = make_regression_data(200)

    col0 = [x[0] for x in X_reg]
    col1 = [x[1] for x in X_reg]
    col0_s, m0, s0 = standardize(col0)
    col1_s, m1, s1 = standardize(col1)
    X_reg_scaled = [[col0_s[i], col1_s[i]] for i in range(len(X_reg))]

    X_tr_r, y_tr_r, X_v_r, y_v_r, X_te_r, y_te_r = train_val_test_split(X_reg_scaled, y_reg)
    reg_model = SimpleLinearRegression(lr=0.01, epochs=500)
    reg_model.fit(X_tr_r, y_tr_r)
    y_pred_r = [reg_model.predict(x) for x in X_te_r]

    print(f"  MSE:       {mse(y_te_r, y_pred_r):.4f}")
    print(f"  RMSE:      {rmse(y_te_r, y_pred_r):.4f}")
    print(f"  MAE:       {mae(y_te_r, y_pred_r):.4f}")
    print(f"  R-squared: {r_squared(y_te_r, y_pred_r):.4f}")

    mean_baseline = [sum(y_tr_r) / len(y_tr_r)] * len(y_te_r)
    print(f"\n  Mean baseline:")
    print(f"    MSE:       {mse(y_te_r, mean_baseline):.4f}")
    print(f"    R-squared: {r_squared(y_te_r, mean_baseline):.4f}")

    print("\n=== Learning Curve ===")
    sizes, train_sc, val_sc = learning_curve(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=200),
        metric_fn=accuracy,
    )
    print(f"  {'Size':>6} {'Train':>8} {'Val':>8}")
    for s, tr, va in zip(sizes, train_sc, val_sc):
        print(f"  {s:>6} {tr:>8.4f} {va:>8.4f}")

    print("\n=== Statistical Model Comparison ===")
    model_a_scores = cross_validate(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=100),
        k=5, metric_fn=accuracy,
    )
    model_b_scores = cross_validate(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=500),
        k=5, metric_fn=accuracy,
    )
    diffs = [a - b for a, b in zip(model_a_scores, model_b_scores)]
    mean_diff = sum(diffs) / len(diffs)
    std_diff = math.sqrt(sum((d - mean_diff) ** 2 for d in diffs) / len(diffs))
    t_stat = mean_diff / (std_diff / math.sqrt(len(diffs))) if std_diff > 0 else 0.0
    print(f"  Model A (100 epochs) mean: {sum(model_a_scores)/len(model_a_scores):.4f}")
    print(f"  Model B (500 epochs) mean: {sum(model_b_scores)/len(model_b_scores):.4f}")
    print(f"  Mean difference: {mean_diff:.4f}")
    print(f"  Paired t-statistic: {t_stat:.4f}")
    print(f"  (|t| > 2.78 for significance at p<0.05 with df=4)")
```

## 实际使用

在 scikit-learn 中，评估已经内置在工作流里：

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_squared_error, r2_score,
)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
scores = cross_val_score(model, X, y, cv=StratifiedKFold(5), scoring="f1")
```

这些从零实现的版本能准确展示交叉验证到底做了什么（没有魔法，本质上只是 for 循环和索引跟踪）、每个指标是如何计算的（本质上就是统计 TP/FP/TN/FN），以及为什么分层很重要（在每一折中保留类别比例）。库版本则额外提供了并行化、更多评分选项，以及与 pipeline 的集成。

## 交付成果

本课会产出：
- `outputs/skill-evaluation.md` - 一个关于分类与回归模型评估策略的技能文档

## 练习

1. 实现 precision-recall 曲线：在不同阈值下绘制精确率对召回率的曲线。计算平均精确率（PR 曲线下面积）。在不平衡数据集上比较 PR 曲线和 ROC 曲线，并解释什么时候各自更有信息量。
2. 构建一个嵌套交叉验证（nested cross-validation）循环：外层循环评估模型性能，内层循环调优超参数。用它来公平地比较两个模型，而不会把验证数据泄漏进评估过程。
3. 为模型比较实现一个置换检验（permutation test）：打乱标签、重新训练、测量性能。重复 100 次以建立零分布（null distribution）。然后基于这个分布计算观测到的模型性能对应的 p 值。

## 关键术语

| 术语 | 人们常说的话 | 实际含义 |
|------|--------------|----------|
| 过拟合（Overfitting） | “把训练数据背下来了” | 模型捕捉了训练数据中的噪声，因此在训练集上表现好，但在未见数据上表现差 |
| 交叉验证（Cross-validation） | “在不同子集上测试” | 系统性地轮换哪一部分数据用于验证，并对所有轮换结果求平均 |
| 精确率（Precision） | “预测为正的里有多少是对的” | TP / (TP + FP)：所有正类预测中真正为正的比例 |
| 召回率（Recall） | “我们找到了多少真实正类” | TP / (TP + FN)：所有真实正类中被正确识别出的比例 |
| AUC-ROC | “模型把类别分开的能力” | 在所有阈值下，真正率对假正率曲线的面积，范围从 0.5（随机）到 1.0（完美） |
| R-squared | “解释了多少方差” | 1 -（残差平方和 / 总平方和）：模型捕捉到的目标方差比例 |
| 数据泄漏（Data leakage） | “模型作弊了” | 在训练时使用了预测时本不该可用的信息，从而导致过于乐观的评估结果 |
| 学习曲线（Learning curve） | “更多数据会让性能怎么变” | 一张展示训练/验证分数随训练集大小变化的图，用来揭示欠拟合或过拟合 |
| 分层切分（Stratified split） | “保持类别比例平衡” | 切分数据时让每个子集都与完整数据集拥有相同的类别比例 |

## 延伸阅读

- [scikit-learn Model Selection Guide](https://scikit-learn.org/stable/model_selection.html) - 关于交叉验证、指标和超参数调优的全面参考
- [Beyond Accuracy: Precision and Recall (Google ML Crash Course)](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall) - 带交互示例的清晰讲解
- [A Survey of Cross-Validation Procedures (Arlot & Celisse, 2010)](https://projecteuclid.org/journals/statistics-surveys/volume-4/issue-none/A-survey-of-cross-validation-procedures-for-model-selection/10.1214/09-SS054.full) - 对不同交叉验证策略何时有效、为何有效的严格讨论
