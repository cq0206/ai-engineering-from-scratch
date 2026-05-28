# 处理类别不平衡数据（Imbalanced Data）

> 当你 99% 的数据都是“正常”时，准确率就是个谎言。

**类型：** 构建
**语言：** Python
**前置要求：** 第 2 阶段，第 01-09 课（尤其是评估指标）
**时间：** 约 90 分钟

## 学习目标

- 从零实现 SMOTE（Synthetic Minority Oversampling Technique），并解释合成过采样与随机复制的区别
- 使用 F1、AUPRC 和马修斯相关系数（Matthews Correlation Coefficient）而不是准确率来评估类别不平衡分类器
- 比较类别加权（class weighting）、阈值调优（threshold tuning）和重采样策略，并为给定的不平衡比例选择合适方法
- 构建一个完整的类别不平衡数据处理流水线，结合 SMOTE、类别权重和阈值优化

## 问题

你做了一个欺诈检测模型。它达到了 99.9% 的准确率。你开始庆祝。然后你意识到，它对每一笔交易都预测“不是欺诈”。

这不是程序错误（bug）。当只有 0.1% 的交易是欺诈时，这其实是模型最“理性”的做法。模型学到的是：永远猜多数类，能够让总体错误最小。它在技术上没错，但在实际中完全没用。

只要真正重要的是分类，这种情况就无处不在。疾病诊断：1% 阳性率。网络入侵：0.01% 攻击。制造缺陷：0.5% 次品。垃圾邮件过滤：20% 垃圾邮件。流失预测：5% 流失用户。少数类越重要，它往往就越稀少。

准确率会失效，因为它把所有正确预测一视同仁。正确标记一笔正常交易和正确抓到一次欺诈，在准确率里都只算 1 分。但抓到欺诈才是这个模型存在的全部意义。我们需要能迫使模型关注稀有但重要类别的指标、技术和训练策略。

## 概念

### 为什么准确率会失效

考虑一个有 1000 个样本的数据集：990 个负类，10 个正类。一个永远预测负类的模型会得到：

|  | 预测为正类 | 预测为负类 |
|--|---|---|
| 实际为正类 | 0 (TP) | 10 (FN) |
| 实际为负类 | 0 (FP) | 990 (TN) |

准确率 = (0 + 990) / 1000 = 99.0%

这个模型一个欺诈都抓不到。一个疾病都诊断不出。一个缺陷都发现不了。但准确率却说它有 99%。这就是为什么在不平衡问题上，准确率很危险。

### 更好的指标

**精确率（Precision）** = TP / (TP + FP)。在所有被标记为正类的样本里，真正为正的有多少？精确率高意味着误报少。

**召回率（Recall）** = TP / (TP + FN)。在所有真实为正类的样本里，我们抓到了多少？召回率高意味着漏报少。

**F1 分数（F1 Score）** = 2 * precision * recall / (precision + recall)。它是调和平均数。与算术平均数相比，它会更严厉地惩罚精确率和召回率之间的极端失衡。

**F-beta 分数（F-beta Score）** = (1 + beta^2) * precision * recall / (beta^2 * precision + recall)。当 beta > 1 时，召回率更重要；当 beta &lt; 1 时，精确率更重要。F2 在欺诈检测中很常见（漏掉欺诈比误报更糟）。

**AUPRC**（Precision-Recall 曲线下面积，Area Under Precision-Recall Curve）。它类似 AUC-ROC，但对不平衡数据更有信息量。随机分类器的 AUPRC 等于正类比例（不像 ROC 那样是 0.5）。这让改进更容易看出来。

**马修斯相关系数（Matthews Correlation Coefficient, MCC）** = (TP * TN - FP * FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))。取值范围是 -1 到 +1。只有当模型在两个类别上都表现良好时，它才会得到高分。即使类别规模差异很大，它也依然保持平衡。

对于上面那个“永远预测负类”的模型：precision = 0/0（未定义，通常记为 0），recall = 0/10 = 0，F1 = 0，MCC = 0。这些指标才正确地指出：这个模型毫无价值。

### 类别不平衡数据处理流水线

```mermaid
flowchart TD
    A[不平衡数据集] --> B{不平衡比例？}
    B -->|轻度：80/20| C[类别权重]
    B -->|中度：95/5| D[SMOTE + 阈值调优]
    B -->|严重：99/1| E[SMOTE + 类别权重 + 阈值]
    C --> F[训练模型]
    D --> F
    E --> F
    F --> G[使用 F1 / AUPRC / MCC 评估]
    G --> H{足够好吗？}
    H -->|否| I[尝试不同策略]
    H -->|是| J[带监控部署]
    I --> B
```

### SMOTE：合成少数类过采样技术

随机过采样（random oversampling）会复制已有的少数类样本。这样确实有效，但也有过拟合风险，因为模型会反复看到完全相同的点。

SMOTE 会创建新的、合理的合成少数类样本，而不是简单复制。算法如下：

1. 对每个少数类样本 x，在其他少数类样本中找到它的 k 个最近邻（k nearest neighbors, k-NN）
2. 随机挑选一个邻居
3. 在线段 x 与该邻居之间生成一个新样本

公式是：`new_sample = x + random(0, 1) * (neighbor - x)`

这相当于在真实少数类点之间做插值，在同一片特征空间区域里生成新样本，而不是简单复制现有数据。

```mermaid
flowchart LR
    subgraph Original["原始少数类样本点"]
        P1["x1 (1.0, 2.0)"]
        P2["x2 (1.5, 2.5)"]
        P3["x3 (2.0, 1.5)"]
    end
    subgraph SMOTE["SMOTE 生成过程"]
        direction TB
        S1["选择 x1，邻居 x2"]
        S2["随机 t = 0.4"]
        S3["new = x1 + 0.4*(x2-x1)"]
        S4["new = (1.2, 2.2)"]
        S1 --> S2 --> S3 --> S4
    end
    Original --> SMOTE
    subgraph Result["扩增后的集合"]
        R1["x1 (1.0, 2.0)"]
        R2["x2 (1.5, 2.5)"]
        R3["x3 (2.0, 1.5)"]
        R4["合成样本 (1.2, 2.2)"]
    end
    SMOTE --> Result
```

### 采样策略对比

**随机过采样**：复制少数类样本，使其数量与多数类一致。
- 优点：简单，不会丢失信息
- 缺点：精确重复会导致过拟合，并增加训练时间

**随机欠采样（random undersampling）**：删除多数类样本，使其数量与少数类一致。
- 优点：训练快，简单
- 缺点：会丢掉可能有用的多数类数据，方差更高

**SMOTE**：通过插值生成合成少数类样本。
- 优点：生成新的数据点，相比随机过采样更不容易过拟合
- 缺点：可能会在决策边界附近生成噪声样本，也没有考虑多数类分布

| 策略 | 数据变化 | 风险 | 适用场景 |
|----------|-------------|------|-------------|
| 过采样 | 复制少数类 | 过拟合 | 小数据集、中度不平衡 |
| 欠采样 | 删除多数类 | 信息损失 | 大数据集、希望快速训练 |
| SMOTE | 增加合成少数类 | 边界噪声 | 中度不平衡，且少数类样本足够做 k 最近邻 |

### 类别权重（Class Weights）

与其改动数据，不如改动模型看待错误的方式。给少数类误分类分配更高权重。

对于一个有 950 个负类样本和 50 个正类样本的二分类问题：
- 负类权重 = n_samples / (2 * n_negative) = 1000 / (2 * 950) = 0.526
- 正类权重 = n_samples / (2 * n_positive) = 1000 / (2 * 50) = 10.0

正类获得了 19 倍的权重。错分 1 个正类样本，代价相当于错分 19 个负类样本。模型因此被迫关注少数类。

在逻辑回归（logistic regression）中，这会修改损失函数：

```
weighted_loss = -sum(w_i * [y_i * log(p_i) + (1-y_i) * log(1-p_i)])
```

其中，w_i 取决于样本 i 所属的类别。

从期望意义上说，类别权重与过采样在数学上是等价的，但它不需要创建新数据点。因此它更快，也能避免重复样本带来的过拟合风险。

### 阈值调优（Threshold Tuning）

大多数分类器都会输出一个概率。默认阈值是 0.5：如果 P(positive) >= 0.5，就预测为正类。但 0.5 其实是任意设定的。当类别不平衡时，最优阈值通常会低得多。

流程如下：
1. 训练一个模型
2. 在验证集上拿到预测概率
3. 将阈值从 0.0 扫到 1.0
4. 在每个阈值下计算 F1（或你选择的指标）
5. 选出让指标最大的阈值

```mermaid
flowchart LR
    A[模型] --> B[预测概率]
    B --> C[扫描 0.0 到 1.0 的阈值]
    C --> D[逐个计算 F1]
    D --> E[选出最佳阈值]
    E --> F[在线上使用]
```

一个模型可能会对某笔欺诈交易输出 P(fraud) = 0.15。在阈值 0.5 下，它会被判成非欺诈；而在阈值 0.10 下，它就能被正确抓住。概率校准的重要性，通常不如排序重要——只要欺诈样本的概率普遍高于非欺诈样本，就一定存在一个阈值能把它们分开。

### 代价敏感学习（Cost-Sensitive Learning）

它是类别权重的泛化版本。不再只使用统一代价，而是为不同的误分类指定具体代价：

| | 预测为正类 | 预测为负类 |
|--|---|---|
| 实际为正类 | 0（正确） | C_FN = 100 |
| 实际为负类 | C_FP = 1 | 0（正确） |

漏掉一笔欺诈交易（FN）的代价，是一次误报（FP）的 100 倍。模型优化的目标因此变成总成本，而不是总错误数。

当你能够估计真实世界代价时，这是最有原则的方法。漏诊癌症的代价，和一次导致额外活检的误报，完全不是一回事。把这些代价明确写出来，才能逼迫模型做出正确的权衡。

### 决策流程图

```mermaid
flowchart TD
    A[开始：不平衡数据集] --> B{不平衡程度如何？}
    B -->|"< 70/30"| C["轻度：先尝试类别权重"]
    B -->|"70/30 到 95/5"| D["中度：SMOTE + 类别权重"]
    B -->|"> 95/5"| E["严重：组合多种策略"]
    C --> F{数据量足够吗？}
    D --> F
    E --> F
    F -->|"< 1000 个样本"| G["过采样或 SMOTE，避免欠采样"]
    F -->|"1000-10000"| H["SMOTE + 阈值调优"]
    F -->|"> 10000"| I["可以欠采样，或使用类别权重"]
    G --> J[训练 + 用 F1/AUPRC 评估]
    H --> J
    I --> J
    J --> K{召回率足够高吗？}
    K -->|否| L[降低阈值]
    K -->|是| M{精确率可以接受吗？}
    M -->|否| N[提高阈值或增加特征]
    M -->|是| O[可以上线]
```

## 动手实现

### 第 1 步：生成一个不平衡数据集

```python
import numpy as np


def make_imbalanced_data(n_majority=950, n_minority=50, seed=42):
    rng = np.random.RandomState(seed)

    X_maj = rng.randn(n_majority, 2) * 1.0 + np.array([0.0, 0.0])
    X_min = rng.randn(n_minority, 2) * 0.8 + np.array([2.5, 2.5])

    X = np.vstack([X_maj, X_min])
    y = np.concatenate([np.zeros(n_majority), np.ones(n_minority)])

    shuffle_idx = rng.permutation(len(y))
    return X[shuffle_idx], y[shuffle_idx]
```

### 第 2 步：从零实现 SMOTE

```python
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def find_k_neighbors(X, idx, k):
    distances = []
    for i in range(len(X)):
        if i == idx:
            continue
        d = euclidean_distance(X[idx], X[i])
        distances.append((i, d))
    distances.sort(key=lambda x: x[1])
    return [d[0] for d in distances[:k]]


def smote(X_minority, k=5, n_synthetic=100, seed=42):
    rng = np.random.RandomState(seed)
    n_samples = len(X_minority)
    k = min(k, n_samples - 1)
    synthetic = []

    for _ in range(n_synthetic):
        idx = rng.randint(0, n_samples)
        neighbors = find_k_neighbors(X_minority, idx, k)
        neighbor_idx = neighbors[rng.randint(0, len(neighbors))]
        t = rng.random()
        new_point = X_minority[idx] + t * (X_minority[neighbor_idx] - X_minority[idx])
        synthetic.append(new_point)

    return np.array(synthetic)
```

### 第 3 步：随机过采样与欠采样

```python
def random_oversample(X, y, seed=42):
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    X_resampled = list(X)
    y_resampled = list(y)

    for cls, count in zip(classes, counts):
        if count < max_count:
            cls_indices = np.where(y == cls)[0]
            n_needed = max_count - count
            chosen = rng.choice(cls_indices, size=n_needed, replace=True)
            X_resampled.extend(X[chosen])
            y_resampled.extend(y[chosen])

    X_out = np.array(X_resampled)
    y_out = np.array(y_resampled)
    shuffle = rng.permutation(len(y_out))
    return X_out[shuffle], y_out[shuffle]


def random_undersample(X, y, seed=42):
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()

    X_resampled = []
    y_resampled = []

    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        chosen = rng.choice(cls_indices, size=min_count, replace=False)
        X_resampled.extend(X[chosen])
        y_resampled.extend(y[chosen])

    X_out = np.array(X_resampled)
    y_out = np.array(y_resampled)
    shuffle = rng.permutation(len(y_out))
    return X_out[shuffle], y_out[shuffle]
```

### 第 4 步：带类别权重的逻辑回归

```python
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def logistic_regression_weighted(X, y, weights, lr=0.01, epochs=200):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(epochs):
        z = X @ w + b
        pred = sigmoid(z)
        error = pred - y
        weighted_error = error * weights

        gradient_w = (X.T @ weighted_error) / n_samples
        gradient_b = np.mean(weighted_error)

        w -= lr * gradient_w
        b -= lr * gradient_b

    return w, b


def compute_class_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    weight_map = {}
    for cls, count in zip(classes, counts):
        weight_map[cls] = n_samples / (n_classes * count)
    return np.array([weight_map[yi] for yi in y])
```

### 第 5 步：阈值调优

```python
def find_optimal_threshold(y_true, y_probs, metric="f1"):
    best_threshold = 0.5
    best_score = -1.0

    for threshold in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_probs >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        if metric == "f1":
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        elif metric == "recall":
            score = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        elif metric == "precision":
            score = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score
```

### 第 6 步：评估函数

```python
def confusion_matrix_values(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp, tn, fp, fn


def compute_metrics(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix_values(y_true, y_pred)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
    }
```

### 第 7 步：比较所有方法

```python
X, y = make_imbalanced_data(950, 50, seed=42)
split = int(0.8 * len(y))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Baseline: no treatment
w_base, b_base = logistic_regression_weighted(
    X_train, y_train, np.ones(len(y_train)), lr=0.1, epochs=300
)
probs_base = sigmoid(X_test @ w_base + b_base)
preds_base = (probs_base >= 0.5).astype(int)

# Oversampled
X_over, y_over = random_oversample(X_train, y_train)
w_over, b_over = logistic_regression_weighted(
    X_over, y_over, np.ones(len(y_over)), lr=0.1, epochs=300
)
preds_over = (sigmoid(X_test @ w_over + b_over) >= 0.5).astype(int)

# SMOTE
minority_mask = y_train == 1
X_minority = X_train[minority_mask]
synthetic = smote(X_minority, k=5, n_synthetic=len(y_train) - 2 * int(minority_mask.sum()))
X_smote = np.vstack([X_train, synthetic])
y_smote = np.concatenate([y_train, np.ones(len(synthetic))])
w_sm, b_sm = logistic_regression_weighted(
    X_smote, y_smote, np.ones(len(y_smote)), lr=0.1, epochs=300
)
preds_smote = (sigmoid(X_test @ w_sm + b_sm) >= 0.5).astype(int)

# Class weights
sample_weights = compute_class_weights(y_train)
w_cw, b_cw = logistic_regression_weighted(
    X_train, y_train, sample_weights, lr=0.1, epochs=300
)
probs_cw = sigmoid(X_test @ w_cw + b_cw)
preds_cw = (probs_cw >= 0.5).astype(int)

# Threshold tuning (tune on held-out validation set, not test set)
probs_val = sigmoid(X_val @ w_cw + b_cw)
best_thresh, best_f1 = find_optimal_threshold(y_val, probs_val, metric="f1")
preds_thresh = (probs_cw >= best_thresh).astype(int)
```

代码文件会把这些内容放进一个脚本里运行，并打印结果。

## 实际使用

使用 scikit-learn 和 imbalanced-learn 时，这些技术几乎都是一行代码：

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

model_weighted = LogisticRegression(class_weight="balanced")
model_weighted.fit(X_train, y_train)
print(classification_report(y_test, model_weighted.predict(X_test)))

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
model_smote = LogisticRegression()
model_smote.fit(X_resampled, y_resampled)
print(classification_report(y_test, model_smote.predict(X_test)))

pipeline = Pipeline([
    ("smote", SMOTE()),
    ("model", LogisticRegression(class_weight="balanced")),
])
pipeline.fit(X_train, y_train)
print(classification_report(y_test, pipeline.predict(X_test)))
```

这些从零实现的版本，能精确展示每种技术到底做了什么。SMOTE 本质上只是对少数类做 k-NN 插值。类别权重本质上是在给损失乘权重。阈值调优本质上只是对不同截断值做一层 for 循环（for-loop）。没有魔法。

## 交付成果

本课会产出：
- `outputs/skill-imbalanced-data.md` -- 一个用于处理不平衡分类问题的决策清单

## 练习

1. **Borderline-SMOTE**：修改 SMOTE 实现，使其只为靠近决策边界的少数类点生成合成样本（也就是其 k 个最近邻中包含多数类样本的点）。在一个类别有重叠的数据集上，把结果与标准 SMOTE 进行比较。

2. **代价矩阵优化（Cost Matrix Optimization）**：实现代价敏感学习，让代价矩阵成为一个参数。创建一个函数，接收代价矩阵并返回能最小化期望代价的最优预测。测试不同的代价比（1:10、1:100、1:1000），并绘制精确率-召回率权衡如何变化。

3. **阈值校准（Threshold Calibration）**：实现 Platt scaling（在模型原始输出上再拟合一个逻辑回归，以生成校准后的概率）。比较校准前后的精确率-召回率曲线。证明校准不会改变排序（AUC 保持不变），但会让概率本身更有意义。

4. **结合平衡袋装的集成（Ensemble with Balanced Bagging）**：训练多个模型，每个模型都在一个平衡的自助采样（bootstrap）样本上训练（全部少数类 + 多数类随机子集）。对它们的预测求平均。把这种方法与单个使用 SMOTE 的模型比较，同时衡量性能和多次运行之间的方差。

5. **不平衡比例实验（Imbalance Ratio Experiment）**：从一个平衡数据集出发，逐步提高不平衡比例（50/50、70/30、90/10、95/5、99/1）。对每个比例，分别在使用和不使用 SMOTE 的情况下训练。绘制两种方法的 F1 随不平衡比例变化的曲线。在哪个比例下，SMOTE 开始带来明显收益？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| 类别不平衡 | “某一个类别的样本多得多” | 数据集中各类别的分布明显偏斜，导致模型偏向多数类 |
| SMOTE | “合成过采样” | 通过在现有少数类样本与其 k 个最近少数类邻居之间做插值，创建新的少数类样本 |
| 类别权重 | “让稀有类别的错误更贵” | 给损失函数乘上类别特定权重，让模型对少数类误分类施加更重惩罚 |
| 阈值调优 | “移动决策边界” | 将分类概率的阈值从默认的 0.5 调整为能优化目标指标的数值 |
| 精确率-召回率权衡 | “两者不可兼得” | 降低阈值会抓到更多正类（更高召回率），但也会带来更多假阳性（更低精确率）；反之亦然 |
| AUPRC | “PR 曲线下面积” | 用一个数字总结精确率-召回率曲线；在类别严重不平衡时，它比 AUC-ROC 更有信息量 |
| 马修斯相关系数 | “平衡指标” | 预测标签与真实标签之间的相关系数；只有在模型对两个类别都表现良好时才会得到高分 |
| 代价敏感学习 | “不同错误有不同代价” | 把真实世界中的误分类成本纳入训练目标，让模型优化总成本，而不是错误数量 |
| 随机过采样 | “复制少数类” | 通过重复少数类样本来平衡类别数量；做法简单，但容易对重复点过拟合 |

## 延伸阅读

- [SMOTE: Synthetic Minority Over-sampling Technique (Chawla et al., 2002)](https://arxiv.org/abs/1106.1813) -- SMOTE 的原始论文，至今仍是不平衡学习领域被引用最多的工作之一
- [Learning from Imbalanced Data (He & Garcia, 2009)](https://ieeexplore.ieee.org/document/5128907) -- 一篇全面的综述，涵盖采样、代价敏感和算法层面的各种方法
- [imbalanced-learn documentation](https://imbalanced-learn.org/stable/) -- Python 库文档，包含 SMOTE 变体、欠采样策略和流水线集成
- [The Precision-Recall Plot Is More Informative than the ROC Plot (Saito & Rehmsmeier, 2015)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432) -- 解释在不平衡问题中何时以及为什么应优先使用 PR 曲线而不是 ROC 曲线
