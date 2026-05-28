# 损失函数（Loss Functions）

> 你的网络做出了预测，真实标签却另有说法。差距有多大？这个数字就是损失。选错损失函数，你的模型会完全优化错误的目标。

**类型：** 构建
**语言：** Python
**前置课程：** 第 03.04 课（激活函数）
**预计时间：** 约 75 分钟

## 学习目标

- 从零实现 MSE、二元交叉熵（binary cross-entropy）、类别交叉熵（categorical cross-entropy）和对比损失（contrastive loss，InfoNCE）及其梯度
- 通过演示"对所有样本预测 0.5"的失效模式，说明 MSE 为何不适用于分类任务
- 对交叉熵应用标签平滑（label smoothing），并描述其如何防止过度自信的预测
- 针对回归、二元分类、多类分类和嵌入学习任务选择正确的损失函数

## 问题所在

一个在分类问题上最小化 MSE 的模型，会对所有输入自信地预测 0.5。它在最小化损失，但毫无用处。

损失函数是你的模型实际优化的唯一目标，不是准确率，不是 F1 分数，也不是你向经理汇报的任何指标。优化器对损失函数求梯度，并调整权重使这个数字变小。如果损失函数无法捕捉你真正关心的目标，模型就会找到数学上最廉价的满足方式，而这种方式几乎从来都不是你想要的。

举一个具体例子。你有一个二元分类任务，两个类别各占 50%。你用 MSE 作为损失函数。模型对每一个输入都预测 0.5。平均 MSE 为 0.25，这是在不学习任何内容的情况下能达到的最小值。模型没有任何判别能力，但它在技术上已经最小化了你的损失函数。换成交叉熵，同样的模型会被迫将预测值推向 0 或 1，因为 -log(0.5) = 0.693 是一个糟糕的损失，而 -log(0.99) = 0.01 则奖励置信且正确的预测。损失函数的选择，是模型能学到东西还是在耍花招的关键所在。

情况还会更糟。在自监督学习中，你甚至没有标签。对比损失完全定义了学习信号：什么算相似，什么算不同，以及模型应该以多大力度将它们分开。对比损失设计错误，你的嵌入会坍缩到一个点——每个输入都映射到同一个向量。技术上损失为零，实际上毫无价值。

## 核心概念

### 均方误差（Mean Squared Error，MSE）

回归任务的默认损失函数。计算预测值与目标值之差的平方，再对所有样本取均值。

```
MSE = (1/n) * sum((y_pred - y_true)^2)
```

为什么要平方：它对大误差施加二次惩罚。误差为 2 的代价是误差为 1 的 4 倍，误差为 10 的代价是 100 倍。这使 MSE 对异常值敏感——一个严重错误的预测会主导整个损失。

实际数字：如果你的模型预测房价，在大多数房子上误差为 1 万美元，但在一栋豪宅上误差为 20 万美元，MSE 会积极地尝试修正那栋豪宅，可能损害其他 99 栋房子的性能。

MSE 关于预测值的梯度为：

```
dMSE/dy_pred = (2/n) * (y_pred - y_true)
```

与误差成线性关系。更大的误差得到更大的梯度。这对于回归是优点（大误差需要大幅修正），对于分类是缺点（你希望对置信错误的预测施以指数级惩罚，而非线性惩罚）。

### 交叉熵损失（Cross-Entropy Loss）

分类任务的损失函数，源自信息论——它衡量预测概率分布与真实分布之间的散度。

**二元交叉熵（Binary Cross-Entropy，BCE）：**

```
BCE = -(y * log(p) + (1 - y) * log(1 - p))
```

其中 y 是真实标签（0 或 1），p 是预测概率。

为什么 -log(p) 有效：当真实标签为 1 且你预测 p = 0.99 时，损失为 -log(0.99) = 0.01；当你预测 p = 0.01 时，损失为 -log(0.01) = 4.6。460 倍的差距正是交叉熵有效的原因。它对置信错误的预测给予严厉惩罚，而对置信正确的预测几乎不施加惩罚。

梯度同样说明了这一点：

```
dBCE/dp = -(y/p) + (1-y)/(1-p)
```

当 y = 1 且 p 趋近于零时，梯度为 -1/p，趋向负无穷。模型得到一个巨大的信号来修正错误。当 p 趋近于 1 时，梯度很小。已经正确，无需修正。

**类别交叉熵（Categorical Cross-Entropy）：**

用于具有 one-hot 编码目标的多类分类。

```
CCE = -sum(y_i * log(p_i))
```

只有真实类别对损失有贡献（因为其他所有 y_i 均为零）。如果有 10 个类别，正确类别获得 0.1 的概率（随机猜测），损失为 -log(0.1) = 2.3；如果获得 0.9 的概率，损失为 -log(0.9) = 0.105。模型学会将概率质量集中在正确答案上。

### MSE 为何不适合分类

```mermaid
graph TD
    subgraph "MSE 用于分类"
        P1["预测类别1的概率为0.5<br/>MSE = 0.25"]
        P2["预测类别1的概率为0.9<br/>MSE = 0.01"]
        P3["预测类别1的概率为0.1<br/>MSE = 0.81"]
    end
    subgraph "交叉熵用于分类"
        C1["预测类别1的概率为0.5<br/>CE = 0.693"]
        C2["预测类别1的概率为0.9<br/>CE = 0.105"]
        C3["预测类别1的概率为0.1<br/>CE = 2.303"]
    end
    P3 -->|"MSE 梯度在<br/>饱和附近<br/>趋于平坦"| Slow["修正缓慢"]
    C3 -->|"CE 梯度在<br/>错误答案附近<br/>急剧增大"| Fast["快速修正"]
```

MSE 的梯度在预测值接近 0 或 1 时趋于平坦（由于 sigmoid 饱和）。交叉熵梯度对此进行了补偿——-log 抵消了 sigmoid 的平坦区域，恰好在最需要的地方提供了强梯度。

### 标签平滑（Label Smoothing）

标准的 one-hot 标签表示"这 100% 是第 3 类，其余类别 0%"。这是一个强主张。标签平滑将其软化：

```
smooth_label = (1 - alpha) * one_hot + alpha / num_classes
```

当 alpha = 0.1，共 10 个类别时：目标从 [0, 0, 1, 0, ...] 变为 [0.01, 0.01, 0.91, 0.01, ...]。模型的目标是 0.91 而非 1.0。

为何有效：试图通过 softmax 输出恰好 1.0 的模型需要将 logits 推向无穷大。这会导致过度自信，损害泛化能力，并使模型对分布偏移变得脆弱。标签平滑将目标上限限制在 0.9（alpha=0.1），将 logits 保持在合理范围内。GPT 和大多数现代模型都使用标签平滑或其等价方法。

### 对比损失（Contrastive Loss）

没有标签，没有类别，只有成对的输入和一个问题：这两个相似还是不同？

**SimCLR 风格的对比损失（NT-Xent / InfoNCE）：**

取一张图片，创建它的两个增强视图（裁剪、旋转、颜色抖动）。这两个是"正样本对"——它们应该有相似的嵌入。批次中的其他每张图片构成"负样本对"——它们应该有不同的嵌入。

```
L = -log(exp(sim(z_i, z_j) / tau) / sum(exp(sim(z_i, z_k) / tau)))
```

其中 sim() 是余弦相似度，z_i 和 z_j 是正样本对，求和遍历所有负样本，tau（温度）控制分布的锐利程度。温度越低 = 负样本越难 = 分离越积极。

实际数字：批大小 256 意味着每个正样本对有 255 个负样本。温度 tau = 0.07（SimCLR 默认值）。损失看起来像相似度上的 softmax——它希望正样本对的相似度在 256 个选项中最高。

**三元组损失（Triplet Loss）：**

接收三个输入：锚点（anchor）、正样本（positive，同类）、负样本（negative，异类）。

```
L = max(0, d(anchor, positive) - d(anchor, negative) + margin)
```

margin（通常为 0.2–1.0）强制正负样本距离之间有最小间隔。如果负样本已经足够远，损失为零——没有梯度，没有更新。这使训练高效，但需要仔细的三元组挖掘（选择靠近锚点的困难负样本）。

### 焦点损失（Focal Loss）

用于不平衡数据集。标准交叉熵对所有正确分类的样本一视同仁。焦点损失对简单样本降低权重：

```
FL = -alpha * (1 - p_t)^gamma * log(p_t)
```

其中 p_t 是真实类别的预测概率，gamma 控制聚焦程度。当 gamma = 0 时，这就是标准交叉熵。当 gamma = 2（默认值）时：

- 简单样本（p_t = 0.9）：权重 = (0.1)^2 = 0.01，实际上被忽略。
- 困难样本（p_t = 0.1）：权重 = (0.9)^2 = 0.81，获得完整梯度信号。

焦点损失由 Lin 等人针对目标检测问题提出，其中 99% 的候选区域是背景（简单负样本）。没有焦点损失，模型会被大量简单背景样本淹没，永远学不会检测目标。有了焦点损失，模型将能力集中在重要的困难模糊样本上。

### 损失函数决策树

```mermaid
flowchart TD
    Start["你的任务是什么？"] --> Reg{"回归？"}
    Start --> Cls{"分类？"}
    Start --> Emb{"学习嵌入？"}

    Reg -->|"是"| Outliers{"对异常值敏感？"}
    Outliers -->|"是，惩罚异常值"| MSE["使用 MSE"]
    Outliers -->|"否，对异常值鲁棒"| MAE["使用 MAE / Huber"]

    Cls -->|"二分类"| BCE["使用二元 CE"]
    Cls -->|"多分类"| CCE["使用类别 CE"]
    Cls -->|"不平衡"| FL["使用焦点损失"]
    CCE -->|"过度自信？"| LS["添加标签平滑"]

    Emb -->|"成对数据"| CL["使用对比损失"]
    Emb -->|"有三元组"| TL["使用三元组损失"]
    Emb -->|"大批量自监督"| NCE["使用 InfoNCE"]
```

### 损失曲面

```mermaid
graph LR
    subgraph "损失曲面形状"
        MSE_S["MSE<br/>平滑抛物线<br/>单一最小值<br/>易于优化"]
        CE_S["交叉熵<br/>错误答案附近陡峭<br/>正确答案附近平坦<br/>在需要的地方提供强梯度"]
        CL_S["对比损失<br/>有许多局部最小值<br/>依赖批次构成<br/>温度控制锐利度"]
    end
    MSE_S -->|"最适合"| Reg2["回归"]
    CE_S -->|"最适合"| Cls2["分类"]
    CL_S -->|"最适合"| Emb2["表示学习"]
```

## 动手实现

### 第一步：MSE 及其梯度

```python
def mse(predictions, targets):
    n = len(predictions)
    total = 0.0
    for p, t in zip(predictions, targets):
        total += (p - t) ** 2
    return total / n

def mse_gradient(predictions, targets):
    n = len(predictions)
    grads = []
    for p, t in zip(predictions, targets):
        grads.append(2.0 * (p - t) / n)
    return grads
```

### 第二步：二元交叉熵

log(0) 的问题是真实存在的。如果模型对一个正例预测恰好为 0，log(0) = 负无穷。截断可以防止这个问题。

```python
import math

def binary_cross_entropy(predictions, targets, eps=1e-15):
    n = len(predictions)
    total = 0.0
    for p, t in zip(predictions, targets):
        p_clipped = max(eps, min(1 - eps, p))
        total += -(t * math.log(p_clipped) + (1 - t) * math.log(1 - p_clipped))
    return total / n

def bce_gradient(predictions, targets, eps=1e-15):
    grads = []
    for p, t in zip(predictions, targets):
        p_clipped = max(eps, min(1 - eps, p))
        grads.append(-(t / p_clipped) + (1 - t) / (1 - p_clipped))
    return grads
```

### 第三步：带 Softmax 的类别交叉熵

Softmax 将原始 logits 转换为概率，然后对 one-hot 目标计算交叉熵。

```python
def softmax(logits):
    max_val = max(logits)
    exps = [math.exp(x - max_val) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]

def categorical_cross_entropy(logits, target_index, eps=1e-15):
    probs = softmax(logits)
    p = max(eps, probs[target_index])
    return -math.log(p)

def cce_gradient(logits, target_index):
    probs = softmax(logits)
    grads = list(probs)
    grads[target_index] -= 1.0
    return grads
```

softmax + 交叉熵的梯度化简得非常简洁：对真实类别为（预测概率 - 1），对所有其他类别为（预测概率）。这种优雅的化简不是偶然——这正是 softmax 与交叉熵配对使用的原因。

### 第四步：标签平滑

```python
def label_smoothed_cce(logits, target_index, num_classes, alpha=0.1, eps=1e-15):
    probs = softmax(logits)
    loss = 0.0
    for i in range(num_classes):
        if i == target_index:
            smooth_target = 1.0 - alpha + alpha / num_classes
        else:
            smooth_target = alpha / num_classes
        p = max(eps, probs[i])
        loss += -smooth_target * math.log(p)
    return loss
```

### 第五步：对比损失（简化版 InfoNCE）

```python
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return dot / (norm_a * norm_b)

def contrastive_loss(anchor, positive, negatives, temperature=0.07):
    sim_pos = cosine_similarity(anchor, positive) / temperature
    sim_negs = [cosine_similarity(anchor, neg) / temperature for neg in negatives]

    max_sim = max(sim_pos, max(sim_negs)) if sim_negs else sim_pos
    exp_pos = math.exp(sim_pos - max_sim)
    exp_negs = [math.exp(s - max_sim) for s in sim_negs]
    total_exp = exp_pos + sum(exp_negs)

    return -math.log(max(1e-15, exp_pos / total_exp))
```

### 第六步：MSE vs 交叉熵在分类上的对比

使用第 04 课的圆形数据集，用两种损失函数训练相同的网络。观察交叉熵收敛更快的过程。

```python
import random

def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))

def make_circle_data(n=200, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        data.append(([x, y], label))
    return data


class LossComparisonNetwork:
    def __init__(self, loss_type="bce", hidden_size=8, lr=0.1):
        random.seed(0)
        self.loss_type = loss_type
        self.lr = lr
        self.hidden_size = hidden_size

        self.w1 = [[random.gauss(0, 0.5) for _ in range(2)] for _ in range(hidden_size)]
        self.b1 = [0.0] * hidden_size
        self.w2 = [random.gauss(0, 0.5) for _ in range(hidden_size)]
        self.b2 = 0.0

    def forward(self, x):
        self.x = x
        self.z1 = []
        self.h = []
        for i in range(self.hidden_size):
            z = self.w1[i][0] * x[0] + self.w1[i][1] * x[1] + self.b1[i]
            self.z1.append(z)
            self.h.append(max(0.0, z))

        self.z2 = sum(self.w2[i] * self.h[i] for i in range(self.hidden_size)) + self.b2
        self.out = sigmoid(self.z2)
        return self.out

    def backward(self, target):
        if self.loss_type == "mse":
            d_loss = 2.0 * (self.out - target)
        else:
            eps = 1e-15
            p = max(eps, min(1 - eps, self.out))
            d_loss = -(target / p) + (1 - target) / (1 - p)

        d_sigmoid = self.out * (1 - self.out)
        d_out = d_loss * d_sigmoid

        for i in range(self.hidden_size):
            d_relu = 1.0 if self.z1[i] > 0 else 0.0
            d_h = d_out * self.w2[i] * d_relu
            self.w2[i] -= self.lr * d_out * self.h[i]
            for j in range(2):
                self.w1[i][j] -= self.lr * d_h * self.x[j]
            self.b1[i] -= self.lr * d_h
        self.b2 -= self.lr * d_out

    def compute_loss(self, pred, target):
        if self.loss_type == "mse":
            return (pred - target) ** 2
        else:
            eps = 1e-15
            p = max(eps, min(1 - eps, pred))
            return -(target * math.log(p) + (1 - target) * math.log(1 - p))

    def train(self, data, epochs=200):
        losses = []
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            for x, y in data:
                pred = self.forward(x)
                self.backward(y)
                total_loss += self.compute_loss(pred, y)
                if (pred >= 0.5) == (y >= 0.5):
                    correct += 1
            avg_loss = total_loss / len(data)
            accuracy = correct / len(data) * 100
            losses.append((avg_loss, accuracy))
            if epoch % 50 == 0 or epoch == epochs - 1:
                print(f"    Epoch {epoch:3d}: loss={avg_loss:.4f}, accuracy={accuracy:.1f}%")
        return losses
```

## 实际应用

PyTorch 内置了所有标准损失函数，并具有数值稳定性：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

predictions = torch.tensor([0.9, 0.1, 0.7], requires_grad=True)
targets = torch.tensor([1.0, 0.0, 1.0])

mse_loss = F.mse_loss(predictions, targets)
bce_loss = F.binary_cross_entropy(predictions, targets)

logits = torch.randn(4, 10)
labels = torch.tensor([3, 7, 1, 9])
ce_loss = F.cross_entropy(logits, labels)
ce_smooth = F.cross_entropy(logits, labels, label_smoothing=0.1)
```

使用 `F.cross_entropy`（而非手动 softmax 后再用 `F.nll_loss`）。它将 log-softmax 和负对数似然合并为一个数值稳定的操作。先单独应用 softmax 再取对数稳定性较差——大指数相减会损失精度。

对于对比学习，大多数团队使用自定义实现或 `lightly`、`pytorch-metric-learning` 等库。核心循环始终相同：计算成对相似度，对正负样本创建 softmax，反向传播。

## 产出物

本课程产出：
- `outputs/prompt-loss-function-selector.md` —— 一个可复用的提示词，用于选择正确的损失函数
- `outputs/prompt-loss-debugger.md` —— 当损失曲线看起来异常时的诊断提示词

## 练习

1. 实现 Huber 损失（平滑 L1 损失），对小误差使用 MSE，对大误差使用 MAE。在预测 y = sin(x) 的回归网络上，将 5% 的训练目标加入随机噪声（异常值），分别用 MSE 和 Huber 训练，比较最终测试误差。

2. 将焦点损失加入二元分类训练循环。创建一个不平衡数据集（90% 第 0 类，10% 第 1 类）。对比标准 BCE 与焦点损失（gamma=2）在 200 轮训练后少数类召回率的差异。

3. 实现带半难负样本挖掘的三元组损失。为 5 个类别生成二维嵌入数据。对于每个锚点，找到比正样本更远但仍然最近的困难负样本（半难）。比较与随机三元组选择的收敛速度。

4. 运行 MSE vs 交叉熵的对比，但同时追踪每层训练过程中的梯度幅度。绘制每轮训练的平均梯度范数。验证交叉熵在模型最不确定的早期轮次中产生更大的梯度。

5. 实现 KL 散度损失，验证当真实分布为 one-hot 时，最小化 KL(true || predicted) 与交叉熵给出相同梯度。然后尝试软目标（如知识蒸馏中），其中"真实"分布来自教师模型的 softmax 输出。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 损失函数（Loss function） | "模型的错误程度" | 将预测值和目标值映射为一个标量的可微函数，优化器最小化该值 |
| MSE | "平均平方误差" | 预测值与目标值之差的平方均值；对大误差施以二次惩罚 |
| 交叉熵（Cross-entropy） | "分类损失" | 用 -log(p) 衡量预测概率分布与真实分布之间的散度 |
| 二元交叉熵（Binary cross-entropy） | "BCE" | 两类别的交叉熵：-(y*log(p) + (1-y)*log(1-p)) |
| 标签平滑（Label smoothing） | "软化目标" | 将硬 0/1 目标替换为软值（如 0.1/0.9），防止过度自信，提升泛化能力 |
| 对比损失（Contrastive loss） | "拉近推远" | 通过使相似对在嵌入空间中更近、不相似对更远来学习表示的损失函数 |
| InfoNCE | "CLIP/SimCLR 损失" | 在相似度分数上进行归一化温度缩放的交叉熵；将对比学习视为分类问题 |
| 焦点损失（Focal loss） | "不平衡数据的解决方案" | 以 (1-p_t)^gamma 加权的交叉熵，对简单样本降权，聚焦于困难样本 |
| 三元组损失（Triplet loss） | "锚点-正样本-负样本" | 在嵌入空间中，将锚点推向比负样本至少有一个 margin 间隔的正样本 |
| 温度（Temperature） | "锐利度旋钮" | 作用于 logits/相似度的标量除数，控制结果分布的峰锐程度；越低越尖锐 |

## 延伸阅读

- Lin et al., "Focal Loss for Dense Object Detection" (2017) -- 针对目标检测中极端类别不平衡引入焦点损失（RetinaNet）
- Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR, 2020) -- 定义了使用 NT-Xent 损失的现代对比学习流程
- Szegedy et al., "Rethinking the Inception Architecture" (2016) -- 将标签平滑作为正则化技术引入，现已成为大多数大型模型的标准
- Hinton et al., "Distilling the Knowledge in a Neural Network" (2015) -- 使用软目标和 KL 散度的知识蒸馏，模型压缩的基础工作
