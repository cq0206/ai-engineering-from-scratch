# 无监督学习 (Unsupervised Learning)

> 没有标签，没有老师。算法会自行发现结构。

**类型：** 构建
**语言：** Python
**前置知识：** 第 1 阶段（范数与距离、概率与分布），第 2 阶段第 1-6 课
**时间：** ~90 分钟

## 学习目标

- 从零实现 K 均值聚类 (K-Means)、DBSCAN 和高斯混合模型 (Gaussian Mixture Models, GMM)，并比较它们的聚类行为
- 使用轮廓系数 (silhouette score) 和肘部法 (elbow method) 评估簇质量并选择最优 K
- 解释何时 DBSCAN 优于 K-Means，并识别哪种算法能处理非球形簇和离群点
- 构建一个利用聚类方法的异常检测 (anomaly detection) 流水线，用于标记偏离正常模式的点

## 问题

到目前为止，每一节机器学习课程都假设数据带有标签：“这是输入，这是正确输出。”但在真实世界里，标签很昂贵。医院有数百万条病人记录，却没人手动为每条记录标注疾病类别。电商网站有数百万次用户会话，却没人手工标好客户分群。安全团队有网络日志，但没人标记出每一个异常。

无监督学习会在没人告诉它该找什么的情况下发现模式。它会把相似的数据点分组，发现隐藏结构，并暴露异常。如果说监督学习像是在看一本附带答案的教材，那么无监督学习就像盯着原始数据，直到模式自己显现出来。

难点在于：没有标签，你就无法直接衡量“对”还是“错”。你需要使用不同的工具来评估算法找到的结构是否真的有意义。

## 概念

### 聚类 (Clustering)：把相似的东西放在一起

聚类会把每个数据点分配到一个组（簇）中，使同一组内的点彼此比与其他组的点更相似。问题始终是：这里的“相似”到底是什么意思？

```mermaid
flowchart LR
    A[原始数据] --> B{选择方法}
    B --> C[K-Means]
    B --> D[DBSCAN]
    B --> E[层次聚类]
    B --> F[GMM]
    C --> G[扁平的球形簇]
    D --> H[任意形状与噪声检测]
    E --> I[嵌套簇的树结构]
    F --> J[软分配与椭圆簇]
```

### K 均值聚类 (K-Means)：主力方法

K-Means 会把数据划分为恰好 K 个簇。每个簇都有一个质心 (centroid)——也就是它的质量中心，而每个点都会归属于最近的质心。

Lloyd 算法：

1. 随机选取 K 个点作为初始质心
2. 将每个数据点分配给最近的质心
3. 将每个质心重新计算为其所属点的均值
4. 重复步骤 2-3，直到分配结果不再变化

目标函数（惯性，inertia）衡量的是每个点到其所属质心的总平方距离。K-Means 会最小化这个值，但只能找到局部最小值。不同的初始化方式可能得到不同结果。

### 选择 K

有两种标准方法：

**肘部法：** 对 K = 1, 2, 3, ..., n 运行 K-Means。绘制 inertia 与 K 的关系图。寻找那个“肘部”——也就是继续增加簇数后，inertia 的下降幅度不再明显的位置。

**轮廓系数：** 对每个点，衡量它与自身簇的相似程度 (a) 和与最近其他簇的相似程度 (b)。轮廓系数为 (b - a) / max(a, b)，取值范围从 -1（分错簇）到 +1（聚得很好）。把所有点的分数取平均，就得到全局评分。

### DBSCAN：基于密度的聚类

K-Means 假设簇是球形的，并且要求你预先选定 K。DBSCAN 两者都不假设。它把簇视为由稠密区域构成，并由稀疏区域彼此分隔。

两个参数：
- **eps**：邻域半径
- **min_samples**：形成稠密区域所需的最少点数

三类点：
- **核心点 (core point)：** 在 eps 距离内至少有 min_samples 个点
- **边界点 (border point)：** 位于某个核心点的 eps 范围内，但自身不是核心点
- **噪声点 (noise point)：** 既不是核心点也不是边界点。这些就是离群点。

DBSCAN 会把彼此距离在 eps 以内的核心点连接为同一个簇。边界点会加入附近某个核心点所在的簇。噪声点不属于任何簇。

优点：能发现任意形状的簇、自动确定簇的数量、识别离群点。缺点：面对密度差异很大的簇时表现较差。

### 层次聚类 (Hierarchical Clustering)

它会构建一棵由嵌套簇组成的树（树状图，dendrogram）。

凝聚式（自底向上）：
1. 从每个点各自作为一个簇开始
2. 合并距离最近的两个簇
3. 重复，直到只剩下一个簇
4. 在想要的层级切开树状图，得到 K 个簇

簇之间的“接近程度”可以这样衡量：
- **单链接 (single linkage)：** 两个簇中任意两点之间的最小距离
- **全链接 (complete linkage)：** 任意两点之间的最大距离
- **平均链接 (average linkage)：** 所有点对距离的平均值
- **Ward 方法 (Ward's method)：** 选择会使簇内总方差增幅最小的合并方式

### 高斯混合模型 (Gaussian Mixture Models, GMM)

K-Means 给出的是硬分配：每个点只属于一个簇。GMM 给出的是软分配 (soft assignments)：每个点属于每个簇都有一个概率。

GMM 假设数据由 K 个高斯分布混合生成，每个分布都有自己的均值和协方差。期望最大化算法 (Expectation-Maximization, EM) 会在以下两步之间交替：

- **E 步 (E-step)：** 计算每个点属于每个高斯分布的概率
- **M 步 (M-step)：** 更新每个高斯分布的均值、协方差和混合权重，以最大化数据似然

GMM 可以建模椭圆形簇（不像 K-Means 只适合球形簇），也能自然处理重叠簇。

### 何时使用哪一种

| 方法 | 最适合 | 避免使用的场景 |
|--------|----------|------------|
| K-Means | 大型数据集、球形簇、已知 K | 形状不规则、存在离群点 |
| DBSCAN | K 未知、任意形状、离群点检测 | 密度差异大、维度非常高 |
| 层次聚类 | 小型数据集、需要树状图、K 未知 | 大型数据集（O(n^2) 内存） |
| GMM | 重叠簇、需要软分配 | 非常大的数据集、维度过多 |

### 利用聚类做异常检测

聚类天然支持异常检测：
- **K-Means：** 离任何质心都很远的点就是异常
- **DBSCAN：** 噪声点按定义就是异常
- **GMM：** 在所有高斯分布下概率都很低的点就是异常

## 动手实现

### 第 1 步：从零实现 K-Means

```python
import math
import random


def euclidean_distance(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def kmeans(data, k, max_iterations=100, seed=42):
    random.seed(seed)
    n_features = len(data[0])

    centroids = random.sample(data, k)

    for iteration in range(max_iterations):
        clusters = [[] for _ in range(k)]
        assignments = []

        for point in data:
            distances = [euclidean_distance(point, c) for c in centroids]
            nearest = distances.index(min(distances))
            clusters[nearest].append(point)
            assignments.append(nearest)

        new_centroids = []
        for cluster in clusters:
            if len(cluster) == 0:
                new_centroids.append(random.choice(data))
                continue
            centroid = [
                sum(point[j] for point in cluster) / len(cluster)
                for j in range(n_features)
            ]
            new_centroids.append(centroid)

        if all(
            euclidean_distance(old, new) < 1e-6
            for old, new in zip(centroids, new_centroids)
        ):
            print(f"  Converged at iteration {iteration + 1}")
            break

        centroids = new_centroids

    return assignments, centroids
```

### 第 2 步：肘部法和轮廓系数

```python
def compute_inertia(data, assignments, centroids):
    total = 0.0
    for point, cluster_id in zip(data, assignments):
        total += euclidean_distance(point, centroids[cluster_id]) ** 2
    return total


def silhouette_score(data, assignments):
    n = len(data)
    if n < 2:
        return 0.0

    clusters = {}
    for i, c in enumerate(assignments):
        clusters.setdefault(c, []).append(i)

    if len(clusters) < 2:
        return 0.0

    scores = []
    for i in range(n):
        own_cluster = assignments[i]
        own_members = [j for j in clusters[own_cluster] if j != i]

        if len(own_members) == 0:
            scores.append(0.0)
            continue

        a = sum(euclidean_distance(data[i], data[j]) for j in own_members) / len(own_members)

        b = float("inf")
        for cluster_id, members in clusters.items():
            if cluster_id == own_cluster:
                continue
            avg_dist = sum(euclidean_distance(data[i], data[j]) for j in members) / len(members)
            b = min(b, avg_dist)

        if max(a, b) == 0:
            scores.append(0.0)
        else:
            scores.append((b - a) / max(a, b))

    return sum(scores) / len(scores)


def find_best_k(data, max_k=10):
    print("Elbow method:")
    inertias = []
    for k in range(1, max_k + 1):
        assignments, centroids = kmeans(data, k)
        inertia = compute_inertia(data, assignments, centroids)
        inertias.append(inertia)
        print(f"  K={k}: inertia={inertia:.2f}")

    print("\nSilhouette scores:")
    for k in range(2, max_k + 1):
        assignments, centroids = kmeans(data, k)
        score = silhouette_score(data, assignments)
        print(f"  K={k}: silhouette={score:.4f}")

    return inertias
```

### 第 3 步：从零实现 DBSCAN

```python
def dbscan(data, eps, min_samples):
    n = len(data)
    labels = [-1] * n
    cluster_id = 0

    def region_query(point_idx):
        neighbors = []
        for i in range(n):
            if euclidean_distance(data[point_idx], data[i]) <= eps:
                neighbors.append(i)
        return neighbors

    visited = [False] * n

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        neighbors = region_query(i)

        if len(neighbors) < min_samples:
            labels[i] = -1
            continue

        labels[i] = cluster_id
        seed_set = list(neighbors)
        seed_set.remove(i)

        j = 0
        while j < len(seed_set):
            q = seed_set[j]

            if not visited[q]:
                visited[q] = True
                q_neighbors = region_query(q)
                if len(q_neighbors) >= min_samples:
                    for nb in q_neighbors:
                        if nb not in seed_set:
                            seed_set.append(nb)

            if labels[q] == -1:
                labels[q] = cluster_id

            j += 1

        cluster_id += 1

    return labels
```

### 第 4 步：高斯混合模型（EM 算法）

```python
def gmm(data, k, max_iterations=100, seed=42):
    random.seed(seed)
    n = len(data)
    d = len(data[0])

    indices = random.sample(range(n), k)
    means = [list(data[i]) for i in indices]
    variances = [1.0] * k
    weights = [1.0 / k] * k

    def gaussian_pdf(x, mean, variance):
        d = len(x)
        coeff = 1.0 / ((2 * math.pi * variance) ** (d / 2))
        exponent = -sum((xi - mi) ** 2 for xi, mi in zip(x, mean)) / (2 * variance)
        return coeff * math.exp(max(exponent, -500))

    for iteration in range(max_iterations):
        responsibilities = []
        for i in range(n):
            probs = []
            for j in range(k):
                probs.append(weights[j] * gaussian_pdf(data[i], means[j], variances[j]))
            total = sum(probs)
            if total == 0:
                total = 1e-300
            responsibilities.append([p / total for p in probs])

        old_means = [list(m) for m in means]

        for j in range(k):
            r_sum = sum(responsibilities[i][j] for i in range(n))
            if r_sum < 1e-10:
                continue

            weights[j] = r_sum / n

            for dim in range(d):
                means[j][dim] = sum(
                    responsibilities[i][j] * data[i][dim] for i in range(n)
                ) / r_sum

            variances[j] = sum(
                responsibilities[i][j]
                * sum((data[i][dim] - means[j][dim]) ** 2 for dim in range(d))
                for i in range(n)
            ) / (r_sum * d)
            variances[j] = max(variances[j], 1e-6)

        shift = sum(
            euclidean_distance(old_means[j], means[j]) for j in range(k)
        )
        if shift < 1e-6:
            print(f"  GMM converged at iteration {iteration + 1}")
            break

    assignments = []
    for i in range(n):
        assignments.append(responsibilities[i].index(max(responsibilities[i])))

    return assignments, means, weights, responsibilities
```

### 第 5 步：生成测试数据并运行所有内容

```python
def make_blobs(centers, n_per_cluster=50, spread=0.5, seed=42):
    random.seed(seed)
    data = []
    true_labels = []
    for label, (cx, cy) in enumerate(centers):
        for _ in range(n_per_cluster):
            x = cx + random.gauss(0, spread)
            y = cy + random.gauss(0, spread)
            data.append([x, y])
            true_labels.append(label)
    return data, true_labels


def make_moons(n_samples=200, noise=0.1, seed=42):
    random.seed(seed)
    data = []
    labels = []
    n_half = n_samples // 2
    for i in range(n_half):
        angle = math.pi * i / n_half
        x = math.cos(angle) + random.gauss(0, noise)
        y = math.sin(angle) + random.gauss(0, noise)
        data.append([x, y])
        labels.append(0)
    for i in range(n_half):
        angle = math.pi * i / n_half
        x = 1 - math.cos(angle) + random.gauss(0, noise)
        y = 1 - math.sin(angle) - 0.5 + random.gauss(0, noise)
        data.append([x, y])
        labels.append(1)
    return data, labels


if __name__ == "__main__":
    centers = [[2, 2], [8, 3], [5, 8]]
    data, true_labels = make_blobs(centers, n_per_cluster=50, spread=0.8)

    print("=== K-Means on 3 blobs ===")
    assignments, centroids = kmeans(data, k=3)
    print(f"  Centroids: {[[round(c, 2) for c in cent] for cent in centroids]}")
    sil = silhouette_score(data, assignments)
    print(f"  Silhouette score: {sil:.4f}")

    print("\n=== Elbow Method ===")
    find_best_k(data, max_k=6)

    print("\n=== DBSCAN on 3 blobs ===")
    db_labels = dbscan(data, eps=1.5, min_samples=5)
    n_clusters = len(set(db_labels) - {-1})
    n_noise = db_labels.count(-1)
    print(f"  Found {n_clusters} clusters, {n_noise} noise points")

    print("\n=== GMM on 3 blobs ===")
    gmm_assignments, gmm_means, gmm_weights, _ = gmm(data, k=3)
    print(f"  Means: {[[round(m, 2) for m in mean] for mean in gmm_means]}")
    print(f"  Weights: {[round(w, 3) for w in gmm_weights]}")
    gmm_sil = silhouette_score(data, gmm_assignments)
    print(f"  Silhouette score: {gmm_sil:.4f}")

    print("\n=== DBSCAN on moons (non-spherical clusters) ===")
    moon_data, moon_labels = make_moons(n_samples=200, noise=0.1)
    moon_db = dbscan(moon_data, eps=0.3, min_samples=5)
    n_moon_clusters = len(set(moon_db) - {-1})
    n_moon_noise = moon_db.count(-1)
    print(f"  Found {n_moon_clusters} clusters, {n_moon_noise} noise points")

    print("\n=== K-Means on moons (will fail to separate) ===")
    moon_km, moon_centroids = kmeans(moon_data, k=2)
    moon_sil = silhouette_score(moon_data, moon_km)
    print(f"  Silhouette score: {moon_sil:.4f}")
    print("  K-Means splits moons poorly because they are not spherical")

    print("\n=== Anomaly detection with DBSCAN ===")
    anomaly_data = list(data)
    anomaly_data.append([20.0, 20.0])
    anomaly_data.append([-5.0, -5.0])
    anomaly_data.append([15.0, 0.0])
    anomaly_labels = dbscan(anomaly_data, eps=1.5, min_samples=5)
    anomalies = [
        anomaly_data[i]
        for i in range(len(anomaly_labels))
        if anomaly_labels[i] == -1
    ]
    print(f"  Detected {len(anomalies)} anomalies")
    for a in anomalies[-3:]:
        print(f"    Point {[round(v, 2) for v in a]}")
```

## 实际使用

使用 scikit-learn 时，这些算法都只需要一行：

```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score as sklearn_silhouette

km = KMeans(n_clusters=3, random_state=42).fit(data)
db = DBSCAN(eps=1.5, min_samples=5).fit(data)
agg = AgglomerativeClustering(n_clusters=3).fit(data)
gmm_model = GaussianMixture(n_components=3, random_state=42).fit(data)
```

这些从零实现的版本会让你清楚看到这些库到底计算了什么。K-Means 在“分配”和“重算”之间迭代。DBSCAN 从高密度种子扩展出簇。GMM 在期望与最大化之间交替。库版本则加入了数值稳定性、更聪明的初始化方式（K-Means++）以及 GPU 加速，但核心逻辑是一样的。

## 交付成果

本课会产出 K-Means、DBSCAN 和 GMM 的可运行从零实现版本。这些聚类代码可以作为更高级无监督方法的基础继续复用。

## 练习

1. 实现 K-Means++ 初始化：不要随机挑选所有质心，而是先随机选第一个质心，之后每个质心按其到最近已有质心的平方距离成比例的概率选取。将其收敛速度与随机初始化进行比较。
2. 在代码中加入层次凝聚聚类。实现 Ward 链接，并生成一个树状图（用嵌套合并列表表示）。在不同层级切开它，并与 K-Means 的结果比较。
3. 构建一个简单的异常检测流水线：在同一份数据上运行 DBSCAN 和 GMM，把两个方法都认定为离群点的样本标记出来（DBSCAN 中的噪声点、GMM 中的低概率点）。衡量它们的重叠程度，并讨论两种方法何时会出现分歧。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| 聚类 | “把相似的东西分在一起” | 按某种特定距离度量把数据划分为多个子集，使组内相似性高于组间相似性 |
| 质心 | “一个簇的中心” | 分配到某个簇的所有点的均值；K-Means 用它作为簇的代表 |
| 惯性 | “簇有多紧密” | 每个点到其所属质心的平方距离之和；越小表示越紧密 |
| 轮廓系数 | “簇彼此分得有多开” | 对每个点，(b - a) / max(a, b)，其中 a 是簇内平均距离，b 是到最近其他簇的平均距离 |
| 核心点 | “稠密区域里的点” | 在 DBSCAN 中，eps 距离内至少有 min_samples 个邻居的点 |
| EM 算法 | “软版 K-Means” | 期望最大化：迭代计算成员归属概率（E 步）并更新分布参数（M 步） |
| 树状图 | “簇的树” | 展示层次聚类中簇按什么顺序、在什么距离上被合并的树形图 |
| 异常点 | “离群点” | 不符合预期模式的数据点，在 DBSCAN 中表现为噪声，在 GMM 中表现为低概率点 |

## 延伸阅读

- [Stanford CS229 - Unsupervised Learning](https://cs229.stanford.edu/notes2022fall/main_notes.pdf) - Andrew Ng 关于聚类和 EM 的课程讲义
- [scikit-learn Clustering Guide](https://scikit-learn.org/stable/modules/clustering.html) - 各类聚类算法的实用对比与可视化示例
- [DBSCAN original paper (Ester et al., 1996)](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf) - 提出基于密度聚类方法的经典论文
