# 面向机器学习的图论

> 图（graph）是关系的数据结构。如果你的数据有连接，你就需要图论。

**类型：** 构建（Build）
**语言：** Python
**前置条件：** 第一阶段，第 01–03 课（线性代数、矩阵）
**时长：** ~90 分钟

## 学习目标

- 构建带有邻接矩阵/邻接表表示的图类，并实现 BFS 和 DFS 遍历
- 计算图拉普拉斯矩阵（graph Laplacian），并用其特征值检测连通分量和聚类节点
- 将一轮 GNN 风格的消息传递实现为归一化邻接矩阵的乘法
- 利用 Fiedler 向量（Fiedler vector）通过谱聚类对图进行分割

## 问题

社交网络、分子、知识库、引文网络、路网——所有这些都是图。传统机器学习将数据视为平铺的表格：每行独立，每个特征是一列。但当连接结构本身携带信息时，表格就失败了。

以社交网络为例。你想预测用户会购买哪种产品。用户的购买历史有用，但朋友的购买历史更重要。连接本身携带信号。

再以分子为例。你想预测它是否与某种蛋白质结合。原子种类重要，但更重要的是原子之间的键合方式。结构本身就是数据。

图神经网络（GNN, Graph Neural Network）是深度学习中增长最快的领域，驱动着药物发现、社交推荐、欺诈检测和知识图谱推理。每个 GNN 都建立在相同的基础上：基础图论。

你需要四样东西：
1. 将图表示为矩阵的方法（以便可以相乘）
2. 探索图结构的遍历算法
3. 拉普拉斯矩阵——谱图论中最重要的矩阵
4. 消息传递——使 GNN 发挥作用的操作

## 概念

### 图：节点与边

图（graph）G = (V, E) 由顶点（节点，vertices/nodes）V 和边（edges）E 组成，每条边连接两个节点。

**有向图与无向图。** 在无向图中，边 (u, v) 表示 u 连接到 v 且 v 也连接到 u。在有向图（digraph）中，边 (u, v) 表示 u 指向 v，但反向不一定成立。

**加权图与无权图。** 在无权图中，边要么存在要么不存在。在加权图中，每条边有一个数值权重——距离、代价或强度。

| 图类型 | 示例 |
|-----------|---------|
| 无向无权图 | Facebook 好友关系网络 |
| 有向无权图 | Twitter 关注网络 |
| 无向加权图 | 路网（距离） |
| 有向加权图 | 网页链接（PageRank 分数） |

### 邻接矩阵

邻接矩阵（adjacency matrix）A 是核心表示。对于有 n 个节点的图：

```
A[i][j] = 1    if there is an edge from node i to node j
A[i][j] = 0    otherwise
```

对于无向图，A 是对称的：A[i][j] = A[j][i]。对于加权图，A[i][j] = 边 (i, j) 的权重。

**示例——三角形：**

```
Nodes: 0, 1, 2
Edges: (0,1), (1,2), (0,2)

A = [[0, 1, 1],
     [1, 0, 1],
     [1, 1, 0]]
```

邻接矩阵是每个 GNN 的输入。对 A 的矩阵运算对应图上的操作。

### 度

节点的度（degree）是与其相连的边数。对于有向图，分为入度（in-degree，指向该节点的边数）和出度（out-degree，从该节点出发的边数）。

度矩阵（degree matrix）D 是对角矩阵：

```
D[i][i] = degree of node i
D[i][j] = 0    for i != j
```

三角形示例中：D = diag(2, 2, 2)，因为每个节点与其他两个节点相连。

度反映节点的重要性。高度数 = 枢纽节点（hub node）。网络的度分布揭示其结构。社交网络遵循幂律分布（少数枢纽，大量叶节点），随机图具有泊松分布的度。

### BFS 和 DFS

两种基本的图遍历算法，两者都需要掌握。

**广度优先搜索（BFS, Breadth-First Search）：** 先探索所有邻居，再探索邻居的邻居。使用队列（FIFO）。

```
BFS from node 0:
  Visit 0
  Queue: [1, 2]        (neighbors of 0)
  Visit 1
  Queue: [2, 3]        (add neighbors of 1)
  Visit 2
  Queue: [3]           (neighbors of 2 already visited)
  Visit 3
  Queue: []            (done)
```

BFS 在无权图中找最短路径。从起点到任意节点的距离等于 BFS 中首次发现该节点的层级。这就是 BFS 用于社交网络跳数距离的原因。

**深度优先搜索（DFS, Depth-First Search）：** 尽可能深入再回溯。使用栈（LIFO）或递归。

```
DFS from node 0:
  Visit 0
  Stack: [1, 2]        (neighbors of 0)
  Visit 2               (pop from stack)
  Stack: [1, 3]         (add neighbors of 2)
  Visit 3               (pop from stack)
  Stack: [1]
  Visit 1               (pop from stack)
  Stack: []             (done)
```

DFS 适用于：
- 查找连通分量（对未访问节点运行 DFS）
- 环检测（DFS 树中的回边）
- 拓扑排序（DFS 完成时间的逆序）

| 算法 | 数据结构 | 查找 | 适用场景 |
|-----------|---------------|-------|----------|
| BFS | 队列 | 最短路径 | 社交网络距离、知识图谱遍历 |
| DFS | 栈 | 连通分量、环 | 连通性、拓扑排序 |

### 图拉普拉斯矩阵

L = D - A。谱图论中最重要的矩阵。

对于三角形：

```
D = [[2, 0, 0],    A = [[0, 1, 1],    L = [[2, -1, -1],
     [0, 2, 0],         [1, 0, 1],         [-1, 2, -1],
     [0, 0, 2]]         [1, 1, 0]]         [-1, -1,  2]]
```

拉普拉斯矩阵具有出色的性质：

1. **L 是正半定的。** 所有特征值 >= 0。

2. **零特征值的数量等于连通分量的数量。** 连通图恰好有一个零特征值；有 3 个不连通分量的图有三个零特征值。

3. **最小非零特征值（Fiedler 值）衡量连通性。** Fiedler 值大意味着图连通性好；Fiedler 值小意味着图有弱点——一个瓶颈。

4. **Fiedler 值对应的特征向量（Fiedler 向量）揭示最佳分割。** 值为正的节点归入一组，值为负的节点归入另一组。这就是谱聚类（spectral clustering）。

```mermaid
graph TD
    subgraph "图到矩阵"
        G["图 G"] --> A["邻接矩阵 A"]
        G --> D["度矩阵 D"]
        A --> L["拉普拉斯矩阵 L = D - A"]
        D --> L
    end
    subgraph "谱分析"
        L --> E["L 的特征值"]
        L --> V["L 的特征向量"]
        E --> C["连通分量（零特征值）"]
        E --> F["连通性（Fiedler 值）"]
        V --> S["谱聚类"]
    end
```

### 谱性质

邻接矩阵和拉普拉斯矩阵的特征值无需任何遍历即可揭示结构性质。

**谱聚类**的步骤如下：
1. 计算拉普拉斯矩阵 L
2. 找出 L 的 k 个最小特征向量（跳过第一个，连通图的第一个特征向量是全 1 向量）
3. 将这些特征向量作为每个节点的新坐标
4. 在这些坐标上运行 k-均值聚类

为什么有效？L 的特征向量编码了图上"最平滑"的函数。连通性好的节点具有相似的特征向量值，被瓶颈分隔的节点具有不同的值。特征向量自然地将簇分开。

**随机游走的联系。** 归一化拉普拉斯矩阵与图上的随机游走相关。随机游走的稳态分布与节点度成比例。混合时间（游走收敛的速度）取决于谱间隙（spectral gap）。

### 消息传递

图神经网络的核心操作。每个节点收集来自邻居的消息，聚合它们，并更新自身状态。

```
h_v^(k+1) = UPDATE(h_v^(k), AGGREGATE({h_u^(k) : u in neighbors(v)}))
```

在最简单的形式中，AGGREGATE = 均值，UPDATE = 线性变换 + 激活函数：

```
h_v^(k+1) = sigma(W * mean({h_u^(k) : u in neighbors(v)}))
```

这本质上是矩阵乘法。如果 H 是所有节点特征的矩阵，A 是邻接矩阵：

```
H^(k+1) = sigma(A_norm * H^(k) * W)
```

其中 A_norm 是归一化邻接矩阵（每行之和为 1）。

一轮消息传递让每个节点"看到"其直接邻居。两轮让它看到邻居的邻居。K 轮给每个节点提供来自 K 跳邻域（K-hop neighborhood）的信息。

```mermaid
graph LR
    subgraph "第 0 轮"
        A0["节点 A：[1,0]"]
        B0["节点 B：[0,1]"]
        C0["节点 C：[1,1]"]
    end
    subgraph "第 1 轮（聚合邻居）"
        A1["节点 A：avg(B,C) = [0.5, 1.0]"]
        B1["节点 B：avg(A,C) = [1.0, 0.5]"]
        C1["节点 C：avg(A,B) = [0.5, 0.5]"]
    end
    A0 --> A1
    B0 --> A1
    C0 --> A1
    A0 --> B1
    C0 --> B1
    A0 --> C1
    B0 --> C1
```

### 概念与机器学习应用

| 概念 | 机器学习应用 |
|---------|---------------|
| 邻接矩阵 | GNN 输入表示 |
| 图拉普拉斯矩阵 | 谱聚类、社区检测 |
| BFS/DFS | 知识图谱遍历、路径查找 |
| 度分布 | 节点重要性、特征工程 |
| 消息传递 | GNN 层（GCN、GAT、GraphSAGE） |
| L 的特征值 | 社区检测、图分割 |
| 谱聚类 | 无监督节点分组 |
| PageRank | 节点重要性、网络搜索 |

## 构建

### 第一步：从零构建图类

```python
class Graph:
    def __init__(self, n_nodes, directed=False):
        self.n = n_nodes
        self.directed = directed
        self.adj = {i: {} for i in range(n_nodes)}

    def add_edge(self, u, v, weight=1.0):
        self.adj[u][v] = weight
        if not self.directed:
            self.adj[v][u] = weight

    def neighbors(self, node):
        return list(self.adj[node].keys())

    def degree(self, node):
        return len(self.adj[node])

    def adjacency_matrix(self):
        import numpy as np
        A = np.zeros((self.n, self.n))
        for u in range(self.n):
            for v, w in self.adj[u].items():
                A[u][v] = w
        return A

    def degree_matrix(self):
        import numpy as np
        D = np.zeros((self.n, self.n))
        for i in range(self.n):
            D[i][i] = self.degree(i)
        return D

    def laplacian(self):
        return self.degree_matrix() - self.adjacency_matrix()
```

邻接表（`self.adj`）高效存储邻居。邻接矩阵转换使用 numpy，因为所有谱运算都需要它。

### 第二步：BFS 和 DFS

```python
from collections import deque

def bfs(graph, start):
    visited = set()
    order = []
    distances = {}
    queue = deque([(start, 0)])
    visited.add(start)
    while queue:
        node, dist = queue.popleft()
        order.append(node)
        distances[node] = dist
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return order, distances


def dfs(graph, start):
    visited = set()
    order = []
    stack = [start]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        for neighbor in reversed(graph.neighbors(node)):
            if neighbor not in visited:
                stack.append(neighbor)
    return order
```

BFS 使用双端队列（deque）实现 O(1) 的 popleft。DFS 使用列表作为栈。两者每个节点恰好访问一次——O(V + E) 时间复杂度。

### 第三步：连通分量与拉普拉斯特征值

```python
def connected_components(graph):
    visited = set()
    components = []
    for node in range(graph.n):
        if node not in visited:
            order, _ = bfs(graph, node)
            visited.update(order)
            components.append(order)
    return components


def laplacian_eigenvalues(graph):
    import numpy as np
    L = graph.laplacian()
    eigenvalues = np.linalg.eigvalsh(L)
    return eigenvalues
```

`eigvalsh` 用于对称矩阵——无向图的拉普拉斯矩阵始终是对称的。它按升序返回特征值。计算零特征值的数量即可得到连通分量的数量。

### 第四步：谱聚类

```python
def spectral_clustering(graph, k=2):
    import numpy as np
    L = graph.laplacian()
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    features = eigenvectors[:, 1:k+1]

    labels = np.zeros(graph.n, dtype=int)
    for i in range(graph.n):
        if features[i, 0] >= 0:
            labels[i] = 0
        else:
            labels[i] = 1
    return labels
```

对于 k=2，Fiedler 向量的符号将图分为两个簇。对于 k>2，应在前 k 个特征向量（排除平凡的全 1 特征向量）上运行 k-均值聚类。

### 第五步：消息传递

```python
def message_passing(graph, features, weight_matrix):
    import numpy as np
    A = graph.adjacency_matrix()
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    A_norm = A / row_sums
    aggregated = A_norm @ features
    output = aggregated @ weight_matrix
    return output
```

这是一轮 GNN 消息传递。每个节点的新特征是其邻居特征的加权平均，经权重矩阵变换。堆叠多轮可以将信息传播得更远。

## 使用

使用 networkx 和 numpy，相同的操作只需一行代码：

```python
import networkx as nx
import numpy as np

G = nx.karate_club_graph()

A = nx.adjacency_matrix(G).toarray()
L = nx.laplacian_matrix(G).toarray()

eigenvalues = np.linalg.eigvalsh(L.astype(float))
print(f"Smallest eigenvalues: {eigenvalues[:5]}")
print(f"Connected components: {nx.number_connected_components(G)}")

communities = nx.community.greedy_modularity_communities(G)
print(f"Communities found: {len(communities)}")

pr = nx.pagerank(G)
top_nodes = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:5]
print(f"Top 5 PageRank nodes: {top_nodes}")
```

networkx 使用优化的 C 后端处理任意大小的图，在生产中使用它。使用你从零实现的版本来理解它的工作原理。

### numpy 谱分析

```python
import numpy as np

A = np.array([
    [0, 1, 1, 0, 0],
    [1, 0, 1, 0, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [0, 0, 0, 1, 0]
])

D = np.diag(A.sum(axis=1))
L = D - A

eigenvalues, eigenvectors = np.linalg.eigh(L)
print(f"Eigenvalues: {np.round(eigenvalues, 4)}")
print(f"Fiedler value: {eigenvalues[1]:.4f}")
print(f"Fiedler vector: {np.round(eigenvectors[:, 1], 4)}")

fiedler = eigenvectors[:, 1]
group_a = np.where(fiedler >= 0)[0]
group_b = np.where(fiedler < 0)[0]
print(f"Cluster A: {group_a}")
print(f"Cluster B: {group_b}")
```

Fiedler 向量承担了主要工作：正值条目归入一个簇，负值条目归入另一个。无需迭代优化——只需一次特征分解。

## 交付

本课输出：
- `outputs/skill-graph-analysis.md` -- 分析图结构数据的技能参考

## 联系

| 概念 | 应用场景 |
|---------|------------------|
| 邻接矩阵 | GCN、GAT、GraphSAGE 输入 |
| 拉普拉斯矩阵 | 谱聚类、ChebNet 滤波器 |
| BFS | 知识图谱遍历、最短路径查询 |
| 消息传递 | 每个 GNN 层，神经消息传递 |
| 谱间隙 | 图连通性、随机游走混合时间 |
| 度分布 | 幂律网络、节点特征工程 |
| 连通分量 | 预处理、处理不连通图 |
| PageRank | 节点重要性排序、注意力初始化 |

GNN 值得特别说明。GCN（Kipf & Welling, 2017）中的图卷积操作使用加了自环的邻接矩阵 A_hat = A + I：

```text
H^(l+1) = sigma(D_hat^(-1/2) * A_hat * D_hat^(-1/2) * H^(l) * W^(l))
```

其中 A_hat = A + I（邻接矩阵加自环），D_hat 是 A_hat 的度矩阵。自环确保每个节点在聚合时包含自身特征。这正是带对称归一化的消息传递。D_hat^(-1/2) * A_hat * D_hat^(-1/2) 是归一化邻接矩阵。拉普拉斯矩阵在这里出现，是因为该归一化与 L_sym = I - D^(-1/2) * A * D^(-1/2) 相关。理解拉普拉斯矩阵意味着理解 GCN 为何有效。

## 练习

1. **从零实现 PageRank。** 从均匀分数开始。每步：score(v) = (1-d)/n + d * sum(score(u)/out_degree(u))，对所有指向 v 的 u 求和。使用 d=0.85。运行至收敛（变化 &lt; 1e-6）。在小型网络图上测试。

2. **用谱聚类查找社区。** 创建一个有两个明显分离簇的图（例如，通过单条边连接的两个团）。运行谱聚类并验证找到了正确的分割。随着增加更多跨簇边，结果如何变化？

3. **实现 Dijkstra 算法**，用于加权图中的最短路径。与在相同图上使用均匀权重的 BFS 结果进行比较。

4. **构建两层消息传递网络。** 用不同的权重矩阵应用消息传递两次。证明经过 2 轮后，每个节点具有来自 2 跳邻域的信息。

5. **分析真实世界图。** 使用空手道俱乐部图（34 个节点，78 条边）。计算度分布、拉普拉斯特征值和谱聚类结果，并与已知的真实分割结果进行比较。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 图（Graph） | "节点和边" | 数学结构 G=(V,E)，编码成对关系 |
| 邻接矩阵（Adjacency matrix） | "连接表" | n×n 矩阵，A[i][j] = 1 表示节点 i 和 j 相连 |
| 度（Degree） | "节点的连接数" | 与节点相连的边数 |
| 拉普拉斯矩阵（Laplacian） | "D 减 A" | L = D - A，其特征值揭示图结构 |
| Fiedler 值（Fiedler value） | "代数连通度" | L 的最小非零特征值，衡量图的连通程度 |
| 广度优先搜索（BFS） | "逐层搜索" | 先访问所有邻居再深入的遍历，找最短路径 |
| 深度优先搜索（DFS） | "优先深入" | 沿一条路径走到底再回溯的遍历 |
| 消息传递（Message passing） | "节点与邻居通信" | 每个节点聚合来自邻居的信息，是 GNN 的核心 |
| 谱聚类（Spectral clustering） | "按特征向量聚类" | 利用拉普拉斯矩阵的特征向量对图进行分割 |
| 连通分量（Connected component） | "一个独立片段" | 最大子图，其中每个节点都能到达其他所有节点 |

## 延伸阅读

- **Kipf & Welling (2017)** -- "Semi-Supervised Classification with Graph Convolutional Networks"（图卷积网络的半监督分类）。开创现代 GNN 的论文，证明谱图卷积简化为消息传递。
- **Spielman (2012)** -- "Spectral Graph Theory" 讲义。拉普拉斯矩阵、谱间隙和图分割的权威入门资料。
- **Hamilton (2020)** -- "Graph Representation Learning"。涵盖从基础到应用的 GNN 著作。
- **Bronstein et al. (2021)** -- "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges"（几何深度学习）。统一框架论文。
- **Veličković et al. (2018)** -- "Graph Attention Networks"（图注意力网络）。通过注意力机制扩展消息传递。
