# Tree of Thoughts 与 LATS：审慎搜索

> 单条思维链（chain-of-thought）轨迹没有回退空间。ToT（Yao 等，2023）把推理变成一棵树，并在每个节点上进行自评估。LATS（Zhou 等，2024）则在蒙特卡洛树搜索（Monte Carlo Tree Search, MCTS）下，把 ToT、ReAct 和 Reflexion 统一起来。Game of 24 从 4%（CoT）提升到 74%（ToT）；LATS 在 HumanEval 上达到 92.7% pass@1。

**类型：** 构建
**语言：** Python（标准库）
**先修要求：** 第 14 阶段 · 01（智能体循环），第 14 阶段 · 03（Reflexion）
**时间：** 约 75 分钟

## 学习目标

- 把推理建模为搜索：节点是“想法（thoughts）”，边是“扩展（expansions）”，价值（value）表示“有多有前景”。
- 仅用标准库实现一个带自评估打分的 ToT 风格 BFS 树搜索。
- 将其扩展为一个玩具版 LATS MCTS 循环，包含选择（select）/扩展（expand）/模拟（simulate）/回传（backpropagate）。
- 判断何时搜索值得付出额外令牌成本（Game of 24、代码生成），何时单条轨迹就足够（简单问答）。

## 问题

思维链是一条线性路径。如果第一步错了，后面的每一步都会建立在错误前提上。在 Game of 24（使用四个数字和 + − × ÷ 得到 24）任务上，GPT-4 CoT 的准确率只有 4%。模型过早选错了子表达式，之后就再也回不来了。

推理真正需要的是：能够提出多个候选、评估它们、选出最有前景的，再在遇到死路时回退。这就是搜索。Tree of Thoughts 和 LATS 是这件事的两个经典表述。

## 概念

### Tree of Thoughts（Yao 等，NeurIPS 2023）

每个节点都是一个连贯的中间步骤（“一个想法（thought）”）。每个节点可以扩展出 K 个子想法。LLM 会使用一个评分提示对每个节点进行自评估。搜索在树中展开——可以是 BFS、DFS 或 beam search（束搜索）。

```
                     (root: "find 24 from 4 6 4 1")
                    /               |            \
           ("6 - 4 = 2")    ("4 + 1 = 5")    ("4 * 6 = 24")  <- Score: HIGH
              /   \              |                  |
          ...    ...          ...                finish
```

自评估是其中真正承重的部分。论文展示了三种变体：`sure / likely / impossible` 分类、`1..10` 数值分数，以及候选之间投票。这三种方法在 Game of 24 上都显著优于 CoT（使用 GPT-4 时从 4% 提升到 74%）。

### LATS（Zhou 等，ICML 2024）

LATS 在 MCTS 框架下统一了 ToT、ReAct 和 Reflexion。LLM 同时扮演三种角色：

- **策略（Policy）**：提出候选的下一步动作（ReAct 风格）。
- **价值函数（Value function）**：为部分轨迹打分（ToT 风格的自评估）。
- **自我反思器（Self-reflector）**：在失败后写出自然语言反思（Reflexion 风格），并用它为未来的模拟轨迹（rollout）重新播种。

环境反馈（观察结果，observations）会混入价值函数中，因此搜索受到真实工具结果的引导，而不只是模型自己的判断。论文时期的结果是：HumanEval 上使用 GPT-4 时 pass@1 达到 92.7%（SOTA），WebShop 上使用 GPT-3.5 时平均分达到 75.9（接近基于梯度的微调）。

### 最小化理解 MCTS

每次迭代有四个阶段：

1. **选择（Select）** —— 使用 UCT（upper confidence bound for trees，树上的上置信界）从根节点走到一个叶子节点。
2. **扩展（Expand）** —— 通过策略生成 K 个子节点。
3. **模拟（Simulate）** —— 从某个子节点继续模拟轨迹（rollout），并用价值函数（或环境奖励）给叶子打分。
4. **回传（Backpropagate）** —— 沿路径向上更新访问计数和值估计。

UCT 公式：`Q(s, a) + c * sqrt(ln N(s) / N(s, a))`。第一项是利用；第二项是探索。`c` 需要按任务调优。

### 成本现实

搜索会让令牌成本爆炸。ToT 在 Game of 24 上使用的令牌数是 CoT 的 100–1000 倍。LATS 也差不多。这不是免费的；请把搜索保留给以下场景：

- 单条轨迹已被证明明显不够（Game of 24、复杂代码）。
- 相比响应时间，正确性更重要。
- 拥有廉价且可靠的价值函数（例如代码的单元测试、数学任务的明确目标）。

如果你的任务只有一个正确答案，但评估器本身很噪声，搜索往往会把事情变得更糟——它会找到一个“得分很高”的错误答案。

### 2026 年的定位

大多数生产级智能体并不会运行 LATS。它们通常运行带有工具落地验证的 ReAct（CRITIC，第 05 课）。搜索主要出现在一些专门领域：

- 使用测试作为价值函数的编码智能体（HumanEval 风格）。
- 探索多条查询路径的深度研究智能体。
- LangGraph 子图中的重规划工作流。

AlphaEvolve（第 11 课）则是 2025 年的极端例子：对代码做进化搜索，使用机器可校验的适应度函数，并取得前沿成果（56 年来第一次改进 4x4 矩阵乘法）。

## 动手构建

`code/main.py` 实现了：

- 一个简化的 ToT BFS，用于风格化的“选择算术运算”任务。
- 同一任务上的玩具版 LATS MCTS 循环（选择 / 扩展 / 模拟 / 回传），使用 UCT 进行选择。
- 一个把符号分数与自评估分数组合起来的价值函数。

运行：

```
python3 code/main.py
```

输出轨迹会展示：ToT 如何在 BFS 下为每个节点扩展三个候选；而 LATS 如何通过 MCTS 收敛到最佳模拟轨迹。两者的令牌计数也会打印出来。

## 使用它

LangGraph 通过子图模式提供 ToT 风格的探索；LangChain 团队关于 LATS 的博客（2024 年 5 月）是参考教程。LlamaIndex 提供了一个 `TreeOfThoughts` 智能体。对于大多数 2026 年的生产级智能体，这种模式通常隐藏在一个 `if task_complexity > threshold: use_search()` 开关之后——可参见第 05 课中的评估者-优化器模式。

## 交付上线

`outputs/skill-search-policy.md` 会根据任务形态、预算和评估器保真度，在线性 ReAct、ToT、LATS 与进化搜索之间做选择。

## 练习

1. 用 UCT c=0.1 与 c=2.0 分别运行玩具版 LATS。轨迹会发生什么变化？
2. 将价值函数替换成噪声更大的评分器（加入随机扰动）。MCTS 还能找到最佳叶子节点吗？它能容忍的最小信噪比是多少？
3. 实现束搜索 ToT（beam search，每层保留前 k 个，即 top-k），并与 BFS 比较。在紧张的令牌预算下，哪一个更好？
4. 阅读 LATS 第 5.1 节。从概念上复现 HumanEval 的轨迹数量：要达到论文报告的 pass@1，需要多少次模拟轨迹？
5. 阅读 LATS 论文中关于“什么时候 LATS 帮助较少”的讨论。写一段决策规则，把任务形态映射到搜索策略。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Tree of Thoughts | “分叉版 CoT” | Yao 等人的方法——带自评估的想法节点树 |
| LATS | “面向 LLM 的 MCTS” | Zhou 等人的方法——在 MCTS 下统一 ToT + ReAct + Reflexion |
| UCT | “上置信界” | 在利用（Q）与探索（ln N / n）之间平衡的选择公式 |
| 价值函数（Value function） | “这个状态有多好” | 由提示驱动的 LLM 分数或环境奖励；用于回传 |
| 策略（Policy） | “动作提议器” | ReAct 风格的生成器；输出候选的下一步想法/动作 |
| 模拟轨迹（Rollout） | “模拟出来的轨迹” | 从某节点走到叶子，沿途使用策略，再由价值函数评分 |
| 回传（Backpropagate） | “更新祖先节点” | 把叶子奖励沿路径向上推送，更新访问次数与 Q 值 |
| 搜索成本（Search cost） | “令牌爆炸” | 在 Game of 24 上达到 CoT 的 100–1000 倍；采用前先做预算 |

## 延伸阅读

- [Yao et al., Tree of Thoughts (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601) —— 经典论文
- [Zhou et al., LATS (arXiv:2310.04406)](https://arxiv.org/abs/2310.04406) —— 带 Reflexion 反馈的 MCTS
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) —— 搜索的子图模式
- [AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131) —— 带程序化评估器的进化搜索
