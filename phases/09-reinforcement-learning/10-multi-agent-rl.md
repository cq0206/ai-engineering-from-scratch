# 多智能体强化学习 (Multi-Agent RL)

> 单智能体强化学习 (RL) 假设环境是平稳的。只要把两个会学习的智能体放进同一个世界，这个假设就会失效：每个智能体都是另一个智能体环境的一部分，而且双方都在变化。多智能体强化学习就是一组技巧，用来在 Markov 假设不再成立时，仍让学习过程收敛。

**类型：** Build
**语言：** Python
**前置要求：** 第 9 阶段 · 04（Q-learning），第 9 阶段 · 06（REINFORCE），第 9 阶段 · 07（Actor-Critic）
**耗时：** ~45 分钟

## 问题

让一个机器人学会在房间里导航，是单智能体 RL 问题。足球队不是。AlphaStar 对战 StarCraft 对手不是。由出价智能体组成的市场不是。两辆车协商通过四向停车路口也不是。现实世界里大量多对多的问题都不是。

在所有多智能体场景里，从任意一个智能体的视角看，其他智能体本身*就是*环境的一部分。随着它们学习并改变行为，环境也随之变成非平稳的。Markov 性质——“下一个状态只取决于当前状态和我的动作”——会被破坏，因为下一个状态还取决于*其他*智能体选择了什么动作，而它们的策略本身又是不断移动的目标。

这会破坏表格方法的收敛证明（Q-learning 的保证假设环境平稳）。它也会破坏朴素的 deep RL：智能体会在互相追逐的循环里打转，始终收敛不到稳定策略。你需要专门针对多智能体的技巧：集中训练 / 分散执行、反事实基线、league play、自博弈。

2026 年的应用包括：机器人群体、交通路由、自动驾驶车队、市场模拟器、多智能体 LLM 系统（第 16 阶段），以及任何存在多个智能玩家的游戏。

## 概念

*MARL 的四种主要范式：独立学习、集中式 critic、自博弈、league*

**形式化：Markov Game。** 它是 MDP 的推广：状态 `S`、联合动作 `a = (a_1, …, a_n)`、转移 `P(s' | s, a)`，以及每个智能体各自的奖励 `R_i(s, a, s')`。每个智能体 `i` 都在自己的策略 `π_i` 下最大化自身回报。如果所有奖励相同，就是**完全协作**。如果是零和，就是**对抗**。如果介于两者之间，就是**一般和 (general-sum)**。

**核心挑战：**

- **非平稳性。** 从智能体 `i` 的视角看，`P(s' | s, a_i)` 还依赖于不断变化的 `π_{-i}`。
- **信用分配。** 当奖励是共享的时，到底是哪一个智能体导致了这个结果？
- **探索协调。** 智能体必须探索互补策略，而不是重复探索同一个状态。
- **可扩展性。** 联合动作空间会随着 `n` 指数级增长。
- **部分可观测性。** 每个智能体只能看到自己的观测；全局状态是隐藏的。

**四种主流范式：**

**1. 独立 Q-learning / 独立 PPO（IQL、IPPO）。** 每个智能体都学习自己的 Q 函数或策略，把其他智能体当作环境的一部分。它很简单，有时也确实能奏效（尤其是在经验回放充当平滑化的 agent-modeling 技巧时）。理论收敛性：没有。实践中：对弱耦合任务还行，对强耦合任务很差。

**2. 集中训练、分散执行 (CTDE, Centralized Training, Decentralized Execution)。** 这是现代最常见的范式。每个智能体都有自己的*策略* `π_i`，只依赖本地观测 `o_i`——部署时就是标准的分散执行。在*训练*期间，则使用一个集中式 critic `Q(s, a_1, …, a_n)`，它以完整的全局状态和联合动作为条件。例子包括：
- **MADDPG**（Lowe 等，2017）：每个智能体都有一个集中式 critic 的 DDPG。
- **COMA**（Foerster 等，2017）：反事实基线 (counterfactual baseline)——问“如果我当时选择的是动作 `a'`，我的奖励会怎样？”——从而隔离出我自己的贡献。
- **MAPPO** / **IPPO** with shared critic（Yu 等，2022）：使用集中式 value function 的 PPO。到 2026 年，它是协作型 MARL 的主流方法。
- **QMIX**（Rashid 等，2018）：价值分解 (value decomposition)——`Q_tot(s, a) = f(Q_1(s, a_1), …, Q_n(s, a_n))`，并使用单调混合。

**3. 自博弈 (Self-play)。** 让同一个智能体的两个拷贝互相对战。对手策略*就是*我过去某个快照里的策略。AlphaGo / AlphaZero / MuZero。OpenAI Five。它最适合零和游戏；因为训练信号是对称的。

**4. League play。** 它是 self-play 在 general-sum / 对抗环境中的扩展：维护一个由过去和当前策略构成的群体，从 league 中采样对手并与之训练。它还会加入 exploiters（专门击败当前最强者）和 main exploiters（专门击败 exploiters）。AlphaStar（StarCraft II）就是典型例子。当游戏存在“石头-剪刀-布”式策略循环时，这种方法是必须的。

**通信。** 允许智能体之间发送学得的消息 `m_i`。这在协作场景中有效。Foerster 等人（2016）证明了可微分的智能体间通信可以端到端训练。今天基于 LLM 的多智能体系统（第 16 阶段）本质上就是在用自然语言通信。

## 动手构建

本课使用一个 6×6 GridWorld，其中有两个协作智能体。它们从相对的角落出发，必须到达同一个共享目标。共享奖励为：只要任意一个智能体仍在移动，每步 `-1`；当两者都到达时，奖励 `+10`。参见 `code/main.py`。

### 第 1 步：多智能体环境

```python
class CoopGridWorld:
    def __init__(self):
        self.size = 6
        self.goal = (5, 5)

    def reset(self):
        return ((0, 0), (5, 0))  # two agents

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done
```

*联合*动作空间是 `|A|² = 16`。全局状态由两个位置组成。

### 第 2 步：独立 Q-learning

每个智能体都运行自己的 Q 表，并以联合状态作为键。每一步中：双方都按 ε-greedy 选择动作，收集联合转移，然后各自使用共享奖励更新自己的 Q。

```python
def independent_q(env, episodes, alpha, gamma, epsilon):
    Q1, Q2 = defaultdict(default_q), defaultdict(default_q)
    for _ in range(episodes):
        s = env.reset()
        while not done:
            a1 = epsilon_greedy(Q1, s, epsilon)
            a2 = epsilon_greedy(Q2, s, epsilon)
            s_next, r, done = env.step(s, (a1, a2))
            target1 = r + gamma * max(Q1[s_next].values())
            target2 = r + gamma * max(Q2[s_next].values())
            Q1[s][a1] += alpha * (target1 - Q1[s][a1])
            Q2[s][a2] += alpha * (target2 - Q2[s][a2])
            s = s_next
```

它在这个任务上能工作，因为奖励稠密且一致。但在强耦合任务里会失败（例如一个智能体必须为另一个智能体*等待*时）。

### 第 3 步：带价值分解更新的集中式 Q

使用一个联合动作上的 Q 函数 `Q(s, a_1, a_2)`。根据共享奖励进行更新。执行时则通过边缘化实现分散化：`π_i(s) = argmax_{a_i} max_{a_{-i}} Q(s, a_1, a_2)`。它用指数级的联合动作空间，换来了一个*正确*的全局视角。

### 第 4 步：简单 self-play（两智能体对抗）

同一个智能体，两个角色。让智能体 A 与智能体 B 对战；每经过 `K` 个 episode，就把 A 的权重复制给 B。训练是对称的，因此能持续进步。这就是缩小版的 AlphaZero 配方。

## 常见陷阱

- **非平稳回放。** 对独立智能体来说，经验回放比单智能体更糟，因为旧转移来自现在已经过时的对手策略。修复方式：重标记，或按新近程度加权。
- **信用分配含糊。** 一个很长的 episode 结束后才给共享奖励，很难说清是谁贡献了什么。修复方式：反事实基线（COMA），或对每个智能体单独做奖励塑形。
- **策略漂移 / 互相追逐。** 每个智能体的最优响应都会随着其他智能体的更新而改变。修复方式：集中式 critic、较小的学习率，或一次只冻结一个智能体。
- **通过协调进行奖励劫持。** 智能体会找到设计者未预料到的协同漏洞。比如拍卖智能体可能收敛到统一报零价。修复方式：仔细设计奖励，加入行为约束。
- **探索冗余。** 两个智能体都在探索同一批状态-动作对。修复方式：按智能体分别加 entropy bonus，或加入角色条件。
- **League 循环。** 纯 self-play 可能卡在支配循环中。修复方式：使用具有多样化对手的 league play。
- **样本爆炸。** `n` 个智能体 × 状态空间 × 联合动作。需要用函数逼近近似，或使用分解式动作空间（每个智能体一个策略输出头）。

## 如何使用

2026 年的 MARL 应用地图如下：

| 领域 | 方法 | 说明 |
|------|------|------|
| 协作导航 / 操作 | MAPPO / QMIX | CTDE；共享 critic + 分散 actor。 |
| 双人游戏（国际象棋、围棋、扑克） | 带 MCTS 的 self-play（AlphaZero） | 零和；训练对称。 |
| 复杂多人游戏（Dota、StarCraft） | League play + imitation pretraining | OpenAI Five、AlphaStar。 |
| 自动驾驶车队 | CTDE MAPPO / 带 attention 的 PPO | 部分可观测；团队规模可变。 |
| 拍卖市场 | 博弈论均衡 + RL | 当 `n` → ∞ 时使用 mean-field RL。 |
| LLM 多智能体系统（第 16 阶段） | 自然语言通信 + 角色条件化 | RL 循环发生在智能体规划层。 |

到 2026 年，MARL 增长最快的方向是基于 LLM 的系统：一群语言模型智能体进行协商、辩论、构建软件。这里的 RL 出现在*轨迹级 (trajectory-level)* 输出上的偏好优化，而不是 token 级（第 16 阶段 · 03）。

## 交付

保存为 `outputs/skill-marl-architect.md`：

```markdown
---
name: marl-architect
description: Pick the right multi-agent RL regime (IPPO, CTDE, self-play, league) for a given task.
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

Given a task with `n` agents, output:

1. Regime classification. Cooperative / adversarial / general-sum. Justify.
2. Algorithm. IPPO / MAPPO / QMIX / self-play / league. Reason tied to coupling tightness and reward structure.
3. Information access. Centralized training (what global info goes to the critic)? Decentralized execution?
4. Credit assignment. Counterfactual baseline, value decomposition, or reward shaping.
5. Exploration plan. Per-agent entropy, population-based training, or league.

Refuse independent Q-learning on tightly-coupled cooperative tasks. Refuse to recommend self-play for general-sum with cycle risks. Flag any MARL pipeline without a fixed-opponent eval (cherry-picked self-play numbers are common).
```

## 练习

1. **简单。** 在 2 智能体协作 GridWorld 上训练独立 Q-learning。平均回报要经过多少个 episode 才会大于 0？画出联合学习曲线。
2. **中等。** 加入一个“协调”任务：只有当两个智能体在同一回合同步踩上目标格时，才算达成目标。独立 Q 还会收敛吗？哪里出了问题？
3. **困难。** 为 MAPPO 风格训练实现一个集中式 critic，并将其在协调任务上的收敛速度与独立 PPO 进行比较。

## 关键术语

| 术语 | 人们常说什么 | 它实际表示什么 |
|------|--------------|----------------|
| Markov game | “多智能体 MDP” | `(S, A_1, …, A_n, P, R_1, …, R_n)`；每个智能体都有自己的奖励。 |
| CTDE | “集中训练，分散执行” | 训练时使用联合 critic；每个智能体的策略只使用本地观测。 |
| IPPO | “Independent PPO” | 每个智能体各自单独运行 PPO。是个简单但常被低估的基线。 |
| MAPPO | “Multi-agent PPO” | 在全局状态条件下使用集中式 value function 的 PPO。 |
| QMIX | “单调价值分解” | `Q_tot = f_monotone(Q_1, …, Q_n)`，从而允许分散式 argmax。 |
| COMA | “反事实多智能体” | Advantage = 我的 Q 减去对我的动作边缘化后的期望 Q。 |
| Self-play | “智能体对战过去的自己” | 一个智能体，两个角色；零和游戏的标准方法。 |
| League play | “群体训练” | 缓存过去策略，从池中采样对手；用于处理策略循环。 |

## 延伸阅读

- [Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275) —— 使用集中式 critic 的 CTDE。
- [Foerster et al. (2017). Counterfactual Multi-Agent Policy Gradients (COMA)](https://arxiv.org/abs/1705.08926) —— 用于信用分配的反事实基线。
- [Rashid et al. (2018). QMIX: Monotonic Value Function Factorisation](https://arxiv.org/abs/1803.11485) —— 带单调性的价值分解。
- [Yu et al. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO)](https://arxiv.org/abs/2103.01955) —— PPO 在 MARL 中强得出人意料。
- [Vinyals et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z) —— 大规模 league play。
- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270) —— 零和游戏中的纯 self-play。
- [Sutton & Barto (2018). Ch. 15 — Neuroscience & Ch. 17 — Frontiers](http://incompleteideas.net/book/RLbook2020.pdf) —— 其中包含了教材对多智能体设定以及 CTDE 旨在解决的非平稳性问题的简要讲解。
- [Zhang, Yang & Başar (2021). Multi-Agent Reinforcement Learning: A Selective Overview](https://arxiv.org/abs/1911.10635) —— 一篇综述，涵盖协作、竞争和混合型 MARL 及其收敛结果。
