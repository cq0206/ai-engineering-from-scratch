# 深度Q网络 (DQN)

> 2013 年：Mnih 在原始像素输入上训练了一个 Q-learning 网络，在 7 个 Atari 游戏中击败了此前所有经典强化学习 (RL) 智能体。2015 年：扩展到 49 个游戏，发表在 Nature 上，点燃了 deep-RL 时代。DQN 就是在 Q-learning 的基础上，加上三个让函数逼近稳定下来的技巧。

**类型：** Build
**语言：** Python
**前置要求：** 第 3 阶段 · 03（反向传播 (Backpropagation)），第 9 阶段 · 04（Q-learning、SARSA）
**耗时：** ~75 分钟

## 问题

表格型 Q-learning 需要为每一个（状态、动作）对单独维护一个 Q 值。国际象棋棋盘大约有 `10⁴³` 个状态。一个 Atari 帧是 210×160×3 = 100,800 个特征。表格型 RL 在几千个状态时就已经撑不住了，更别说几十亿个状态。

事后看，修复办法很显然：用神经网络替代 Q 表，也就是 `Q(s, a; θ)`。但这种“事后显然”的想法其实花了几十年才落地。把朴素函数逼近直接和 Q-learning 结合，会在“致命三元组 (deadly triad)”条件下发散——函数逼近 + 自举 (bootstrapping) + 离策略学习 (off-policy learning)。Mnih 等人（2013、2015）识别出了三个能稳定训练的工程技巧：

1. **经验回放 (Experience replay)** 解除样本转移之间的相关性。
2. **目标网络 (Target network)** 冻结自举目标。
3. **奖励裁剪 (Reward clipping)** 规范化梯度幅度。

Atari 上的 DQN 是第一次证明：单一架构配合一组统一超参数，就能从原始像素输入解决几十个控制问题。此后几乎所有 deep-RL 方法——DDQN、Rainbow、Dueling、Distributional、R2D2、Agent57——都是叠加在这三个基础技巧之上的。

## 概念

*DQN 训练循环：环境、回放缓冲区、在线网络、目标网络、Bellman TD 损失*

**目标。** DQN 在神经 Q 函数上最小化一步 TD 损失：

`L(θ) = E_{(s,a,r,s')~D} [ (r + γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))² ]`

`θ` = 在线网络，通过梯度下降在每一步更新。`θ^-` = 目标网络，周期性地从 `θ` 复制参数（大约每 10,000 步一次）。`D` = 存放过去状态转移的回放缓冲区。

**这三个技巧，按重要性排序：**

**经验回放。** 一个容量约为 `10⁶` 条转移的环形缓冲区。每次训练时都从中均匀随机采样一个小批量。这样可以打破时间相关性（连续帧几乎一模一样），让网络能多次学习那些稀有但高奖励的转移，同时也让相邻梯度更新彼此去相关。如果没有它，在 Atari 上把 on-policy TD 和神经网络直接结合会发散。

**目标网络。** 如果在 Bellman 方程两侧都使用同一个网络 `Q(·; θ)`，目标值就会在每次更新时一起移动——相当于“追着自己的尾巴跑”。解决方法是保留第二个权重冻结的网络 `Q(·; θ^-)`。每经过 `C` 步，就执行一次 `θ → θ^-`。这样回归目标会在数千个梯度步内保持稳定。软更新 `θ^- ← τ θ + (1-τ) θ^-`（用于 DDPG、SAC）则是更平滑的变体。

**奖励裁剪。** Atari 的奖励幅度从 1 到 1000+ 不等。把奖励裁剪到 `{-1, 0, +1}`，可以防止某一个游戏的奖励尺度主导梯度。当奖励大小本身很重要时，这么做是不对的；但在 Atari 里通常只关心奖励符号，因此可行。

**Double DQN。** Hasselt（2016）修复了最大化偏差：让在线网络负责*选择*动作，让目标网络负责*评估*这个动作。

`target = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ^-)`

这是即插即用的替换，而且几乎总是更好。默认就该使用它。

**其他改进（Rainbow，2017）：** 优先经验回放 (prioritized replay)（更多采样高 TD 误差的转移）、对偶网络架构 (dueling architecture)（拆分 `V(s)` 和 advantage 头）、噪声网络 (noisy networks)（学习式探索）、n-step returns、分布式 Q (distributional Q)（C51/QR-DQN）、多步自举。每一项通常都能再带来几个百分点的提升，而且增益大体可以叠加。

## 动手构建

这里的代码只使用标准库，不依赖 numpy——我们在一个很小的连续型 GridWorld 上手写了一个单隐藏层 MLP，因此每一步训练都只需要微秒级时间。算法本身与大规模 Atari DQN 完全一致。

### 第 1 步：回放缓冲区

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)
```

在 Atari 中容量通常约为 50,000；对我们的玩具环境来说，5,000 就够了。

### 第 2 步：一个很小的 Q 网络（手写 MLP）

```python
class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h
```

前向传播流程就是：linear → ReLU → linear。整个网络就这么简单。

### 第 3 步：DQN 更新

```python
def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))
```

它的结构和第 04 课里的 Q-learning 一样，只是有两个区别：(a) 我们对可微的 `Q(·; θ)` 进行反向传播，而不是查表；(b) 目标值使用的是 `Q(·; θ^-)`。

### 第 4 步：外层循环

对每个 episode，都基于 `Q(·; θ)` 按 ε-greedy 执行动作，把转移放入缓冲区，采样一个小批量，做一次梯度更新，并周期性同步 `θ^- ← θ`。模式如下：

```python
for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next
```

在我们这个使用 16 维 one-hot 状态表示的小型 GridWorld 中，智能体大约在 500 个 episode 内就能学到接近最优的策略。放到 Atari 上时，只需要把规模扩展到 2 亿帧，并加上一个 CNN 特征提取器。

## 常见陷阱

- **致命三元组。** 函数逼近 + 离策略 + 自举可能导致发散。DQN 用目标网络 + 回放缓冲区来缓解；这两个都不能删。
- **探索。** ε 必须衰减，典型做法是在训练前约 10% 的阶段里从 1.0 降到 0.01。早期探索不足时，Q 网络会收敛到局部盆地。
- **高估偏差。** 对带噪声的 Q 值取 `max` 会产生向上的偏差。生产环境里应始终使用 Double DQN。
- **奖励尺度。** 要么裁剪奖励，要么做归一化；梯度幅度与奖励幅度成正比。
- **回放缓冲区冷启动。** 在缓冲区里没有积累几千条转移之前，不要开始训练。用约 20 个样本得到的早期梯度会严重过拟合。
- **目标同步频率。** 太频繁 ≈ 没有目标网络；太稀疏 ≈ 目标过时。Atari DQN 使用每 10,000 个环境步同步一次。经验法则：大约每个训练时域的 1/100 同步一次。
- **观测预处理。** Atari DQN 会堆叠 4 帧，使状态近似满足 Markov 性。任何包含速度信息的环境，都需要帧堆叠或循环状态。

## 如何使用

到 2026 年，DQN 很少还是最先进的方法，但仍然是离策略算法的参考基线：

| 任务 | 首选方法 | 为什么不用 DQN？ |
|------|----------|------------------|
| 类 Atari 的离散动作任务 | Rainbow DQN 或 Muesli | 同一框架，但技巧更多。 |
| 连续控制 | SAC / TD3（第 9 阶段 · 07） | DQN 没有策略网络。 |
| on-policy / 高吞吐训练 | PPO（第 9 阶段 · 08） | 不需要回放缓冲区；更容易扩展。 |
| Offline RL | CQL / IQL / Decision Transformer | Q 目标更保守，不容易出现自举爆炸。 |
| 大规模离散动作空间（推荐系统） | 带 action embedding 的 DQN，或 IMPALA | 可以用；细节设计很关键。 |
| LLM RL | PPO / GRPO | 它是序列级而不是步级问题；损失函数不同。 |

这些经验仍然在迁移。回放和目标网络出现在 SAC、TD3、DDPG、SAC-X、AlphaZero 的自博弈缓冲区，以及所有 offline RL 方法中。奖励裁剪则在 PPO 中以 advantage normalization 的形式延续下来。这套架构就是蓝图。

## 交付

保存为 `outputs/skill-dqn-trainer.md`：

```markdown
---
name: dqn-trainer
description: Produce a DQN training config (buffer, target sync, ε schedule, reward clipping) for a discrete-action RL task.
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

Given a discrete-action environment (observation shape, action count, horizon, reward scale), output:

1. Network. Architecture (MLP / CNN / Transformer), feature dim, depth.
2. Replay buffer. Capacity, minibatch size, warmup size.
3. Target network. Sync strategy (hard every C steps or soft τ).
4. Exploration. ε start / end / schedule length.
5. Loss. Huber vs MSE, gradient clip value, reward clipping rule.
6. Double DQN. On by default unless explicit reason to disable.

Refuse to ship a DQN with no target network, no replay buffer, or ε held at 1. Refuse continuous-action tasks (route to SAC / TD3). Flag any reward range > 10× per-step mean as needing clipping or scale normalization.
```

## 练习

1. **简单。** 运行 `code/main.py`。画出每个 episode 的回报曲线。运行均值要经过多少个 episode 才会超过 -10？
2. **中等。** 禁用目标网络（Bellman 目标两侧都使用在线网络）。测量训练不稳定性——回报会振荡还是直接发散？
3. **困难。** 加入 Double DQN：用在线网络选择 `argmax a'`，用目标网络做评估。在一个带噪声奖励的 GridWorld 上，训练 1,000 个 episode 后，对比启用与不启用 Double DQN 时 `Q(s_0, best_a)` 相对真实 `V*(s_0)` 的偏差。

## 关键术语

| 术语 | 人们常说什么 | 它实际表示什么 |
|------|--------------|----------------|
| DQN | “Deep Q-learning” | 使用神经 Q 函数、回放缓冲区和目标网络的 Q-learning。 |
| Experience replay | “打乱后的转移样本” | 每个梯度步都从环形缓冲区中均匀采样；用于去相关。 |
| Target network | “冻结的自举目标” | 在 Bellman 目标中使用的 Q 的周期性拷贝；用于稳定训练。 |
| Deadly triad | “RL 为什么会发散” | 函数逼近 + 自举 + 离策略 = 没有收敛保证。 |
| Double DQN | “修复最大化偏差的方法” | 在线网络选择动作，目标网络评估动作。 |
| Dueling DQN | “V 和 A 两个头” | 将 Q 分解为 V + A - mean(A)；输出相同，但梯度流更好。 |
| Rainbow | “所有技巧全都上” | DDQN + PER + dueling + n-step + noisy + distributional 的组合。 |
| PER | “Prioritized Replay” | 按 TD 误差幅度成比例地采样转移。 |

## 延伸阅读

- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) —— 2013 年 NeurIPS workshop 论文，拉开了 deep RL 的序幕。
- [Mnih et al. (2015). Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) —— 发表在 Nature 上的 49 游戏 DQN 论文。
- [Hasselt, Guez, Silver (2016). Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) —— DDQN。
- [Wang et al. (2016). Dueling Network Architectures](https://arxiv.org/abs/1511.06581) —— dueling DQN。
- [Hessel et al. (2018). Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) —— 把各种技巧叠加在一起的经典论文。
- [OpenAI Spinning Up — DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html) —— 一份清晰的现代讲解。
- [Sutton & Barto (2018). Ch. 9 — On-policy Prediction with Approximation](http://incompleteideas.net/book/RLbook2020.pdf) —— 对“致命三元组”（函数逼近 + 自举 + 离策略）的教材式讲解；DQN 的目标网络和回放缓冲区正是为缓解这一问题而设计的。
- [CleanRL DQN implementation](https://docs.cleanrl.dev/rl-algorithms/dqn/) —— 研究消融实验时常用的单文件 DQN 参考实现；很适合和本课的从零实现版本对照阅读。
