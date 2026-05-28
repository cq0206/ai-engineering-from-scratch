# 近端策略优化 (PPO)

> A2C 在每次更新后都会丢弃整段 rollout。PPO 把策略梯度包进一个裁剪后的重要性比率里，因此你可以在同一批数据上做 10 个以上的 epoch，而不会让策略直接炸掉。Schulman 等人（2017）。到 2026 年，它仍然是默认的策略梯度算法。

**类型：** Build
**语言：** Python
**前置要求：** 第 9 阶段 · 06（REINFORCE），第 9 阶段 · 07（Actor-Critic）
**耗时：** ~75 分钟

## 问题

A2C（第 07 课）是 on-policy 的：梯度 `E_{π_θ}[A · ∇ log π_θ]` 要求数据必须来自*当前*的 `π_θ`。只要做过一次更新，`π_θ` 就变了；你刚才用的数据此时就成了 off-policy。继续复用它，梯度就会带偏。

收集 rollout 很贵。在 Atari 上，8 个环境 × 128 步的一次 rollout = 1024 条转移，通常要花十几秒环境时间。只做一个梯度步就把这批数据扔掉，非常浪费。

信赖域策略优化 (TRPO, Trust Region Policy Optimization，Schulman 2015) 是第一个修复方案：约束每次更新，使旧策略和新策略之间的 KL 散度不超过 `δ`。理论上很干净，但每次更新都需要解一个共轭梯度问题。到 2026 年，几乎没人再跑 TRPO。

PPO（Schulman 等，2017）把硬性的 trust-region 约束替换成了一个简单的裁剪目标。只多一行代码。每个 rollout 可训练十个 epoch。无需共轭梯度。理论保证也足够好。九年之后，它仍然是从 MuJoCo 到 RLHF 的默认策略梯度算法。

## 概念

*PPO 裁剪替代目标：在 1 ± ε 处裁剪比率*

**重要性比率。**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

它表示新策略相对于采集数据时所用策略的似然比。`r_t = 1` 表示没有变化。`r_t = 2` 表示新策略选择 `a_t` 的概率是旧策略的两倍。

**裁剪替代目标。**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

这里有两项：

- 如果优势 `A_t > 0`，并且比率想继续增长到超过 `1 + ε`，裁剪会把梯度压平——不要把一个好动作的概率推到比旧策略高出 `+ε` 以上。
- 如果优势 `A_t &lt; 0`，并且比率朝着超过 `1 - ε` 的方向移动（意味着相对其被裁剪后的降低幅度，我们会让一个坏动作变得更可能），裁剪就会封顶梯度——不要把坏动作的更新推到超出 `-ε`。

`min` 负责处理另一个方向：如果比率是往*有利*的方向移动，你仍然能得到梯度（不会在对你有害的那一侧进行裁剪）。

典型设置是 `ε = 0.2`。如果把目标函数画成 `r_t` 的函数，它会是一条分段线性曲线：在“好的一侧”有一个平顶，在“坏的一侧”有一个平底。

**完整的 PPO 损失。**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

它和 A2C 一样，仍是 actor-critic 结构。三个系数通常取 `c_v = 0.5`、`c_e = 0.01`、`ε = 0.2`。

**训练循环。**

1. 在 `N` 个并行环境中各运行 `T` 步，收集 `N × T` 条转移。
2. 计算优势（GAE），并把它们冻结为常量。
3. 把 `π_{θ_old}` 冻结为当前 `π_θ` 的一个快照。
4. 对 `K` 个 epoch 中的每个 minibatch `(s, a, A, V_target, log π_old(a|s))`：
   - 计算 `r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`。
   - 应用 `L^{CLIP}` + value loss + entropy。
   - 做一次梯度更新。
5. 丢弃这段 rollout。回到第 1 步。

`K = 10`、minibatch 大小为 64，是一组标准超参数。PPO 很鲁棒：只要在 ±50% 范围内，具体数值通常都不太敏感。

**KL 惩罚变体。** 原论文还提出了另一种使用自适应 KL 惩罚的形式：`L = L^{PG} - β · KL(π_θ || π_old)`，其中 `β` 会根据观测到的 KL 动态调整。后来裁剪版本成为主流；KL 版本则保留在 RLHF 中（因为在那里，对参考策略的 KL 本来就是一个你总会需要的独立约束）。

## 动手构建

### 第 1 步：在 rollout 时记录 `log π_old(a | s)`

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

这个快照只会在 rollout 时记录一次。在后续多个更新 epoch 中，它都不会变化。

### 第 2 步：计算 GAE 优势（第 07 课）

和 A2C 一样。对整个 batch 做归一化。

### 第 3 步：裁剪替代目标更新

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # backprop -surrogate, add value loss, subtract entropy
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # clipped
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

“被裁剪 → 梯度归零”这一模式就是 PPO 的核心。如果新策略已经朝有利方向漂得太远，更新就会停下来。

### 第 4 步：value 和 entropy

像 A2C 一样，在 critic 目标上加入标准 MSE，并在 actor 上加入 entropy bonus。

### 第 5 步：诊断指标

每次更新都要盯住三件事：

- **平均 KL** `E[log π_old - log π_θ]`。应当保持在 `[0, 0.02]` 之间。如果冲到 `0.1` 以上，就降低 `K_EPOCHS` 或 `LR`。
- **Clip fraction**——比率落在 `[1-ε, 1+ε]` 之外的样本占比。理想范围应为 `~0.1-0.3`。如果接近 `0`，说明裁剪几乎从不触发 → 提高 `LR` 或 `K_EPOCHS`。如果接近 `0.5+`，说明你对 rollout 过拟合了 → 把它们降下来。
- **Explained variance** `1 - Var(V_target - V_pred) / Var(V_target)`。这是衡量 critic 质量的指标。随着 critic 学会，它应逐步接近 1。

## 常见陷阱

- **Clip 系数调错。** `ε = 0.2` 是事实标准。降到 `0.1` 会让更新过于保守；升到 `0.3+` 则容易不稳定。
- **Epoch 太多。** `K > 20` 通常会导致不稳定，因为策略会漂离 `π_old` 太远。特别是大网络，更要限制 epoch 数。
- **没有奖励归一化。** 奖励尺度过大时，会侵占 clip 的有效范围。计算优势之前，应先对奖励做归一化（运行中的标准差）。
- **忘记做优势归一化。** 对每个 batch 做零均值 / 单位标准差归一化是标准做法。跳过这一步会让 PPO 在大多数基准上表现崩坏。
- **学习率没有衰减。** PPO 很受益于把 LR 线性衰减到 0。使用恒定 LR 往往更差。
- **重要性比率计算错误。** 为了数值稳定性，始终使用 `exp(log_new - log_old)`，不要直接算 `new / old`。
- **梯度符号写反。** 最大化 surrogate = *最小化* `-L^{CLIP}`。符号翻转是最常见的 PPO bug。

## 如何使用

PPO 是 2026 年许多领域里的默认 RL 算法，这一点比你想象的更普遍：

| 使用场景 | PPO 变体 |
|----------|----------|
| MuJoCo / 机器人控制 | 使用 Gaussian policy 和 GAE(0.95) 的 PPO |
| Atari / 离散游戏 | 使用 categorical policy、滚动 128 步 rollout 的 PPO |
| 用于 LLM 的 RLHF | 带参考模型 KL 惩罚的 PPO，奖励由响应末尾的 RM 给出 |
| 大规模游戏智能体 | IMPALA + PPO（AlphaStar、OpenAI Five） |
| 推理型 LLM | GRPO（第 12 课）——不带 critic 的 PPO 变体 |
| 仅有偏好数据 | DPO——对 PPO+KL 的闭式坍缩，无需在线采样 |

PPO 的*损失形状*——裁剪替代目标 + value + entropy——构成了 DPO、GRPO 以及几乎所有 RLHF 流水线的脚手架。

## 交付

保存为 `outputs/skill-ppo-trainer.md`：

```markdown
---
name: ppo-trainer
description: Produce a PPO training config and a diagnostic plan for a given environment.
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

Given an environment and training budget, output:

1. Rollout size. `N` envs × `T` steps.
2. Update schedule. `K` epochs, minibatch size, LR schedule.
3. Surrogate params. `ε` (clip), `c_v`, `c_e`, advantage normalization on.
4. Advantage. GAE(`λ`) with explicit `γ` and `λ`.
5. Diagnostics plan. KL, clip fraction, explained variance thresholds with alerts.

Refuse `K > 30` or `ε > 0.3` (unsafe trust region). Refuse any PPO run without advantage normalization or KL/clip monitoring. Flag clip fraction sustained above 0.4 as drift.
```

## 练习

1. **简单。** 在 4×4 GridWorld 上运行 PPO，设置 `ε=0.2, K=4`。在环境步数相同的前提下，将它的样本效率与 A2C（每个 rollout 只训练一个 epoch）做对比。
2. **中等。** 扫描 `K ∈ {1, 4, 10, 30}`。绘制 return 随 env steps 变化的曲线，并跟踪每次更新的平均 KL。在这个任务上，KL 会从哪个 `K` 开始爆炸？
3. **困难。** 用自适应 KL 惩罚替换裁剪替代目标（如果 `KL > 2·target` 就把 `β` 翻倍，如果 `KL &lt; target/2` 就减半）。比较最终回报、稳定性，以及“无裁剪性”。

## 关键术语

| 术语 | 人们常说什么 | 它实际表示什么 |
|------|--------------|----------------|
| Importance ratio | “r_t(θ)” | `π_θ(a\|s) / π_old(a\|s)`；相对采集数据时策略的偏离程度。 |
| Clipped surrogate | “PPO 的核心技巧” | `min(r·A, clip(r, 1-ε, 1+ε)·A)`；超过裁剪阈值后，在有利一侧梯度会变平。 |
| Trust region | “TRPO / PPO 的意图” | 限制每次更新的 KL，以保证单调改进。 |
| KL penalty | “软 trust region” | PPO 的另一种形式：`L - β · KL(π_θ \|\| π_old)`，其中 `β` 是自适应的。 |
| Clip fraction | “裁剪多久触发一次” | 诊断指标——应在 0.1-0.3；超出范围通常表示参数没调好。 |
| Multi-epoch training | “复用数据” | 每个 rollout 训练 K 个 epoch；用更高方差换更高样本效率。 |
| On-policy-ish | “大体上还是 on-policy” | PPO 名义上是 on-policy，但当 K>1 时会安全地使用轻微 off-policy 的数据。 |
| PPO-KL | “另一个 PPO” | KL 惩罚变体；常用于 RLHF，因为对参考策略的 KL 本就是约束。 |

## 延伸阅读

- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) —— 原论文。
- [Schulman et al. (2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) —— TRPO，PPO 的前身。
- [Andrychowicz et al. (2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990) —— 系统消融了 PPO 的各项超参数。
- [Ouyang et al. (2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) —— InstructGPT；PPO 在 RLHF 中的经典配方。
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) —— 使用 PyTorch 的清晰现代讲解。
- [CleanRL PPO implementation](https://github.com/vwxyzjn/cleanrl) —— 许多论文都会采用的单文件 PPO 参考实现。
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) —— 在语言模型上使用 PPO 的生产级配方；适合与第 09 课（RLHF）对照阅读。
- [Engstrom et al. (2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) —— 那篇“37 个代码级优化”的论文；哪些 PPO 技巧是真正承重的，哪些只是经验之谈。
