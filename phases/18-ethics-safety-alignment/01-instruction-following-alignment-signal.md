# 指令遵循作为对齐信号

> 后续所有对 RLHF 的批评，反对的都是这条流水线。在你学习优化压力如何扭曲代理指标之前，你得先看见这个代理指标。InstructGPT（Ouyang et al., 2022）定义了参考架构：在指令-响应对上做监督微调 (supervised fine-tuning, SFT)，训练一个基于成对偏好排序的奖励模型 (reward model, RM)，再针对该奖励模型用带有相对 SFT 策略 KL 惩罚项的 PPO 进行优化。一个 1.3B 的 InstructGPT 在人类偏好评估中胜过了 175B 的 GPT-3。正是这个单一结果，让 2026 年的每一家前沿实验室仍在交付 RLHF 形态的后训练流水线。

**类型：** 学习
**语言：** Python（stdlib，玩具三阶段流水线）
**先修：** Phase 10 · 06 (SFT), Phase 10 · 07 (RLHF), Phase 10 · 08 (DPO)
**时间：** ~45 分钟

## 学习目标

- 说出 InstructGPT 流水线的三个阶段，以及每个阶段使用的损失函数。
- 解释为什么一个 1.3B 的指令微调模型会在人类偏好评估中击败原始的 175B GPT-3。
- 说明第 3 阶段中的 KL 惩罚项在防止什么，以及为什么移除它会坍缩为寻模行为。
- 描述对齐税 (alignment tax)，以及 Ouyang et al. 用 PPO-ptx 缓解它的方式。

## 问题

预训练语言模型会续写文本，不会回答问题。你让 GPT-3 “write a Python function that reverses a list”，它常常会回给你另一个 prompt，因为训练分布的大部分都是网页文本，而网页文本后面通常还是更多网页文本。模型是在完成它的工作——只是这个工作本身不对。

所有严肃实验室用来修复这一点的代理信号都是人类偏好。把两个 completion 交给标注者；标注者选更好的那个；奖励模型学习标注者的判断。然后一个 RL 循环把策略推向奖励模型打分更高的输出。这三句话就是 InstructGPT 的全部论点。论文剩下的内容基本都是工程实现。

## 概念

### 阶段 1：监督微调 (SFT)

收集 prompt-response 对，其中 response 是一个善意人类会写出的内容。Ouyang et al. 使用了来自标注者和 OpenAI API 的 13k 个 prompts。然后用标准交叉熵损失 (cross-entropy loss) 在这些数据上微调基础模型。

SFT 带来的东西：模型现在会回答问题，而不是继续续写。它带不来的东西：当多个答案都说得通时，标注者究竟偏好哪一个，这一点它没有提供任何信号。

### 阶段 2：奖励模型 (RM)

对每个 prompt，从 SFT 模型里采样 K 个 completions。让标注者对它们排序。训练一个奖励模型，为任意 prompt-response 对打分，使得对于 `y_w` 优于 `y_l` 的样本对：

```
L_RM = -log sigmoid(r(x, y_w) - r(x, y_l))
```

这就是 Bradley-Terry 成对偏好损失 (pairwise preference loss)。RM 通常从 SFT 模型初始化，只是把 LM head 换成一个 scalar head。

奖励模型很小：对 175B 的 InstructGPT 来说，6B 就够了。它们也很脆弱——论文第 5 节的大部分内容都在讲小规模实验中出现的奖励黑客行为。

### 阶段 3：带 KL 惩罚项的 PPO

定义目标：

```
J(pi) = E_{x~D, y~pi(.|x)} [ r(x, y) ] - beta * KL(pi(.|x) || pi_SFT(.|x))
```

用 PPO 最大化它。KL 项会阻止 `pi` 偏离 SFT 策略太远。没有它，优化器就会找到对抗样本 (adversarial examples)——这些字符串之所以在 RM 下得分很高，只是因为 RM 从没见过它们，而不是因为人类真的更喜欢它们。

KL 系数 `beta` 是 RLHF 中最重要的超参数。太低：奖励黑客。太高：相对 SFT 没有提升。

### 对齐税

做完 RLHF 后，模型更受人类偏好，但在标准基准（SQuAD、HellaSwag、DROP）上会退步。Ouyang et al. 把这称为对齐税，并用 PPO-ptx 修复：把预训练梯度混入 RL 目标中，这样模型就不会忘掉那些它从未被奖励过、但原本会做的下游任务。

```
J_ptx(pi) = J(pi) + gamma * E_{x~D_pretrain} [ log pi(x) ]
```

PPO-ptx 后来成了标准做法。Anthropic、DeepMind 和 Meta 都用了某种变体。

### 结果

一个 1.3B 的 InstructGPT（SFT + RM + PPO-ptx）在人类标注者那里，大约有 70% 的时间比 175B 的基础 GPT-3 更受偏好。在来自生产流量的隐藏测试 prompts 上，这个差距还会更大。可以从这个数字读出两件事：

1. 对齐与能力是不同维度。175B 模型能力更强；1.3B 模型对齐更好；标注者更喜欢对齐更好的那个。
2. 能力下限由基础模型决定。你不可能靠 RLHF 让一个基础模型知道它从未见过的事实。

### 为什么这是第 18 阶段的参考点

后续课程中的每一种批评——奖励黑客（第 2 课）、DPO（第 3 课）、谄媚（第 4 课）、CAI（第 5 课）、休眠代理（第 7 课）、对齐伪装（第 9 课）——都在反对这条流水线中的某一部分。奖励黑客攻击第 2 阶段。DPO 把第 2 和第 3 阶段折叠在一起。CAI 用 AI 取代人类标注者。谄媚说明标注者本身是有偏信号。对齐伪装说明策略甚至可以完全绕开第 3 阶段。如果你的脑中还没有这条流水线，就没法真正理解这些批评。

## 实操

`code/main.py` 在玩具偏好数据上模拟这三个阶段。基础“策略”是动作 {A, B, C} 上的一枚带偏硬币。第 1 阶段 SFT 在 200 个 prompts 上模仿标注者动作。第 2 阶段用 500 组成对排序拟合一个 Bradley-Terry 奖励模型。第 3 阶段运行一个简化版 PPO 更新，并对 SFT 策略加入 KL 惩罚项。你可以看到奖励上升、KL 散度增长、策略漂移——也可以关掉 KL 项，在 50 次更新之内看到奖励黑客出现。

要观察的内容：

- `beta = 0.1` 与 `beta = 0.0` 时的奖励轨迹。
- 训练步数上的 `KL(pi || pi_SFT)`。
- 与标注者偏好相比的最终动作分布。

## 交付

本课会产出 `outputs/skill-instructgpt-explainer.md`。给定一段 RLHF 流水线描述或论文摘要，它会识别三阶段中哪一阶段被修改了、每个阶段使用了什么损失，以及是否存在 KL 惩罚项或等价正则器。

## 练习

1. 运行 `code/main.py`。设定 `beta = 0.0`，报告 200 步 PPO 之后的动作分布。用一段话解释这种寻模行为。

2. 给奖励模型对动作 B 加一个 +0.5 偏置（模拟奖励 bug）。在 `beta = 0.1` 下运行 PPO。KL 惩罚项能阻止策略利用这个偏置吗？在什么 `beta` 下这种利用开始变得明显？

3. 阅读 Ouyang et al.（arXiv:2203.02155）图 1。通过运行 1、5、20、100 步 PPO 并测量相对 SFT 模型的偏好，复现标注者偏好曲线。

4. 论文第 4.3 节报告，1.3B 的 InstructGPT 大约有 70% 的时间胜过 175B 的 GPT-3。为什么在隐藏的生产 prompts 上，这个比例会比在标注者自己写的 prompts 上更高？

5. 在相同偏好数据上，把 PPO 损失替换为 DPO（Phase 10 · 08）。比较最终策略漂移（相对 SFT 的 KL）和最终奖励。在相同奖励水平下，哪种方法漂得更远？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| SFT | “指令微调” | 第 1 阶段：在 prompt-response 对上做交叉熵微调 |
| Reward model | “RM” | 在 `(prompt, response)` 上的标量回归器，使用 Bradley-Terry 在成对标签上训练 |
| Bradley-Terry | “成对偏好损失” | `-log sigmoid(r_w - r_l)`；把成对排序约简为二分类 |
| KL penalty | “正则项” | `beta * KL(pi \|\| pi_SFT)` —— 让 RL 策略保持在 SFT 锚点附近 |
| PPO-ptx | “带预训练混合的 PPO” | 在 PPO 目标中加入一部分预训练对数似然，以抵消对齐税 |
| Alignment tax | “RLHF 回归” | RLHF 后在标准基准上的下降，而 RLHF 并未直接优化这些基准 |
| Labeler preference | “真实值” | 人类排序样本；RM 是它的统计代理，而不是“人类价值”的代理 |

## 延伸阅读

- [Ouyang et al. — Training language models to follow instructions with human feedback (arXiv:2203.02155)](https://arxiv.org/abs/2203.02155) —— InstructGPT 论文，也是后续所有 RLHF 流水线的基础
- [Stiennon et al. — Learning to summarize from human feedback (arXiv:2009.01325)](https://arxiv.org/abs/2009.01325) —— RLHF 用于摘要的前身工作
- [Christiano et al. — Deep reinforcement learning from human preferences (arXiv:1706.03741)](https://arxiv.org/abs/1706.03741) —— 最早的基于偏好 RL 表述
- [Bai et al. — Training a Helpful and Harmless Assistant with RLHF (arXiv:2204.05862)](https://arxiv.org/abs/2204.05862) —— Anthropic 对 InstructGPT 流水线的 HH 扩展
