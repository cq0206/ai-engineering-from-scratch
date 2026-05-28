# Reflexion：语言式强化学习

> 基于梯度的强化学习（gradient-based RL）需要成千上万次试验和一整个 GPU 集群，才能修复一种失败模式。Reflexion（Shinn 等，NeurIPS 2023）则用自然语言来做到这件事：每次试验失败后，智能体都会写下一段反思（reflection），把它存入情景记忆（episodic memory），并在下一次试验中以这段记忆为条件继续尝试。这正是 Letta 的睡眠期计算（sleep-time compute）、Claude Code 的 CLAUDE.md 经验条目（learnings），以及 pro-workflow 的 learn-rule 背后的模式。

**类型：** 构建
**语言：** Python（标准库）
**先修要求：** 第 14 阶段 · 01（智能体循环），第 14 阶段 · 02（ReWOO）
**时间：** 约 60 分钟

## 学习目标

- 说出 Reflexion 的三个组件——执行者（Actor）、评估器（Evaluator）、自我反思器（Self-Reflector）——以及情景记忆的作用。
- 仅用标准库实现一个 Reflexion 循环，包含二元评估器、反思缓冲区和全新的重新尝试。
- 针对给定任务，选择标量、启发式和自评估三类反馈源中的合适方案。
- 解释为什么语言式强化能捕捉到那些基于梯度的强化学习（RL）需要成千上万次试验才能修复的错误。

## 问题

一个智能体在某个任务上失败了。在标准强化学习（RL）中，你会再运行成千上万次试验、计算梯度、更新权重。成本高、速度慢，而且大多数生产级智能体并没有为每一次失败都准备训练预算。

Reflexion（Shinn 等，arXiv:2303.11366）问了一个不同的问题：如果智能体只是想一想自己为什么失败，并把这个想法放进下一次尝试的提示里，会怎样？不更新权重。不算梯度。只是在试验之间保存自然语言。

结果是：在 ALFWorld 上，它击败了 ReAct 和其他未微调基线；在 HotpotQA 上，它优于 ReAct；在代码生成（HumanEval/MBPP）上，它在当时达到了最先进水平（SOTA）。而这一切都没有用到任何一次梯度更新。

## 概念

### 三个组件

```
Actor         : generates a trajectory (ReAct-style loop)
Evaluator     : scores the trajectory — binary, heuristic, or self-eval
Self-Reflector: writes a natural-language reflection on the failure
```

再加上一个数据结构：

```
Episodic memory: list of prior reflections, prepended to the next trial's prompt
```

一次试验由执行者运行。评估器为它打分。如果分数低，自我反思器就会生成一段反思（例如：“我选错了工具，因为我把问题误读成在问 X，而它其实在问 Y”）。这段反思会进入情景记忆。下一次试验会重新开始，但会看到这段反思。

### 三种评估器类型

1. **标量型（Scalar）** —— 外部二元信号。ALFWorld 成功或失败。HumanEval 测试通过或失败。最简单，信号也最强。
2. **启发式型（Heuristic）** —— 预定义的失败特征。“如果智能体连续两次产生相同动作，就标记为卡住。”“如果轨迹超过 50 步，就标记为低效。”
3. **自评估型（Self-evaluated）** —— 由 LLM 给自己的轨迹打分。当没有真实标签（ground truth）时需要它。信号较弱；与基于工具落地验证（tool-grounded verification，见第 05 课——CRITIC）配合效果最好。

2026 年的默认做法是混合：有标量信号时用标量，没有时用自评估，再用启发式规则做安全护栏。

### 为什么它能泛化

Reflexion 与其说是一种新算法，不如说是一种被命名的模式。几乎每个生产级“自愈”智能体都在运行某种变体：

- Letta 的睡眠期计算（sleep-time compute，第 08 课）：一个独立智能体反思过去的对话，并写入记忆块（memory blocks）。
- Claude Code 的 `CLAUDE.md` / “save memory” 模式：将反思记录为经验总结（learnings），并预置到未来会话中。
- pro-workflow 的 `/learn-rule` 命令：把修正显式记录为规则。
- LangGraph 的反思节点（reflection nodes）：一个节点先给输出打分，如果需要就路由到改进（refine）步骤。

它们都源于同一个洞见：自然语言已经足够丰富，足以在多次运行之间承载“我从失败中学到了什么”。

### 什么时候有效，什么时候无效

Reflexion 在以下情况下有效：

- 有清晰的失败信号（测试失败、工具错误、答案错误）。
- 任务类别具有可复现性（同类型问题还会再次出现）。
- 反思确实有空间改善轨迹（动作预算足够）。

Reflexion 在以下情况下帮助不大：

- 智能体第一次就已经成功。
- 失败来自外部（网络中断、工具损坏）——对“网络断了”的反思并不能帮助未来运行。
- 反思演变成迷信——把一次偶发性不稳定运行（flaky run）编造成叙事并保存下来。

2026 年的一个陷阱是：记忆腐化（memory rot）。反思不断累积；有些已经过时或本身就是错的；随着情景缓冲区增长，重新运行会越来越慢。缓解方法：定期压缩（第 06 课）、给反思加 TTL，或使用独立的睡眠期清理智能体（Letta）。

## 动手构建

`code/main.py` 在一个玩具谜题上实现了 Reflexion：生成一个 3 元素列表，使其和等于目标值。执行者输出候选列表；评估器检查其和；自我反思器写一行说明哪里出了问题。这段反思会被写入情景记忆，供下一次试验使用。

组件包括：

- `Actor` —— 一个脚本化策略，在看到反思时会改进。
- `Evaluator.binary()` —— 根据目标和判断通过/失败。
- `SelfReflector` —— 生成一行对失败的诊断。
- `EpisodicMemory` —— 一个带 TTL 语义的有界列表。

运行：

```
python3 code/main.py
```

输出轨迹会显示三次试验。第 1 次失败，保存一条反思；第 2 次看到反思后有所改进，但仍然失败；第 3 次成功。与基线运行（无反思）对比——它会一直卡在第 1 次试验的答案上。

## 使用它

LangGraph 将反思作为一种节点模式提供。Claude Code 的 `/memory` 命令和 pro-workflow 的 `/learn-rule` 会把情景缓冲区外化为一个 Markdown 文件。Letta 的 sleep-time compute 会在空闲时运行自我反思器，因此主智能体可以保持低延迟。OpenAI Agents SDK 并不直接提供 Reflexion；你需要用自定义 Guardrail（护栏）来按分数拒绝轨迹，并使用能跨运行持久化的记忆 `Session` 来自行构建。

## 交付上线

`outputs/skill-reflexion-buffer.md` 会创建并维护一个带有反思捕获、TTL 和去重功能的情景缓冲区。给定任务类别和一次失败，它会输出一条真正能帮助下一次试验的反思，而不是一句泛泛的“更小心一点”。

## 练习

1. 把二元评估器切换为返回距离度量的标量评估器（离目标有多远）。它会更快收敛吗？
2. 给反思添加 10 次试验的 TTL。超过这个点之后，旧反思是在帮助还是在拖后腿？
3. 实现启发式评估器：如果同一动作重复出现，就将该次试验标记为卡住。这会如何影响自我反思器？
4. 用一个会忽略反思的对抗式执行者（Actor）运行 Reflexion。什么样的最小反思提示工程，才能迫使执行者注意到这些反思？
5. 阅读 Reflexion 论文第 4 节关于 ALFWorld 的内容。从概念上复现 130% 成功率提升：相对于原始 ReAct，关键差异是什么？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Reflexion | “自我纠错” | Shinn 等人在 2023 年提出——执行者、评估器、自我反思器加情景记忆 |
| 语言强化 | “无需梯度的学习” | 把自然语言反思预置到下一次试验的提示中 |
| 情景记忆 | “按任务保存的反思” | 针对某一任务类别的有界反思缓冲区 |
| 标量评估器 | “二元成功信号” | 来自真实标签的通过/失败或数值分数 |
| 启发式评估器 | “基于模式的检测器” | 预定义失败特征（如卡死循环、步骤过多） |
| 自评估器 | “LLM 充当自己轨迹的裁判” | 当没有真实标签时使用的弱信号回退方案——需配合工具落地验证 |
| 记忆腐化 | “陈旧反思” | 情景缓冲区堆满过时条目；可通过压缩/TTL 修复 |
| 睡眠期反思 | “异步自我反思” | 将自我反思器（Self-Reflector）放到热路径之外运行，让主智能体保持快速 |

## 延伸阅读

- [Shinn et al., Reflexion: Language Agents with Verbal Reinforcement Learning (arXiv:2303.11366)](https://arxiv.org/abs/2303.11366) —— 经典论文
- [Letta, Sleep-time Compute](https://www.letta.com/blog/sleep-time-compute) —— 生产环境中的异步反思
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) —— 将情景缓冲区作为上下文的一部分来管理
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) —— 反思节点模式
