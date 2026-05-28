# 自动化对齐研究 (Automated Alignment Research, Anthropic AAR)

> Anthropic 在彼此独立的沙箱中运行了并行团队，由 Claude Opus 4.6 自动化对齐研究员 (Autonomous Alignment Researchers) 组成，并通过一个共享论坛 (shared forum) 协调；该论坛的日志位于所有沙箱之外（因此代理无法删除自己的记录）。在“弱到强训练”问题上，AAR 的表现优于人类研究者。Anthropic 自己的总结还指出，预设工作流 (prescribed workflows) 往往会限制 AAR 的灵活性并降低性能。自动化对齐研究，是把时间线压缩到 RSP 试图检测的那类失配风险上的“压缩步骤”。

**类型：** 学习
**语言：** Python（stdlib、parallel-research-forum 模拟器）
**前置条件：** Phase 15 · 05（AI Scientist v2）、Phase 15 · 04（DGM）
**时长：** ~60 分钟

## 问题

对齐研究 (alignment research) 很消耗人类研究者时间。像可扩展监督 (scalable oversight)、奖励规范 (reward specification) 或弱到强训练 (weak-to-strong training) 这类问题，每轮实验往往都需要数周。随着前沿能力不断推进，对齐工作量增长得比合格研究者的供给更快。

自动化对齐研究（AAR）提出的问题是：那些在能力上已经跑赢对齐进度的前沿模型，是否也能帮助缩小这个差距。Anthropic 在 2026 年发布的 AAR 弱到强训练研究报告（alignment.anthropic.com/2026/automated-w2s-researcher/），是这一类已部署系统最早公开的成果之一。

这个结果确实是积极的：AAR 在一个研究问题上做得比人类基线更好。但这个结果也把本阶段讨论过的所有担忧集中到了同一个系统中。如果对齐研究可以自动化，那么削弱保障机制的那些部分也同样可以自动化。RSP 中针对 AI 研发能力的阈值，就是围绕这个闭环来写的。

## 概念

### 架构（按公开描述）

- **并行代理。** 多个 Claude Opus 4.6 实例作为 AAR 运行，每个都在独立沙箱中。沙箱会限制文件系统、网络和进程暴露面。
- **共享论坛。** 研究发现会发布到一个共享论坛，其存储后端位于所有沙箱之外。代理可以读取论坛；但代理无法从自己的沙箱内部删除或编辑论坛记录。
- **任务分配。** 由人类设计的任务队列把研究问题分配给 AAR。AAR 会提出分解方案、运行实验并撰写结果。
- **聚焦弱到强训练。** 已发表研究处理的是弱到强训练——即用较弱的监督模型去训练更强的学生模型。AAR 在这项任务上的表现优于人类研究者。

### 为什么“沙箱外日志”很重要

日志完整性 (log integrity) 这个属性，是“受控实验”和“可能已被破坏的实验”之间的分界线。如果 AAR 能编辑自己的日志，它们就可以掩盖失败或可疑的实验。Anthropic 直接强调了这一点：如果你想信任研究产出，这个属性是必要条件，而且它并不容易验证。

用基础设施术语来说，对应的是会直写到外部存储的追加式日志 (append-only logs)（例如 S3 object-lock、不可变的 Datadog 日志）。代理没有权限修改过去的记录。这与 2026 年企业安全团队用来构建抗勒索软件审计轨迹的模式完全一致。

### 预设工作流的权衡

Anthropic 的报告指出，预设工作流往往会限制 AAR 的灵活性并降低性能。换句话说：如果人类设计者把研究计划一步一步写死，AAR 的表现会比“只给高层目标、允许自由分解”更差。这与 agentic research（LangGraph、Microsoft Agent Framework）中的结果一致——过度具体的脚手架会拖累有能力的代理。

这里的安全含义很微妙。能够提升 AAR 表现的那种灵活性，也正是允许未授权分解、全新工具用法和目标漂移 (goal-drift) 的灵活性。这个权衡本质上是一个设计空间决策：要给代理多少灵活性，才能获得足以超越人类的能力，同时又不至于让它在分布外泛化目标？

### 压缩风险

RSP v3.0（第 19 课）引入了一个 AI 研发能力阈值：以相对于“人类 + AI 工具”具有竞争力的成本，完整自动化整个 AI 研发流水线。DeepMind 的 FSF v3 里也有一个类似的 ML 研发自主等级。两个框架都把这个阈值视为应升级控制措施的触发点。

AAR 距离这个阈值只差一步：它自动化了流水线中的一部分（针对特定且范围明确任务的对齐研究），但还没有自动化端到端的能力开发闭环。真正的问题是，这个差距会以多快的速度被填平。

被压缩的时间线，是“复合失效”担忧的核心。如果对齐研究和能力研究以相近速率叠加增长，那么失配风险面至少会与能力增长一样快。如果能力增长更快（这也是历史趋势），那么差距就会扩大。这就是为什么 AAR 是一种“有条件的好事”：每多一个对齐成果，只有在研究过程本身值得信任的前提下，才真正能缩小差距。

### AAR 不能取代什么

人类研究者负责设定任务队列、审查结果，并掌握宪制层面的最终权威。AAR 加速的是流水线中间部分，而不是两端。Anthropic 公开发布的输出，既包含 AAR 的贡献，也包含人类研究者对“哪些该发布、哪些该撤回、哪些该继续打磨”的判断。

这与第 15 课中的 propose-then-commit 模式一致，只不过它被应用到了研究本身：AAR 负责 propose；人类负责 commit。

## 使用它

`code/main.py` 会模拟一个多 AAR 研究论坛。三个彼此独立的“研究代理”并行运行，把研究发现发布到一个追加式论坛中，并可选择尝试篡改。

请观察：

- 沙箱外日志如何捕获篡改日志的尝试。
- 在更难的问题上，允许自由分解的任务分配如何优于固定工作流。
- 如果日志不是可感知篡改的，单个被攻陷的 AAR 如何污染论坛信号。

## 交付它

`outputs/skill-aar-deployment-review.md` 是一个自动化对齐研究流水线的部署前审查。

## 练习

1. 运行 `code/main.py`。比较 “fixed-workflow” 与 “free-decomposition” 设置。自由分解是否总是更优，还是说某些问题类别下固定工作流也有竞争力？

2. 修改模拟器，让其中一个代理尝试篡改日志。确认追加式日志能检测到它。再写一段话，准确描述日志里呈现出的检测特征是什么。

3. 阅读 Anthropic 的弱到强 AAR 报告。找出 AAR 击败人类研究者的具体子任务。是什么让它适合自动化？

4. 设计一种任务队列分配策略，在 AAR 灵活性（结果更好）与预设工作流约束（更容易审计）之间取得平衡。说明你会如何对这两者做 A/B 测试。

5. 阅读 RSP v3.0 中的 AI R&D-4 阈值。用一段话说明：有哪些事情会跨过这一阈值，而当前的 AAR 还做不到？

## 关键术语

| 术语 | 人们常说什么 | 实际含义 |
|---|---|---|
| AAR | “自动化对齐研究员” | 在对齐问题上自主运行的 Claude Opus 4.6 实例 |
| 弱到强训练 | “用较弱的监督者训练更强的模型” | AAR 在其上优于人类的经典可扩展监督基准 |
| 共享论坛 | “代理发布发现的地方” | 位于沙箱外的追加式存储 |
| 沙箱外日志 | “代理无法编辑自己的记录” | 直写到外部存储、具备防篡改可见性的日志 |
| 预设工作流 | “人类设计者给出的逐步计划” | 会约束 AAR；相较自由分解通常会降低性能 |
| 自由分解 | “由代理决定如何拆解任务” | 能力更强，但更难审计 |
| AI 研发阈值 | “RSP/FSF 能力等级” | 以有竞争力的成本完整自动化研发流水线 |
| 压缩时间线 | “对齐与能力的竞速” | 如果能力增长快于对齐，失配风险就会增加 |

## 延伸阅读

- [Anthropic — Automated Weak-to-Strong Researcher](https://alignment.anthropic.com/2026/automated-w2s-researcher/) —— 一手来源。
- [Anthropic Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) —— AI 研发阈值的 framing。
- [Anthropic — Measuring AI agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) —— 更广义的代理自主性 framing。
- [DeepMind Frontier Safety Framework v3](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) —— 与 RSP 平行的 ML 研发自主等级。
- [Burns et al. (2023). Weak-to-Strong Generalization (OpenAI)](https://openai.com/index/weak-to-strong-generalization/) —— AAR 所攻克问题的底层背景。
