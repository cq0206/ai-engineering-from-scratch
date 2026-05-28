# Anthropic 负责任扩展政策 v3.0

> RSP v3.0 于 2026 年 2 月 24 日生效，取代了 2023 年版政策。它采用双层缓解结构：一层是 Anthropic 将单方面采取的措施，另一层是被表述为全行业建议的内容（包括 RAND SL-4 安全标准）。它把 Frontier Safety Roadmaps 和 Risk Reports 提升为常设文档，而不再是一次性交付物。它删除了 2023 年的暂停承诺。它还引入了 AI R&D-4 阈值：一旦跨过，Anthropic 就必须发布一份肯定性论证 (affirmative case)，说明失配风险及其缓解措施。Claude Opus 4.6 尚未跨过这一阈值。Anthropic 在 v3.0 公告中表示，“要有把握地排除这一点正变得困难”。SaferAI 给 2023 年版 RSP 的评分是 2.2；他们将 v3.0 下调到 1.9，使 Anthropic 与 OpenAI、DeepMind 一起被归入“弱”RSP 类别。定性阈值取代了 2023 年的定量承诺；而移除暂停条款，则是最明显的倒退。

**类型：** 学习
**语言：** Python（stdlib、RSP 阈值决策引擎）
**前置条件：** Phase 15 · 06（AAR）、Phase 15 · 07（RSI）
**时长：** ~45 分钟

## 问题

前沿实验室发布的扩展政策，一部分是技术文档，一部分是治理文档，还有一部分是给监管者看的信号。RSP v3.0 是 Anthropic 当前的文档。认真阅读它很重要，不是因为它具有强制约束力（并没有），而是因为它的表述方式会塑造实验室如何理解灾难性风险，以及如何向公众传达取舍。

真正有用的分析单位，是 v3.0 与 v2.0 的差异。新增了什么：Frontier Safety Roadmaps、Risk Reports、AI R&D-4 阈值。删除了什么：2023 年的暂停承诺。重新框定了什么：把缓解时间表拆成“Anthropic 单边措施”和“行业建议”两层。外部评审机构 SaferAI 把评分从 2.2（v2）下调到 1.9（v3.0）。这说明，一份扩展政策完全可能在看起来更精致的同时，实际却变得更不严格。

## 概念

### 双层缓解时间表

- **Anthropic 单边行动**：无论其他实验室做什么，Anthropic 都会自行采取的措施。包括在某个阈值之上停止训练、实施特定安全措施、设置特定部署闸门。
- **全行业建议**：Anthropic 认为整个行业应共同采取的措施。包括 RAND SL-4 安全标准。这些不是 Anthropic 自己的承诺；它们属于政策倡议。

这种双层结构在 v2 中并不存在。这意味着读者必须查看每一项承诺究竟落在哪一列。一个安全措施如果出现在“全行业建议”这一列，它就不是 Anthropic 的承诺，而只是 Anthropic 的期待。

### AI R&D-4 阈值

这是 RSP v3.0 明确指出的下一个重要能力阈值。具体来说：如果某个模型能够以有竞争力的成本，自动化相当大比例的 AI 研究工作，就达到了这一水平。一旦 Anthropic 认为某个模型跨过该阈值，他们就必须在继续扩展之前，发布一份肯定性论证，说明失配风险与缓解措施。

根据 v3.0 公告，Claude Opus 4.6 尚未跨过这一阈值。文档还补充说：“要有把握地排除这一点正变得困难。” 这个措辞很重要；它承认这一阈值已经足够接近，成为现实关切，而不再只是推测性的上限。

第 6 课（自动化对齐研究）和第 7 课（递归自我改进）都会直接推高这一阈值。自动化对齐研究员一旦跨过“可用于研究”的质量门槛，就表明 AI R&D-4 阈值正在逼近。

### Frontier Safety Roadmaps 与 Risk Reports

v3.0 把两类产物提升为常设文档：

- **Frontier Safety Roadmap**：前瞻性文档，用来描述计划中的安全工作、能力预期和缓解研究。
- **Risk Report**：回顾性文档，用于在模型发布后描述已观察到的能力与残余风险 (residual risk)。

两者都是公开的，也都会按声明好的节奏更新。它们的价值在于：读者可以跟踪 Anthropic 在 Roadmap 中说要做什么，并将其与他们在 Risk Report 中实际报告了什么进行对照。

### 移除暂停条款

2023 年版 RSP 包含一条明确的暂停承诺：如果模型跨过某些特定能力阈值，训练将暂停，直到缓解措施就位。v3.0 用更柔和的表述替换了这条明确暂停条款（发布肯定性论证；如果缓解措施足够，就继续推进）。SaferAI 和其他分析者都明确指出，这是新文档中最显著的倒退。

支持这一改动的政策论点是：2023 年的定量阈值到了 2026 年已难以匹配能力基准，因为基准本身也被重新缩放了。反方观点则是：扩展政策中的暂停条款，本质上是一种承诺装置 (commitment device)；移除它，就移除了政策的可信度。

### SaferAI 的降级

SaferAI 是一家对 RSP 类文档进行评分的独立机构。他们的公开评分如下：2023 年版 Anthropic RSP 得分为 2.2（在一套以 4.0 为当前最佳、1.0 为名义值的量表中）；v3.0 得分为 1.9。这使 Anthropic 从“中等”跌入“弱”，与 OpenAI 和 DeepMind 一起进入弱类别。

SaferAI 给出的降级因素包括：
- 定性阈值取代了定量阈值。
- 暂停承诺被移除。
- AI R&D-4 阈值的缓解措施被描述为“肯定性论证”，而不是具体措施。
- 评审机制依赖 Anthropic 自己的 Safety Advisory Group，独立监督有限。

### 本课不是什么

这不是一节合规课。RSP v3.0 不是法规；没有任何机制强迫 Anthropic 遵守它。本课的重点，是学会以它应得的那种具体性与怀疑精神来阅读文档。对于前沿实验室而言，扩展政策是它们向外界发出的关于灾难性风险姿态的核心公共信号。能读懂它们，是任何工作依赖前沿能力的人都需要掌握的实用技能。

## 使用它

`code/main.py` 实现了一个小型决策引擎，模拟 RSP 阈值评估的形态：给定一个候选模型和一组能力测量值，返回它是否跨过 AI R&D-4 阈值、所需的肯定性论证章节，以及部署是否可以继续。它被刻意做得很简单；重点是把文档中的逻辑明确化。

## 交付它

`outputs/skill-scaling-policy-review.md` 会按 v3.0 参考框架审查一份扩展政策（Anthropic、OpenAI、DeepMind 或内部版本）：双层结构、阈值、暂停承诺、独立评审。

## 练习

1. 运行 `code/main.py`。输入三个能力水平不同的合成模型。确认阈值评估器表现符合预期，并能产出正确的肯定性论证模板。

2. 完整阅读 RSP v3.0（32 页）。找出所有落在“全行业建议”层中的承诺。其中哪些承诺在 v2 中本应属于 “Anthropic 单边行动”？

3. 阅读 SaferAI 的 RSP 评分方法。把他们的评分规则应用到文档上，复现 v3.0 的 1.9 分。是哪一条评分项最明显地推动了这次降级？

4. 2023 年的暂停承诺被移除了。提出一个替代性承诺：既保留政策的可信度，又承认 2026 年“基准重新缩放”带来的问题。

5. 将 RSP v3.0 与 OpenAI Preparedness Framework v2（第 20 课）进行比较。选一个 v3.0 更强的方面，再选一个 Preparedness Framework 更强的方面。

## 关键术语

| 术语 | 人们常说什么 | 实际含义 |
|---|---|---|
| RSP | “Anthropic 的扩展政策” | Responsible Scaling Policy；v3.0 于 2026 年 2 月 24 日生效 |
| AI R&D-4 | “研究自动化阈值” | 以有竞争力的成本自动化相当大比例 AI 研究的能力 |
| 肯定性论证 | “安全性论证” | 公开发布的论证，说明风险已被识别且缓解措施足够 |
| Frontier Safety Roadmap | “前瞻计划” | 关于计划中安全工作与预期能力的常设文档 |
| Risk Report | “模型回顾报告” | 关于模型发布后已观察能力与残余风险的常设文档 |
| 双层缓解 | “单边 vs 行业” | Anthropic 承诺与行业建议被明确区分 |
| 暂停承诺 | “2023 条款” | 明确承诺暂停训练；在 v3.0 中被移除 |
| SaferAI 评分 | “独立 RSP 评级” | 第三方评分规则；v3.0 得分 1.9（v2 为 2.2） |

## 延伸阅读

- [Anthropic — Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) —— 完整 32 页政策文本。
- [Anthropic — RSP v3.0 announcement](https://www.anthropic.com/news/responsible-scaling-policy-v3) —— 从 v2 到 v3 的变更摘要。
- [Anthropic — Frontier Safety Roadmap](https://www.anthropic.com/research/frontier-safety) —— RSP v3.0 中链接的常设文档。
- [Anthropic — Risk Report: Claude Opus 4.6](https://www.anthropic.com/research/risk-report-claude-opus-4-6) —— 对当前前沿模型的回顾报告。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) —— 将 AI R&D-4 与可测量自主性连接起来。
