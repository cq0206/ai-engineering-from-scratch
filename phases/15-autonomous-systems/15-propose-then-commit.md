# 人在回路中：先提议后提交 (Propose-Then-Commit)

> 2026 年对 HITL 的共识是非常具体的。它不是“代理发问，用户点击 Approve”。真正的模式是先提议后提交：拟议动作会被持久化到一个持久存储 (durable store) 中，并带有幂等键 (idempotency key)；随后呈现给审查者，同时展示意图、数据沿袭 (data lineage)、触及的权限、影响半径 (blast radius) 以及回滚计划；只有在收到明确的正向确认后才会真正提交；执行后还要再验证，确认副作用确实发生了。LangGraph 的 `interrupt()` 加 PostgreSQL 检查点、Microsoft Agent Framework 的 `RequestInfoEvent`，以及 Cloudflare 的 `waitForApproval()`，实现的其实都是同一种形态。它最典型的失效模式，就是“橡皮图章式批准”：界面只问一句 “Approve?”，然后用户不经审查就点掉。文档里给出的缓解方式，是带明确核对清单的挑战-响应 (challenge-and-response)。

**类型：** 学习
**语言：** Python（stdlib、带幂等性的 propose-then-commit 状态机）
**前置条件：** Phase 15 · 12（持久化执行）、Phase 15 · 14（Tripwires）
**时长：** ~60 分钟

## 问题

代理要采取一个动作。用户必须决定：批准还是不批准。如果这个决定是在瞬间做出的，那它大概率不算真正的审查。如果这个决定是结构化的，它会更慢，但也更值得信任。工程上的问题在于：如何让“结构化审查”成为阻力最小的路径。

2023 年式的 HITL 模式，是同步弹窗提示：“代理想给 X 发一封正文为 Y 的邮件——是否批准？” 用户点击 Approve。所有人都会觉得系统是安全的。可在实践中，这种界面极容易被橡皮图章化：用户批准得很快，批准结果几乎没有预测价值，而且当代理出错时，审计轨迹里只会留下长长一串用户自己都不记得的批准记录。

2026 年的模式——先提议后提交——把 HITL 搬到了一个持久化基底 (durable substrate) 上，附带结构化元数据，并要求明确提交。所有受管代理 SDK 都提供了某种版本：LangGraph `interrupt()`、Microsoft Agent Framework `RequestInfoEvent`、Cloudflare `waitForApproval()`。API 名字不同；形态完全相同。

## 概念

### 先提议后提交的状态机

1. **提议（Propose）。** 代理生成一个拟议动作。它会被持久化到持久存储中（PostgreSQL、Redis、Durable Object）。内容包括：
   - 意图（为什么代理要这样做）
   - 数据沿袭（是什么源数据导致了这个提议）
   - 触及的权限（涉及哪些 scope / file / endpoint）
   - 影响半径（最坏情况下会造成什么影响）
   - 回滚计划（如果提交了，如何撤销）
   - 幂等键（每个提议唯一；重复提交返回同一条记录）
2. **呈现（Surface）。** 审查者会看到这条提议以及全部元数据。审查者必须是人，而不是代理自己审自己。
3. **提交（Commit）。** 明确的正向确认。然后动作才会执行。
4. **验证（Verify）。** 执行之后，重新读取副作用并确认。如果验证失败，系统就处在一个“已知的不良状态”中，必须触发告警。

### 幂等键

如果没有幂等键，那么在一次瞬时故障之后重试，就可能让一个已经批准的动作被执行两次。一个具体例子：用户批准“从 A 向 B 转账 100 美元”。网络抖动了。工作流重试。用户只批准过一次，但转账却执行了两次。幂等键会把这次批准绑定到一个唯一的副作用上；第二次执行就会变成 no-op。

这和 Stripe 与 AWS API 使用的幂等模式是同一种东西。Microsoft Agent Framework 文档明确指出，代理批准流程也要复用这一模式。

### 持久性：为什么批准必须比进程活得更久

等待批准的“房间”本身是一段代理并不拥有的状态。工作流在这里暂停（第 12 课）。等批准到来后，工作流会从完全相同的位置恢复。这也是为什么 LangGraph 会把 `interrupt()` 与 PostgreSQL 检查点配对，而不是只依赖内存状态——哪怕两天后批准才到，工作流仍然能完整恢复。

### 橡皮图章式批准与挑战-响应缓解措施

HITL 的默认 UI（“Approve” / “Reject” 按钮）往往会产生快速批准，而非真正审查。文档中记录的缓解方法，是一份挑战-响应核对清单：只有在审查者对特定问题给出肯定回答之后，Approve 按钮才会启用。具体形式如下：

- "Do you understand what resource this touches? [ ]"
- "Have you verified the blast radius is acceptable? [ ]"
- "Do you have a rollback plan if this fails? [ ]"

这并不是为了官僚流程本身，而是一种强制函数 (forcing function)。如果审查者连这些框都没法勾上，要么就该要求进一步澄清（升级处理），要么就该直接拒绝（安全默认）。Anthropic 关于代理安全的研究明确把“由核对清单驱动的 HITL”列为缓解橡皮图章式批准的手段。

### 什么算“有后果的”动作

并不是每一个动作都需要先提议后提交。2026 年的指导意见如下：

- **有后果的动作**（始终需要 HITL）：不可逆写入、金融交易、对外通信、生产数据库变更、破坏性文件系统操作。
- **可逆动作**（有时需要 HITL）：本地文件编辑、预发环境变更、具备明确回滚方案的可逆写入。
- **读取与检查**（从不需要 HITL）：读取文件、列出资源、调用只读 API。

### 动作后的验证

“提交过程运行了”并不等于“副作用真的发生了”。网络分区 (network partition) 与竞争条件 (race conditions) 可能让工作流以为自己成功了，但后端实际上没有持久化。验证步骤会在提交之后重新读取目标资源进行确认。这与数据库事务里的 `RETURNING` 子句，或 AWS 在 `PutObject` 之后再做一次 `GetObject`，是同样的模式。

### 欧盟 AI 法案第 14 条

第 14 条要求欧盟境内高风险 AI 系统具备有效的人类监督。“有效”不是摆设。监管语言明确排除了橡皮图章式模式。在 Microsoft Agent Governance Toolkit 的合规文档中，能够通过第 14 条审查的形态，就是带挑战-响应机制的先提议后提交。

## 使用它

`code/main.py` 用 Python 标准库实现了一个先提议后提交状态机。持久存储是一个 JSON 文件。幂等键是 `(thread_id, action_signature)` 的哈希。驱动程序会模拟三种情况：一条干净的批准流程、一次瞬时故障后的重试（必须不能重复执行），以及“橡皮图章式默认流程”与“挑战-响应流程”的对比。

## 交付它

`outputs/skill-hitl-design.md` 会审查一个拟议中的 HITL 工作流是否符合先提议后提交形态，并标出缺失的元数据、幂等性、验证或挑战-响应层。

## 练习

1. 运行 `code/main.py`。确认：对于一条已批准提议的重试，会复用持久记录而不会再次执行。然后把幂等键改成包含时间戳，并展示重试会导致重复执行。

2. 给提议记录增加一个 `rollback` 字段。模拟一次执行完成但验证失败的情况。展示回滚会自动触发。

3. 阅读 Microsoft Agent Framework 的 `RequestInfoEvent` 文档。找出一个 API 中包含、但这个玩具引擎还缺少的元数据字段。把它加进去，并解释它防范的是什么风险。

4. 为一个具体动作设计一份挑战-响应核对清单（例如“向公开 Twitter 账号发帖”）。审查者必须回答哪三个问题？为什么是这三个？

5. 找一个同步 “Approve?” 提示就已经足够的场景（不需要持久存储）。解释为什么足够，以及你接受的是哪一类风险。

## 关键术语

| 术语 | 人们常说什么 | 实际含义 |
|---|---|---|
| 先提议后提交 | “两阶段批准” | 持久化提议 + 明确提交 + 验证 |
| 幂等键 | “可安全重试的令牌” | 每个提议唯一；第二次执行会变成 no-op |
| 数据沿袭 | “它是从哪里来的” | 导致该提议出现的具体源内容 |
| 影响半径 | “最坏情况” | 一旦动作出错，其影响范围有多大 |
| 橡皮图章 | “快速批准” | 没有真正审查就点击了 “Approve” |
| 挑战-响应 | “强制核对清单” | 审查者必须对特定问题做出明确确认 |
| `RequestInfoEvent` | “MS Agent Framework 原语” | 带结构化元数据的持久化 HITL 请求 |
| `interrupt()` / `waitForApproval()` | “框架原语” | LangGraph / Cloudflare 中的对应实现 |

## 延伸阅读

- [Microsoft Agent Framework — Human in the loop](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) —— `RequestInfoEvent`、持久化批准。
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) —— `waitForApproval()` 与 Durable Objects。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) —— 将 HITL 视为长时程风险缓解手段。
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) —— 高风险系统的监管基线。
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) —— 围绕监督的宪制化 framing。
