# LangGraph：有状态图与持久化执行

> LangGraph 是 2026 年低层有状态编排的参考实现。智能体是一个状态机；节点是函数；边是状态转移；状态是不可变的，并且每一步之后都会做检查点。无论在哪一步失败，都可以精确地从中断处继续恢复。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**先修要求：** 第 14 阶段 · 01（智能体循环）、第 14 阶段 · 12（工作流模式）
**时间：** 约 75 分钟

## 学习目标

- 描述 LangGraph 的核心模型：带有不可变状态、函数节点、条件边以及步后检查点的状态机。
- 说出文档重点强调的四项能力：持久化执行（durable execution）、流式输出（streaming）、人类参与回路（human-in-the-loop）、全面记忆（comprehensive memory）。
- 解释 LangGraph 支持的三种编排拓扑：监督者（supervisor）、点对点（peer-to-peer / swarm）、分层式（hierarchical，嵌套子图）。
- 仅使用标准库实现一个带不可变状态、条件边以及检查点/恢复循环的状态图。

## 问题

智能体和工作流有一个共同问题：当一次 40 步运行在第 38 步失败时，你希望从第 38 步继续，而不是从头开始。把状态当作二等公民的模型，会让运维人员不得不围绕那些默认“每次都是全新运行”的库，硬写各种重试逻辑。

LangGraph 的设计答案是：状态是一等、带类型的对象；状态变更是显式的；每个节点之后都持久化检查点。恢复只需要调用 `load_state(session_id)`。

## 概念

### 图

一个图由以下部分定义：

- **状态类型。** 一个带类型的字典（或 Pydantic 模型），每个节点都会读取并修改它。
- **节点。** 纯函数 `(state) -> state_update`。函数返回后，更新会被合并进状态。
- **边。** 节点之间的条件转移或直接转移。
- **入口与出口。** `START` 和 `END` 哨兵节点标记图的边界。

例如：一个包含 `classify`、`refund`、`bug`、`sales`、`done` 节点的智能体——本质上就是一个以图表示的路由工作流。

### 持久化执行

每当一个节点返回，运行时就会序列化状态并写入检查点器（checkpointer）（SQLite、Postgres、Redis 或自定义后端）。如果在第 N 步失败，运行时可以调用 `resume(session_id)`，并以精确状态从第 N+1 步继续。

LangGraph 文档明确点名了几个在生产中非常看重这一点的用户：Klarna、Uber、J.P. Morgan。其核心主张不只是“图的形状”，而是“图的形状 + 检查点机制”让恢复成本变得很低。

### 流式输出

每个节点都可以产出部分结果。图会把按节点增量生成的事件流式传给调用方，这样 UI 能随着图的执行实时更新。

### 人类参与回路

可以在节点之间检查并修改状态。典型实现方式是：在关键节点前暂停，把状态展示给人类，接受修改，再恢复执行。检查点器让这件事变得简单，因为状态本来就已经被序列化了。

### 记忆

短期记忆（一次运行内部——状态中的对话历史）和长期记忆（跨运行——通过检查点器加上独立的长期存储来持久化）。LangGraph 通过工具与外部记忆系统（如 Mem0 或自定义系统）集成。

### 三种拓扑

1. **监督者。** 中央路由 LLM 将任务分发给专长型子智能体。`langgraph-supervisor` 中提供 `create_supervisor()`（不过 LangChain 团队在 2026 年建议直接通过工具调用来完成这类模式，以获得更强的上下文控制）。
2. **群体式（Swarm）/ 点对点。** 智能体通过共享工具界面直接交接，没有中央路由器。
3. **分层式。** 监督者管理子监督者，以嵌套子图的方式实现。

### 这种模式会在哪些地方出问题

- **检查点太小。** 如果只对对话轮次做检查点，工具状态和记忆写入就无法恢复。必须序列化完整状态。
- **非确定性节点。** 恢复默认假设同样的节点输入会产生同样的状态更新。随机种子、系统时钟、外部 API 响应都必须被捕获。
- **过度使用条件边。** 如果每条边都是条件边，这个状态机就无法被人理解。优先使用线性链路，只在必要处做分支。

## 动手构建

`code/main.py` 实现了一个基于标准库的有状态图：

- `State` —— 一个带类型的字典，包含 `messages`、`step`、`route`、`output`、`human_approval`。
- `Node` —— 一个可调用对象，接收状态并返回更新字典。
- `StateGraph` —— 节点 + 边 + 条件边 + 运行 + 恢复。
- `SQLiteCheckpointer`（内存中的伪实现）—— 每个节点后序列化状态；`load(session_id)` 负责恢复。
- 一个演示图：classify -> branch(refund / bug / sales) -> 人工闸门 -> send。

运行：

```
python3 code/main.py
```

执行轨迹会展示：第一次运行在人工闸门处失败、状态被持久化、随后恢复并生成最终输出。

## 使用

- **LangGraph** —— 参考实现，具备生产可用性。可以使用 `create_react_agent`、`create_supervisor`，或自己构建图。
- **AutoGen v0.4**（第 14 课）—— 适用于高并发场景的演员模型替代方案。
- **Claude Agent SDK**（第 17 课）—— 带内置会话存储的托管式运行支架（harness）。
- **自定义实现** —— 当你需要精确控制状态形状或检查点后端时。

## 交付

`outputs/skill-state-graph.md` 会在任意目标运行时中生成一个 LangGraph 风格的状态图，并接好检查点与恢复能力。

## 练习

1. 当分类置信度低于某个阈值时，从 `classify` 增加一条通往 `end` 的条件边。随后在人类手动设置 `route` 后恢复运行。
2. 用真实的 SQLite 检查点器替换类 SQLite 的伪实现。衡量每一步的序列化开销。
3. 实现并行边：两个节点并发运行，再通过自定义归并器（reducer）合并。不可变状态在这里带来了什么？
4. 阅读 `langgraph-supervisor` 参考资料。把这个玩具示例迁移到 `create_supervisor`。比较执行轨迹形状。
5. 增加流式输出：每个节点在运行时产出部分状态。按到达顺序打印这些增量。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| 状态图 | “把智能体当作状态机” | 带类型的状态 + 节点 + 边 + 归并器 |
| 检查点器 | “持久化后端” | 每个节点后序列化状态；支持恢复 |
| 归并器（Reducer） | “状态合并器” | 将当前状态与节点更新组合起来的函数 |
| 条件边 | “分支” | 根据状态函数选择的边 |
| 子图 | “嵌套图” | 作为节点嵌入另一个图中的图 |
| 持久化执行 | “从故障处恢复” | 从最近一次成功节点开始，以精确状态继续 |
| 监督者 | “路由 LLM” | 面向专长型子智能体的中央分发器 |
| 群体式（Swarm） | “P2P 智能体” | 智能体通过共享工具交接；没有中央路由器 |

## 延伸阅读

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) —— 参考文档
- [langgraph-supervisor reference](https://reference.langchain.com/python/langgraph/supervisor/) —— 监督者模式 API
- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) —— 演员模型替代方案
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) —— 会话存储与子智能体
