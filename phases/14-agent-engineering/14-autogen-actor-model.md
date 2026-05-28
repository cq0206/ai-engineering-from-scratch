# AutoGen v0.4：演员模型与智能体框架

> AutoGen v0.4（Microsoft Research，2025 年 1 月）围绕演员模型重新设计了智能体编排。异步消息交换、事件驱动智能体、故障隔离、天然并发。该框架现已进入维护模式，而 Microsoft Agent Framework（2025 年 10 月公开预览）正成为它的继任者。

**类型：** 学习 + 构建
**语言：** Python（标准库，stdlib）
**前置条件：** 第 14 阶段 · 01（智能体循环）、第 14 阶段 · 12（工作流模式）
**耗时：** ~75 分钟

## 学习目标

- 描述演员模型：智能体即演员（actor），消息是唯一的进程间通信（IPC），故障按演员隔离。
- 说出 AutoGen v0.4 的三层 API —— Core、AgentChat、Extensions —— 以及各自用途。
- 解释为什么将消息投递与消息处理解耦，能够带来故障隔离和天然并发。
- 在 Python 中用标准库（stdlib）实现一个演员运行时，并把一个双智能体代码审查流程迁移到其上。

## 问题

大多数智能体框架都是同步的：一个智能体产出，另一个智能体消费，运行在同一调用栈中。出现故障时，整条调用栈都会崩。并发只是后加的补丁。要做分布式时，往往得重写。

AutoGen v0.4 的答案是：演员模型。每个智能体都是一个拥有私有收件箱的演员。消息是唯一交互方式。运行时将投递与处理解耦。故障被隔离在单个演员内部。并发是原生能力。分布式只不过是换一种传输层。

## 概念

### 演员

一个演员包含：

- 私有状态（外部永远不能直接修改）。
- 收件箱（消息队列）。
- 处理器：`receive(message) -> effects`，其中 effects 可以是“回复”“发送给其他演员”“生成新演员”“更新状态”“停止自己”。

两个演员不能共享内存。它们只能发送消息。

### AutoGen v0.4 的三层 API

1. **Core。** 低层演员框架。`AgentRuntime`、`Agent`、`Message`、`Topic`。异步消息交换，事件驱动。
2. **AgentChat。** 面向任务的高层 API（替代 v0.2 的 ConversableAgent）。`AssistantAgent`、`UserProxyAgent`、`RoundRobinGroupChat`、`SelectorGroupChat`。
3. **Extensions。** 各类集成 —— OpenAI、Anthropic、Azure、工具、记忆。

### 为什么解耦很重要

在 v0.2 模型里，同步调用 `agent_a.chat(agent_b)` 会一直阻塞 agent_a，直到 agent_b 返回。而在 v0.4 中，`send(agent_b, msg)` 只是把消息放入 agent_b 的收件箱后立即返回。运行时稍后再完成投递。这会带来三个结果：

- **故障隔离。** 智能体 B 崩溃不会拖垮智能体 A —— 运行时会捕获 B 处理器中的失败，并决定如何处理（记录日志、重试、死信）。
- **天然并发。** 可以同时有很多消息在飞行；演员会并发处理各自的收件箱。
- **天然适配分布式。** 无论演员是在进程内还是在另一台主机上，“收件箱 + 传输层”都是同一个抽象。

### 拓扑

- **RoundRobinGroupChat。** 智能体按固定轮转顺序依次发言。
- **SelectorGroupChat。** 由一个选择器智能体根据对话上下文决定下一位发言者。
- **Magentic-One。** 面向网页浏览、代码执行、文件处理的参考多智能体团队。构建于 AgentChat 之上。

### 可观测性

内置支持 OpenTelemetry。每条消息都会产生一个追踪跨度（span）；工具调用会按照 2026 OTel GenAI 语义约定（第 23 课）附带 `gen_ai.*` 属性。

### 当前状态：维护模式

到 2026 年初，AutoGen v0.7.x 已经足够稳定，适合研究和原型开发。Microsoft 已将活跃开发转向 Microsoft Agent Framework（2025 年 10 月 1 日公开预览；计划于 2026 年第一季度末发布 1.0 GA）。AutoGen 的模式可以平滑迁移到后者——真正持久的思想是演员模型。

## 动手构建

`code/main.py` 实现了一个基于标准库（stdlib）的演员运行时：

- `Message` —— 带类型的负载，包含 `sender`、`recipient`、`topic`、`body`。
- `Actor` —— 抽象基类，提供 `receive(message, runtime)`。
- `Runtime` —— 带共享队列、消息投递和故障隔离的事件循环。
- 一个双演员演示：`ReviewerAgent` 负责审查代码，`ChecklistAgent` 负责执行检查清单；二者通过消息交换直到达成共识。

运行：

```
python3 code/main.py
```

执行轨迹会展示消息投递、一个演员中的模拟故障如何不影响另一个演员，以及它们最终如何收敛到共同结论。

## 使用

- **AutoGen v0.4/v0.7**（维护中）—— 适合研究、原型开发和多智能体模式实验。
- **Microsoft Agent Framework**（公开预览）—— 面向未来的路径；采用相同的演员模型思想，但 API 是重新设计过的。
- **LangGraph swarm 拓扑**（第 13 课）—— 通过共享工具交接实现的类似模式。
- **自定义演员运行时** —— 当你需要特定传输层（NATS、RabbitMQ、gRPC）时。

## 交付

`outputs/skill-actor-runtime.md` 会为给定的多智能体任务生成一个最小演员运行时，以及一个团队模板（RoundRobin 或 Selector）。

## 练习

1. 增加一个死信队列（DLQ）：当处理器抛出异常时，把失败消息暂存起来供人工检查。在你的玩具示例中，DLQ 会被命中多少次？
2. 实现 `SelectorGroupChat`：由一个选择器演员根据对话状态决定下一条消息由谁处理。
3. 增加分布式传输：把进程内队列替换为基于 JSON-over-HTTP 的服务端，让演员可以运行在不同进程中。
4. 为每条消息接入一个 OTel 跨度（span）（或一个空操作（no-op）占位实现）。按照第 23 课输出 `gen_ai.agent.name`、`gen_ai.operation.name`。
5. 阅读 AutoGen v0.4 的架构文章。把你的玩具示例迁移到真实的 `autogen_core` API 上。有哪些在生产环境里不能跳过、但这里省略掉了的部分？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| 演员 | “智能体” | 私有状态 + 收件箱 + 处理器；无共享内存 |
| 消息 | “事件” | 带类型的负载；演员之间唯一的交互方式 |
| 收件箱 | “邮箱” | 每个演员独有的待处理消息队列 |
| 运行时 | “智能体宿主” | 路由消息并隔离故障的事件循环 |
| Topic | “频道” | 演员之间具名的发布-订阅路径 |
| 故障隔离 | “让它崩溃” | 一个演员失败不会拖垮其他演员 |
| RoundRobinGroupChat | “固定轮转团队” | 智能体按顺序轮流行动 |
| SelectorGroupChat | “上下文路由团队” | 由选择器决定下一位 |
| Magentic-One | “参考团队” | 面向网页 + 代码 + 文件的多智能体小队 |

## 延伸阅读

- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) —— 重构设计文章
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) —— 图形化替代方案
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) —— AutoGen 默认会输出的追踪跨度（span）
