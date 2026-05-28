# OpenAI Agents SDK：交接、护栏与追踪

> OpenAI Agents SDK 是基于 Responses API 构建的轻量多智能体框架。五个原语：Agent、Handoff、Guardrail、Session、Tracing。交接（handoff）会表现为名为 `transfer_to_&lt;agent>` 的工具。护栏（guardrail）会在输入或输出阶段触发。追踪默认开启。

**类型：** 学习 + 构建
**语言：** Python（标准库，stdlib）
**前置条件：** 第 14 阶段 · 01（智能体循环）、第 14 阶段 · 06（工具使用）
**耗时：** ~75 分钟

## 学习目标

- 说出 OpenAI Agents SDK 的五个原语。
- 解释交接：为什么它被建模为工具、模型看到的名称是什么样、以及上下文如何转移。
- 区分输入护栏、输出护栏与工具护栏；解释 `run_in_parallel` 与阻塞模式的区别。
- 用标准库（stdlib）实现一个包含交接 + 护栏 + 追踪跨度（span）风格追踪的运行时。

## 问题

不能干净委派的智能体，最后往往会把所有内容都塞进一个提示里。没有护栏的智能体，会把 PII 发出去、输出违反策略的内容，或者无限循环。OpenAI 的 SDK 把让多智能体可控落地的三个关键原语编码了下来。

## 概念

### 五个原语

1. **智能体（Agent）。** LLM + 指令 + 工具 + 交接。
2. **交接（Handoff）。** 将任务委派给另一个智能体。在模型看来，它表现为一个名为 `transfer_to_<agent_name>` 的工具。
3. **护栏（Guardrail）。** 对输入（仅第一个智能体）、输出（仅最后一个智能体）或工具调用（每个函数工具）做校验。
4. **会话（Session）。** 自动维护跨轮次的对话历史。
5. **追踪（Tracing）。** 内置追踪跨度（span），覆盖 LLM 生成、工具调用、交接和护栏。

### 把交接当作工具

模型会在工具列表里看到 `transfer_to_billing_agent`。调用它就表示运行时需要：

1. 复制对话上下文（或通过测试版特性 `nest_handoff_history` 折叠上下文）。
2. 用目标智能体自己的指令初始化它。
3. 让目标智能体继续接管这次运行。

这就是产品化后的监督者模式（第 13 课 / 第 28 课）。

### 护栏

有三种风格：

- **输入护栏。** 运行在第一个智能体收到的输入上。在任何 LLM 调用之前，先拒绝不安全或超出范围的请求。
- **输出护栏。** 运行在最后一个智能体的输出上。捕获 PII 泄露、策略违规和格式错误响应。
- **工具护栏。** 针对每个函数工具运行。负责校验参数、检查权限、审计执行。

模式：

- **并行**（默认）。护栏 LLM 与主 LLM 同时运行。尾延迟更低。如果触发，主 LLM 的工作会被丢弃（浪费令牌）。
- **阻塞**（`run_in_parallel=False`）。护栏 LLM 先运行。如果触发，就不会浪费主调用的令牌。

触发器会抛出 `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered`。

### 追踪

默认开启。每一次 LLM 生成、工具调用、交接和护栏检查都会发出一个追踪跨度（span）。设置 `OPENAI_AGENTS_DISABLE_TRACING=1` 可以关闭。`add_trace_processor(processor)` 可以把追踪跨度同时扇出到你自己的后端以及 OpenAI 后端。

### 会话

`Session` 会把对话历史存入某个后端（SQLite、Redis 或自定义实现）。`Runner.run(agent, input, session=session)` 会自动加载并追加历史。

### 这种模式会在哪些地方出问题

- **交接漂移。** 智能体 A 交给智能体 B，智能体 B 又交回给智能体 A。加一个跳转（hop）计数器。
- **护栏绕过。** 工具护栏只会对函数工具生效；内置工具（文件读取器、网页抓取）需要额外策略。
- **过度追踪。** 敏感内容被写进 跨度（span）。要配合 OTel GenAI 的内容捕获规则（第 23 课）——把内容外置存储，仅通过 ID 引用。

## 动手构建

`code/main.py` 用标准库（stdlib）实现了 SDK 的整体形状：

- `Agent`、`FunctionTool`、`Handoff`（作为具有转移语义的函数工具）。
- `Runner`，支持输入/输出/工具护栏、交接分发和 hop 计数器。
- 一个简单的追踪跨度（span）发射器，用来展示追踪的形状。
- 一个分诊智能体，会根据用户问题把请求交给计费或支持智能体；其中一个输入会触发护栏。

运行：

```
python3 code/main.py
```

执行轨迹会展示两次成功交接、一次输入护栏触发，以及一棵与真实 SDK 类似的跨度（span）树。

## 使用

- **OpenAI Agents SDK** 适合以 OpenAI 为中心的产品。
- **Claude Agent SDK**（第 17 课）适合以 Claude 为中心的产品。
- **LangGraph**（第 13 课）适合你需要显式状态与可恢复执行时。
- **自定义实现** 适合你需要绝对控制（语音、多提供商、联邦部署）时。

## 交付

`outputs/skill-agents-sdk-scaffold.md` 会搭建一个 Agents SDK 应用脚手架，内含分诊智能体、交接、输入/输出/工具护栏、会话存储以及追踪（trace）处理器。

## 练习

1. 增加一个交接 hop 计数器：超过 N 次转移后拒绝继续。把行为轨迹打印出来。
2. 将 `nest_handoff_history` 实现为一个可选项——在转移前把历史消息压缩成一条摘要。
3. 写一个阻塞式输出护栏。比较会触发它的提示与能通过的提示之间的延迟差异。
4. 把 `add_trace_processor` 接到一个 JSON 日志记录器上。每个追踪跨度（span）会输出什么形状？
5. 阅读 SDK 文档。把你的 stdlib 玩具示例迁移到 `openai-agents-python`。你有哪些地方建模错了？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| 智能体 | “LLM + 指令” | SDK 中的智能体类型；拥有工具和交接 |
| 交接 | “转交” | 模型调用的工具，用于把任务委派给另一个智能体 |
| 护栏 | “策略检查” | 针对输入 / 输出 / 工具调用的校验 |
| 触发器（Tripwire） | “护栏触发” | 护栏拒绝时抛出的异常 |
| 会话 | “历史存储” | 在多次运行之间持久化的对话记忆 |
| 追踪 | “跨度（span）” | 覆盖 LLM + 工具 + 交接 + 护栏的内置可观测性 |
| 阻塞式护栏 | “顺序检查” | 护栏先运行；触发时不浪费令牌 |
| 并行护栏 | “并发检查” | 护栏并行运行；延迟更低，但触发时会浪费令牌 |

## 延伸阅读

- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) —— 原语、交接、护栏、追踪
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) —— Claude 风格的对应方案
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) —— 何时才值得使用交接（handoff）
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) —— Agents SDK 追踪跨度（span）对应的标准
