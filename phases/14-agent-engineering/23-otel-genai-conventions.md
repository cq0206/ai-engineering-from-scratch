# OpenTelemetry GenAI 语义约定

> OpenTelemetry 的 GenAI SIG（于 2024 年 4 月启动）定义了智能体遥测的标准模式。追踪跨度（span）名称、属性和内容捕获规则正在跨厂商收敛，因此在 Datadog、Grafana、Jaeger 和 Honeycomb 中，智能体追踪的含义是一致的。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**先修要求：** 第 14 阶段 · 13（LangGraph），第 14 阶段 · 24（可观测性平台）
**耗时：** ~60 分钟

## 学习目标

- 说出 GenAI 追踪跨度（span）的类别：模型/客户端、智能体、工具。
- 区分 `invoke_agent` 的 CLIENT 跨度与 INTERNAL 跨度，以及它们各自适用的场景。
- 列出顶层 GenAI 属性：提供方名称、请求模型、数据源 ID。
- 解释内容捕获契约：选择加入、`OTEL_SEMCONV_STABILITY_OPT_IN`、外部引用建议。

## 问题

每个厂商都发明自己的跨度命名方式。运维团队最后不得不为每个框架单独构建仪表盘。OpenTelemetry 的 GenAI SIG 通过定义一个全生态都可以对齐的统一标准来解决这个问题。

## 概念

### 追踪跨度（Span）类别

1. **模型 / 客户端追踪跨度（span）。** 覆盖原始 LLM 调用。由提供方 SDK（Anthropic、OpenAI、Bedrock）和框架的模型适配器发出。
2. **智能体追踪跨度。** `create_agent`（构建智能体时）和 `invoke_agent`（运行智能体时）。
3. **工具追踪跨度。** 每次工具调用一个；通过父子关系连接到智能体追踪跨度。

### 智能体跨度命名

- 跨度名称：如果已命名，则为 `invoke_agent {gen_ai.agent.name}`；否则回退为 `invoke_agent`。
- 跨度类型：
  - **CLIENT** —— 用于远程智能体服务（OpenAI Assistants API、Bedrock Agents）。
  - **INTERNAL** —— 用于进程内智能体框架（LangChain、CrewAI、本地 ReAct）。

### 关键属性

- `gen_ai.provider.name` —— `anthropic`、`openai`、`aws.bedrock`、`google.vertex`。
- `gen_ai.request.model` —— 模型 ID。
- `gen_ai.response.model` —— 实际解析后的模型（可能因路由而不同于请求模型）。
- `gen_ai.agent.name` —— 智能体标识符。
- `gen_ai.operation.name` —— `chat`、`completion`、`invoke_agent`、`tool_call`。
- `gen_ai.data_source.id` —— 对于 RAG：查询了哪个语料库或存储。

此外，还存在面向 Anthropic、Azure AI Inference、AWS Bedrock、OpenAI 的技术特定约定。

### 内容捕获

默认规则是：插桩库**不应**默认捕获输入/输出。内容捕获需要通过以下字段选择加入：

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

推荐的生产模式：将内容存储在外部（S3、你的日志存储），只在跨度上记录引用（例如外部内容 ID，而不是正文内容）。这是第 27 课中的内容投毒防御，接到了可观测性体系里。

### 稳定性

截至 2026 年 3 月，大多数约定仍处于实验阶段。可通过以下方式选择加入稳定预览：

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+ 会将 GenAI 属性原生映射到它的 LLM Observability 结构。其他后端（Grafana、Honeycomb、Jaeger）支持原始属性。

### 这种模式会在哪些地方出错

- **在跨度中捕获完整提示词。** 追踪里会出现运维可读的 PII、密钥和客户数据。应改为外部存储。
- **没有 `gen_ai.provider.name`。** 缺少归因时，多提供商仪表盘会失效。
- **跨度没有父链接。** 工具跨度会变成孤儿。始终传播上下文。
- **没有设置稳定性选择加入。** 后端升级时，你的属性名可能会被重命名。

## 动手构建

`code/main.py` 实现了一个符合 GenAI 约定的标准库追踪发射器：

- 带有 GenAI 属性结构的 `Span`。
- 带 `start_span` 和嵌套上下文的 `Tracer`。
- 一个脚本化的智能体运行，会发出：`create_agent`、`invoke_agent`（INTERNAL）、每个工具的追踪跨度，以及用于 LLM 调用的 `chat` 追踪跨度。
- 一种内容捕获模式：将提示词外部存储，并在跨度上记录 ID。

运行它：

```
python3 code/main.py
```

输出内容：一个包含所有必需 GenAI 属性的跨度树，以及一个展示选择加入内容引用的“外部存储”。

## 使用它

- **Datadog LLM Observability**（v1.37+）可原生映射这些属性。
- **Langfuse / Phoenix / Opik**（第 24 课）—— 为整个生态自动插桩。
- **Jaeger / Honeycomb / Grafana Tempo** —— 原始 OTel 追踪（trace）；基于 GenAI 属性构建仪表盘。
- **自托管** —— 运行带有 GenAI 处理器的 OTel Collector。

## 交付它

`outputs/skill-otel-genai.md` 会把 OTel GenAI 追踪接入现有智能体，并配置内容捕获默认值和外部引用存储。

## 练习

1. 为你第 01 课的 ReAct 循环添加 `invoke_agent`（INTERNAL）和每个工具跨度。发送到一个 Jaeger 实例。
2. 以“仅引用”模式添加内容捕获：把提示词写入 SQLite，追踪跨度属性只携带行 ID。
3. 阅读 `gen_ai.data_source.id` 的规范。把它接入你第 09 课的 Mem0 搜索。
4. 设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`，并验证你的属性不会被收集器重命名。
5. 构建一个仪表盘：仅通过 GenAI 属性回答“哪些工具错误与哪些模型相关”。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| GenAI SIG | “OpenTelemetry GenAI 小组” | 定义该语义结构的 OTel 工作组 |
| invoke_agent | “智能体跨度” | 表示一次智能体运行的跨度名称 |
| CLIENT 跨度 | “远程调用” | 调用远程智能体服务时的跨度 |
| INTERNAL 跨度 | “进程内” | 进程内智能体运行的跨度 |
| gen_ai.provider.name | “提供方” | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | “RAG 来源” | 某次检索命中了哪个语料库/存储 |
| 内容捕获 | “提示词日志” | 对消息进行选择加入式捕获；生产环境中外部存储 |
| 稳定性选择加入 | “预览模式” | 用于固定实验性约定的环境变量 |

## 延伸阅读

- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 规范本身
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 默认带 GenAI 跨度
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 内置 OTel 跨度
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — W3C 追踪上下文传播
