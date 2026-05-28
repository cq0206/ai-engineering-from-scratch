# 函数调用深入 — OpenAI、Anthropic、Gemini

> 三大前沿提供商在 2024 年收敛到了相同的工具调用循环，然后在其他所有方面分道扬镳。OpenAI 使用 `tools` 和 `tool_calls`。Anthropic 使用 `tool_use` 和 `tool_result` 块。Gemini 使用 `functionDeclarations` 和唯一 ID 关联。本课将三者并排对比，让你在一个提供商上发布的代码迁移到其他提供商时不会出错。

**类型：** 构建实战
**语言：** Python（标准库，schema 转换器）
**前置要求：** 第 13 阶段 · 01（工具接口）
**时间：** 约 75 分钟

## 学习目标

- 说出 OpenAI、Anthropic 和 Gemini 函数调用载荷的三个结构差异（声明、调用、结果）。
- 将一个工具声明在三种提供商格式间互相转换，并预测严格模式约束在哪些地方会不同。
- 使用每个提供商的 `tool_choice` 来强制、禁止或自动选择工具调用。
- 了解每个提供商的硬性限制（工具数量、schema 深度、参数长度）以及违反限制时各自发出的错误信号。

## 问题背景

函数调用请求的格式因提供商而异。以下是 2026 年生产环境中的三个具体示例：

**OpenAI Chat Completions / Responses API。** 你传入 `tools: [{type: "function", function: {name, description, parameters, strict}}]`。模型的响应包含 `choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]`，其中 `arguments` 是一个需要你解析的 JSON 字符串。严格模式（`strict: true`）通过约束解码来强制 schema 合规。

**Anthropic Messages API。** 你传入 `tools: [{name, description, input_schema}]`。响应以 `content: [{type: "text"}, {type: "tool_use", id, name, input}]` 形式返回。`input` 已经是解析好的对象（不是字符串）。你需要回复一条新的 `user` 消息，包含 `{type: "tool_result", tool_use_id, content}` 块。

**Google Gemini API。** 你传入 `tools: [{functionDeclarations: [{name, description, parameters}]}]`（嵌套在 `functionDeclarations` 下）。响应以 `candidates[0].content.parts: [{functionCall: {name, args, id}}]` 形式到达，其中 `id` 在 Gemini 3 及以上版本中是唯一的，用于并行调用关联。你回复 `{functionResponse: {name, id, response}}`。

相同的循环。不同的字段名、不同的嵌套方式、不同的字符串/对象约定、不同的关联机制。一个在 OpenAI 上写了天气 Agent 的团队，迁移到 Anthropic 需要两天，再迁移到 Gemini 还要一天 — 仅仅是管道代码。

本课构建一个转换器，将三种格式统一为一个规范的工具声明并在边缘路由。第 13 阶段 · 17 将相同模式推广为 LLM 网关。

## 核心概念

### 共同结构

每个提供商都需要五样东西：

1. **工具列表。** 每个工具的名称、描述和输入 schema。
2. **工具选择。** 强制指定工具、禁止工具，或让模型决定。
3. **调用发出。** 命名工具和参数的结构化输出。
4. **调用 ID。** 将响应关联到正确的调用（对并行调用很重要）。
5. **结果注入。** 将结果绑定回调用的消息或块。

### 逐字段结构差异

| 方面 | OpenAI | Anthropic | Gemini |
|------|--------|-----------|--------|
| 声明信封 | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| Schema 字段 | `parameters` | `input_schema` | `parameters` |
| 响应容器 | 助手消息上的 `tool_calls[]` | `content[]` 中类型为 `tool_use` | `parts[]` 中类型为 `functionCall` |
| 参数类型 | 字符串化的 JSON | 已解析的对象 | 已解析的对象 |
| ID 格式 | `call_...`（OpenAI 生成） | `toolu_...`（Anthropic） | UUID（Gemini 3+） |
| 结果块 | 角色 `tool`，`tool_call_id` | `user` 中的 `tool_result`，`tool_use_id` | `functionResponse` 带匹配 `id` |
| 强制工具 | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| 禁止工具 | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| 严格 schema | `strict: true` | schema 即 schema（始终强制） | 请求级别的 `responseSchema` |

### 你实际会触及的限制

- **OpenAI。** 每个请求 128 个工具。Schema 深度 5。参数字符串 &lt;= 8192 字节。严格模式要求无 `$ref`、`oneOf`/`anyOf`/`allOf` 无重叠、每个属性都列在 `required` 中。
- **Anthropic。** 每个请求 64 个工具。Schema 深度实际上无限制但实际限制为 10。无严格模式标志；schema 是一个契约，模型倾向于遵守。
- **Gemini。** 每个请求 64 个函数。Schema 类型是 OpenAPI 3.0 子集（与 JSON Schema 2020-12 略有差异）。Gemini 3 起支持并行调用唯一 ID。

### `tool_choice` 行为

三种所有人都支持但命名不同的模式：

- **Auto。** 模型选择工具或文本。默认。
- **Required / Any。** 模型必须至少调用一个工具。
- **None。** 模型不能调用工具。

加上每个提供商独有的一种模式：

- **OpenAI。** 按名称强制指定特定工具。
- **Anthropic。** 按名称强制指定特定工具；`disable_parallel_tool_use` 标志区分单次和多次调用。
- **Gemini。** `mode: "VALIDATED"` 无论模型意图如何，都将每个响应通过 schema 验证器。

### 并行调用

OpenAI 的 `parallel_tool_calls: true`（默认）在一条助手消息中发出多个调用。你全部执行后用批量 tool 角色消息回复，每个 `tool_call_id` 一条。Anthropic 历史上是单次调用；`disable_parallel_tool_use: false`（Claude 3.5 起的默认值）启用多次调用。Gemini 2 允许并行调用但没有稳定的 ID；Gemini 3 添加了 UUID，使乱序响应可以干净地关联。

### 流式传输

三者都支持流式工具调用。线路格式不同：

- **OpenAI。** `tool_calls[i].function.arguments` 的增量 delta 块逐步到达。你累积直到 `finish_reason: "tool_calls"`。
- **Anthropic。** block-start / block-delta / block-stop 事件。`input_json_delta` 块携带部分参数。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 新增）发出带 `functionCallId` 的块，使多个并行调用可以交错。

第 13 阶段 · 03 深入讲解并行 + 流式重组。本课聚焦声明和单次调用的格式。

### 错误和修复

无效参数错误看起来也不同：

- **OpenAI（非严格）。** 模型返回 `arguments: "{bad json}"`，你的 JSON 解析失败，你注入错误消息并重新调用。
- **OpenAI（严格）。** 验证在解码过程中进行；无效 JSON 不可能出现，但可能出现 `refusal`。
- **Anthropic。** `input` 可能包含意外字段；schema 是建议性的。在服务器端验证。
- **Gemini。** OpenAPI 3.0 怪癖：对象字段上的 `enum` 被静默忽略；自行验证。

### 转换器模式

你代码中的规范工具声明长这样（格式由你选择）：

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

三个小函数将其转换为三种提供商格式。`code/main.py` 中的工具正是这样做的，然后通过每个提供商的响应格式往返一个模拟工具调用。无需网络 — 本课教的是格式，不是 HTTP。

生产团队将这个转换器封装在 `AbstractToolset`（Pydantic AI）、`UniversalToolNode`（LangGraph）或 `BaseTool`（LlamaIndex）中。第 13 阶段 · 17 发布了一个网关，在三者之上暴露 OpenAI 格式的 API。

## 实际应用

`code/main.py` 定义了一个规范的 `Tool` 数据类和三个转换器，分别生成 OpenAI、Anthropic 和 Gemini 的声明 JSON。然后解析每种格式手工制作的提供商响应为相同的规范调用对象，证明语义在底层是一致的。运行它并并排对比三种声明。

重点关注：

- 三种声明块仅在信封和字段名上不同。
- 三种响应块在调用位置上不同（顶层 `tool_calls`、`content[]` 块、`parts[]` 条目）。
- 一个 `canonical_call()` 函数从三种响应格式中提取 `{id, name, args}`。

## 交付物

本课产出 `outputs/skill-provider-portability-audit.md`。给定一个针对某个提供商的函数调用集成，该技能生成可移植性审计：它依赖哪些提供商限制、哪些字段需要重命名、以及迁移到其他每个提供商时会出什么问题。

## 练习

1. 运行 `code/main.py` 并验证三种提供商声明 JSON 都序列化了同一个底层 `Tool` 对象。修改规范工具添加一个 enum 参数，确认只有 Gemini 转换器需要处理 OpenAPI 怪癖。

2. 为每个提供商添加 `ListToolsResponse` 解析器，提取模型在 `list_tools` 或发现调用后返回的工具列表。OpenAI 原生没有这个；注意这个不对称性。

3. 实现 `tool_choice` 转换：将规范的 `ToolChoice(mode="force", tool_name="x")` 映射到三种提供商格式。然后映射 `mode="any"` 和 `mode="none"`。核对本课的对比表。

4. 选择三个提供商之一，从头到尾阅读其函数调用指南。找出其 schema 规范中另外两个不支持的一个字段。候选项：OpenAI `strict`、Anthropic `disable_parallel_tool_use`、Gemini `function_calling_config.allowed_function_names`。

5. 编写一个测试向量：一个参数违反声明 schema 的工具调用。通过每个提供商的验证器运行它（第 01 课的标准库验证器可以作为代理），记录哪些错误被触发。记录你在生产环境中会选择哪个提供商来保证严格性。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| 函数调用 (Function calling) | "工具使用" | 提供商级别的结构化工具调用发出 API |
| 工具声明 (Tool declaration) | "工具规范" | 名称 + 描述 + JSON Schema 输入载荷 |
| `tool_choice` | "强制/禁止" | Auto / required / none / 指定名称模式 |
| 严格模式 (Strict mode) | "Schema 强制执行" | OpenAI 标志，约束解码以匹配 schema |
| `tool_use` 块 | "Anthropic 的调用格式" | 包含 id、name、input 的内联内容块 |
| `functionCall` 部分 | "Gemini 的调用格式" | 包含 name、args 和 id 的 `parts[]` 条目 |
| 参数即字符串 | "字符串化 JSON" | OpenAI 以 JSON 字符串而非对象返回参数 |
| 并行工具调用 | "一轮中的扇出" | 一条助手消息中的多个工具调用 |
| 拒绝 (Refusal) | "模型拒绝" | 严格模式专有的拒绝块，替代调用 |
| OpenAPI 3.0 子集 | "Gemini schema 怪癖" | Gemini 使用类似 JSON Schema 的方言，有细微差异 |

## 延伸阅读

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — 权威参考，包括严格模式和并行调用
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — `tool_use` 和 `tool_result` 块语义
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) — 并行调用、唯一 ID 和 OpenAPI 子集
- [Vertex AI — Function calling reference](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) — Gemini 的企业级接口
- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — 严格模式 schema 强制执行细节
