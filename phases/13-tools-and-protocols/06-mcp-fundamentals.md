# MCP 基础 — 原语、生命周期与 JSON-RPC 底层

> MCP 之前的每个集成都是一次性的。模型上下文协议 (Model Context Protocol) 由 Anthropic 于 2024 年 11 月首次发布，现由 Linux 基金会的 Agentic AI 基金会管理，它标准化了发现和调用流程，使任何客户端都能与任何服务器通信。2025-11-25 规范定义了六个原语（三个服务器端、三个客户端）、一个三阶段生命周期和 JSON-RPC 2.0 线路格式。学会这些，本阶段后续的 MCP 章节就是顺理成章的事。

**类型：** 理论学习
**语言：** Python（标准库，JSON-RPC 解析器）
**前置要求：** 第 13 阶段 · 01 至 05（工具接口和函数调用）
**时间：** 约 45 分钟

## 学习目标

- 说出全部六个 MCP 原语（服务器端：tools、resources、prompts；客户端：roots、sampling、elicitation）并各举一个用例。
- 走通三阶段生命周期（初始化、操作、关闭），说明每个阶段由谁发送哪条消息。
- 解析和生成 JSON-RPC 2.0 的请求、响应和通知信封。
- 解释 `initialize` 阶段的能力协商是什么，以及没有它会出什么问题。

## 问题背景

在 MCP 之前，每个使用工具的 Agent 都有自己的协议。Cursor 有一个类似 MCP 但不兼容的工具系统。Claude Desktop 配备了另一个不同的系统。VS Code 的 Copilot 扩展又是第三种。一个构建了"Postgres 查询"工具的团队需要写三遍同样的工具，每次都要适配不同宿主的 API。复用它们需要复制代码。

结果就是一次性集成的寒武纪大爆发，以及生态系统发展速度的天花板。

MCP 通过标准化线路格式解决了这个问题。单个 MCP 服务器可以在所有 MCP 客户端中工作：Claude Desktop、ChatGPT、Cursor、VS Code、Gemini、Goose、Zed、Windsurf，到 2026 年 4 月已有 300+ 个客户端。每月 1.1 亿次 SDK 下载。10,000+ 个公开服务器。Linux 基金会于 2025 年 12 月在新成立的 Agentic AI 基金会下接管了管理权。

本阶段使用的规范版本是 **2025-11-25**。它新增了异步 Tasks（SEP-1686）、URL 模式的信息获取（SEP-1036）、带工具的 sampling（SEP-1577）、增量范围同意（SEP-835）和 OAuth 2.1 资源指示器语义。第 13 阶段 · 09 到 16 涵盖这些扩展。本课停留在基础层面。

## 核心概念

### 三个服务器端原语

1. **Tools（工具）。** 可调用的操作。与第 13 阶段 · 01 中相同的四步循环。
2. **Resources（资源）。** 暴露的数据。通过 URI 可寻址的只读内容：`file:///path`、`db://query/...`、自定义 scheme。
3. **Prompts（提示模板）。** 可复用的模板。在宿主 UI 中表现为斜杠命令；服务器提供模板，客户端填充参数。

### 三个客户端原语

4. **Roots（根目录）。** 服务器被允许访问的 URI 集合。客户端声明它们；服务器遵守。
5. **Sampling（采样）。** 服务器请求客户端的模型执行一次补全。使服务器托管的 Agent 循环无需服务器端 API 密钥。
6. **Elicitation（信息获取）。** 服务器在执行过程中向客户端的用户请求结构化输入。表单或 URL（SEP-1036）。

MCP 中的每个能力都恰好属于这六个原语之一。第 13 阶段 · 10 到 14 深入讲解每一个。

### 线路格式：JSON-RPC 2.0

每条消息都是一个 JSON 对象，包含以下字段：

- 请求：`{jsonrpc: "2.0", id, method, params}`。
- 响应：`{jsonrpc: "2.0", id, result | error}`。
- 通知：`{jsonrpc: "2.0", method, params}` — 没有 `id`，不期待响应。

基础规范约有 15 个方法，按原语分组。重要的有：

- `initialize` / `initialized`（握手）
- `tools/list`、`tools/call`
- `resources/list`、`resources/read`、`resources/subscribe`
- `prompts/list`、`prompts/get`
- `sampling/createMessage`（服务器到客户端）
- `notifications/tools/list_changed`、`notifications/resources/updated`、`notifications/progress`

### 三阶段生命周期

**阶段 1：初始化 (initialize)。**

客户端发送 `initialize`，附带其 `capabilities` 和 `clientInfo`。服务器响应自己的 `capabilities`、`serverInfo` 和它支持的规范版本。客户端消化响应后发送 `notifications/initialized`。此后，双方可以根据协商的能力发送请求。

**阶段 2：操作 (operation)。**

双向通信。客户端调用 `tools/list` 发现工具，然后调用 `tools/call` 执行。如果服务器声明了该能力，它可以发送 `sampling/createMessage`。当工具集变化时，服务器可以发送 `notifications/tools/list_changed`。当用户更改根目录范围时，客户端可以发送 `notifications/roots/list_changed`。

**阶段 3：关闭 (shutdown)。**

任一方关闭传输层。MCP 中没有结构化的关闭方法；传输层（stdio 或 Streamable HTTP，第 13 阶段 · 09）承载连接结束信号。

### 能力协商

`initialize` 握手中的 `capabilities` 就是契约。服务器的示例：

```json
{
  "tools": {"listChanged": true},
  "resources": {"subscribe": true, "listChanged": true},
  "prompts": {"listChanged": true}
}
```

服务器声明它可以发出 `tools/list_changed` 通知并支持 `resources/subscribe`。客户端通过声明自己的能力来回应：

```json
{
  "roots": {"listChanged": true},
  "sampling": {},
  "elicitation": {}
}
```

如果客户端没有声明 `sampling`，服务器就不能调用 `sampling/createMessage`。对称地：如果服务器没有声明 `resources.subscribe`，客户端就不能尝试订阅。

这就是防止生态系统分裂的机制。不支持 sampling 的客户端仍然是有效的 MCP 客户端；不调用 `sampling` 的服务器仍然是有效的 MCP 服务器。它们只是不一起使用该功能。

### 结构化内容与错误格式

`tools/call` 返回一个 `content` 数组，包含类型化的块：`text`、`image`、`resource`。第 13 阶段 · 14 将 MCP Apps（`ui://` 交互式 UI）加入了该列表。

错误使用 JSON-RPC 错误码。规范定义的附加码：`-32002` "资源未找到"、`-32603` "内部错误"，以及 MCP 特定的 `error.data` 错误数据。

### 客户端能力 vs 工具调用细节

一个常见的混淆：`capabilities.tools` 是指客户端是否支持工具列表变更通知。客户端是否会调用特定工具是由其模型在运行时决定的，不是一个能力标志。能力标志是规范层面的契约。模型的选择与此正交。

### 为什么用 JSON-RPC 而不是 REST？

JSON-RPC 2.0（2010）是一个轻量级的双向协议。REST 是客户端发起的。MCP 需要服务器发起的消息（sampling、通知），所以具有对称请求/响应结构的 JSON-RPC 是自然的选择。JSON-RPC 还可以在 stdio 和 WebSocket/Streamable HTTP 上干净地组合，无需重新发明 HTTP 的请求格式。

## 实际应用

`code/main.py` 提供了一个最小的 JSON-RPC 2.0 解析器和生成器，然后手动执行 `initialize` → `tools/list` → `tools/call` → `shutdown` 序列，打印每条消息。没有真实的传输层；只有消息格式。与延伸阅读中链接的规范对比，验证每个信封。

重点关注：

- `initialize` 双向声明能力；响应包含 `serverInfo` 和 `protocolVersion: "2025-11-25"`。
- `tools/list` 返回 `tools` 数组；每个条目有 `name`、`description`、`inputSchema`。
- `tools/call` 使用 `params.name` 和 `params.arguments`。
- 响应的 `content` 是 `{type, text}` 块的数组。

## 交付物

本课产出 `outputs/skill-mcp-handshake-tracer.md`。给定一个 pcap 风格的 MCP 客户端-服务器交互记录，该技能标注每条消息属于哪个原语、哪个生命周期阶段以及它依赖哪个能力。

## 练习

1. 运行 `code/main.py`。找到能力协商发生的那一行，描述如果服务器没有声明 `tools.listChanged` 会有什么变化。

2. 扩展解析器以处理 `notifications/progress`。消息格式：`{method: "notifications/progress", params: {progressToken, progress, total}}`。在长时间运行的 `tools/call` 过程中发出它，确认客户端处理器能显示进度条。

3. 从头到尾阅读 MCP 2025-11-25 规范 — 整个文档约 80 页。找出大多数服务器不需要的那一个能力标志。提示：它与资源订阅有关。

4. 在纸上画出假设的"定时任务"功能应该属于哪个原语。（提示：服务器希望客户端在预定时间调用它。现有的六个原语都不适合。）MCP 的 2026 路线图中有一个相关的 SEP 草案。

5. 解析 GitHub 上某个开源 MCP 服务器的一个会话日志。统计请求 vs 响应 vs 通知消息数量。计算生命周期流量与操作流量的比例。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| MCP | "模型上下文协议" | 用于模型到工具的发现和调用的开放协议 |
| 服务器原语 | "服务器暴露什么" | tools（操作）、resources（数据）、prompts（模板） |
| 客户端原语 | "客户端让服务器使用什么" | roots（范围）、sampling（LLM 回调）、elicitation（用户输入） |
| JSON-RPC 2.0 | "线路格式" | 对称的请求/响应/通知信封 |
| `initialize` 握手 | "能力协商" | 第一对消息；服务器和客户端声明各自支持的功能 |
| `tools/list` | "发现" | 客户端向服务器查询当前的工具集 |
| `tools/call` | "调用" | 客户端请求服务器用参数执行工具 |
| `notifications/*_changed` | "变更事件" | 服务器告知客户端其原语列表已变化 |
| Content block | "类型化结果" | 工具结果中的 `{type: "text" \| "image" \| "resource" \| "ui_resource"}` |
| SEP | "规范演进提案" | 命名的草案提案（如 SEP-1686 用于异步 Tasks） |

## 延伸阅读

- [Model Context Protocol — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 权威规范文档
- [Model Context Protocol — Architecture concepts](https://modelcontextprotocol.io/docs/concepts/architecture) — 六原语心智模型
- [Anthropic — Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) — 2024 年 11 月发布公告
- [MCP blog — First MCP anniversary](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — 一周年回顾和 2025-11-25 规范变更
- [WorkOS — MCP 2025-11-25 spec update](https://workos.com/blog/mcp-2025-11-25-spec-update) — SEP-1686、1036、1577、835 和 1724 的总结
