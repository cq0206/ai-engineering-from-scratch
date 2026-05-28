# Claude Agent SDK：子智能体与会话存储

> Claude Agent SDK 是 Claude Code 运行支架的库形态。它内置工具、用于上下文隔离的子智能体、钩子、W3C trace 传播，以及与会话存储对等的能力。Claude Managed Agents 则是面向长时间异步工作的托管式替代方案。

**类型：** 学习 + 构建
**语言：** Python（stdlib）
**前置条件：** Phase 14 · 01（Agent Loop）、Phase 14 · 10（Skill Libraries）
**耗时：** ~75 分钟

## 学习目标

- 解释 Anthropic Client SDK（原始 API）与 Claude Agent SDK（运行支架形态）之间的区别。
- 描述子智能体——并行化与上下文隔离——以及何时应该使用它们。
- 说出 Python SDK 的会话存储接口（`append`、`load`、`list_sessions`、`delete`、`list_subkeys`）以及 `--session-mirror` 的作用。
- 用 stdlib 实现一个运行支架，包含内置工具、带隔离上下文的子智能体生成、生命周期钩子，以及会话存储。

## 问题

原始 LLM API 只能给你一次往返。而一个生产级智能体需要工具执行、MCP 服务器、生命周期钩子、子智能体生成、会话持久化，以及 trace 传播。Claude Agent SDK 把这种整体形态以库的方式提供出来——也就是 Claude Code 正在使用的同一套运行支架，只不过对自定义智能体开放了。

## 概念

### 客户端 SDK 与 Agent SDK

- **Client SDK（`anthropic`）。** 原始 Messages API。循环、工具和状态都由你自己负责。
- **Agent SDK（`claude-agent-sdk`）。** 内置工具执行、MCP 连接、生命周期钩子、子智能体生成、会话存储。也就是“作为库提供的 Claude Code 循环”。

### 内置工具

SDK 开箱即用提供 10+ 个工具：文件读写、shell、grep、glob、web fetch 等。自定义工具则通过标准的工具结构接口注册。

### 子智能体

Anthropic 文档给出了两个用途：

1. **并行化。** 并发运行彼此独立的工作。“为这 20 个模块分别找到测试文件”就是 20 个可并行的子智能体任务。
2. **上下文隔离。** 子智能体使用自己的上下文窗口；只有结果返回给协调者。协调者的上下文预算因此得以保留。

Python SDK 最近新增了：`list_subagents()`、`get_subagent_messages()`，用于读取子智能体的执行记录。

### 会话存储

与 TypeScript 协议保持对等：

- `append(session_id, message)` —— 追加一轮消息。
- `load(session_id)` —— 恢复对话。
- `list_sessions()` —— 枚举会话。
- `delete(session_id)` —— 删除，并级联删除子智能体会话。
- `list_subkeys(session_id)` —— 列出子智能体键。

`--session-mirror`（CLI 标志）会在流式输出过程中，把对话转录同步写入一个外部文件，便于调试。

### 生命周期钩子

你可以注册的生命周期钩子包括：

- `PreToolUse`、`PostToolUse` —— 对工具调用做闸门控制或审计。
- `SessionStart`、`SessionEnd` —— 执行初始化与清理。
- `UserPromptSubmit` —— 在模型看到用户输入之前先进行处理。
- `PreCompact` —— 在上下文压缩前运行。
- `Stop` —— 智能体退出时清理。
- `Notification` —— 旁路通知。

钩子就是这类生产工作流（以及第 14 阶段中类似系统）添加横切能力的方式。

### W3C 跟踪上下文

调用方上活跃的 OTel span 会通过 W3C trace 上下文请求头传播到 CLI 子进程中。这样，整个多进程执行轨迹会在你的后端里表现为同一条追踪。

### Claude Managed Agents

这是托管式替代方案（测试版请求头：`managed-agents-2026-04-01`）。适合长时间异步任务，内置提示缓存，也内置上下文压缩。你用对控制力的让渡，换取托管基础设施。

### 这种模式会在哪些地方出问题

- **子智能体过度生成。** 为 100 个极小任务生成 100 个子智能体。最终开销会压倒收益。应改为分批。
- **钩子蔓延。** 每个团队都往里加钩子；启动时间越来越长。应按季度审查钩子。
- **会话膨胀。** 会话越积越多，体积持续增长。结合 `list_sessions` 和过期策略来治理。

## 动手构建

`code/main.py` 用 stdlib 实现了 SDK 的整体形态：

- `Tool`、`ToolRegistry`，内置 `read_file`、`write_file`、`list_dir`。
- `Subagent` —— 私有上下文、隔离运行、返回结果。
- `SessionStore` —— append、load、list、delete、list_subkeys。
- `Hooks` —— `pre_tool_use`、`post_tool_use`、`session_start`、`session_end`。
- 一个演示：主智能体并行生成 3 个子智能体（各自隔离），聚合结果，并持久化会话。

运行：

```
python3 code/main.py
```

执行轨迹会展示子智能体的上下文隔离（协调者的上下文规模保持有界）、钩子执行，以及会话持久化。

## 使用

- **Claude Agent SDK** 适合以 Claude 为中心、希望获得 Claude Code 运行支架形态的产品。
- **Claude Managed Agents** 适合托管式长时间异步工作。
- **OpenAI Agents SDK**（第 16 课）适合作为以 OpenAI 为中心的对应方案。
- **LangGraph + 自定义工具** 适合你更想要图形化状态机时。

## 交付

`outputs/skill-claude-agent-scaffold.md` 会搭建一个 Claude Agent SDK 应用脚手架，包含子智能体、钩子、会话存储、MCP 服务器挂接，以及 W3C trace 传播。

## 练习

1. 增加一个子智能体生成器：把 20 个任务按每批 5 个并行子智能体进行分组。衡量协调者上下文大小，与“一任务一个子智能体”相比有何差异。
2. 实现一个 `PreToolUse` 钩子，对 `write_file` 调用做限流（每个会话每分钟 5 次）。把行为轨迹打印出来。
3. 将 `list_subkeys` 接上可视化，渲染出一棵子智能体树。深层嵌套会是什么样？
4. 把这个玩具示例迁移到真实的 `claude-agent-sdk` Python 包。工具注册方式发生了什么变化？
5. 阅读 Claude Managed Agents 文档。你会在什么情况下从自托管切换到托管？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| Agent SDK | “作为库的 Claude Code” | 运行支架形态：工具、MCP、钩子、子智能体、会话存储 |
| 子智能体 | “子代理” | 独立上下文、独立预算；结果向上汇总 |
| 会话存储 | “对话数据库” | 持久化、加载、枚举、删除消息，并级联处理子智能体 |
| 钩子 | “生命周期回调” | 工具前/后、会话、提示提交、压缩、停止等回调 |
| W3C 跟踪上下文 | “跨进程追踪” | 父 span 会传播到 CLI 子进程 |
| Managed Agents | “托管式运行支架” | 由 Anthropic 托管的长时间异步工作环境 |
| `--session-mirror` | “转录镜像” | 在流式输出时把会话轮次写入外部文件 |
| MCP 服务器 | “工具界面” | 挂接到智能体上的外部工具/资源来源 |

## 延伸阅读

- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) —— 作为库提供的 Claude Code
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) —— 生产模式
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) —— 托管式替代方案
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) —— 对应方案
