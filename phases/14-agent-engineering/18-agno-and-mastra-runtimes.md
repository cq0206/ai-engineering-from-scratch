# Agno 与 Mastra：生产运行时

> Agno（Python）与 Mastra（TypeScript）是 2026 年的一组生产运行时搭档。Agno 追求微秒级智能体实例化，以及无状态的 FastAPI 后端。Mastra 则在 Vercel AI SDK 基础上提供智能体、工具、工作流、统一模型路由和复合存储。

**类型：** 学习
**语言：** Python、TypeScript
**前置条件：** 第 14 阶段 · 01（智能体循环）、第 14 阶段 · 13（LangGraph）
**耗时：** ~45 分钟

## 学习目标

- 识别 Agno 的性能目标，以及它们在什么场景下才重要。
- 说出 Mastra 的三个原语——Agents、Tools、Workflows——以及它支持的服务器适配器。
- 解释为什么“无状态、会话作用域（session-scoped）的 FastAPI 后端”是 Agno 推荐的生产路径。
- 根据技术栈（Python 优先还是 TypeScript 优先）选择 Agno 或 Mastra。

## 问题

LangGraph、AutoGen、CrewAI 都偏重框架。那些只想要“把智能体循环放进我的运行时里，越快越好”的团队，会选择 Agno（Python）或 Mastra（TypeScript）。两者都用一部分框架自带原语，换取原始速度和与周边技术栈更紧密的贴合。

## 概念

### Agno

- Python 运行时，前身是 Phi-data。
- “没有图、没有链、没有繁琐模式——只有纯 Python。”
- 文档中的性能目标：约 ~2μs 智能体实例化、每个智能体约 ~3.75 KiB 内存、约 ~23 个模型提供方。
- 生产路径：无状态、会话作用域的 FastAPI 后端。每个请求都启动一个全新智能体；会话状态存放在数据库里。
- 原生支持多模态（文本、图像、音频、视频、文件）以及智能体式 RAG（agentic RAG）。

当你每秒有成千上万个短生命周期智能体时（如聊天汇聚、评估流水线），这些性能目标就很重要。当一个智能体要跑 10 分钟时，它们就没那么重要了。

### Mastra

- TypeScript，构建于 Vercel AI SDK 之上。
- 三个原语：**Agents**、**Tools**（Zod 类型化）、**Workflows**。
- 统一模型路由——横跨 94 个提供方的 3,300+ 模型（2026 年 3 月）。
- 复合存储：将记忆、工作流、可观测性分别写入不同后端；在大规模可观测性场景下推荐 ClickHouse。
- 采用 Apache 2.0，但源码中的 `ee/` 目录使用源码可读企业许可证。
- 提供 Express、Hono、Fastify、Koa 服务器适配器；对 Next.js 与 Astro 提供一等集成。
- 自带 Mastra Studio（localhost:4111）用于调试。
- 在 1.0（2026 年 1 月）时拥有 22k+ GitHub star 和 300k+ 周 npm 下载量。

### 定位

它们都不是在试图成为 LangGraph。它们竞争的是：

- **语言契合度。** Agno 适合以 Python 为主的团队；Mastra 适合以 TypeScript 为主的团队。
- **运行时体验。** Agno = 接近零开销；Mastra = 与 Vercel 生态深度集成。
- **可观测性。** 两者都能与 Langfuse / Phoenix / Opik（第 24 课）集成，但 Mastra Studio 是官方原生能力。

### 何时选择各自

- **Agno** —— Python 后端、短生命周期智能体很多、性能要求强、团队本身就是 FastAPI 技术栈。
- **Mastra** —— TypeScript 后端、部署在 Next.js / Vercel 上、需要统一的多提供方模型路由、需要 Zod 类型化工具。
- **LangGraph**（第 13 课）—— 当持久状态与显式图推理比原始速度更重要时。
- **OpenAI / Claude Agent SDK** —— 当你更想要提供方已经产品化的整体形态时（第 16–17 课）。

### 这种模式会在哪些地方出问题

- **为了性能而性能。** 因为 “2μs” 听起来很厉害就选择 Agno，但实际工作负载是每个请求只跑一次很慢的智能体调用。此时开销根本不是瓶颈。
- **生态锁定。** Mastra 的 Vercel 风格集成在 Vercel 上是优势，换到别处可能就是负担。
- **企业许可证混淆。** Mastra 的 `ee/` 目录是源码可读许可，不是 Apache 2.0。如果你打算分叉（fork），一定要仔细阅读许可证。

## 动手构建

本课主要是对比型内容——没有哪一个单独的代码制品能同时公平呈现这两个框架。请查看 `code/main.py` 中的并排玩具示例：同一个“运行智能体、流式输出、持久化会话”的最小流程分别实现了两遍（一次采用 Agno 形状，一次采用 Mastra 形状）。

运行：

```
python3 code/main.py
```

你会看到两条结构不同、但功能等价的执行轨迹。

## 使用

- **Agno** —— 适合需要速度和 FastAPI 形状的 Python 后端。
- **Mastra** —— 适合拥有大量提供方与工作流原语需求的 TypeScript 后端。
- 两者都提供原生可观测性挂钩，也都能与 Langfuse 集成。

## 交付

`outputs/skill-runtime-picker.md` 会根据技术栈、延迟预算和运维形态，在 Agno、Mastra、LangGraph 或某个提供方 SDK 之间做选择。

## 练习

1. 阅读 Agno 文档。把标准库版（stdlib）的 ReAct 循环（第 01 课）迁移到 Agno。有哪些东西消失了？哪些保留了？
2. 阅读 Mastra 文档。把同一个循环迁移到 Mastra。工具类型化发生了什么变化（Zod 与无类型）？
3. 做基准测试：在你的技术栈上测量智能体实例化延迟。Agno 的 2μs 对你的工作负载真的重要吗？
4. 设计一次迁移：如果你现在在 Python 中运行的是 CrewAI，迁移到 Agno 会破坏什么？
5. 阅读 Mastra 的 `ee/` 许可证条款。哪些限制会影响开源分叉（fork）？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| Agno | “快速 Python 智能体” | 无状态、会话作用域的智能体运行时 |
| Mastra | “基于 Vercel AI SDK 的 TypeScript 智能体” | 智能体 + 工具 + 工作流 + 模型路由 |
| 统一模型路由 | “多提供方访问” | 一个客户端访问 94 个提供方的 3,300+ 模型 |
| 复合存储 | “多个后端” | 记忆 / 工作流 / 可观测性分别接到不同存储 |
| Mastra Studio | “本地调试器” | 位于 localhost:4111、用于检查智能体的界面 |
| 源码可读许可 | “不是 OSS” | 许可证允许阅读源码，但限制商业使用 |

## 延伸阅读

- [Agno Agent Framework docs](https://www.agno.com/agent-framework) —— 性能目标与 FastAPI 集成
- [Mastra docs](https://mastra.ai/docs) —— 原语、服务器适配器、Model Router
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) —— 有状态图替代方案
- [Comet Opik](https://www.comet.com/site/products/opik/) —— Mastra 集成中引用的可观测性对比
