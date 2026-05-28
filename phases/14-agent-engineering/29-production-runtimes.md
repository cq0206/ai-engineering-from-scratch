# 生产运行时：队列、事件与定时调度

> 生产级智能体会运行在六种运行时形态上：请求-响应、流式、持久执行、基于队列的后台执行、事件驱动，以及定时调度。先选形态，再选框架。可观测性在每一种形态里都是承重层。

**类型：** 学习
**语言：** Python（标准库）
**前置条件：** 第 14 阶段 · 13（LangGraph），第 14 阶段 · 22（语音）
**时长：** ~60 分钟

## 学习目标

- 说出六种生产运行时形态，并将每一种对应到一个框架 / 产品模式。
- 解释为什么持久执行（LangGraph）对长时程任务很重要。
- 描述事件驱动运行时，以及 Claude Managed Agents 何时适合它。
- 解释为何“可观测性是承重层”对多步智能体成立。

## 问题

生产环境中的智能体会以 Jupyter Notebook 无法显露的方式失败：第 37 步网络超时、用户在语音通话中途挂断、定时任务在机器重启时中止、后台工作进程内存耗尽。运行时形态决定了哪些故障是可以存活下来的。

## 概念

### 请求—响应

- 同步 HTTP。用户等待完成。
- 只适用于短任务（&lt;30s）。
- 技术栈：Agno（Python + FastAPI）、Mastra（TypeScript + Express/Hono/Fastify/Koa）。
- 可观测性：标准 HTTP 访问日志与 OTel 跨度（span）。

### 流式

- 使用 SSE 或 WebSocket 进行渐进式输出。
- LiveKit 将其扩展到语音 / 视频的 WebRTC（第 22 课）。
- 技术栈：任何支持流式输出的框架 + 能处理 SSE/WS 的前端。
- 可观测性：逐块耗时、首个令牌延迟、尾延迟。

### 持久执行

- 每一步之后都对状态做检查点；失败后自动恢复。
- AutoGen v0.4 的 actor 模型将故障隔离到单个智能体（第 14 课）。
- LangGraph 的核心差异化能力（第 13 课）。
- 当步骤数量未知且恢复成本高时，它是必需的。

### 队列式 / 后台

- 任务进入队列，由工作进程取走处理，结果通过 webhook 或发布/订阅回流。
- 对长时程智能体至关重要（每个任务有数十到数百步，见 Anthropic 的 Computer Use 公告）。
- 技术栈：Celery（Python）、BullMQ（Node）、SQS + Lambda（AWS）、自定义方案。
- 可观测性：队列深度、单任务延迟分布、DLQ 大小。

### 事件驱动

- 智能体订阅触发器：新邮件、PR 打开、定时触发。
- Claude Managed Agents 原生覆盖了这种形态（第 17 课）。
- CrewAI Flows（第 15 课）把事件驱动的确定性工作流组织起来。
- 可观测性：触发源、事件到启动的延迟、智能体延迟。

### 定时调度

- 周期性运行的定时调度型智能体。
- 与持久执行结合，这样夜间任务失败后能在下一个周期接着跑。
- 技术栈：Kubernetes CronJob + 持久执行框架；托管方案（Render cron、Vercel cron）。

### 2026 年部署模式

- **CrewAI Flows** 用于事件驱动的生产环境。
- **Agno** 的无状态 FastAPI 适合 Python 微服务。
- **Mastra** 的服务器适配器（Express、Hono、Fastify、Koa）适合嵌入式集成。
- **Pipecat Cloud / LiveKit Cloud** 用于托管语音（第 22 课）。
- **Claude Managed Agents** 用于托管的长时运行异步任务。

### 可观测性是承重层

如果没有 OpenTelemetry GenAI 追踪跨度（spans，第 23 课）再加上 Langfuse/Phoenix/Opik 后端（第 24 课），你就无法调试一个在第 40 步失败的多步智能体。这对生产环境不是可选项，而是必需品。它决定了你是“快速定位问题”，还是“只能从头重放并加更多日志”。

### 生产运行时会在哪里失败

- **形态选错。** 给一个 5 分钟任务选了请求-响应。用户挂断；工作进程堆积；重试叠加。
- **没有 DLQ。** 队列工作进程没有死信队列。失败任务直接消失。
- **后台工作不透明。** 后台智能体运行时不导出运行轨迹。要等用户报错你才知道失败。
- **跳过持久状态。** 任何超过 30 秒且你无法接受从头重启的运行，都需要持久执行。

## 动手构建

`code/main.py` 是一个仅用标准库实现的多形态演示：

- 请求-响应端点（普通函数）。
- 流式处理器（生成器）。
- 带 DLQ 的队列工作进程。
- 事件触发器注册表。
- 定时调度器。

运行它：

```bash
python3 code/main.py
```

输出：五段运行轨迹，展示同一任务在每种形态下的行为。智能体逻辑相同，只是外层壳不同。持久执行（第六种形态）有意放在第 13 课中通过 LangGraph 的检查点机制讲解。

## 如何使用

- **请求-响应** 用于聊天式 UX。
- **流式** 用于渐进式响应。
- **持久执行** 用于长时程任务。
- **队列** 用于批处理 / 异步 / 长时间运行任务。
- **事件** 用于让智能体对外部事件作出反应。
- **定时调度（Cron）** 用于例行维护（记忆整合、评估、成本报告）。

## 交付

`outputs/skill-runtime-shape.md` 会为一个任务选择合适的运行时形态，并接好相应的可观测性要求。

## 练习

1. 把你第 01 课的 ReAct 循环迁移到你技术栈里的六种形态。哪一种形态对应哪一种产品界面？
2. 给队列演示加上 DLQ。模拟 10% 的任务失败；把 DLQ 大小展示出来。
3. 写一个由定时任务触发的评估智能体，每晚对当天最重要的 20 条运行轨迹运行评估。
4. 实现带背压（backpressure）的流式输出：如果客户端很慢，就暂停智能体。这会如何影响轮次预算？
5. 阅读 Claude Managed Agents 文档。你会在什么情况下把自托管的长时程智能体迁移到托管方案？

## 关键术语

| 术语 | 人们常说什么 | 它实际意味着什么 |
|------|--------------|------------------|
| 请求-响应 | “同步” | 用户等待；只适合短任务 |
| 流式 | “SSE / WS” | 渐进式输出；UX 更好；可逐块观察延迟 |
| 持久执行 | “从故障恢复” | 状态已写入检查点；从上一步重新开始 |
| 基于队列 | “后台任务” | 生产者 / 工作进程池 / DLQ |
| 事件驱动 | “触发式” | 智能体对外部事件作出反应 |
| DLQ | “死信队列” | 失败任务的停放区 |
| Claude Managed Agents | “托管式运行支架” | Anthropic 托管的长时运行异步方案，带缓存 + 压缩 |

## 延伸阅读

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 持久执行细节
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — 托管的长时异步运行
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — “每个任务往往需要数十到数百步”
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — actor 模型的故障隔离

