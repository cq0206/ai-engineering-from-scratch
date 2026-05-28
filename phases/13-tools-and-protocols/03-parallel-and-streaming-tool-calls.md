# 并行工具调用 (Parallel Tool Calls) 与工具流式处理 (Streaming with Tools)

> 把三个彼此独立的天气查询串行执行，就需要三次往返 (round trips)。把它们并行运行，总耗时就会收敛到最慢的那一次调用。现在所有前沿模型提供商都会在单轮中发出多个工具调用。收益很真实；但底层实现并不简单。本课会把两部分都讲清楚：并行扇出 (fan-out) 与流式参数重组 (streamed-argument reassembly)，重点说明 id 关联陷阱 (id-correlation trap)。

**类型：** 构建
**语言：** Python（stdlib、线程池 + 流式处理框架）
**前置条件：** Phase 13 · 02（函数调用深度解析）
**时长：** ~75 分钟

## 学习目标

- 解释为什么需要 `parallel_tool_calls: true`，以及何时应关闭它。
- 在并行扇出期间，把流式参数块关联到正确的工具调用 id。
- 将部分 `arguments` 字符串重组成完整 JSON，而不是过早解析。
- 运行一个三城市天气基准测试，展示串行与并行的延迟差异。

## 问题

如果没有并行调用，一个代理回答“Bengaluru、Tokyo 和 Zurich 的天气如何”时，会这样做：

```
user -> LLM
LLM -> call get_weather(Bengaluru)
host -> run executor, reply with result
LLM -> call get_weather(Tokyo)
host -> run executor, reply with result
LLM -> call get_weather(Zurich)
host -> run executor, reply with result
LLM -> final text answer
```

这需要三次 LLM 往返，每一次还要承担执行器延迟 (executor latency)。总墙钟时间大约是理想值的 4 倍。

如果使用并行调用：

```
user -> LLM
LLM -> call get_weather(Bengaluru); call get_weather(Tokyo); call get_weather(Zurich)
host -> run all three executors concurrently, reply with three results
LLM -> final text answer
```

只需要一次 LLM 往返。执行器耗时取三者中的最大值，而不是总和。OpenAI、Anthropic 和 Gemini 的生产基准测试显示，在扇出型工作负载 (fan-out workloads) 上，墙钟时间通常可减少 60% 到 70%。

代价是关联复杂度 (correlation complexity)。当三个调用乱序完成时，你返回的结果必须携带对应的 `tool_call_id`，这样模型才能正确对齐。当结果以流式方式返回时，你必须先把零散的参数片段组装成完整 JSON，再去执行。Gemini 3 添加唯一 id，部分原因正是为了解决现实中的一个问题：对同一个工具发起两次并行调用时，之前无法区分它们。

## 概念

### 启用并行

- **OpenAI。** `parallel_tool_calls: true` 默认开启。设为 `false` 可强制串行。
- **Anthropic。** 通过 `disable_parallel_tool_use: false` 启用并行（Claude 3.5 及以上默认开启）。设为 `true` 则改为串行。
- **Gemini。** 始终支持并行；`tool_config.function_calling_config.mode = "AUTO"` 让模型自行决定。

当工具存在顺序依赖（例如 `create_file` 然后 `write_file`）、某个调用的输出会影响另一个调用的输入，或限流器 (rate limiter) 无法承受扇出时，应关闭并行。

### Id 关联

模型发出的每个调用都有一个 `id`。宿主返回的每个结果也必须包含同一个 id。否则结果就会产生歧义。

- **OpenAI。** 每条 tool-role message 上带有 `tool_call_id`。
- **Anthropic。** 每个 `tool_result` block 上带有 `tool_use_id`。
- **Gemini。** 每个 `functionResponse` 上带有 `id`（Gemini 3 及以上如此；Gemini 2 通过名称匹配，这会在同名并行调用时失效）。

### 并发运行调用

宿主会让每个调用的执行器运行在各自的线程、协程 (coroutine) 或远程工作进程上。最简单的框架 (harness) 会使用线程池；生产环境更常用带 `asyncio.gather` 的 asyncio，或结构化并发 (structured concurrency)。调用完成的顺序不可预测——真正的标识符是 id。

一个常见 bug 是：按调用列表顺序返回结果，而不是按完成顺序返回。通常这也能工作，因为模型主要依赖 `tool_call_id`，但如果某个结果丢失或重复，乱序提交会让调试更困难。更好的做法是：按完成顺序返回，并显式附带 id。

### 流式工具调用

当模型以流式方式输出时，`arguments` 会分块到达。三个并行调用各自的参数块会在同一条链路上交错出现。你需要为每个 id 准备一个累加器 (accumulator)。

按提供商划分的形式如下：

- **OpenAI。** 每个块位于 `choices[0].delta.tool_calls[i].function.arguments`（部分字符串）。该块会携带 `index`（调用列表中的位置）。你要按 index 累加，在 `id` 首次出现时读取它，并在 `finish_reason = "tool_calls"` 时解析 JSON。
- **Anthropic。** 流事件依次是 `message_start`，然后每个 block 各有一个 `content_block_start`，其类型为 `tool_use`（包含 id、name 和空 input）。`content_block_delta` 事件携带 `input_json_delta` 片段。`content_block_stop` 用于结束各个 block。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 及以上）会发出带 `functionCallId` 的参数块，因此多个调用可以在流中清晰交错。Gemini 3 之前，流式返回一次只会给出一个完整调用。

### 部分 JSON 与过早解析陷阱 (parse-early trap)

在 `arguments` 尚未完整之前，你不能解析它。像 `{"city": "Beng` 这样的部分 JSON 并不合法，会直接报错。正确的门槛信号是提供商给出的“调用结束”标记：OpenAI 的 `finish_reason = "tool_calls"`、Anthropic 的 `content_block_stop`，或 Gemini 的流结束事件。只有在那之后才应尝试 `json.loads`。更稳健的方法是使用增量式 JSON 解析器 (incremental JSON parser)，在结构完整时逐步产出事件；OpenAI 的流式指南就推荐这样做，以支持展示实时 “thinking” 指示器的 UX。用括号计数来判断完整性并不可靠（引号中的括号或转义内容都会导致误判），最多只能当作非正式的调试启发。

### 乱序完成

```
call_A: fast API, returns first
call_B: slow API, returns second
call_C: median API, returns third
```

宿主的回复仍然必须带上这些 id：

```
[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

在 OpenAI 或 Anthropic 中，回复里的顺序不会影响正确性。只要 id 匹配，Gemini 也接受任意顺序。

### 基准测试：串行 vs 并行

`code/main.py` 中的框架会模拟三个执行器，延迟分别为 400、600 和 800 毫秒。串行执行总共需要 1800 毫秒。并行执行则是 max(400, 600, 800) = 800 毫秒。这个差值是恒定收益，而不是按比例变化，因此工具数量越多，节省越明显。

现实中的一个注意点是：并行调用会给下游 API 带来压力。对一个受限流保护的服务做 10 路扇出，很可能会失败。Phase 13 · 17 会讲网关级背压 (backpressure)；重试语义会在未来的阶段中讨论。

### 流式扇出的墙钟时间

如果模型本身也是流式输出，那么一旦某个调用的参数已经完整，你就可以立刻开始执行，而不必等所有调用全部结束。这是 OpenAI 文档里提到的一种优化，但并非所有 SDK 都暴露了这个能力。本课中的框架就采用了这种方式：只要模拟流产出了一个完整参数对象，宿主就立即启动该调用。

## 使用它

`code/main.py` 分为两部分。第一部分使用 `concurrent.futures.ThreadPoolExecutor` 串行和并行运行三个模拟天气调用，并打印墙钟时间。第二部分回放一个伪造的流式响应——三个并行调用的 `arguments` 片段交错出现在同一条流上——然后使用 `StreamAccumulator` 按 id 进行重组。不需要 LLM，不需要网络，只有重组逻辑本身。

你应该关注：

- 串行计时器会到 1.8 秒。相同的模拟延迟下，并行计时器会到 0.8 秒。
- 累加器会按 id 分别缓冲，因此即使块乱序到达，也能在每个调用的 JSON 完整后再解析。
- 某个 id 的参数一旦定稿，执行器就会立即启动，而不是等所有流都结束。

## 交付它

本课会产出 `outputs/skill-parallel-call-safety-check.md`。给定一个工具注册表，这个技能会审计哪些工具适合并行化、哪些存在顺序依赖、哪些会压垮下游限流——然后返回一个修订后的注册表，并为每个工具附上 `parallel_safe` 标记。

## 练习

1. 运行 `code/main.py` 并改变模拟延迟。确认并行/串行比值大致接近 `max/sum`（真实运行会因为线程调度、序列化和框架开销而略微偏离理想值）。在什么样的延迟分布下，并行开始变得不重要？

2. 扩展累加器，处理“某个调用在流式过程中被取消”的情况：丢弃其缓冲区，并发出 `cancelled` 事件。哪个提供商明确记录了这种情况？请查看 Anthropic 的 `content_block_stop` 语义，以及 OpenAI 的 `finish_reason: "length"` 行为。

3. 把线程池替换为 `asyncio.gather`。对两者做基准测试。你应该只会在执行器执行真实 I/O 时，看到 async 因上下文切换成本更低而带来的小幅收益。

4. 选两个不应并行化的工具（例如 `create_file` 然后 `write_file`）。给注册表增加一个 `ordering_dependency` 图，并据此控制并行扇出。这是“依赖感知调度”所需的最小机制，未来的 agent engineering 阶段会将其系统化。

5. 阅读 OpenAI 关于并行函数调用的章节，以及 Anthropic 的 `disable_parallel_tool_use` 文档。找出 Anthropic 建议关闭并行性的那一种真实世界工具类型。（提示：对同一资源执行有后果的变更操作。）

## 关键术语

| 术语 | 人们常说什么 | 实际含义 |
|------|--------------|----------|
| 并行工具调用 | “一轮内扇出” | 模型在单条 assistant message 中发出多个工具调用 |
| `parallel_tool_calls` | “OpenAI 的开关” | 启用或禁用多调用发射 |
| `disable_parallel_tool_use` | “Anthropic 的反向开关” | 选择退出；默认启用并行 |
| 工具调用 id | “关联句柄” | 结果消息必须回显的单次调用标识符 |
| 累加器 | “流缓冲区” | 用于保存部分 `arguments` 片段的按 id 字符串缓冲区 |
| 乱序完成 | “最快的先回来” | 并行调用会以不可预测的顺序完成；id 是粘合剂 |
| 依赖图 | “顺序约束” | 某些工具的输出会成为其他工具的输入；不能并行化 |
| 过早解析陷阱 | “JSON.parse 爆了” | 试图解析尚未完整的 `arguments` 字符串 |
| `streamFunctionCallArguments` | “Gemini 3 特性” | 为每次调用提供唯一 id 的流式参数块 |
| 按完成顺序回复 | “不要等全部结束” | 结果一到就按 id 返回 |

## 延伸阅读

- [OpenAI — Parallel function calling](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) —— 默认行为与退出开关
- [Anthropic — Tool use: implementing tool use](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implementing-tool-use) —— `disable_parallel_tool_use` 与结果批处理
- [Google — Gemini function calling parallel section](https://ai.google.dev/gemini-api/docs/function-calling) —— Gemini 3 中基于 id 关联的并行调用
- [OpenAI — Streaming responses with tools](https://platform.openai.com/docs/api-reference/responses-streaming) —— OpenAI 流中的分块参数重组
- [Anthropic — Streaming messages](https://docs.anthropic.com/en/api/messages-streaming) —— 带 `input_json_delta` 的 `content_block_delta`
