# 为什么要用多智能体？

> 一个智能体撞上墙时，聪明的做法不是换一个更大的智能体——而是使用更多智能体。

**Type:** 学习
**Languages:** TypeScript
**Prerequisites:** 第 14 阶段（Agent Engineering）
**Time:** ~60 分钟

## 学习目标

- 识别单智能体上限 (single-agent ceiling)（上下文溢出、混合专长、串行瓶颈），并解释何时应将任务拆分为多个智能体
- 比较编排模式 (orchestration patterns)（流水线、并行扇出、监督者、分层），并为给定任务结构选择合适模式
- 设计具有清晰角色边界、共享状态 (shared state) 和通信契约 (communication contract) 的多智能体系统
- 分析多智能体复杂性 (multi-agent complexity)（延迟、成本、调试难度）与单智能体简洁性之间的权衡

## 问题

你在第 14 阶段构建了一个单智能体。它能工作。它可以读取文件、运行命令、调用 API，并对结果进行推理。然后你把它指向一个真实代码库：200 个文件、三种语言、依赖基础设施的测试，以及在写代码前需要先研究外部 API 的需求。

智能体开始吃不消。不是因为 LLM 很笨，而是因为任务已经超出了一个智能体循环所能处理的范围。上下文窗口被文件内容塞满。智能体忘记了它在 40 次工具调用前读过什么。它试图同时扮演研究员、程序员和审查员，结果三个角色都做得很差。

这就是单智能体上限。每当任务需要以下任一条件时，你都会撞上它：

- **超出单个窗口可容纳的上下文** —— 读取 50 个文件会轻松超过 200k tokens
- **不同阶段需要不同专长** —— 研究所需的提示方式与代码生成不同
- **可以并行进行的工作** —— 既然可以同时读取三个文件，为什么还要顺序读？

## 概念

### 单智能体上限

单个智能体只有一个循环、一个上下文窗口、一个系统提示。想象一下：

```
┌─────────────────────────────────────────┐
│              单智能体                   │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │           上下文窗口              │  │
│  │                                   │  │
│  │  研究笔记                         │  │
│  │  + 代码文件                       │  │
│  │  + 测试输出                       │  │
│  │  + 审查反馈                       │  │
│  │  + API 文档                       │  │
│  │  + ...                            │  │
│  │                                   │  │
│  │  ██████████████████████ 已满 ███  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  一个系统提示试图同时覆盖               │
│  研究 + 编码 + 审查 + 测试              │
│                                         │
│  结果：样样都会，样样平庸               │
└─────────────────────────────────────────┘
```

会有三件事出问题：

1. **上下文饱和** —— 工具结果不断堆积。到第 30 轮时，智能体已经消耗了 150k tokens 的文件内容、命令输出和先前推理。第 5 轮中的关键细节会丢失。

2. **角色混乱** —— 一个写着“你是研究员、程序员、审查员和测试员”的系统提示，会产出一个半研究、半写码、且永远审不完代码的智能体。

3. **串行瓶颈** —— 智能体先读文件 A，再读文件 B，再读文件 C。三次串行 LLM 调用，三次串行工具执行，没有并行性。

### 多智能体解决方案

把工作拆开。给每个智能体一个任务、一个上下文窗口，以及一个为该任务专门调优的系统提示：

```
┌──────────────────────────────────────────────────────────┐
│                        编排器                            │
│                                                          │
│  “构建一个用于用户管理的 REST API”                       │
│                                                          │
│         ┌──────────┬──────────┬──────────┐               │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │ 研究员   │ │ 程序员   │ │ 审查员   │ │ 测试员   │  │
│   │          │ │          │ │          │ │          │  │
│   │ 阅读     │ │ 编写     │ │ 检查     │ │ 运行     │  │
│   │ 文档、   │ │ 代码     │ │ 代码     │ │ 测试、   │  │
│   │ 查找     │ │ 基于研究 │ │ 质量、   │ │ 报告     │  │
│   │ 模式     │ │ + 规格   │ │ 发现     │ │ 结果     │  │
│   │          │ │          │ │ bug      │ │          │  │
│   └─────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│         │           │            │             │         │
│         └───────────┴────────────┴─────────────┘         │
│                          │                               │
│                       合并结果                            │
└──────────────────────────────────────────────────────────┘
```

每个智能体都有：
- 一个聚焦的系统提示（“你是代码审查员。你唯一的工作就是找 bug。”）
- 自己独立的上下文窗口（不会被其他智能体的工作污染）
- 明确的输入/输出契约（接收研究笔记，输出代码）

### 这样做的真实系统

**Claude Code subagents** —— 当 Claude Code 用 `Task` 生成 subagent 时，它会创建一个带作用域任务的子智能体。父智能体保持自身上下文整洁。子智能体完成聚焦工作并返回摘要。

**Devin** —— 运行一个规划智能体、一个编码智能体和一个浏览器智能体。规划智能体把工作拆成步骤。编码智能体写代码。浏览器智能体研究文档。它们各自拥有独立上下文。

**多智能体编码团队（SWE-bench）** —— 在 SWE-bench 上表现最好的系统，会使用一个阅读代码库的研究员、一个设计修复方案的规划者，以及一个实现修复的程序员。单智能体系统得分更低。

**ChatGPT Deep Research** —— 并行生成多个搜索智能体，每个从不同角度探索，最后再综合结果。

### 光谱

多智能体不是二元选择，而是一条光谱：

```
简单 ───────────────────────────────────────────── 复杂

 单智能体      子智能体       流水线        团队         群体

  ┌───┐       ┌───┐        ┌───┐───┐    ┌───┐───┐    ┌─┐┌─┐┌─┐
  │ A │       │ A │        │ A │ B │    │ A │ B │    │ ││ ││ │
  └───┘       └─┬─┘        └───┘─┬─┘    └─┬─┘─┬─┘    └┬┘└┬┘└┬┘
                │                │        │   │       ┌┴──┴──┴┐
              ┌─┴─┐          ┌───┘───┐    │   │       │共享状态│
              │ a │          │ C │ D │  ┌─┴───┴─┐    └───────┘
              └───┘          └───┘───┘  │ 消息总线│
                                         │       │
  1 个循环      父任务 +      分阶段      └───────┘    N 个对等体，
  1 个上下文    子任务        执行                       涌现行为
                                         显式角色
```

**单智能体** —— 一个循环，一个提示。适合简单任务。

**子智能体** —— 父智能体为聚焦子任务生成子智能体。父智能体维护计划。子智能体回报结果。这就是 Claude Code 的做法。

**流水线** —— 智能体按顺序运行。智能体 A 的输出成为智能体 B 的输入。适合分阶段工作流：研究 -> 代码 -> 审查 -> 测试。

**团队** —— 智能体通过共享消息总线并行运行。每个都有自己的角色。由一个编排器协调。当不同技能需要同时发挥作用时非常适合。

**群体** —— 许多相同或近似相同的智能体共享状态。没有固定的编排器。智能体从队列中领取工作。适合高吞吐并行任务。

### 四种多智能体模式

#### 模式 1：流水线

```
输入 ──▶ 智能体 A ──▶ 智能体 B ──▶ 智能体 C ──▶ 输出
            （研究）      （代码）      （审查）
```

每个智能体转换数据并将其向前传递。容易理解。如果某个阶段失败，后续阶段都会被阻塞。

#### 模式 2：扇出 / 扇入

```
                 ┌──▶ 智能体 A ──┐
                 │                │
输入 ──▶ 拆分 ──┼──▶ 智能体 B ──┼──▶ 合并 ──▶ 输出
                 │                │
                 └──▶ 智能体 C ──┘
```

将工作拆分到多个并行智能体上，再合并结果。适合可以分解为独立子任务的工作。

#### 模式 3：编排器-工作者

```
                     ┌──────────┐
                     │ 编排器   │
                     └──┬───┬───┘
                  任务 │   │ 任务
                  ┌─────┘   └─────┐
                  ▼               ▼
            ┌──────────┐   ┌──────────┐
            │ 工作者 A │   │ 工作者 B │
            └──────────┘   └──────────┘
```

一个聪明的编排器决定要做什么，把任务委派给工作者，并综合结果。编排器本身也是一个智能体，并拥有生成工作者的工具。

#### 模式 4：对等群体

```
         ┌───┐ ◄──── 消息 ────▶ ┌───┐
         │ A │                   │ B │
         └─┬─┘                   └─┬─┘
           │                        │
      消息 │    ┌───────────┐       │ 消息
           └───▶│  共享状态  │◄─────┘
                │   / 队列   │
           ┌───▶│           │◄────┐
           │    └───────────┘     │
      消息 │                        │ 消息
         ┌─┴─┐                   ┌─┴─┐
         │ C │ ◄──── 消息 ────▶ │ D │
         └───┘                   └───┘
```

没有中央编排器。智能体点对点通信。决策从交互中涌现。更难调试，但可扩展到大量智能体。

### 什么时候**不要**使用多智能体

多智能体会增加复杂性。智能体之间的每条消息都可能成为失败点。调试会从“读一段对话”变成“追踪五个智能体之间的消息”。

**以下情况应保持单智能体：**
- 任务能装进一个上下文窗口（工作数据低于约 100k tokens）
- 不需要为不同阶段使用不同系统提示
- 串行执行已经足够快
- 任务足够简单，拆分带来的开销大于价值

**复杂性成本：**
- 每个智能体边界都是一次有损压缩：智能体 A 的完整上下文会被总结成一条消息发给智能体 B
- 协调逻辑（谁做什么、何时做、按什么顺序做）本身就是 bug 来源
- 延迟会上升：N 个智能体至少意味着 N 次串行 LLM 调用，如果它们还要来回沟通，次数会更多
- 成本会倍增：每个智能体都会独立消耗 tokens

经验法则：如果一个任务少于 20 次工具调用，且能装进 100k tokens，就保持单智能体。

## 动手构建

### 第 1 步：过载的单智能体

下面是一个试图包办一切的单智能体。它有一个巨大的系统提示，以及一个同时存放研究、代码和审查内容的上下文窗口：

```typescript
type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `You are a full-stack developer. You must:
1. Research the requirements
2. Write the code
3. Review the code for bugs
4. Write tests
Do ALL of these in a single conversation.`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `Research: ${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `Given this research:\n${contextWindow.join("\n")}\n\nNow write code for: ${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `Given all previous context:\n${contextWindow.join("\n")}\n\nReview the code.`
  );
  contextWindow.push(review.output);
  totalTokens += review.tokens;
  totalToolCalls += review.calls;

  return {
    content: contextWindow.join("\n---\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

这种方法的问题：
- 上下文窗口会随着每个阶段不断膨胀。到审查步骤时，它同时包含研究笔记、代码和先前推理。
- 系统提示过于泛化，无法针对每个阶段单独调优。
- 没有任何部分能并行运行。

### 第 2 步：专家型智能体

现在把它拆开。每个智能体只负责一个任务：

```typescript
type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

function createSpecialist(name: string, systemPrompt: string): SpecialistAgent {
  return {
    name,
    systemPrompt,
    run: async (input: string) => {
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

const researcher = createSpecialist(
  "researcher",
  "You are a technical researcher. Read documentation, find patterns, and summarize findings. Output only the facts needed for implementation."
);

const coder = createSpecialist(
  "coder",
  "You are a senior TypeScript developer. Given requirements and research notes, write clean, tested code. Nothing else."
);

const reviewer = createSpecialist(
  "reviewer",
  "You are a code reviewer. Find bugs, security issues, and logic errors. Be specific. Cite line numbers."
);
```

每个专家都有聚焦的提示。每个专家都拥有干净的上下文窗口，只接收自己需要的输入。

### 第 3 步：通过消息协调

用显式消息传递把这些专家连接起来：

```typescript
type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

async function multiAgentApproach(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const researchResult = await researcher.run(task);
  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "coder",
    to: "reviewer",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  const reviewerInput = messages
    .filter((m) => m.to === "reviewer")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const reviewResult = await reviewer.run(reviewerInput);
  messages.push({
    from: "reviewer",
    to: "orchestrator",
    content: reviewResult.content,
    timestamp: Date.now(),
  });
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages.map((m) => `[${m.from} -> ${m.to}]: ${m.content}`).join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

每个智能体只接收发给自己的消息。没有上下文污染。研究员阅读的 50k tokens 文档内容永远不会进入审查员的上下文。

### 第 4 步：比较

```typescript
async function compare() {
  const task = "Build a rate limiter middleware for an Express.js API";

  console.log("=== Single Agent ===");
  const single = await singleAgentApproach(task);
  console.log(`Tokens: ${single.tokensUsed}`);
  console.log(`Tool calls: ${single.toolCalls}`);

  console.log("\n=== Multi-Agent ===");
  const multi = await multiAgentApproach(task);
  console.log(`Tokens: ${multi.tokensUsed}`);
  console.log(`Tool calls: ${multi.toolCalls}`);
}
```

多智能体版本会使用更多总 tokens（三个智能体，三次独立 LLM 调用），但每个智能体的上下文都保持干净。由于系统提示是专门化的，每个阶段的质量都会提高。

## 使用它

本课会产出一个可复用提示，用于判断何时该切换到多智能体。参见 `outputs/prompt-multi-agent-decision.md`。

## 练习

1. 添加第四个专家：一个“测试员”智能体，它从程序员接收代码、从审查员接收审查反馈，然后编写测试
2. 修改流水线，让审查员可以把反馈发回给程序员，形成修订循环（最多 2 轮）
3. 将串行流水线改成扇出：并行运行研究员和“需求分析员”智能体，在把结果交给程序员之前先合并它们的输出

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 群体 | “AI 智能体的蜂巢思维” | 一组具有共享状态且没有固定领导者的对等智能体。行为从局部交互中涌现。 |
| 编排器 | “老板智能体” | 一种工具中包含生成和管理其他智能体能力的智能体。它负责规划和委派，但不一定亲自完成实际工作。 |
| 协调器 | “交通警察” | 一个非智能体组件（通常只是代码，而不是 LLM），按照规则在智能体之间路由消息。 |
| 共识 | “智能体达成一致” | 一种在继续推进前必须由多个智能体达成一致的协议。用于解决相互冲突的输出。 |
| 涌现行为 | “智能体自己想出来了” | 由智能体交互产生、但未被显式编程的系统级模式。它可能有用，也可能有害。 |
| 扇出 / 扇入 | “智能体版 map-reduce” | 将任务拆分给并行智能体（扇出），然后合并它们的结果（扇入）。 |
| 消息传递 | “智能体彼此交谈” | 智能体之间的通信机制：把结构化数据从一个智能体发给另一个智能体，以此替代共享上下文窗口。 |

## 延伸阅读

- [The Landscape of Emerging AI Agent Architectures](https://arxiv.org/abs/2409.02977) - 多智能体模式综述
- [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155) - Microsoft 的多智能体对话框架
- [Claude Code subagents documentation](https://docs.anthropic.com/en/docs/claude-code) - Claude Code 如何用 Task 进行委派
- [CrewAI documentation](https://docs.crewai.com/) - 基于角色的多智能体框架
