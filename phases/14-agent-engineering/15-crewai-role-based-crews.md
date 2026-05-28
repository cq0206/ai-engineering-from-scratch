# CrewAI：基于角色的团队与流程

> CrewAI 是 2026 年面向角色分工的多智能体框架。四个原语：Agent、Task、Crew、Process。两种顶层形态：团队（Crew，自主、基于角色的协作）和流程（Flow，事件驱动、确定性）。文档说得很直接：“任何面向生产的应用，都应从 Flow 开始。”

**类型：** 学习 + 构建
**语言：** Python（stdlib）
**前置条件：** Phase 14 · 12（Workflow Patterns）、Phase 14 · 14（Actor Model）
**耗时：** ~75 分钟

## 学习目标

- 说出 CrewAI 的四个原语（Agent、Task、Crew、Process）以及各自负责什么。
- 区分 Sequential、Hierarchical 和计划中的 Consensus 过程，并能为不同工作负载选择合适方案。
- 区分 Crews（自主的角色协作）与 Flows（事件驱动的确定性流程），并解释为什么文档推荐在生产中优先使用 Flow。
- 使用 `@tool` 装饰器和 `BaseTool` 子类接入工具；理解结构化输出与自由文本的取舍。
- 说出 CrewAI 的四类记忆以及各自何时值得使用。
- 用 stdlib 实现一个三智能体团队（研究员、写作者、编辑）来产出一份简报。
- 识别 CrewAI 的三种常见失效模式：提示膨胀、管理者 LLM 成本税、脆弱交接。

## 问题

采用多智能体框架的团队总会撞上同一堵墙。“自主协作”在演示里听起来很棒。但随后客户报了一个缺陷，你需要可确定重放。或者财务问：一个由 LLM 路由的团队每次运行到底花多少钱？又或者值班工程师凌晨 3 点需要知道究竟是哪个智能体卡住了。

自由形态、由 LLM 路由的 Crew 无法干净地回答这些问题。纯 DAG 能全部回答，但又失去了头脑风暴型智能体所需要的探索式形态。

CrewAI 对这种取舍是坦诚的：Crews 适合协作式、角色化、探索式工作；Flows 适合事件驱动、代码掌控、可审计的生产系统。同一个框架，两种形态，针对不同界面分别选择。

## 概念

### 四个原语

CrewAI 的表层接口很小。把这四个记住，剩下的基本都是配置。

- **Agent。** `role + goal + backstory + tools + (optional) llm`。其中 `backstory` 不是装饰，而是关键负载。它会塑造语气、判断方式以及智能体何时停止。工具（tool）是智能体可以调用的函数（下文详述）。
- **Task。** `description + expected_output + agent + (optional) context + (optional) output_pydantic`。一个可复用的工作单元。`expected_output` 是契约。`context` 列出其依赖的上游任务，后者输出会被传入。`output_pydantic` 会强制输出满足结构化形状。
- **Crew。** 容器。拥有 `agents` 列表、`tasks` 列表、`process`，以及可选的 `memory`、`verbose`、`manager_llm` 设置。
- **Process。** 执行策略。Sequential、Hierarchical、Consensus（计划中）。它决定一次运行的形状。

智能体之间不会直接看见彼此。Task 引用 Agent。Crew 负责按顺序组织 Task。Process 决定由谁选择下一项 Task。这就是完整的心智模型。

> **已按以下版本核对：** CrewAI 0.86（2026-05）。更新版本可能会重命名或合并流程类型；在依赖某个具体形态前，请先检查 [CrewAI Processes docs](https://docs.crewai.com/concepts/processes)。

### 顺序式、分层式与共识式

- **Sequential。** Task 按声明顺序运行。任务 N 的输出可以作为 `context` 提供给任务 N+1。成本最低，也最可预测。适用于顺序固定的场景。
- **Hierarchical。** 由一个管理者智能体（额外的一次 LLM 调用）在多个专长角色之间路由。CrewAI 会根据你的 `manager_llm` 配置或默认值创建这个管理者。它每一轮都会选择下一项 Task，也可以拒绝或改道。适用于你拥有四个及以上专家角色，且顺序确实依赖前面输出时。
- **Consensus。** 已规划，但当前公共 API 尚未实现。文档保留了这个名称，用于未来的投票式过程。今天不要依赖它。

Hierarchical 会在每一次专家调用之前额外增加一次管理者的 LLM 调用。五步运行中，令牌成本可能翻到三倍。只有在你真的需要动态路由时，才值得为它付费。

### 团队模式与流程模式

这是 2026 年文档重点采用的框架。

- **Crew。** 由 LLM 驱动的自主性。框架会在运行时决定具体形态。适合：研究、头脑风暴、初稿生成，以及任何“路径本身就是答案一部分”的场景。难以重放，难以测试，但原型开发便宜。
- **Flow。** 由你掌控的事件驱动图。`@start` 标记入口。`@listen(topic)` 标记当其他步骤发出该 topic 时触发的步骤。每一步都是普通 Python（内部也可以调用 Crew）。适合：生产。可观测、可测试、确定性强。

文档在 2026 年给出的生产建议是：从 Flow 开始。当自主性真正值得其成本时，再在 Flow 步骤内部通过 `Crew.kickoff()` 嵌入 Crews。Flow 提供审计轨迹，Crew 提供探索能力。组合使用，而不是二选一。

### 工具集成

给 Agent 提供工具有三种方式。选择能满足需求的最简单方案。

1. **`@tool` 装饰器。** 纯函数可以直接变成工具。函数签名就是参数结构；文档字符串（docstring）就是 LLM 能看到的说明。最适合一次性的辅助函数。

   ```python
   from crewai.tools import tool

   @tool("Search the web")
   def search(query: str) -> str:
       """Return top results for the query."""
       return run_search(query)
   ```

2. **`BaseTool` 子类。** 基于类的工具，支持显式参数结构、异步、重试。适用于工具本身带状态（例如客户端、缓存）或需要结构化参数时。

   ```python
   from crewai.tools import BaseTool
   from pydantic import BaseModel

   class SearchArgs(BaseModel):
       query: str
       limit: int = 10

   class SearchTool(BaseTool):
       name = "web_search"
       description = "Search the web and return top results."
       args_schema = SearchArgs

       def _run(self, query: str, limit: int = 10) -> str:
           return self.client.search(query, limit=limit)
   ```

3. **内置工具包。** CrewAI 自带一方适配器：`SerperDevTool`、`FileReadTool`、`DirectoryReadTool`、`CodeInterpreterTool`、`RagTool`、`WebsiteSearchTool`。导入即可使用。

结构化输出基于 Pydantic。只要在 Task 上传入 `output_pydantic=MyModel` 即可。CrewAI 会用该模型校验 LLM 响应，并在必要时做强制转换或重试。最好配合严格的 `expected_output` 字符串一起使用。自由文本输出适合草稿；而结构化输出才是下游 Flow 可以稳定消费的内容。

### 记忆挂钩

CrewAI 开箱即用提供四类记忆。它们可以组合：一个 Crew 可以同时启用四种。

> **已按以下版本核对：** CrewAI 0.86（2026-05）。近期版本把所有能力统一收敛到一个 `Memory` 系统里，用来包装这四类存储。下面的概念模型依然成立，但公开类接口在新版本中可能会收缩为单一 `Memory` 入口；请查看 [CrewAI memory docs](https://docs.crewai.com/concepts/memory) 了解当前 API。

- **短期记忆。** 单次运行中的对话缓冲区。运行结束后清空。
- **长期记忆。** 跨运行持久化。存放在向量数据库中（默认 Chroma，可替换）。根据与当前任务的相似度检索。
- **实体记忆。** 面向实体保存事实。例如“客户 X 使用企业版套餐”。以实体为键，而非以相似度为键。可跨运行保留。
- **上下文记忆。** 在组装阶段即时检索。不是预先加载，而是在 Agent 真正需要时拉取相关记忆。

可以通过在 Crew 上设置 `memory=True` 或按类型单独配置来启用。其底层依赖你配置的嵌入（embedding）提供方（默认 OpenAI，也可以替换为本地）。记忆是 CrewAI 相比轻量框架真正体现价值的地方之一；纯 LangGraph 需要你把这些都自己接起来。

### CrewAI 适用的场景

- 三到六个具名角色、带协作工作流的团队。起草、审阅、规划、头脑风暴。
- 下一步该做什么本身就依赖 LLM 判断的路由场景（Hierarchical）。
- 团队成员更愿意阅读 `role + goal + backstory`，而不是阅读图定义的地方。

### CrewAI 不适用的场景

- 顺序严格、确定性的 DAG。请使用 LangGraph（第 13 课）。图结构才是正确抽象；CrewAI 的角色叙事反而成了摩擦。
- 亚秒级延迟预算。Hierarchical 会增加往返；即便是 Sequential，也要串行发送包含 backstory 和历史输出的提示。
- 单智能体循环。跳过框架吧；一个智能体循环（第 1 课）加上工具注册表会更短。

第 17 课（智能体框架权衡）会用矩阵方式展开这一点。简版结论是：CrewAI 位于“协作式、基于角色”的象限。

### 依赖形态

独立于 LangChain。支持 Python 3.10 到 3.13。使用 `uv`。Star 数量：见 [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI)（2026-05 快照）。文档中写明了 AWS Bedrock 集成；一些厂商基准称其在 QA 工作负载上比 LangGraph 快很多，但其方法学（数据集、硬件、评估指标）并未公开，因此请只把框架厂商数据当作方向性参考。

### 这种模式会在哪些地方出问题

- **背景设定导致提示膨胀。** 每个智能体 2000 字背景设定、五个智能体组成团队，在第一次工具调用之前就把上下文预算烧掉了。将背景设定控制在 200 字以内。复用措辞，不要把团队风格重复五遍。
- **管理者 LLM 的令牌税。** Hierarchical 流程会在每次专家调用前额外加入一次管理者 LLM 调用。在一个五任务的 Crew 中，这就从五次 LLM 调用变成六次，而且管理者调用还会携带完整任务列表和历史输出。除非路由真的依赖输出，否则切回 Sequential。
- **脆弱交接。** 任务 N 的 `expected_output` 是“一个大纲”。任务 N+1 通过 `context` 读取它，并尝试按三个部分解析。结果 LLM 生成了四个部分。下游 Agent 只能临场发挥。解决办法是在任务 N 上使用 `output_pydantic`，让任务 N+1 读取带类型对象，而不是自由文本。
- **把 Crew 直接当生产系统。** 在没有 Flow 包装的情况下把自由形态 Crew 直接上线。输出波动很大；无法重放；值班人员无法比较一次坏运行和一次好运行的差异。请用 Flow 包起来。

## 动手构建

`code/main.py` 用 stdlib 实现了这两种形态，以及一个三智能体 Crew。

形态：

- `Agent`、`Task` 数据类，对齐 CrewAI 的表层接口。
- `SequentialCrew.kickoff(inputs)` 按声明顺序运行任务，并把输出作为 `context` 串起来。
- `HierarchicalCrew.kickoff(topic)` 增加一个管理者智能体，在每一轮选择下一位专家，并在输出 “done” 时停止。
- `Flow` 提供 `@start` 与 `@listen(topic)` 装饰器、一个极小事件循环，以及一条执行轨迹。
- `tool(name)` 装饰器，模拟 CrewAI 的 `@tool` 形式。
- `Memory` 提供 `short_term`、`long_term`、`entity` 三种存储；相似度检索用 numpy 做模拟。
- 模拟 LLM 响应是基于角色和输入前缀硬编码的字符串。无网络。确定性。

具体演示：由研究员、写作者、编辑组成的团队，围绕 “agent engineering 2026” 生成一份简报。研究员拉取（模拟的）资料来源，写作者起草，编辑收紧润色。随后同一个团队通过 Flow 再运行一次，以展示确定性形态。

运行：

```bash
python3 code/main.py
```

执行轨迹覆盖：Sequential Crew 如何通过 `context` 串接输出；Hierarchical Crew 中管理者的选择过程（研究员、写作者、编辑，最后是 “done”）；Flow 如何用显式 topic（`researched`、`drafted`、`edited`）运行同样三步；工具调用如何通过 `@tool` 路由；以及长期记忆如何在两次 kickoff 之间持续存在。

Crew 的执行轨迹是流动的；理论上管理者可以重新排序。Flow 的执行轨迹是固定的。这个选择本身就是本课要传达的重点。

## 使用

- **CrewAI Flow** 用于生产。即使 Flow 只有一步，而这一步内部只是调用 `Crew.kickoff()`。Flow 提供审计边界。
- **CrewAI Crew（Sequential）** 用于顺序清晰的协作工作，尤其适合初稿生成和审阅循环。
- **CrewAI Crew（Hierarchical）** 用于顺序依赖输出、且拥有四个及以上专家角色的场景。
- **LangGraph**（第 13 课）适合显式状态机、可恢复执行和严格顺序。
- **AutoGen v0.4**（第 14 课）适合演员模型并发与故障隔离。
- **OpenAI Agents SDK**（第 16 课）适合以 OpenAI 为中心、需要交接和护栏的产品。
- **Claude Agent SDK**（第 17 课）适合以 Claude 为中心、需要子智能体和会话存储的产品。

## 交付

`outputs/skill-crew-or-flow.md` 会为一个任务选择 Crew 还是 Flow，并搭建最小实现。它会强制拒绝以下情况：没有背景设定的 Crew、没有显式 topic 的 Flow、以及专家少于三个却使用 Hierarchical 的方案。

## 易错点

- **把背景设定当风味文本。** 它真的会塑造输出。每个智能体至少测试三种变体；差异是实际存在的。选定一个后就冻结。
- **跳过 `expected_output`。** 如果每个 Task 没有契约，下游任务只能接手 LLM 随机产出的内容。Crew 也许能跑通，但审计一定过不了。
- **记忆永远开启。** 每次运行都写长期记忆，向量数据库会不断变大，检索噪声也会升高。只在事实具有持久价值的任务上写入。
- **管理者提示漂移。** Hierarchical 的管理者提示是隐式的。如果路由开始变怪，就打开 verbose 模式，把它打印出来认真读。
- **在 Crew 中使用有副作用的工具。** Crew 可能比你预期多次调用某个工具。POST、DELETE、支付这类操作应该放在 Flow 步骤里，绝不应该做成 Crew 工具。

## 练习

1. 把 Sequential Crew 改写成一个 Flow。统计有哪些触点让变异性下降了。再记录哪些地方的可读性变差了。
2. 为 Crew 增加实体记忆：关于某个客户的事实能跨多次 `kickoff` 保留。验证检索时能拉到正确实体。
3. 实现一个 Hierarchical 流程：在写作者的输出少于三段之前，管理者拒绝把任务路由给编辑。把重试轨迹打印出来。
4. 为（模拟的）网页搜索实现一个 `BaseTool` 子类。将其与 `@tool` 装饰器版本的轨迹形状做比较。
5. 为 editor 任务添加 `output_pydantic=Brief`，其中 `Brief` 包含 `title`、`summary`、`sections`。让 writer 任务故意输出一次格式错误的 JSON；验证执行轨迹中 CrewAI 的重试行为。
6. 阅读 CrewAI 文档简介。把这个玩具示例迁移到真实的 `crewai` API。stdlib 版本跳过了哪些保证？
7. 把 AgentOps 或 Langfuse（第 24 课）接到一次真实运行上。stdlib 版本漏掉了哪些轨迹？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| 智能体 | “人格设定” | 角色 + 目标 + 背景设定 + 工具 |
| 任务 | “工作单元” | 描述 + 预期输出 + 负责人 + 可选结构化输出 |
| 团队 | “智能体团队” | 智能体 + 任务 + 流程的容器 |
| 流程 | “执行策略” | 顺序式 / 分层式 / 共识式（计划中） |
| Flow | “确定性工作流” | 事件驱动、代码掌控、可测试 |
| 背景设定 | “角色提示” | 用于塑造智能体语气与判断的背景设定 |
| `@tool` | “函数工具” | 把函数变成 Agent 可调用工具的装饰器 |
| `BaseTool` | “类工具” | 支持参数结构、重试、异步的类式工具 |
| 实体记忆 | “按实体存事实” | 以客户 / 账户 / 问题单为边界的记忆 |
| 长期记忆 | “跨运行记忆” | 基于向量存储、可在多次 kickoff 间保留 |
| 上下文记忆 | “即时检索” | 在智能体需要时才拉取的记忆 |
| 管理者 LLM | “路由智能体” | 分层式流程中负责挑选下一项任务的额外 LLM |
| `expected_output` | “任务契约” | 告诉智能体（以及审计）应返回什么形状的字符串 |

## 延伸阅读

- [CrewAI docs introduction](https://docs.crewai.com/en/introduction)：概念与推荐的生产路径
- [CrewAI Flows guide](https://docs.crewai.com/en/concepts/flows)：事件驱动形态、`@start`、`@listen`
- [CrewAI tools reference](https://docs.crewai.com/en/concepts/tools)：`@tool`、`BaseTool`、内置工具包
- [CrewAI memory](https://docs.crewai.com/en/concepts/memory)：短期、长期、实体、上下文记忆
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)：何时多智能体有帮助，何时没有
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)：状态机式替代方案
