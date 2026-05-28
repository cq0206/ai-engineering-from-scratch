# Voyager：技能库与终身学习

> Voyager（Wang 等，TMLR 2024）把可执行代码视为一种技能（skill）。技能有名字、可检索、可组合，并能通过环境反馈不断改进。这是 Claude Agent SDK skills、skillkit 以及 2026 年技能库模式的参考架构。

**类型：** 构建
**语言：** Python（标准库）
**先修要求：** 第 14 阶段 · 07（MemGPT），第 14 阶段 · 08（Letta 记忆块）
**时间：** 约 75 分钟

## 学习目标

- 说出 Voyager 的三个组件——自动课程（automatic curriculum）、技能库（skill library）、迭代式提示（iterative prompting）——以及各自的作用。
- 解释为什么 Voyager 把动作空间（action space）定义为代码，而不是原始命令。
- 仅用标准库实现一个支持注册、检索、组合和基于失败改进的技能库。
- 将 Voyager 的模式映射到 2026 年的 Claude Agent SDK skills 与 skillkit 生态。

## 问题

那些在每次会话里都从头重建全部能力的智能体，会同时犯三类错误：

1. **浪费令牌。** 每个任务都要重新诱发同样的推理。
2. **丢失进展。** 在会话 A 中学到的修正，无法迁移到会话 B。
3. **在长时程组合上失败。** 复杂任务需要能力层级；一次性提示无法表达这种层级。

Voyager 的答案是：把每个可复用能力都看作一个具名代码块，存入库中，可按相似度检索、与其他技能组合，并通过执行反馈不断改进。

## 概念

### 三个组件

Voyager（arXiv:2305.16291）围绕以下三部分组织智能体：

1. **自动课程（Automatic curriculum）**。一个由好奇心驱动的提议器，会根据智能体当前的技能集合和环境状态挑选下一个任务。探索是自底向上的。
2. **技能库（Skill library）**。每个技能都是可执行代码。任务成功时，就会新增技能。技能按“查询到描述”的相似度进行检索。
3. **迭代式提示机制（Iterative prompting mechanism）**。任务失败时，智能体会收到执行错误、环境反馈和自验证输出，然后据此改进技能。

Minecraft 评测（Wang 等，2024）的结果是：相较基线，独特物品数量提高 3.3 倍，石制工具获取速度提高 8.5 倍，铁制工具获取速度提高 6.4 倍，地图穿越距离提高 2.3 倍。这些数字属于 Minecraft 场景，但模式本身具有迁移性。

### 动作空间 = 代码

大多数智能体输出原始命令。Voyager 输出 JavaScript 函数。一个技能可能是：

```
async function craftIronPickaxe(bot) {
  await mineIron(bot, 3);
  await mineStick(bot, 2);
  await placeCraftingTable(bot);
  await craft(bot, 'iron_pickaxe');
}
```

它由更小的子技能组合而成。以描述和嵌入（embedding）为键存储。检索出来的是程序，而不是提示词。

这正是 2026 年 Claude Agent SDK 中 skill（技能）的样子：一个具名、可检索的代码块，加上一组按需加载的说明。

### 技能检索

面对新任务 “制作一把钻石镐”，智能体会：

1. 对任务描述做嵌入。
2. 在技能库中查询前 k 个（top-k）相似技能。
3. 取回 `craftIronPickaxe`、`mineDiamond`、`placeCraftingTable` 等技能。
4. 用这些取回的原语技能加上一些新逻辑，组合出新的技能。

这正是 MCP 资源（resources，第 13 阶段）和 Agent SDK skills 所实现的模式：在知识/代码层面做检索，并将范围限制在当前任务内。

### 迭代式改进

Voyager 的反馈循环如下：

1. 智能体编写一个技能。
2. 技能在环境中运行。
3. 返回三种信号之一：`success`、`error`（附带栈追踪）、`self-verification failure`（自验证失败）。
4. 智能体以该信号为上下文重写技能。
5. 循环直到成功或达到最大轮数。

这就是 Self-Refine（第 05 课）应用到代码生成上的样子，并以环境落地验证作为依据。CRITIC（第 05 课）则是同一种模式，只不过由外部工具来做验证器。

### 课程与探索

Voyager 的课程模块会根据智能体已经拥有的内容和仍未完成的事情，提出像“在湖边建一个避难所”这样的任务。提议器利用环境状态和技能清单，选择一个略高于当前能力的任务——这就是探索的甜蜜点。

对于生产级智能体，这通常会转化为一种“缺什么”的操作：给定当前技能库和某个领域，我们还缺哪些技能？团队通常会手动把这件事实现为课程评审。

### 这种模式会在哪里出问题

- **技能库腐化。** 同一个技能被用稍有差异的描述重复添加 10 次。应在写入时做去重；检索时只返回一个。
- **组合技能漂移。** 父技能依赖一个后来被改进过的子技能。应给技能做版本管理；固定在 v1 的父技能不会自动拾取 v3。
- **检索质量。** 当库增长到几百条以上时，仅靠对技能描述做向量检索会退化。应补充标签过滤和硬约束（例如“只要 `category=tooling` 的技能”）。

## 动手构建

`code/main.py` 实现了一个标准库技能库：

- `Skill` —— name、description、code（字符串形式）、version、tags、dependencies。
- `SkillLibrary` —— 提供注册、搜索（令牌重叠）、组合（对依赖做拓扑排序）和改进（更新时提升版本号）。
- 一个脚本化智能体：它注册三个原始技能，组合出第四个，遇到一次失败，再做改进。

运行：

```
python3 code/main.py
```

输出轨迹会展示库写入、检索、组合、一次执行失败，以及 v2 版本改进——完整跑通 Voyager 的循环。

## 使用它

- **Claude Agent SDK skills**（Anthropic）—— 2026 年的参考实现：每个 skill（技能）都有描述、代码和说明；在智能体会话中按需加载。
- **skillkit**（npm: skillkit）—— 面向 32+ AI 编码智能体的跨智能体技能管理。
- **自定义技能库** —— 面向特定领域（例如数据智能体的 SQL 技能、基础设施智能体的 Terraform 技能）。Voyager 模式可以向下缩放。
- **OpenAI Agents SDK `tools`** —— 位于更底层；每个工具都可以看作一个轻量技能。

## 交付上线

`outputs/skill-skill-library.md` 会为任何目标运行时生成一个 Voyager 形态的技能库，并接好注册、检索、版本控制和改进逻辑。

## 练习

1. 给 `compose()` 添加依赖环检测。如果技能 A 依赖 B，而 B 又依赖 A，会发生什么？报错还是警告？
2. 实现按技能粒度的版本固定。当父技能组合子技能 `crafting@1` 时，子技能改进到 `crafting@2` 不应悄悄升级父技能。
3. 用 sentence-transformers 嵌入（或一个标准库版 BM25 实现）替换基于令牌重叠的检索。在一个 50 技能的玩具库上测 retrieval@5。
4. 添加一个“课程”智能体：给定当前技能库和领域描述，提出 5 个缺失技能。每周运行一次。
5. 阅读 Anthropic 的 Claude Agent SDK skill 文档。把这个玩具库迁移到该 SDK 的 skill 模式（schema）。可发现性会发生什么变化？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| 技能 | “可复用能力” | 带名字和描述的代码块，可按相似度检索 |
| 技能库 | “智能体对做事方法的记忆” | 持久化的技能存储，可搜索、可组合 |
| 课程生成器 | “任务提议器” | 由当前能力缺口驱动的自底向上目标生成器 |
| 组合 | “技能 DAG” | 技能调用技能；执行时按拓扑顺序排序 |
| 迭代改进 | “自我纠错循环” | 将环境反馈、错误和自验证折叠进下一个版本 |
| 以代码作为动作空间 | “程序化动作” | 输出函数，而不是原始命令，以表达跨时间的行为 |
| 写入时去重 | “技能塌缩” | 近似重复的描述被折叠成一个规范技能 |

## 延伸阅读

- [Wang et al., Voyager (arXiv:2305.16291)](https://arxiv.org/abs/2305.16291) —— 原始技能库论文
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) —— skills 作为 2026 年产品化形态
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) —— skills 与 subagents 的实践
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) —— Voyager 底层的改进循环
