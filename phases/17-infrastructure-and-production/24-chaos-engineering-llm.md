# 面向 LLM 生产环境的混沌工程

> 到 2026 年，面向 LLM 的混沌工程已经成为一门独立学科。在生产环境运行实验前的前提条件包括：明确的 SLI/SLO、trace+metric+log 可观测性、自动回滚、runbook、值班 on-call。架构包含四个平面：控制（实验调度器）、目标（服务、基础设施、数据存储）、安全（防护 + 中止 + 流量过滤器）、可观测性（指标 + trace + 日志），以及反馈（回流到 SLO 调整中）。护栏 (guardrails) 是强制要求：如果每日错误预算消耗速率超过预期的 2x，burn-rate 告警会暂停实验；抑制窗口 + trace-ID 关联可对告警噪声进行去重。节奏是：每周小型 canary + SLO 复盘；每月 game day + postmortem；每季度跨团队韧性审计 + 依赖映射。LLM 专属实验包括：内存过载、网络故障、provider 宕机、畸形 prompt、KV cache 驱逐风暴。工具包括：Harness Chaos Engineering（LLM 派生建议、爆炸半径缩放、MCP 工具集成）；LitmusChaos（CNCF）；Chaos Mesh（CNCF Kubernetes 原生）。

**Type:** 学习
**Languages:** Python（stdlib、玩具级 chaos experiment runner）
**Prerequisites:** 第 17 阶段 · 23（SRE for AI），第 17 阶段 · 13（Observability）
**Time:** ~60 分钟

## 学习目标

- 说出混沌工程的五个前提条件（SLI/SLO、可观测性、回滚、runbook、on-call），并解释为什么缺少任意一个都会让实践失效。
- 画出四个平面（控制、目标、安全、可观测性）以及回流到 SLO 的反馈环。
- 列举五种 LLM 专属实验（内存过载、网络故障、provider 宕机、畸形 prompt、KV 驱逐风暴）。
- 根据技术栈选择工具 —— Harness、LitmusChaos、Chaos Mesh。

## 问题

传统技术栈中的混沌测试已经相对成熟。LLM 技术栈则引入了新的失效模式。一个带有有毒字符的 4K-token prompt 会让 tokenizer 卡住 12 秒。上游 provider 返回 429；你的 gateway 重试；你的服务在被重试放大的并发下 OOM。突发负载下的 KV cache 驱逐风暴会导致重新 prefill 级联，进而把计算资源打满。

这些问题都不会出现在单元测试里。混沌工程就是你在用户之前发现它们的方法。

## 概念

### 前提条件

在生产环境中运行混沌实验之前，不要缺少以下任何一项：

1. **SLI/SLO** —— 已定义的服务级指标与目标。
2. **可观测性 (observability)** —— trace、指标、日志，并接入仪表盘。
3. **自动回滚** —— 第 17 阶段 · 20 的策略标志回滚。
4. **Runbook** —— 结构化 runbook，第 17 阶段 · 23。
5. **On-call** —— 必须有人响应。

缺少任何一项，混沌实验都会变成真实事故。

### 四个平面 + 反馈

**控制平面** —— 实验调度器（Litmus workflow、Chaos Mesh schedule、Harness UI）。

**目标平面** —— 服务、pod、节点、负载均衡器、数据存储。

**安全平面** —— 紧急停止开关、抑制窗口、爆炸半径限制、错误预算闸门。

**可观测性平面** —— 常规指标 + trace-ID 关联，用于区分混沌实验导致的故障与自然故障。

**反馈环** —— 将发现回流到 SLO 调整、runbook 更新和代码修复中。

### 护栏是强制要求

- **Burn-rate 告警**：如果每日错误预算消耗超过预期的 2x，则暂停实验。
- **抑制窗口**：在实验期间，对爆炸半径范围内的非实验告警进行静默处理。
- **Trace-ID 关联**：所有由实验引起的错误都带上标签，以便 on-call 去重。

### 五种 LLM 专属实验

1. **内存过载** —— 通过发送高并发的长上下文请求，强制触发 KV cache 抢占风暴。观察：服务会优雅降载，还是直接崩溃？

2. **网络故障** —— 切断 inference gateway 与 provider 之间的连通性。观察：fallback 能否在 SLA 内生效？（第 17 阶段 · 19）

3. **Provider 宕机模拟** —— 让 OpenAI 100% 返回 429。观察：路由是否会故障切换到 Anthropic？（第 17 阶段 · 16、19）

4. **畸形 prompt** —— 注入会让 tokenizer 卡住的负载（例如深度嵌套的 unicode、超大的 UTF-8 codepoint）。观察：单个请求会不会锁死一个 worker？

5. **KV 驱逐风暴** —— 通过打满 vLLM 的 block budget 强制触发驱逐。观察：LMCache 会恢复，还是服务会持续退化？

### 节奏

- **每周** —— 在 staging 中做小型 canary 实验，必要时可带 5% 生产流量。
- **每月** —— 针对特定场景安排一次 game day；跨团队参与；产出 postmortem。
- **每季度** —— 跨团队韧性审计；更新依赖地图。

### 工具

- **Harness Chaos Engineering** —— 商业产品；提供 AI 派生实验建议、爆炸半径缩放、MCP 工具集成。
- **LitmusChaos** —— CNCF 毕业项目；基于 Kubernetes 工作流。
- **Chaos Mesh** —— CNCF sandbox；Kubernetes 原生 CRD 风格。
- **Gremlin** —— 商业产品；支持面广。
- **AWS FIS** / **Azure Chaos Studio** —— 托管式云产品。

### 从小处开始

第一个实验：在稳定流量下，让一个 decode 副本 pod 下线。观察重路由与恢复。如果这看起来可控且安全，再升级到网络混沌。

第一个 LLM 专属实验：在 5 分钟内让某个 provider 持续返回 429。观察 fallback。大多数团队都会发现，他们的 fallback 根本没有被完整测试过。

### 你应该记住的数字

- 四个平面：控制、目标、安全、可观测性。
- Burn-rate 暂停阈值：预期每日预算消耗的 2x。
- 节奏：每周 canary、每月 game day、每季度审计。
- 五种 LLM 实验：内存、网络、provider、畸形 prompt、KV 风暴。

## 使用它

`code/main.py` 模拟了三个带安全平面闸门的混沌实验。它会报告哪些实验会触发 burn-rate 中止。

## 交付它

本课会产出 `outputs/skill-chaos-plan.md`。给定技术栈和成熟度后，它会选择前三个实验以及对应工具。

## 练习

1. 运行 `code/main.py`。哪个实验触发了 burn-rate 闸门？为什么？
2. 为一个基于 vLLM 的 RAG 服务设计前五个混沌实验。请包含成功标准。
3. 你的 burn-rate 告警暂停了一个实验。你要如何判断根因来自混沌实验还是自然故障？
4. 论证混沌实验应该在生产环境运行，还是只应在 staging 运行。什么时候生产环境才是正确答案？
5. 说出三种通用网络混沌无法复现的 LLM 专属失效模式。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| SLI / SLO | “服务目标” | 指标 + 目标；属于必要前提条件 |
| 爆炸半径 | “范围” | 受实验影响的一组服务 / 用户 |
| Burn-rate 告警 | “预算闸门” | 当错误预算消耗速率 > 预期的 2x 时触发 |
| Game day | “每月演练” | 预先安排的跨团队混沌演练 |
| LitmusChaos | “CNCF 工作流” | 已毕业的 CNCF Kubernetes 混沌工具 |
| Chaos Mesh | “CNCF CRD” | CNCF sandbox 的 Kubernetes 原生混沌工具 |
| Harness CE | “商业化 AI 辅助” | 带 AI 建议的 Harness 混沌工程 |
| 畸形 prompt | “tokenizer 炸弹” | 会让 tokenization 卡住的输入 |
| KV 驱逐风暴 | “抢占级联” | 大规模驱逐触发的重新 prefill 级联 |

## 延伸阅读

- [DevSecOps School — Chaos Engineering 2026 Guide](https://devsecopsschool.com/blog/chaos-engineering/)
- [Ankush Sharma — Observability for LLMs (book)](https://www.amazon.com/Observability-Large-Language-Models-Engineering-ebook/dp/B0DJSR65TR)
- [LitmusChaos (CNCF)](https://litmuschaos.io/)
- [Chaos Mesh (CNCF)](https://chaos-mesh.org/)
- [Harness Chaos Engineering](https://www.harness.io/products/chaos-engineering)
- [AWS FIS](https://aws.amazon.com/fis/)
