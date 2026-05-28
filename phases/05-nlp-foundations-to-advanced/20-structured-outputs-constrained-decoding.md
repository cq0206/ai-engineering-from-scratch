# 结构化输出（Structured Outputs）与约束解码（Constrained Decoding）

> 让 LLM 返回 JSON。大多数时候都会得到 JSON。但在生产环境中，“大多数”就是问题。约束解码通过在采样前修改 logits，把“大多数”变成“总是”。

**类型：** 构建
**语言：** Python
**前置要求：** 第 5 阶段 · 17（聊天机器人），第 5 阶段 · 19（子词分词）
**时长：** ~60 分钟

## 问题

一个分类器向 LLM 提示：“返回 {positive, negative, neutral} 中的一个。” 模型却返回：“情感是 positive——这条评论压倒性地偏正面，因为顾客明确表示他们……” 你的解析器直接崩溃，分类器的 F1 也变成 0.0。

自由形式生成（free-form generation）不是契约，而只是建议。生产系统需要契约。

到 2026 年，常见有三层方案。

1. **提示工程（Prompting）。** 礼貌地请求：“只返回 JSON 对象。” 在前沿模型上大约有 ~80% 的成功率，在更小的模型上更低。
2. **原生结构化输出 API（Native structured output APIs）。** OpenAI 的 `response_format`、Anthropic 的 tool use、Gemini 的 JSON mode。对受支持的 schema 很可靠，但会被厂商锁定。
3. **约束解码（Constrained decoding）。** 在每一步生成时修改 logits，让模型*无法*输出无效 token。通过构造保证 100% 有效。适用于任何本地模型。

本课会帮助你建立对这三种方式的直觉，并说明分别该在什么场景使用。

## 概念

*在每一步对无效 token 做掩码的约束解码*

**约束解码如何工作。** 在每个生成步骤，LLM 都会对完整词表（约 10 万 token）输出一个 logit 向量。采样器前会插入一个*logit 处理器（logit processor）*。它根据目标语法中的当前位置——JSON Schema、正则表达式、上下文无关文法——计算哪些 token 是合法的，然后把所有非法 token 的 logits 设为负无穷。对剩余 logits 做 softmax 后，概率质量只会分配给合法续写。

2026 年的实现方式：

- **Outlines。** 把 JSON Schema 或正则编译成有限状态机（finite-state machine, FSM）。每个 token 都能以 O(1) 查询合法的下一个 token。基于 FSM，因此递归 schema 需要先展开。
- **XGrammar / llguidance。** 上下文无关文法（context-free grammar, CFG）引擎。能处理递归 JSON Schema，解码开销几乎为零。OpenAI 在 2025 年的结构化输出实现中致谢了 llguidance。
- **vLLM guided decoding。** 内置 `guided_json`、`guided_regex`、`guided_choice`、`guided_grammar`，后端可选 Outlines、XGrammar 或 lm-format-enforcer。
- **Instructor。** 基于 Pydantic 的任意 LLM 包装器。验证失败时重试。支持跨供应商，但不修改 logits——它依赖“重试 + 面向结构化输出的提示”。

### 反直觉的结果

约束解码往往比无约束生成（unconstrained generation）*更快*。原因有二：第一，它缩小了下一 token 的搜索空间；第二，聪明的实现会直接跳过那些被强制确定的 token（比如 `{"name": "` 这种脚手架）的生成——每个字节都已经确定。

### 真正让你付出代价的坑

字段顺序很重要。如果把 `answer` 放在 `reasoning` 前面，模型会在思考之前先提交答案。JSON 仍然合法，但答案会错，而且没有任何验证能捕获这一点。

```json
// BAD
{"answer": "yes", "reasoning": "because ..."}

// GOOD
{"reasoning": "... therefore ...", "answer": "yes"}
```

Schema 中字段顺序不是格式问题，而是逻辑问题。

## 动手构建

### 步骤 1：从零实现基于正则的约束生成

请看 `code/main.py` 中的独立 FSM 实现。核心思想 30 行就够：

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

FSM 会跟踪当前已经满足了语法中的哪些部分。`valid_tokens(state, tokenizer)` 负责计算：在不离开可接受路径的前提下，词表中哪些 token 能推进 FSM。

### 步骤 2：用 Outlines 处理 JSON Schema

```python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

不会再出现验证错误。FSM 让无效输出根本不可达。

### 步骤 3：用 Instructor 做跨供应商的 Pydantic

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

机制不同。Instructor 不会碰 logits。它会把 schema 格式化进提示词，解析输出，并在验证失败时重试（默认 3 次）。它适用于任何供应商。重试会增加延迟和成本，而跨供应商可移植性就是它的卖点。

### 步骤 4：使用厂商原生 API

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

这是服务端约束解码。对受支持 schema 的可靠性与 Outlines 基本等价，不需要管理本地模型，但会把你锁定在厂商生态里。

## 常见陷阱

- **递归 schema。** Outlines 会把递归展开到固定深度。树状输出（嵌套评论、AST）需要 XGrammar 或 llguidance（基于 CFG）。
- **超大枚举。** 1 万选项的枚举编译很慢，甚至会超时。改用检索器：先预测 top-k 候选，再对这些候选做约束。
- **语法过严。** 如果强制 `date: "YYYY-MM-DD"` 正则，模型就无法在日期缺失时输出 `"unknown"`。它会通过编造一个日期来自我补偿。要允许 `null` 或哨兵值。
- **过早承诺。** 见上面的字段顺序陷阱。始终把 reasoning 放在前面。
- **没有 schema 的厂商 JSON mode。** 纯 JSON mode 只保证 JSON 合法，不保证*对你的用例*合法。一定要提供完整 schema。

## 如何使用

2026 年的技术栈：

| 场景 | 选择 |
|-----------|------|
| OpenAI/Anthropic/Google 模型，schema 简单 | 厂商原生结构化输出 |
| 任意供应商、Pydantic 工作流、能接受重试 | Instructor |
| 本地模型，需要 100% 有效性，schema 扁平 | Outlines（FSM） |
| 本地模型，schema 递归 | XGrammar 或 llguidance |
| 自托管推理服务器 | vLLM guided decoding |
| 可接受重试的批处理 | Instructor + 最便宜的模型 |

## 交付

保存为 `outputs/skill-structured-output-picker.md`：

```markdown
---
name: structured-output-picker
description: Choose a structured output approach, schema design, and validation plan.
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

Given a use case (provider, latency budget, schema complexity, failure tolerance), output:

1. Mechanism. Native vendor structured output, Instructor retries, Outlines FSM, or XGrammar CFG. One-sentence reason.
2. Schema design. Field order (reasoning first, answer last), nullable fields for "unknown", enum vs regex, required fields.
3. Failure strategy. Max retries, fallback model, graceful `null` handling, out-of-distribution refusal.
4. Validation plan. Schema compliance rate (target 100%), semantic validity (LLM-judge), field-coverage rate, latency p50/p99.

Refuse any design that puts `answer` or `decision` before reasoning fields. Refuse to use bare JSON mode without a schema. Flag recursive schemas behind an FSM-only library.
```

## 练习

1. **简单。** 对 `Review(sentiment, confidence, evidence_span)`，在不使用约束解码的情况下，提示一个小型开放权重模型（例如 Llama-3.2-3B）。在 100 条评论上测量其中有多少比例能被解析为合法 JSON。
2. **中等。** 在同一语料上使用 Outlines JSON mode。比较合规率、延迟和语义准确率。
3. **困难。** 从零实现一个针对电话号码（`\d{3}-\d{3}-\d{4}`）的正则约束解码器。在 1000 个样本上验证无效输出为 0。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 约束解码 | 强制得到有效输出 | 在每一步生成时对无效 token 的 logits 做掩码。 |
| Logit 处理器 | 负责加约束的那个东西 | 函数：`(logits, state) -> masked_logits`。 |
| FSM | 有限状态机 | 编译后的语法表示；能以 O(1) 查询合法下一 token。 |
| CFG | 上下文无关文法 | 能处理递归的文法；比 FSM 慢，但表达力更强。 |
| Schema 字段顺序 | 这也重要吗？ | 重要——第一个字段会促使模型先做承诺；始终把 reasoning 放在 answer 前。 |
| Guided decoding | vLLM 对它的叫法 | 同一个概念，只是集成在推理服务器里。 |
| JSON mode | OpenAI 的早期版本 | 只保证 JSON 语法；**不**保证匹配 schema。 |

## 延伸阅读

- [Willard, Louf (2023). Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702) —— Outlines 论文。
- [XGrammar paper (2024)](https://arxiv.org/abs/2411.15100) —— 快速的基于 CFG 的约束解码。
- [vLLM — Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html) —— 推理服务器集成。
- [OpenAI — Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs) —— API 参考与注意事项。
- [Instructor library](https://python.useinstructor.com/) —— 跨供应商的 Pydantic + 重试。
- [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868) —— 6 个约束解码框架的基准评测。
