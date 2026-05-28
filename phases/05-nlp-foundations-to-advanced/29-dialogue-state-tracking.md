# 对话状态跟踪（Dialogue State Tracking，DST）

> “我想找一家北边的便宜餐馆……等等，改成中等价位……再加上意大利菜。” 三轮对话，三次状态更新。DST 的工作就是让槽位-值字典始终保持同步，这样预订才能成功。

**类型：** 构建
**语言：** Python
**前置要求：** 第 5 阶段 · 17（聊天机器人），第 5 阶段 · 20（结构化输出）
**时长：** ~75 分钟

## 问题

在任务型对话系统中，用户目标会被编码为一组槽位-值对：`{cuisine: italian, area: north, price: moderate}`。每一轮用户发言都可能新增、修改或删除一个槽位。系统必须阅读整个对话，并正确输出当前状态。

只要有一个槽位出错，系统就可能订错餐馆、安排错航班，或者扣错信用卡。DST 正是“用户说了什么”和“后端实际执行什么”之间的铰链。

为什么即使到了 2026 年、LLM 已经普及，它仍然重要：

- 对合规敏感的领域（银行、医疗、机票预订）需要确定性的槽位值，而不是自由生成文本。
- 工具调用智能体（tool-use agent）在调用 API 前，仍然需要先把槽位解析出来。
- 多轮纠错并没有看起来那么简单：“其实不对，改成周四。”

现代流水线是：经典 DST 概念 + LLM 抽取器 + 结构化输出护栏。

## 概念

*DST：对话历史 → 槽位-值状态*

**任务结构。** 一个 schema 定义了领域（餐馆、酒店、出租车）及其槽位（菜系、区域、价格、人数）。每个槽位可以为空、填入一个封闭集合中的值（price: {cheap, moderate, expensive}），或一个自由文本值（name: "The Copper Kettle"）。

**两种 DST 形式化。**

- **分类式。** 对每个 `(slot, candidate_value)` 对预测 yes/no。适合封闭词表槽位。是 2020 年前的标准方案。
- **生成式。** 给定对话，直接生成自由文本形式的槽位值。适合开放词表槽位。是现代默认方案。

**指标。** 联合目标准确率（Joint Goal Accuracy, JGA）——一轮对话中*所有*槽位都正确的比例。它是全有或全无的指标。到 2026 年，MultiWOZ 2.4 榜单最高大约在 83%。

**常见架构。**

1. **基于规则（槽位正则 + 关键词）。** 在窄领域里是很强的基线，且易于调试。
2. **TripPy / BERT-DST。** 用 BERT 编码的复制式生成。是 LLM 前时代的标准。
3. **LDST（LLaMA + LoRA）。** 经过指令微调的 LLM，并用领域-槽位提示（domain-slot prompting）驱动。在 MultiWOZ 2.4 上能达到接近 ChatGPT 的质量。
4. **无本体方案（Ontology-free，2024–26）。** 直接生成槽位名和值，跳过预定义 schema，可处理开放域。
5. **提示 + 结构化输出（2024–26）。** LLM 配合 Pydantic schema + 约束解码。只需 5 行代码，就可以用于生产。

### 经典失败模式

- **跨轮共指。** “就用第一个选项吧。” 需要知道“第一个”指哪个选项。
- **覆盖还是追加。** 用户说“再加上意大利菜”。你是替换 cuisine，还是追加？
- **隐式确认。** “好，没问题。”——这算接受系统给出的预订吗？
- **纠正。** “其实改成晚上 7 点。”必须只更新时间，而不能清空其他槽位。
- **指向前一轮系统话语的共指。** “对，就那个。”这里的“那个”到底是哪一个？

## 动手构建

### 步骤 1：基于规则的槽位抽取器

请看 `code/main.py`。正则 + 同义词词典，在窄领域里能覆盖 70% 的标准表达：

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

一旦离开标准词表，它就会变得脆弱。但对确定性的槽位确认仍然有效。

### 步骤 2：状态更新循环

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

有三个不变式：

- 不要重置用户没有触及的槽位。
- 显式否定（“算了，不要菜系限制”）必须清空对应槽位。
- 用户纠正（“其实……”）必须覆盖，而不是追加。

### 步骤 3：使用结构化输出的 LLM-DST

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

Instructor + Pydantic 保证输出的是合法状态对象：没有正则、没有 schema 不匹配，也没有幻觉出来的新槽位。

### 步骤 4：JGA 评估

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

要校准的问题是：系统有多少轮能把所有槽位都预测正确？在 MultiWOZ 2.4 上，2026 年最强系统大约为 80-83%。如果你的窄词表、领域内系统都做不到比这更高，那还不如直接用 LLM 基线。

### 步骤 5：处理纠正

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

一旦检测到纠正，应该覆盖最近更新的槽位，而不是追加。这个逻辑如果没有 LLM，很难完全处理正确。现代模式通常是：每一轮都让 LLM 基于完整历史重新生成整个状态，而不是增量更新——这样会更自然地处理纠错。

## 常见陷阱

- **全历史重生成的成本。** 如果每轮都让 LLM 从头生成状态，总 token 成本会呈 O(n²) 增长。要限制历史长度，或把旧轮次先摘要化。
- **Schema 漂移。** 事后新增槽位会破坏旧训练数据。要对 schema 做版本管理。
- **大小写敏感。** “Italian”“italian”“ITALIAN”——需要统一归一化。
- **隐式继承。** 如果用户之前已经说过“4 个人”，那后来只改时间时不应该把人数清空。必须传入完整历史。
- **自由文本 vs 封闭集合。** 名字、时间、地址需要自由文本槽位；菜系和区域属于封闭集合。schema 中两者应混合存在。

## 如何使用

2026 年的技术栈：

| 场景 | 方法 |
|-----------|----------|
| 窄领域（一个或两个意图） | 基于规则 + 正则 |
| 广领域，有标注数据 | LDST（在 MultiWOZ 风格数据上对 LLaMA + LoRA 微调） |
| 广领域，无标注，且可直接上线 | LLM + Instructor + Pydantic schema |
| 语音 / 口语对话 | ASR + 归一化器 + LLM-DST |
| 多领域预订流程 | 模式（schema）引导的 LLM + 按领域拆分的 Pydantic 模型 |
| 合规敏感 | 规则主系统，LLM 兜底，并加确认流程 |

## 交付

保存为 `outputs/skill-dst-designer.md`：

```markdown
---
name: dst-designer
description: Design a dialogue state tracker — schema, extractor, update policy, evaluation.
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

Given a use case (domain, languages, vocab openness, compliance needs), output:

1. Schema. Domain list, slots per domain, open vs closed vocabulary per slot.
2. Extractor. Rule-based / seq2seq / LLM-with-Pydantic. Reason.
3. Update policy. Regenerate-whole-state / incremental; correction handling; negation handling.
4. Evaluation. Joint Goal Accuracy on a held-out dialogue set, slot-level precision/recall, confusion on the hardest slot.
5. Confirmation flow. When to explicitly ask the user to confirm (destructive actions, low-confidence extractions).

Refuse LLM-only DST for compliance-sensitive slots without a rule-based secondary check. Refuse any DST that cannot roll back a slot on user correction. Flag schemas without version tags.
```

## 练习

1. **简单。** 为 3 个槽位（cuisine、area、price）在 `code/main.py` 中构建规则式状态跟踪器。用 10 段手工编写的对话测试，并测量 JGA。
2. **中等。** 在同一数据集上使用 Instructor + Pydantic + 一个小型 LLM。比较 JGA，并检查最难的那些轮次。
3. **困难。** 同时实现两者并做路由：规则系统为主，当规则系统输出 `&lt;2 slots` 且置信度不足时，回退到 LLM。测量组合后的 JGA 以及每轮推理成本。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| DST | 对话状态跟踪 | 在多轮对话中维护槽位-值字典。 |
| 槽位（Slot） | 用户意图的一个单元 | 后端执行所需的命名参数（菜系、日期）。 |
| 领域（Domain） | 任务范围 | 餐馆、酒店、出租车——各自对应一组槽位。 |
| JGA | 联合目标准确率 | 一轮中所有槽位都正确的比例；全有或全无。 |
| MultiWOZ | 那个基准 | 多领域 Wizard-of-Oz 数据集；DST 的标准评测。 |
| 无本体 DST | 没有 schema | 不使用固定列表，直接生成槽位名和值。 |
| 纠正（Correction） | “Actually...” | 会覆盖此前已填写槽位的一轮发言。 |

## 延伸阅读

- [Budzianowski et al. (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) —— 经典基准。
- [Feng et al. (2023). Towards LLM-driven Dialogue State Tracking (LDST)](https://arxiv.org/abs/2310.14970) —— 用于 DST 的 LLaMA + LoRA 指令微调。
- [Heck et al. (2020). TripPy — A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/abs/2005.02877) —— 复制式 DST 主力方法。
- [King, Flanigan (2024). Unsupervised End-to-End Task-Oriented Dialogue with LLMs](https://arxiv.org/abs/2404.10753) —— 基于 EM 的无监督 TOD。
- [MultiWOZ leaderboard](https://github.com/budzianowski/multiwoz) —— 标准 DST 结果榜单。
