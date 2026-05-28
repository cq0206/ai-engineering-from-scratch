# LLM 中的偏见与表征性伤害

> Gallegos、Rossi、Barrow、Tanjim、Kim、Dernoncourt、Yu、Zhang、Ahmed（Computational Linguistics 2024，arXiv:2309.00770）给出了 2024 年的基础性综述：区分表征性伤害 (representational harms) 与分配性伤害 (allocational harms)，并把评估指标分成基于 embedding、基于概率、基于生成文本三类。2024-2025 年的实证工作中，An et al.（PNAS Nexus，2025 年 3 月）测量了 GPT-3.5 Turbo、GPT-4o、Gemini 1.5 Flash、Claude 3.5 Sonnet、Llama 3-70B 在 20 个初级岗位自动简历评估中的性别 x 种族交叉偏见。WinoIdentity（COLM 2025，arXiv:2508.07111）引入了基于不确定性的交叉公平评估。Yu & Ananiadou 2025 在 MLP 层中识别出 gender neurons；Ahsan & Wallace 2025 用 SAE 揭示临床种族偏见；Zhou et al. 2024（UniBias）通过操纵 attention heads 来做去偏。元批评（arXiv:2508.11067）指出：过去 10 年的文献过度聚焦于二元性别偏见。

**类型：** 构建
**语言：** Python（stdlib，玩具 embedding 偏见探针）
**先修：** Phase 05 (word embeddings), Phase 18 · 01 (instruction following)
**时间：** ~60 分钟

## 学习目标

- 定义表征性伤害与分配性伤害，并各举一个在 LLM 部署中的例子。
- 说出 Gallegos et al. 2024 提出的三类评估指标，并分别描述每一类中的一个指标。
- 描述交叉性 (intersectionality)，并解释为什么 WinoIdentity 基于不确定性的公平测量弥补了单轴偏见评估的缺口。
- 描述两种偏见的机制可解释性方法（gender neurons、SAE 特征、attention-head 操纵中的任意两种）。

## 问题

前面的课程主要覆盖有意伤害（越狱、谋划）和安全治理。偏见则是一种无需主观意图也会出现的伤害——它来自训练数据分布、来自 prompt framing，也来自不断叠加的设计选择。测量并减少它，是一个与对抗鲁棒性不同的方法论挑战。

## 概念

### 表征性 vs 分配性

- **表征性伤害。** 刻板印象、抹除、贬损性描绘。一个把护士描绘成清一色女性的 LLM，就在产生表征性伤害。
- **分配性伤害。** 不平等的物质结果。一个系统性地给 Black 求职者简历更低分的 LLM，就在产生分配性伤害。

这两者并不相同。一个模型可以在“表征上不偏”（生成多样化描绘）的同时，在“分配上有偏”（做出不平等推荐）。评估必须同时测量二者。

### 三类评估指标（Gallegos et al. 2024）

- **基于 embedding。** 在 pre-RLHF embedding 上做 WEAT 风格测试。它测量身份词与属性词之间的统计关联。局限在于：它测的是表征，不是行为。
- **基于概率。** 比较刻板印象确认型 completion 与违背刻板印象 completion 的对数似然。这是 decoder 侧测量，能捕捉到一部分行为偏见。
- **基于生成文本。** 在生成文本的下游任务上测量，如简历打分、推荐写作、对话。生态有效性最高，但最难复现。

### 交叉性

只评估“gender”偏见，会漏掉那些只在（gender, race）组合上触发的偏见。An et al. 2025 发现，在简历评分任务中，GPT-4o 对 Black women 的惩罚比对 Black men 更严重，也比对白人女性更严重。单轴评估捕捉不到这种现象。

WinoIdentity（COLM 2025）提出了基于不确定性的交叉公平性。它测量的不是不同交叉身份元组上的点预测是否不同，而是模型对结果的不确定性是否不同。这样就能抓住一种情况：模型对各组的点预测同样错误，但对某些组更不确定，而这会带来不同的下游分配行为。

### 机制性方法

2024-2025 年的可解释性工作，让偏见开始可以被做机制性干预：

- **Gender neurons（Yu & Ananiadou 2025）。** 某些特定 MLP 神经元与性别特定行为相关。消融这些神经元后，性别差距指标会下降，而能力代价有限。
- **通过 SAE 揭示临床种族偏见（Ahsan & Wallace 2025）。** 稀疏自编码器 (sparse autoencoder, SAE) 特征把内部表征分解为可解释维度；研究者可以识别并抑制与种族相关的特征。
- **UniBias（Zhou et al. 2024）。** 通过操纵 attention heads 来做零样本去偏。某些特定 heads 会放大身份类别敏感性；将这些 heads 归零或重加权，可以在不做微调的情况下减轻偏见。

### 元批评

那篇对 10 年文献的综述（arXiv:2508.11067，2025）发现，这个领域过度聚焦二元性别偏见。其他轴——残障、宗教、迁移身份、多语言身份——得到的关注少得多。这个元批评认为，过窄的焦点会以“忽视”的方式伤害边缘化群体：一个在二元性别上去偏得很好的模型，可能在从未被检查过的维度上依然严重有偏。

### 这在第 18 阶段中的位置

第 20-21 课从形式上讨论偏见与公平。第 22 课讨论隐私。第 23 课讨论 watermarking。它们构成用户伤害层，与前面关于欺骗 / 安全的那一层相互补充。

## 实操

`code/main.py` 会构建一个玩具版的基于 embedding 的偏见探针：在一个简单的共现 embedding 中，测量身份词与属性词之间的 WEAT 风格距离。你可以人为注入一种偏见，然后观察指标被触发；再应用一个简单的去偏操作，观察它如何部分恢复。

## 交付

本课会产出 `outputs/skill-bias-eval.md`。给定一个模型卡或公平性声明，它会从三类指标（embedding、probability、generated-text）、交叉性覆盖范围，以及任意去偏干预的机制解释三个方面来审计评估是否充分。

## 练习

1. 运行 `code/main.py`。报告去偏步骤前后的 WEAT 风格偏见分数。解释为什么这个指标不会降到零。

2. 扩展这个探针，加入一个交叉测试：(gender, race) x (career, family)。报告跨轴偏见分数。

3. 阅读 An et al. 2025（PNAS Nexus）。指出他们报告的两种交叉效应，而这些效应是单轴性别评估会漏掉的。

4. Yu & Ananiadou 2025 识别出了 gender neurons。勾勒一个证伪实验，用来区分“这些神经元导致了性别偏见”和“这些神经元只是与性别偏见相关”。

5. 元批评认为，这个领域对二元性别的关注过于狭窄。选择一个研究不足的轴，并为它设计一个表征性伤害测量协议。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Representational harm | “刻板印象 / 抹除” | 对某个群体的偏置性描绘 |
| Allocational harm | “不平等决策” | 对某个群体造成偏置性的物质结果 |
| WEAT | “embedding 测试” | Word Embedding Association Test；一种基于共现的偏见探针 |
| Intersectionality | “组合身份效应” | 在多个身份轴交叉处出现的偏见 |
| Gender neurons | “MLP 偏见神经元” | 一些激活与性别特定行为相关的特定神经元 |
| SAE feature | “可解释维度” | 由 sparse autoencoder 识别出的特征；可用于机制性偏见分析 |
| UniBias | “attention-head 去偏” | 通过重加权 attention heads 来做零样本去偏 |

## 延伸阅读

- [Gallegos et al. — Bias and Fairness in LLMs: A Survey (arXiv:2309.00770, Computational Linguistics 2024)](https://arxiv.org/abs/2309.00770) —— 经典综述
- [An et al. — Intersectional resume-evaluation bias (PNAS Nexus, March 2025)](https://academic.oup.com/pnasnexus/article/4/3/pgaf089/8111343) —— 五模型交叉偏见研究
- [WinoIdentity — uncertainty-based intersectional fairness (arXiv:2508.07111, COLM 2025)](https://arxiv.org/abs/2508.07111) —— 新基准
- [UniBias — attention-head manipulation (Zhou et al. 2024, ACL)](https://arxiv.org/abs/2405.20612) —— 零样本去偏
