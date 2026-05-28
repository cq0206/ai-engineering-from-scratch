# 百万 Token 上下文中的长视频理解

> 一段 1 小时、4K、24 FPS 的视频，在切成 patch 并做嵌入后，大约会产生 6000 万个 token。一段 2 小时的播客，转录后约为 30000 个 token。一整部蓝光电影，即便经过激进池化压缩，也仍然有数十万 token。Google 的 Gemini 1.5（2024 年 3 月）以 1000 万 token 上下文开启了这个时代，能够在长达数小时的视频中可靠完成“大海捞针（needle-in-a-haystack）”式召回。LWM（Liu 等，2024 年 2 月）展示了 ring attention 的扩展路径。LongVILA 和 Video-XL 进一步扩大了可摄入规模。VideoAgent 则用 agentic retrieval 替代了原始上下文。每条路线都是在计算成本、召回率和工程复杂度之间做不同权衡。本课会把它们并排讲清楚。

**类型：** 构建
**语言：** Python（stdlib，needle-in-haystack simulator + agentic-retrieval router）
**先修要求：** 第 12 阶段 · 17（video temporal tokens）
**时间：** ~180 分钟

## 学习目标

- 计算长视频在不同 FPS 与池化设置下的总视觉 token 数。
- 解释三条扩展路径：蛮力上下文（brute context，Gemini 1.5）、ring attention（LWM）、token 压缩（LongVILA / Video-XL）。
- 比较原始上下文视频 VLM 与 agentic retrieval 视频 VLM（VideoAgent）在准确率和延迟上的差异。
- 为一段 30 分钟视频设计大海捞针测试，并测量特定分钟位置上的召回率。

## 问题

在 Qwen2.5-VL 级别的 patch 设置下，一帧 384 原生分辨率图像约有 729 个 token。若做 3x3 池化，则每帧为 81 个 token。一段 30 分钟视频，按 1 FPS 采样 = 1800 帧 = 145800 个 token。到 2025 年的开源 VLM 还算可做，但已经很紧。若升到 2 FPS，则是 291600 个 token——只有最大上下文模型才能装下。

一部 2 小时电影在 1 FPS 下有 58.3 万个 token。已经超出大多数 2026 年开源模型的能力；需要 Gemini 2.5 Pro，或者更激进的池化。

于是出现了三条扩展路径。

## 概念

### 路径 1：蛮力上下文（Gemini 1.5、Claude Opus）

直接往问题上砸硬件。把上下文扩展到数百万 token，在一次前向传播中处理全部内容。

Gemini 1.5 Pro 发布时支持 100 万 token；Gemini 1.5 Ultra 达到 1000 万；到 2026 年，Gemini 2.5 Pro 已能稳定处理数小时视频。论文（arXiv:2403.05530）记录了其在约 950 万 token 以内的大海捞针召回率高达 99.7%。

工程实现上：使用自定义注意力实现，结合内存层次结构（local + global + sparse）以及 MoE expert routing，以提升长上下文效率。完整细节并未公开。也不是开源实现。

### 路径 2：Ring attention（LWM、LongVILA）

Ring attention 会把长序列分布到多个设备上，形成一个“环”，每个设备持有其中一个片段。跨全序列的注意力通过一种环形模式完成：每个设备把自己的片段发送给下一个设备，计算部分注意力，再进行聚合。

LWM（Liu 等，2024）就是用这种方式训练了一个 100 万 token 上下文模型。训练计算量随上下文长度线性增长，而不是二次增长——注意力的二次成本被分摊到了环中的多个设备上。

LongVILA（arXiv:2408.10188）把这种模式适配到了 VLM。1400 帧视频、每帧 192 个 token = 26.8 万上下文，并通过 8 路并行上的 ring attention 完成训练。

### 路径 3：Token 压缩（Video-XL、LongVA）

比蛮力上下文更便宜：在序列送入 LLM 之前先做激进压缩。

Video-XL（arXiv:2409.14485）使用视觉摘要 token（visual summary token）：每个由 N 帧组成的片段会生成一个“摘要” token，该 token 对这 N 帧进行注意力聚合。推理时，LLM 每个片段只看到一个摘要 token，从而大幅压缩上下文。

LongVA 则通过“long context transfer”技术，把 LLM 上下文从 20 万扩展到 200 万。先在长上下文文本上训练，再通过共享表示把能力迁移到长上下文视频。

Token 压缩是在“特定时间戳召回”与“可扩展性”之间做交换。模型大致知道发生了什么，但有时会错过精确帧位置。

### 路径 4：Agentic retrieval（VideoAgent）

不要把整段视频都喂给 LLM。相反，把视频当作数据库，并让 LLM 去查询它。

VideoAgent（arXiv:2403.10517）：

1. LLM 读取问题。
2. LLM 请求一个检索工具来找相关片段（“show me segments with a cat”）。
3. 工具返回匹配片段的时间戳。
4. LLM 通过 VLM 读取这些片段。
5. LLM 组合答案，或者继续发起追问。

这就是把“LLM 作为 agent”的模式应用到长视频上。推理更便宜（只编码相关片段），但工程更复杂（检索质量会成为瓶颈）。

### 大海捞针基准

标准长上下文测试是：在视频的随机位置插入一个独特的视觉或文本标记，然后提出一个必须回忆该标记的问题。

指标：跨视频长度和标记位置的 Recall@k。

Gemini 2.5 Pro 在最长 90 分钟视频上可达到 >99% 的召回。开源 72B 模型（Qwen2.5-VL-72B、InternVL3-78B）在 30 分钟时约为 85-90%，超过 60 分钟后开始退化。

VideoAgent 在 2 小时以上内容上可能追平甚至超过原始上下文模型，因为只要工具足够好，检索就能命中“针”。

### 该选哪条路径

如果你要在 15 分钟片段上追求前沿精度：开源 72B + 原生上下文通常就够。选 Qwen2.5-VL-72B。

如果内容长度在 30 分钟到 1 小时：开源可选 LongVILA 或 Video-XL；闭源可选 Gemini 2.5 Pro。质量门槛很重要——前沿表现通常还是闭源更强。

如果内容长度在 2 小时以上：选 VideoAgent 或类似检索模式。或者先做分层摘要，再把更小块的摘要送入模型。

### 2026 年生产模式

在实践中，生产级长视频流水线通常是混合式的：

1. 对整段视频运行动态 FPS 采样 + 激进池化（得到一个 10 万 token 级的全局表示）。
2. 把它送入 72B VLM，生成全局摘要。
3. 如果用户提出细节问题，再利用这个摘要作为索引，执行 agentic retrieval。

这样就把“蛮力上下文”的全局理解和“检索模式”的局部细节结合起来了。

## 使用它

`code/main.py`：

- 计算从 1 分钟到 3 小时视频在不同 FPS + 池化设置下的 token 预算。
- 模拟一次大海捞针运行：在随机时间戳注入标记、提出问题、评估召回。
- 包含一个 agentic-retrieval router 模拟器，用于挑选特定片段送入下游 VLM。

运行预算表，亲自感受规模鸿沟。

## 交付它

本课会生成 `outputs/skill-long-video-strategy-planner.md`。给定视频时长和查询复杂度，它会在 brute-context、compression 与 agentic retrieval 之间做选择，并计算延迟与质量预期。

## 练习

1. 一段 45 分钟讲座，1 FPS、每帧 81 个 token。总 token 数是多少？能装进哪些模型的上下文？

2. 设计一个大海捞针测试：你会在第几分钟注入标记？查询格式具体是什么？

3. 比较 brute-context Qwen2.5-VL-72B（80k context）与 VideoAgent（Claude 3.5 + retrieval）在 1 小时视频上的表现。谁的召回更高？谁的延迟更低？

4. Ring attention 的内存成本会随序列长度线性增长，也会随设备数量线性增长。解释原因，并说明如果去掉 ring-rotation 阶段会先出什么问题。

5. 阅读 Gemini 1.5 第 5 节关于大海捞针的内容。论文在 100 万与 1000 万 token 边界上，对召回率有什么发现？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|----------|
| Brute context | “就是加更多 token” | 把 LLM 上下文扩展到数百万 token；一遍处理全部内容 |
| Ring attention | “LWM 风格并行” | 一种分布式注意力模式：每个设备持有一个片段并轮转计算 |
| Token compression | “摘要 token” | 在送入 LLM 之前，用学习到的压缩器减少每个片段的 token 数 |
| Needle-in-haystack | “NIH test” | 在随机位置插入唯一标记，并在测试时要求模型回忆它 |
| Agentic retrieval | “LLM 作为查询规划器” | LLM 请求检索工具找相关片段，经 VLM 读取后再组合答案 |
| VideoAgent | “视频检索模式” | 典型的 agentic-retrieval 设计：question -> tool -> clip -> answer |

## 延伸阅读

- [Gemini Team — Gemini 1.5 (arXiv:2403.05530)](https://arxiv.org/abs/2403.05530)
- [Liu et al. — LWM / RingAttention (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Xue et al. — LongVILA (arXiv:2408.10188)](https://arxiv.org/abs/2408.10188)
- [Shu et al. — Video-XL (arXiv:2409.14485)](https://arxiv.org/abs/2409.14485)
- [Wang et al. — VideoAgent (arXiv:2403.10517)](https://arxiv.org/abs/2403.10517)
