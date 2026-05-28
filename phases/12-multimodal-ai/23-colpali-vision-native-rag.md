# ColPali 与视觉原生（Vision-Native）文档 RAG

> 传统 RAG 会先把 PDF 解析为文本，再切成 chunk，接着为 chunk 做嵌入并存成向量。每一步都在丢失信号：OCR 会丢掉图表数据，chunking 会拆断表格行，文本嵌入会忽略图形。ColPali（Faysse 等，2024 年 7 月）提出了一个更简单的问题：为什么一定要先抽取文本？直接通过 PaliGemma 嵌入页面图像，用 ColBERT 风格的后期交互（late interaction）做检索，并保留文档中的布局、图形、字体和格式信号。公开 benchmark 显示：在视觉信息丰富的文档上，端到端准确率比 text-RAG 高 20-40%。ColQwen2、ColSmol 和 VisRAG 进一步扩展了这一模式。本课会解释 vision-native RAG 的核心论点，并构建一个迷你版 ColPali 索引器。

**类型：** 构建
**语言：** Python（stdlib，multi-vector indexer + MaxSim scorer）
**先修要求：** 第 11 阶段（LLM Engineering — RAG basics），第 12 阶段 · 05（LLaVA）
**时间：** ~180 分钟

## 学习目标

- 解释双编码器检索（bi-encoder retrieval，每文档一个向量）与后期交互检索（late-interaction retrieval，每文档多个向量）之间的区别。
- 描述 ColBERT 的 MaxSim 操作，以及 ColPali 如何把它从文本 token 泛化到图像 patch。
- 构建一个迷你版 ColPali 索引器：页面 → patch embeddings → 对 query-term embeddings 做 MaxSim → top-k 页面。
- 比较 ColPali + Qwen2.5-VL 生成器 与 text-RAG + GPT-4 在发票 / 财务报告场景中的表现。

## 问题

对 PDF 做 text-RAG 会丢掉文档中的大部分信息。财报里 Q3 营收增长通常在图表中；医疗报告的发现常常存在于带标注图像中；法律合同的签名区块是一种布局事实，而不是文本事实。

Text-RAG 流水线如下：

1. PDF → 通过 OCR / pdftotext 转成文本。
2. 文本 → 切成 300-500 token 的 chunk。
3. Chunk → bi-encoder embedding（一个向量）。
4. 用户查询 → embedding → 余弦相似度 → top-k chunk。
5. Chunk + 查询 → LLM。

五个有损步骤。图表无法被捕捉。表格被拆散到不同 chunk。多栏布局被压平。图中注释消失。

ColPali 的修复方式是：跳过 OCR，直接嵌入页面图像。再使用 ColBERT 风格的后期交互检索，让模型能够在查询时关注细粒度 patch。

## 概念

### ColBERT（2020）

ColBERT（Khattab & Zaharia，arXiv:2004.12832）是一种文本检索方法。它不是“每个文档一个向量”，而是“每个 token 一个向量”。在查询时：

- 查询 token 各自得到独立嵌入（N_q 个向量）。
- 文档 token 得到嵌入（N_d 个向量，通常会缓存）。
- 分数 = 对每个查询 token，在所有文档 token 中找最大余弦相似度后求和：Σ_i max_j cos(q_i, d_j)。

这就是 MaxSim 操作。每个查询 token 都会“挑出”自己最匹配的文档 token。最终分数就是这些匹配分数之和。

优点：召回强，能处理词项级语义。缺点：每个文档需要存 N_d 个向量，存储昂贵。

### ColPali

ColPali（Faysse 等，arXiv:2407.01449）把 ColBERT 模式应用到了图像上。

- 每个页面由 PaliGemma（ViT + language）编码成 patch embeddings：每页 N_p 个向量。
- 每个用户查询（文本）被编码为 query-token embeddings：N_q 个向量。
- 分数 = Σ_i max_j cos(q_i, p_j)，也就是在查询文本 token 与页面图像 patch 之间做 MaxSim。
- 根据总分检索 top-k 页面。

在文档摄入（document-ingestion）阶段：用 PaliGemma 嵌入每一页，并存储所有 patch embeddings。在查询阶段：嵌入查询 token，与所有已存储页面嵌入计算 MaxSim，返回 top-k 页面。

优点：在视觉信息丰富的文档上，端到端效果比 text-RAG 高 20-40%。每个 patch 向量都携带局部布局与内容信息。

缺点：每页需要 N_p 个 patch × 4-byte float × D 维向量，存储增长很快。通常通过 PQ / OPQ quantization 缓解。

### ColQwen2 与 ColSmol

ColQwen2（illuin-tech，2024-2025）把 PaliGemma 替换为 Qwen2-VL。基础编码器更强，检索也更强。

ColSmol 是面向本地 / 边缘使用的小规模变体。一个约 1B 参数的 ColSmol 检索器可以在消费级 GPU 上运行。

### VisRAG

VisRAG（Yu 等，arXiv:2410.10594）采用了另一种变体：不是对 patch 做 MaxSim，而是先用 VLM 把每一页池化成单个向量，再用 bi-encoder 做检索。索引更快、存储更小，但召回更弱。

这是一种质量与成本的权衡：ColPali 追求质量，VisRAG 追求规模。

### M3DocRAG

M3DocRAG（Cho 等，arXiv:2411.04952）把多模态检索扩展到了多页、多文档推理。它会跨文档检索页面，并为 VLM 组合一个多页上下文。

### ViDoRe —— 基准

这是 ColPali 配套的 benchmark，全称是 Visual Document Retrieval Evaluation。任务包括财务报告、科学论文、行政文档、医疗记录、手册。指标：nDCG@5。

ColPali-v1 在 ViDoRe 上约为 80% nDCG@5；而相同文档上的 text-RAG 大约只有 50-60%。

### 端到端 RAG 流水线

对于 vision-native RAG：

1. 摄入：PDF → 页面图像 → PaliGemma 编码 → 存储所有 patch embeddings。
2. 查询：用户文本 → query-token embeddings → 对所有索引页面做 MaxSim → top-k 页面。
3. 生成：top-k 页面图像 + 查询 → VLM（Qwen2.5-VL 或 Claude）→ 答案。

全程都不需要 OCR。图形、图表、字体、布局都会一路流入最终答案。

### 存储数学

一份 50 页的财务报告，如果每页 729 个 patch、128 维嵌入：

- ColPali：50 * 729 * 128 * 4 bytes = 原始约 18 MB，经过 PQ 后约 4 MB。
- Text-RAG：50 个 chunk * 768 维 * 4 bytes = 约 150 kB。

ColPali 每份文档的存储大约是 30 倍。到了大规模场景，OPQ / PQ 通常能把它压到约 5-10 倍，通常仍可接受。

### 什么时候 text-RAG 仍然更优

- 纯文本文档，没有布局信号（wiki 文章、聊天日志）。Text-RAG 更简单，存储也更省。
- 数百万页级别的档案库，其中存储成本是主导因素。
- 严格监管要求检索时必须同时提供可提取的 OCR 文本。

对于 2026 年的其他大多数场景——财务报告、科学论文、法律合同、医疗记录、UX 文档——vision-native RAG 都更优。

## 使用它

`code/main.py`：

- 玩具版 patch encoder：把一个“页面”（小型特征向量网格）映射为一组 patch embeddings。
- MaxSim scorer：计算查询 token 嵌入集与页面 patch 集之间的 ColBERT 风格分数。
- 索引 5 个玩具页面，执行 3 个查询，并返回带分数的 top-k 结果。

## 交付它

本课会生成 `outputs/skill-vision-rag-designer.md`。给定一个 document-RAG 项目，它会在 ColPali / ColQwen2 / VisRAG / text-RAG 之间做选择，并估算存储规模。

## 练习

1. 一份 200 页年报，每页 729 个 patch、128 维 emb、4-byte float。请计算原始存储，以及 PQ 压缩（8x）后的存储。

2. MaxSim 是 Σ_i max_j cos(q_i, p_j)。相比简单的平均相似度，这个求和捕捉到了什么？

3. ColPali 把页面索引为 patch 集。如果我们改为按词级别建立索引（像 ColBERT 那样），会发生什么变化？权衡是什么？

4. 为一个 100 万页语料库设计端到端流水线，查询延迟预算为每次 500ms。请在 ColQwen2 / VisRAG 中做选择并说明理由。

5. 阅读 M3DocRAG（arXiv:2411.04952）。描述其多页注意力模式，以及它与单页 ColPali 检索有何不同。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|----------|
| Late interaction | “ColBERT 风格” | 检索时使用逐 token 或逐 patch 嵌入 + MaxSim，而不是单一文档向量 |
| MaxSim | “在 patch 上取最大值” | 对每个查询 token，选取相似度最高的文档 token；再对查询维度求和 |
| Bi-encoder | “单向量” | 每个文档一个向量；更快，但会损失细粒度信息 |
| Multi-vector | “每个文档多个向量” | 每个文档 / 页面存储 N_p 个向量；存储成本增加，但召回改善 |
| Patch embedding | “页面特征” | 由 VLM 编码器为每个图像 patch 产生的一个向量，按页缓存 |
| ViDoRe | “视觉文档基准” | ColPali 的视觉文档检索 benchmark 套件 |
| PQ quantization | “乘积量化” | 在缩小存储约 8 倍的同时，尽量保持向量相似性的压缩方法 |

## 延伸阅读

- [Faysse et al. — ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449)
- [Khattab & Zaharia — ColBERT (arXiv:2004.12832)](https://arxiv.org/abs/2004.12832)
- [Yu et al. — VisRAG (arXiv:2410.10594)](https://arxiv.org/abs/2410.10594)
- [Cho et al. — M3DocRAG (arXiv:2411.04952)](https://arxiv.org/abs/2411.04952)
- [illuin-tech/colpali GitHub](https://github.com/illuin-tech/colpali)
