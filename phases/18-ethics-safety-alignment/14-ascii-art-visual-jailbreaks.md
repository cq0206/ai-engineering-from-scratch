# ASCII Art 与视觉越狱

> Jiang、Xu、Niu、Xiang、Ramasubramanian、Li、Poovendran，《ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs》（ACL 2024，arXiv:2402.11753）。把有害请求中与安全相关的 token 遮掉，用相同字母的 ASCII-art 渲染来替换，然后把这个伪装过的 prompt 发出去。GPT-3.5、GPT-4、Gemini、Claude、Llama-2 都无法稳健识别 ASCII-art token。这种攻击可以绕过 PPL（困惑度过滤器）、Paraphrase 防御和 Retokenization。相关工作中，ViTC 基准测量模型对非语义视觉 prompts 的识别能力；StructuralSleight 则把这一类攻击推广到非常见文本编码结构 (Uncommon Text-Encoded Structures, UTES)，包括树、图、嵌套 JSON 等一整个编码攻击家族。

**类型：** 构建
**语言：** Python（stdlib，ArtPrompt token-masking harness）
**先修：** Phase 18 · 12 (PAIR), Phase 18 · 13 (MSJ)
**时间：** ~60 分钟

## 学习目标

- 描述 ArtPrompt 攻击：词识别步骤、ASCII-art 替换、最终伪装 prompt。
- 解释为什么标准防御（PPL、Paraphrase、Retokenization）会对 ArtPrompt 失效。
- 定义 ViTC，并描述它测量的内容。
- 将 StructuralSleight 描述为一种对任意非常见文本编码结构的推广。

## 问题

通过改写和角色扮演进行的攻击（第 12 课）以及通过超长上下文进行的攻击（第 13 课），都作用在文本级模式上。ArtPrompt 作用在识别级别：模型并不是在解析被禁词，而是在解析一个由字符画出来的图像。安全过滤器看到的是无害的标点符号；模型看到的是一个词。

## 概念

### ArtPrompt：两步走

第 1 步：词识别。给定一个有害请求，攻击者使用 LLM 识别其中与安全相关的词（例如在 “how to make a bomb” 中识别出 “bomb”）。

第 2 步：伪装 prompt 生成。把每个被识别出的词替换成它的 ASCII-art 渲染形式（一个 7x5 或 7x7 的字符块，用字符构成字母形状）。模型收到的是一片由标点和空格组成的网格；对足够强的模型来说，它能把这片网格识别成那个词；而安全过滤器看到的只是网格本身。

结果：GPT-4、Gemini、Claude、Llama-2、GPT-3.5 全都会失效。在它们的基准子集上，攻击成功率超过 75%。

### 为什么标准防御会失效

- **PPL（困惑度过滤器）。** ASCII art 的困惑度很高——但所有新奇输入的困惑度都很高。一个能挡住 ArtPrompt 的阈值，也会挡住合法的结构化输入。
- **Paraphrase。** 对 prompt 做改写会破坏 ASCII art。但在实践里，用于改写的 LLM 往往会保留甚至重建这些字符画。
- **Retokenization。** 把 token 切分方式改掉，并不能改变模型是在识别字母形状这件事。

根本问题在于：安全过滤器工作在 token 或语义层面；而 ArtPrompt 工作在视觉识别层面。

### ViTC 基准

用于衡量模型识别非语义视觉 prompts 的能力。它测量模型读取 ASCII-art、wingdings 以及其他非文本语义视觉内容的能力。ArtPrompt 的有效性与 ViTC 准确率相关：模型越擅长“读懂”视觉文本，ArtPrompt 在它身上就越有效。这是一个能力-安全权衡。

### StructuralSleight

它把 ArtPrompt 推广到非常见文本编码结构（UTES）。树、图、嵌套 JSON、CSV-in-JSON、diff 风格代码块。只要一种结构在安全训练数据中足够罕见，但模型又能解析它，它就可以被用来隐藏有害内容。

这对防御的启示是：安全机制必须能泛化到模型可以解析的各种结构化表示之上。而这个集合既很大，还在继续增长。

### 图像模态中的对应物

视觉 LLM（GPT-5.2、Gemini 3 Pro、Claude Opus 4.5、Grok 4.1）进一步扩大了攻击面。用真实图片来实现 ArtPrompt 风格的攻击，会比 ASCII-art 类比更强，因为图像编码器能产出更丰富的信号。

### 这在第 18 阶段中的位置

第 12-14 课描述了三种彼此正交的攻击向量：迭代细化（PAIR）、上下文长度（MSJ），以及编码（ArtPrompt / StructuralSleight）。第 15 课会从以模型为中心的攻击转向以系统边界为中心的攻击（间接 prompt injection）。第 16 课则描述防御工具链的响应。

## 实操

`code/main.py` 会构建一个玩具版 ArtPrompt。你可以把有害查询中的特定词用 ASCII-art 字形伪装起来，验证伪装后的字符串能够通过关键词过滤器，并且（可选地）用一个简单识别器把伪装字符串再解码回来。

## 交付

本课会产出 `outputs/skill-encoding-audit.md`。给定一份 jailbreak 防御报告，它会枚举其中覆盖的编码攻击家族（ASCII art、base64、leet-speak、UTF-8 homoglyph、UTES），以及能捕获每一种攻击的防御层。

## 练习

1. 运行 `code/main.py`。验证伪装后的字符串能够通过一个简单关键词过滤器。报告为此所需的字符级修改。

2. 为同一个目标词实现第二种编码：base64。比较它与 ArtPrompt 在过滤器绕过率上的差异，以及恢复原词的难度。

3. 阅读 Jiang et al. 2024 第 4.3 节（五模型结果）。提出一个理由，解释为什么 Claude 在同一基准上的 ArtPrompt 抵抗性高于 Gemini。

4. 设计一个生成前防御，用于检测 prompt 中具有 ASCII-art 形状的区域。测量它在合法代码、表格和数学符号上的误报率。

5. StructuralSleight 列出了 10 种编码结构。勾勒一个能够同时处理这 10 种结构的通用防御，并估算每个受保护 prompt 的计算成本。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| ArtPrompt | “ASCII-art 攻击” | 一种两步越狱：用 ASCII-art 渲染遮蔽安全相关词 |
| Cloaking | “把词藏起来” | 用模型能读、过滤器却读不出来的视觉表示替换被禁 token |
| UTES | “非常见结构” | 非常见文本编码结构——树、图、嵌套 JSON 等——可被用来走私内容 |
| ViTC | “视觉文本能力” | 测量模型读取非语义视觉编码的能力的基准 |
| Perplexity filter | “PPL 防御” | 拒绝高困惑度 prompt；之所以失败，是因为合法结构化输入也常常高困惑度 |
| Retokenization | “tokenizer 转移防御” | 用不同 tokenizer 预处理 prompt；之所以失败，是因为识别过程本质上是视觉性的 |
| Homoglyph | “长得一样的字符” | 看起来与拉丁字母相同的 Unicode 字符；可绕过子串检查 |

## 延伸阅读

- [Jiang et al. — ArtPrompt (ACL 2024, arXiv:2402.11753)](https://arxiv.org/abs/2402.11753) —— ASCII-art 越狱论文
- [Li et al. — StructuralSleight (arXiv:2406.08754)](https://arxiv.org/abs/2406.08754) —— UTES 推广工作
- [Chao et al. — PAIR (Lesson 12, arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) —— 互补的迭代式攻击
- [Anil et al. — Many-shot Jailbreaking (Lesson 13)](https://www.anthropic.com/research/many-shot-jailbreaking) —— 互补的长度攻击
