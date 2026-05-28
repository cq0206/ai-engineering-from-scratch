# 从零构建分词器 (tokenizer)

> 第 01 课给了你一个玩具。这一课给你一件武器。

**类型：** 构建
**语言：** Python
**前置条件：** 第 10 阶段，第 01 课（分词器：BPE、WordPiece、SentencePiece）
**时间：** ~90 分钟

## 学习目标

- 构建一个生产级 BPE 分词器 (tokenizer)，能够处理 Unicode、空白归一化 (whitespace normalization) 和特殊标记 (special tokens)
- 实现字节级回退 (byte-level fallback)，让分词器在没有未知词元 (token) 的情况下编码任何输入（包括 emoji、CJK 和代码）
- 添加预分词正则模式 (pre-tokenization regex patterns)，在应用 BPE 合并前先按词边界切分文本
- 在语料库上训练自定义分词器，并在多语言文本上将其压缩率 (compression ratio) 与 tiktoken 对比评估

## 问题

你在第 01 课里做的 BPE 分词器可以处理英文文本。现在拿日语试试。或者 emoji。或者混用制表符和空格的 Python 代码。

它会坏掉。

不是因为 BPE 错了，而是因为实现还不完整。一个生产级分词器要能处理任意编码下的原始字节，在切分前完成 Unicode 归一化，管理那些永远不会参与合并的特殊标记，把预分词和子词切分串起来，并且这一切都要足够快，不能成为处理 15 万亿 token 的训练流水线瓶颈。

GPT-2 的分词器有 50,257 个 token。Llama 3 有 128,256 个。GPT-4 大约有 100,000 个。这些都不是玩具级数字。支撑这些词表的合并表 (merge table) 是在数百 GB 文本上训练出来的，而周围那整套机制——归一化、预分词、特殊标记注入、聊天模板格式化——才是真正把一个只能处理 “hello world” 的分词器，和一个能处理整个互联网的分词器区分开的东西。

你现在就要把这套机制搭起来。

## 概念

### 完整流水线

一个生产级分词器并不是单一算法，而是一条包含五个阶段的流水线，每个阶段解决不同的问题。

```mermaid
graph LR
    A[原始文本] --> B[规范化]
    B --> C[预分词]
    C --> D[BPE 合并]
    D --> E[特殊标记]
    E --> F[Token ID]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
```

每个阶段都有明确职责：

| 阶段 | 作用 | 为什么重要 |
|-------|-------------|----------------|
| 归一化 | NFKC Unicode，可选小写化，可选去除重音 | `ﬁ` 连字（U+FB01）会变成 `fi`（两个字符）。否则，同一个词会得到不同 token。 |
| 预分词 | 在 BPE 之前先把文本切成片段 | 防止 BPE 跨词边界合并。`the cat` 绝不应该产生 `e c` 这样的 token。 |
| BPE 合并 | 对字节序列应用已学习的合并规则 | 压缩的核心。把原始字节变成子词 token。 |
| 特殊标记 | 注入 [BOS]、[EOS]、[PAD]、聊天模板标记 | 这些 token 有固定 ID，永远不会参与 BPE 合并。模型需要它们来表达结构。 |
| ID 映射 | 把 token 字符串转换为整数 ID | 模型看到的是整数，不是字符串。 |

### 字节级 BPE (Byte-Level BPE)

第 01 课中的分词器基于 UTF-8 字节工作。这是正确选择。但我们跳过了一个重要问题：如果这些字节不是合法的 UTF-8，会发生什么？

字节级 BPE 的做法是把每一个可能的字节值（0-255）都当作合法 token。你的基础词表恰好就是 256 个条目。任何文件——文本、二进制、损坏数据——都可以被分词，而不会产生未知 token。

GPT-2 又加了一个技巧：把每个字节映射成一个可打印的 Unicode 字符，这样词表对人类来说仍然可读。在他们的映射里，字节 0x20（空格）会变成字符 “G”。这纯粹是表现形式上的处理，算法本身并不在意。

真正的威力在于：字节级 BPE 能处理地球上的每一种语言。中文字符各自是 3 个 UTF-8 字节。日语通常是 3-4 个字节。阿拉伯语、天城文、emoji——本质上都只是字节序列。BPE 算法在这些字节序列里寻找模式的方式，与它在英文 ASCII 字节里寻找模式完全一样。

### 预分词 (Pre-Tokenization)

在 BPE 接触文本之前，你需要先把它切成若干片段。这能防止合并算法创建跨越词边界的 token。

GPT-2 使用一个正则表达式 (regex) 模式来切分文本：

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

这个模式会把文本拆成缩写形式（例如 `don't` 会变成 `don` + `'t`）、可带前导空格的单词、数字、标点和空白。前导空格会继续附着在单词上——所以 `the cat` 会变成 `[" the", " cat"]`，而不是 `["the", " ", "cat"]`。

Llama 使用 SentencePiece，完全跳过了正则表达式。它把原始字节流视作一整条长序列，让 BPE 算法自己推断边界。这种方式更简单，但也给了 BPE 更多自由去创建跨词 token。

这个选择很重要。GPT-2 的正则会阻止分词器学到“一个词末尾的 `the` 和下一个词开头的 `the` 应该合并”。SentencePiece 则允许这种情况，因此有时会带来更高的压缩效率，但 token 的可解释性更差。

### 特殊标记 (Special Tokens)

每个生产级分词器都会为结构标记预留 token ID：

| 标记 | 用途 | 使用方 |
|-------|---------|---------|
| `[BOS]` / `<<s>` | 序列开始 | Llama 3, GPT |
| `[EOS]` / `<</s>` | 序列结束 | 所有模型 |
| `[PAD]` | 用于 batch 对齐的填充 | BERT, T5 |
| `[UNK]` | 未知 token（字节级 BPE 可以消除它） | BERT, WordPiece |
| `&lt;\|im_start\|>` | 聊天消息边界起始 | ChatGPT, Qwen |
| `&lt;\|im_end\|>` | 聊天消息边界结束 | ChatGPT, Qwen |
| `&lt;\|user\|>` | 用户轮次标记 | Llama 3 |
| `&lt;\|assistant\|>` | 助手轮次标记 | Llama 3 |

特殊标记永远不会被 BPE 切分。它们会在合并算法运行前被精确匹配，替换为固定 ID，而周围文本仍按常规方式分词。

### 聊天模板 (Chat Templates)

这是大多数人最容易困惑、也是大多数实现最容易出错的地方。

当你向聊天模型发送消息时，API 接收的是一组消息列表：

```
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

模型并不会看到 JSON。它看到的是一条扁平的 token 序列。聊天模板会使用特殊标记把消息转换成这条扁平序列。每个模型的做法都不同：

```
Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>

ChatGPT:
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

只要模板写错，模型输出就会变成垃圾。它是在某一种精确格式上训练出来的。任何偏差——少一个换行、调换一个 token、多一个空格——都会让输入落到训练分布之外。

### 速度

Python 对生产环境分词来说太慢了。

`tiktoken`（OpenAI）是用 Rust 写的，并提供 Python 绑定。HuggingFace 的 `tokenizers` 也是 Rust。SentencePiece 是 C++。与纯 Python 相比，它们能带来 10-100 倍的速度提升。

举个量级上的例子：如果以每秒 100 万 token（对 Python 已经很快）对 Llama 3 预训练所需的 15 万亿 token 进行分词，需要 174 天。若以每秒 1 亿 token（Rust）处理，则只需 1.7 天。

你在这里用 Python 构建，是为了理解算法。在生产环境里，你会使用编译后的实现，并且通常只会接触 Python 包装层。

## 动手构建

### 第 1 步：字节级编码

这是基础。把任意字符串转换成字节序列，为显示目的把每个字节映射成可打印字符，然后再反向还原。

```python
def bytes_to_tokens(text):
    return list(text.encode("utf-8"))

def tokens_to_text(token_bytes):
    return bytes(token_bytes).decode("utf-8", errors="replace")
```

用多语言文本测试一下，看看字节数：

```python
texts = [
    ("English", "hello"),
    ("Chinese", "你好"),
    ("Emoji", "🔥"),
    ("Mixed", "hello你好🔥"),
]

for label, text in texts:
    b = bytes_to_tokens(text)
    print(f"{label}: {len(text)} chars -> {len(b)} bytes -> {b}")
```

`hello` 是 5 个字节。`你好` 是 6 个字节（每个字符 3 个）。火焰 emoji 是 4 个字节。字节级分词器根本不在乎它是什么语言。字节就是字节。

### 第 2 步：带正则的预分词器

使用 GPT-2 的正则模式把文本切成片段。之后，每个片段都会独立地交给 BPE 分词。

```python
import re

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

def pre_tokenize(text):
    return [match.group() for match in GPT2_PATTERN.finditer(text)]
```

`regex` 模块支持 Unicode 属性转义（字母用 `\p{L}`，数字用 `\p{N}`）。标准库里的 `re` 模块不支持，所以我们退回到 ASCII 字符类。对于生产级多语言分词器，请安装 `regex`。

试一下：

```python
print(pre_tokenize("Hello, world! Don't stop."))
# [' Hello', ',', ' world', '!', " Don", "'t", ' stop', '.']
```

前导空格会继续附着在单词上。缩写会在撇号处拆开。标点会变成独立片段。BPE 永远不会跨这些边界进行合并。

### 第 3 步：在字节序列上运行 BPE

核心算法和第 01 课一样，只不过现在它要在预分词后的各个片段上独立运行。

```python
from collections import Counter

def get_byte_pairs(chunks):
    pairs = Counter()
    for chunk in chunks:
        byte_seq = list(chunk.encode("utf-8"))
        for i in range(len(byte_seq) - 1):
            pairs[(byte_seq[i], byte_seq[i + 1])] += 1
    return pairs

def apply_merge(byte_seq, pair, new_id):
    merged = []
    i = 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return merged
```

### 第 4 步：处理特殊标记

特殊标记需要精确匹配和固定 ID。它们会完全绕过 BPE。

```python
class SpecialTokenHandler:
    def __init__(self):
        self.special_tokens = {}
        self.pattern = None

    def add_token(self, token_str, token_id):
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self.pattern = re.compile("|".join(escaped))

    def split_with_specials(self, text):
        if not self.pattern:
            return [(text, False)]
        parts = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            parts.append((match.group(), True))
            last_end = match.end()
        if last_end < len(text):
            parts.append((text[last_end:], False))
        return parts
```

### 第 5 步：完整分词器类

把一切串起来：归一化、按特殊标记切分、预分词、BPE 合并、映射到 ID。

```python
import unicodedata

class ProductionTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id = 256

    def normalize(self, text):
        return unicodedata.normalize("NFKC", text)

    def train(self, text, num_merges):
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            pairs = Counter()
            for seq in chunk_bytes:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunk_bytes = [apply_merge(seq, best, new_id) for seq in chunk_bytes]

    def add_special_token(self, token_str):
        token_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, token_id)
        self.vocab[token_id] = token_str.encode("utf-8")
        return token_id

    def encode(self, text):
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_handler.special_tokens[part_text])
            else:
                for chunk in pre_tokenize(part_text):
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    all_ids.extend(byte_seq)
        return all_ids

    def decode(self, ids):
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)
```

### 第 6 步：多语言测试

真正的测试来了。把英文、中文、emoji 和代码都扔进去。

```python
corpus = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Deep learning transforms how we build software. "
    "def train(model, data): return model.fit(data) "
    "def predict(model, x): return model(x) "
)

tok = ProductionTokenizer()
tok.train(corpus, num_merges=50)

bos = tok.add_special_token("<|begin|>")
eos = tok.add_special_token("<|end|>")

test_texts = [
    "The quick brown fox.",
    "你好世界",
    "Hello 🌍 World",
    "def foo(x): return x + 1",
    f"<|begin|>Hello<|end|>",
]

for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"Input:   {text}")
    print(f"Tokens:  {len(ids)} ids")
    print(f"Decoded: {decoded}")
    print()
```

中文字符各自会产生 3 个字节。emoji 会产生 4 个字节。它们都不会让分词器崩溃，也都不会产生未知 token。这就是字节级 BPE 的力量。

## 实际使用

### 对比真实分词器

加载 Llama 3、GPT-4 和 Mistral 的真实分词器，看看它们如何处理同一段多语言文本。

```python
import tiktoken

gpt4_enc = tiktoken.get_encoding("cl100k_base")

test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"

tokens = gpt4_enc.encode(test_paragraph)
pieces = [gpt4_enc.decode([t]) for t in tokens]
print(f"GPT-4 ({len(tokens)} tokens): {pieces}")
```

```python
from transformers import AutoTokenizer

llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

for name, tok in [("Llama 3", llama_tok), ("Mistral", mistral_tok)]:
    tokens = tok.encode(test_paragraph)
    pieces = tok.convert_ids_to_tokens(tokens)
    print(f"{name} ({len(tokens)} tokens): {pieces[:20]}...")
```

对于同一段文本，你会看到不同的 token 数量。拥有 128K 词表的 Llama 3 在合并常见模式时更激进。拥有 100K 词表的 GPT-4 处于中间。拥有 32K 词表的 Mistral 会产出更多 token，但它的嵌入层更小。

这个权衡始终不变：词表越大，序列越短，但参数也越多。

## 交付成果

这一课会产出一个用于构建和调试生产级分词器的提示词。参见 `outputs/prompt-tokenizer-builder.md`。

## 练习

1. **简单：** 添加一个 `get_token_bytes(id)` 方法，显示任意 token ID 对应的原始字节。用它检查你那些最常见的合并 token 到底表示什么。
2. **中等：** 实现一个 Llama 风格的预分词器：按空白和数字切分，但保留前导空格。把它和 GPT-2 的正则方案放在同一语料上比较词表差异。
3. **困难：** 添加一个聊天模板方法，接收一个 `{"role": ..., "content": ...}` 消息列表，并为 Llama 3 聊天格式生成正确的 token 序列。再拿 HuggingFace 的实现做对比测试。

## 关键术语

| 术语 | 人们常说什么 | 它真正的含义 |
|------|----------------|----------------------|
| 字节级 BPE | “基于字节工作的分词器” | 以 256 个字节值为基础词表的 BPE——可处理任何输入而不产生未知 token |
| 预分词 | “在 BPE 之前先切分” | 基于正则或规则的切分方式，防止 BPE 跨词边界合并 |
| NFKC 归一化 | “Unicode 清理” | 先做规范分解，再做兼容组合——`ﬁ` 连字会变成 `fi`，全角 `Ａ` 会变成 `A` |
| 聊天模板 | “消息如何变成 token” | 把一组 role/content 消息转换成扁平 token 序列的精确格式——它依赖具体模型，并且必须与训练格式一致 |
| 特殊标记 | “控制 token” | 被预留的 token ID，会绕过 BPE——[BOS]、[EOS]、[PAD]、聊天标记——在合并前先被精确匹配 |
| Fertility | “每个词对应多少 token” | 输出 token 数与输入词数的比值——GPT-4 的英文约为 1.3，韩语约为 2-3；越高表示上下文浪费越多 |
| tiktoken | “OpenAI 分词器” | Rust 实现的 BPE，并带有 Python 绑定——比纯 Python 快 10-100 倍 |
| Merge table | “词表” | 训练过程中学到的有序字节对合并列表——这就是分词器真正学到的知识 |

## 延伸阅读

- [OpenAI tiktoken source](https://github.com/openai/tiktoken) -- GPT-3.5/4 使用的 Rust BPE 实现
- [HuggingFace tokenizers](https://github.com/huggingface/tokenizers) -- 支持 BPE、WordPiece、Unigram 的 Rust 分词器库
- [Llama 3 paper (Meta, 2024)](https://arxiv.org/abs/2407.21783) -- 关于 128K 词表与分词器训练的细节
- [SentencePiece (Kudo & Richardson, 2018)](https://arxiv.org/abs/1808.06226) -- 与语言无关的分词方法
- [GPT-2 tokenizer source](https://github.com/openai/gpt-2/blob/master/src/encoder.py) -- 最初的字节到 Unicode 映射
