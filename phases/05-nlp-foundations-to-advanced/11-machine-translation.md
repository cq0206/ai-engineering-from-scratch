# 机器翻译（Machine Translation）

> 翻译这项任务，三十年来一直在为 NLP 研究买单，现在仍然如此。

**类型：** 构建
**语言：** Python
**前置条件：** 第 5 阶段 · 10（注意力机制），第 5 阶段 · 04（GloVe、FastText、Subword）
**时间：** ~75 分钟

## 问题

一个模型读取一种语言中的句子，并生成另一种语言中的句子。长度会变。词序会变。有些源词会映射到多个目标词，反之亦然。习语根本不接受一一映射。法语里，“I miss you” 是 “tu me manques”——字面上更接近“你之于我，是缺失的”。这种情况下，词级对齐根本存活不下来。

机器翻译（Machine Translation, MT）是那个逼着 NLP 发明了编码器-解码器（encoder-decoder）、注意力（attention）、Transformer，最终连整个 LLM 范式都推出来的任务。每一步前进，都是因为翻译质量可以被衡量，而机器与人之间的差距又异常顽固。

本课跳过历史课，直接讲 2026 年仍然实用的工作流水线：预训练多语言编码器-解码器（pretrained multilingual encoder-decoder，如 NLLB-200 或 mBART）、子词分词（subword tokenization）、束搜索（beam search）、BLEU 和 chrF 评估，以及那几种直到今天仍会悄悄漏进生产环境的失败模式。

## 概念

*MT 流水线：分词 → 编码 → 带注意力的解码 → 反分词*

现代 MT 是在平行文本上训练的 Transformer 编码器-解码器。编码器用该语言对应的分词方式读取源文本。解码器利用编码器输出，通过交叉注意力（cross-attention，第 10 课）一次生成一个子词（subword）组成目标文本。解码通常使用束搜索，以避免贪心解码（greedy decoding）的陷阱。输出随后会被反分词、恢复真实大小写（detruecase），并与参考译文进行打分。

有三个操作层面的选择，决定了真实世界里的 MT 质量。

- **分词器（Tokenizer）。** 在混合语言语料上训练的 SentencePiece BPE。跨语言共享词表，正是 NLLB 能支持 zero-shot 语言对的原因。
- **模型大小（Model size）。** NLLB-200 distilled 600M 能在笔记本上跑。NLLB-200 3.3B 是公开发表的生产默认。54.5B 是研究上限。
- **解码（Decoding）。** 一般内容用 4-5 的束宽。用长度惩罚（length penalty）避免输出过短。需要术语一致性时，用受约束解码（constrained decoding）。

## 动手构建

### 第 1 步：一次预训练 MT 调用

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

src = "The cats are running."
inputs = tok(src, return_tensors="pt")

out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
    num_beams=5,
    length_penalty=1.0,
    max_new_tokens=64,
)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

```text
Les chats courent.
```

这里有三件事很重要。`src_lang` 告诉分词器应使用哪种书写系统和分词方式。`forced_bos_token_id` 告诉解码器应生成哪种语言。这两点都是 NLLB 特有的小技巧；mBART 和 M2M-100 有各自的约定，彼此并不能互换。

### 第 2 步：BLEU 和 chrF

BLEU 衡量输出与参考译文之间的 n-gram 重叠。它使用 1-4 阶参考 n-gram、精确率的几何平均，以及针对过短输出的简短惩罚（brevity penalty）。分数范围是 [0, 100]。它很常用，但解释起来很别扭：30 BLEU 算“能用”；40 算“不错”；50 算“非常出色”；小于 1 BLEU 的差异通常只是噪声。

chrF 衡量字符级 F-score。对于形态变化丰富的语言，它比 BLEU 更敏感，因为 BLEU 往往会少算匹配。通常会和 BLEU 一起报告。

```python
import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")
```

始终使用 `sacrebleu`。它会标准化分词，因此分数才能跨论文可比。自己手搓 BLEU 计算，往往就是误导性基准测试的开端。

### 三层评估层级（2026）

现代 MT 评估使用三类互补的指标。至少带上其中两类再上线。

- **启发式（Heuristic）。** BLEU、chrF。快、基于参考译文、可解释，但对释义不敏感。用于历史对比和回归检测。
- **学习型（Learned）。** COMET、BLEURT、BERTScore。基于人类判断训练的神经模型，用来比较译文与源文、参考译文之间的语义相似性。自 2023 年以来，COMET 在 MT 研究中的相关性最高，到 2026 年已是质量敏感场景的生产默认。
- **LLM 评委（LLM-as-judge，无参考）。** 提示一个大型模型，从流畅度、充分性、语气、文化恰当性等方面给译文打分。当评分 rubric 设计得好时，GPT-4-as-judge 与人工一致率约为 80%。适用于没有参考译文的开放式内容。

2026 年的实用组合：用 `sacrebleu` 计算 BLEU 和 chrF，用 `unbabel-comet` 计算 COMET，再用一个经提示设计的 LLM 提供最终、面向人的判断信号。在信任任何指标之前，先用 50-100 个人工标注样本做校准。

无参考指标（reference-free metrics）如 COMET-QE、BLEURT-QE、LLM-as-judge，让你在没有参考译文的情况下评估翻译，这对长尾语言对尤其重要，因为这些场景往往根本没有参考译文。

### 第 3 步：生产环境里会坏掉的地方

上面的流水线在 80% 的时候会翻得很流畅，而剩下 20% 会默默失败。下面是几个有名字的失败模式：

- **幻觉（Hallucination）。** 模型编造了源文里没有的内容。在陌生领域词汇下很常见。症状：输出很流畅，但声称了源文并未表达的事实。缓解：对领域术语做受约束解码；对受监管内容做人审；监控那些远长于输入的输出。
- **跑偏生成（Off-target generation）。** 模型翻成了错误的语言。NLLB 在稀有语言对上出人意料地容易出现这个问题。缓解：核对 `forced_bos_token_id`，并始终用语言识别（language-ID）模型检查输出。
- **术语漂移（Terminology drift）。** “Sign up” 在文档 1 里被译为 “s'inscrire”，在文档 2 里却成了 “créer un compte”。对 UI 文案和用户可见字符串来说，一致性比裸质量更重要。缓解：词汇表约束解码（glossary-constrained decoding）或后编辑字典。
- **正式程度不匹配（Formality mismatch）。** 法语里的 “tu” vs “vous”，日语里的敬语等级。模型通常会选训练数据中更常见的形式。对面向客户的内容，这通常是错的。缓解：如果模型支持，就在提示前缀中加入表示正式程度的 token；否则在只含正式语料的小模型上微调。
- **短输入上的长度爆炸（Length explosion on short input）。** 很短的输入句子常会产生过长译文，因为当源序列长度低于 ~5 个 token 时，长度惩罚会突然失效。缓解：使用与源长度成比例的硬性最大长度上限。

### 第 4 步：面向领域的微调

预训练模型是通才。法律、医疗或游戏对话的翻译，在领域平行数据上微调后会有可测的提升。配方并不神秘：

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

pairs = [
    {"src": "The defendant pleaded guilty.", "tgt": "L'accusé a plaidé coupable."},
]

ds = Dataset.from_list(pairs)


def preprocess(ex):
    return tok(
        ex["src"],
        text_target=ex["tgt"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )


ds = ds.map(preprocess, remove_columns=["src", "tgt"])

args = TrainingArguments(output_dir="out", per_device_train_batch_size=4, num_train_epochs=3, learning_rate=3e-5)
Trainer(model=model, args=args, train_dataset=ds).train()
```

几千条高质量平行样本，胜过几十万条噪声很大的网页抓取样本。训练数据质量，是生产里最大的杠杆。

## 使用它

2026 年 MT 的生产栈：

| 用例 | 推荐起点 |
|---------|---------------------------|
| 任意到任意，200 种语言 | `facebook/nllb-200-distilled-600M`（笔记本）或 `nllb-200-3.3B`（生产） |
| 以英语为中心，高质量，50 种语言 | `facebook/mbart-large-50-many-to-many-mmt` |
| 短任务、低成本推理、英语-法语/德语/西班牙语 | Helsinki-NLP / Marian models |
| 浏览器侧延迟敏感 | ONNX 量化 Marian（约 50 MB） |
| 追求最高质量，愿意付费 | 带翻译提示的 GPT-4 / Claude / Gemini |

截至 2026 年，LLM 已经在若干语言对上超越了专用 MT 模型，尤其是在习惯用法内容和长上下文上。权衡点是按 token 计费的成本与延迟。当上下文长度、风格一致性或通过提示进行领域适配比吞吐量更重要时，选 LLM。

## 交付它

保存为 `outputs/skill-mt-evaluator.md`：

```markdown
---
name: mt-evaluator
description: Evaluate a machine translation output for shipping.
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

Given a source text and a candidate translation, output:

1. Automatic score estimate. BLEU and chrF ranges you would expect. State whether a reference is available.
2. Five-point human-verifiable check list: (a) content preservation (no hallucinations), (b) correct language, (c) register / formality match, (d) terminology consistency with glossary if provided, (e) no truncation or length explosion.
3. One domain-specific issue to probe. E.g., for legal: named entities and statute citations. For medical: drug names and dosages. For UI: placeholder variables `{name}`.
4. Confidence flag. "Ship" / "Ship with review" / "Do not ship". Tie to the severity of issues found in step 2.

Refuse to ship a translation without a language-ID check on output. Refuse to evaluate without a reference unless the user explicitly opts in to reference-free scoring (COMET-QE, BLEURT-QE). Flag any content over 1000 tokens as likely needing chunked translation.
```

## 练习

1. **简单。** 用 `nllb-200-distilled-600M` 把一个 5 句的英文段落翻成法语，再回译成英语。测量往返翻译与原文的接近程度。你应该会看到语义基本保留，但措辞会发生漂移。
2. **中等。** 使用 `fasttext lid.176` 或 `langdetect` 对翻译输出实现语言识别检查。把它集成进 MT 调用里，这样跑偏生成会在返回前被拦住。
3. **困难。** 在你选择的一个 5,000 对领域语料上微调 `nllb-200-distilled-600M`。测量微调前后在留出集上的 BLEU。报告哪些类型的句子变好了，哪些退化了。

## 关键术语

| 术语 | 大家常说 | 实际含义 |
|------|----------|----------|
| BLEU | 翻译分数 | 带简短惩罚的 n-gram 精确率。[0, 100]。 |
| chrF | 字符 F-score | 字符级 F-score。对形态丰富语言更敏感。 |
| NMT | 神经 MT | 在平行文本上训练的 Transformer 编码器-解码器。2017 年后的默认方案。 |
| NLLB | No Language Left Behind | Meta 的 200 语种 MT 模型家族。 |
| 受约束解码（Constrained decoding） | 受控输出 | 强制某些 token 或 n-gram 在输出中出现 / 不出现。 |
| 幻觉（Hallucination） | 编造内容 | 模型输出中没有得到源文支持的内容。 |

## 延伸阅读

- [Costa-jussà et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) —— NLLB 论文。
- [Post (2018). A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/) —— 为什么 `sacrebleu` 是报告 BLEU 的唯一正确方式。
- [Popović (2015). chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/) —— chrF 论文。
- [Hugging Face MT guide](https://huggingface.co/docs/transformers/tasks/translation) —— 实用的微调入门指南。
