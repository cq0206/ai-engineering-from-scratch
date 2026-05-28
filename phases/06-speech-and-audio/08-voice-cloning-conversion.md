# 语音克隆 (Voice Cloning) 与语音转换 (Voice Conversion)

> 语音克隆会用别人的声音朗读你的文本。语音转换则会在保留你说话内容的前提下，把你的声音改写成另一个人的声音。两者都依赖同一个核心分解：把说话人身份与内容分开。

**类型：** 构建
**语言：** Python
**前置条件：** 第 6 阶段 · 06（说话人识别），第 6 阶段 · 07（TTS）
**时长：** ~75 分钟

## 问题

到了 2026 年，只要一段 5 秒音频，再加上一块消费级 GPU，就足以克隆出任何人的高质量声音。ElevenLabs、F5-TTS、OpenVoice v2、VoiceBox 都已经支持零样本 (zero-shot) 或少样本 (few-shot) 克隆。这项技术既是福音（无障碍 TTS、配音、辅助语音），也是武器（诈骗电话、政治深度伪造 (deepfake)、知识产权盗用）。

这里有两个高度相关的任务：

- **语音克隆（TTS 侧）：** 文本 + 5 秒参考音色 → 输出该音色的语音。
- **语音转换（语音侧）：** 源音频（A 说 X）+ B 的参考音色 → 输出 B 说 X 的语音。

它们都会把波形拆分为（内容、说话人、韵律 (prosody)），然后将一个来源的内容与另一个来源的说话人信息重新组合。

你在 2026 年必须面对的关键约束是：**在欧盟（AI Act，自 2026 年 8 月起可执行）和加州（AB 2905，自 2025 年起生效），水印 (watermarking) 与同意门禁 (consent gate) 都是法律要求。** 你的流水线必须输出不可听见的水印，并拒绝未经同意的克隆请求。

## 概念

*语音克隆与语音转换：分解、替换说话人、重新组合*

**零样本克隆 (Zero-shot cloning)。** 将一段 5 秒音频送入一个在数千名说话人上训练过的模型。说话人编码器 (speaker encoder) 会把这段音频映射为说话人嵌入 (speaker embedding)，而 TTS 解码器则基于该嵌入和文本共同生成语音。

代表模型有：F5-TTS（2024）、YourTTS（2022）、XTTS v2（2024）、OpenVoice v2（2024）。

**少样本微调 (Few-shot fine-tuning)。** 录制目标说话人 5–30 分钟的语音，用 LoRA 在基础模型上微调一小时，质量就可能从“还行”跃升到“几乎无法区分”。Coqui 和 ElevenLabs 都支持这种模式，社区也常把它用于 F5-TTS。

**语音转换（VC）。** 主要有两大路线：

- **识别-合成 (Recognition-synthesis)。** 先运行类似 ASR 的模型，提取内容表示（例如 soft phoneme posteriors、PPGs），再结合目标说话人嵌入重新合成。它对语言和口音更稳健。代表方法有 KNN-VC（2023）、Diff-HierVC（2023）。
- **解耦 (Disentanglement)。** 训练一个自编码器，在瓶颈层的潜空间中把内容、说话人和韵律拆开。推理时替换说话人嵌入。质量通常略低，但速度更快。代表方法有 AutoVC（2019）和各类 VITS-VC 变体。

**基于神经 codec 的克隆 (Neural codec-based cloning, 2024+)。** VALL-E、VALL-E 2、NaturalSpeech 3、VoiceBox 会把音频视为来自 SoundStream / EnCodec 的离散 tokens，并在这些 codec tokens 上训练大型自回归或 flow-matching 模型。对短提示词来说，质量已可与 ElevenLabs 相当。

### 伦理不是附加项

**水印。** PerTh（Perth）和 SilentCipher（2024）可以在音频中嵌入约 16–32 bit、几乎不可察觉的 ID。即使经过重新编码、流式传输和常见编辑后，仍然能被检测到。并且已有可用于生产的开源实现。

**同意门禁。** 每一段克隆输出都必须绑定一条可验证的授权记录，例如：“我，Rohit，于 2026-04-22 授权此声音用于 X 用途。” 这些记录应保存在防篡改日志中。

**检测。** AASIST、RawNet2 和 Wav2Vec2-AASIST 都可以作为检测器使用。ASVspoof 2025 challenge 公布的结果显示，面对 ElevenLabs、VALL-E 2 和 Bark 生成的语音，最先进检测器的 EER 在 0.8–2.3% 之间。

### 数据指标（2026）

| 模型 | 支持 Zero-shot？ | SECS（目标相似度） | WER（可懂度） | 参数量 |
|-------|-----------|--------------------|--------------|--------|
| F5-TTS | 是 | 0.72 | 2.1% | 335M |
| XTTS v2 | 是 | 0.65 | 3.5% | 470M |
| OpenVoice v2 | 是 | 0.70 | 2.8% | 220M |
| VALL-E 2 | 是 | 0.77 | 2.4% | 370M |
| VoiceBox | 是 | 0.78 | 2.1% | 330M |

通常来说，SECS > 0.70 时，大多数听众已经很难把它和目标音色区分开。

## 动手构建

### 第 1 步：用识别-合成路线做分解（`main.py` 中只有代码演示）

```python
def clone_pipeline(ref_audio, text, target_embedder, tts_model):
    speaker_emb = target_embedder.encode(ref_audio)
    mel = tts_model(text, speaker=speaker_emb)
    return vocoder(mel)
```

概念上很简单；真正复杂的实现主要集中在 `tts_model` 和说话人编码器上。

### 第 2 步：用 F5-TTS 做零样本克隆

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="rohit_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please add milk and bread to my list.",
)
```

参考转录文本必须与音频完全一致，连标点都要对上；否则对齐会出问题。

### 第 3 步：用 KNN-VC 做语音转换

```python
import torch
from knnvc import KNNVC  # 2023 model, https://github.com/bshall/knn-vc
vc = KNNVC.load("wavlm-base-plus")
out_wav = vc.convert(source="my_voice.wav", target_pool=["alice_1.wav", "alice_2.wav"])
```

KNN-VC 会运行 WavLM，为源语音和目标语音池提取逐帧嵌入，然后把每个源帧替换成目标池中最近邻的帧。它是非参数方法，用一分钟左右的目标语音就能工作。

### 第 4 步：嵌入水印

```python
from silentcipher import SilentCipher
sc = SilentCipher(model="2024-06-01")
payload = b"consent_id:abc123;ts:1745353200"
watermarked = sc.embed(wav, sr=24000, message=payload)
detected = sc.detect(watermarked, sr=24000)   # returns payload bytes
```

大约可嵌入 32 bit 的载荷 (payload)，即使经过 MP3 重新编码和轻微噪声干扰后仍能检测出来。

### 第 5 步：加入同意门禁

```python
def cloned_inference(text, ref_audio, consent_record):
    assert verify_signature(consent_record), "Signed consent required"
    assert consent_record["speaker_id"] == hash_speaker(ref_audio)
    wav = tts.infer(ref_file=ref_audio, gen_text=text)
    wav = watermark(wav, payload=consent_record["id"])
    return wav
```

## 如何使用

2026 年的常见选型如下：

| 场景 | 选择 |
|-----------|------|
| 5 秒 zero-shot 开源克隆 | F5-TTS 或 OpenVoice v2 |
| 商业级生产克隆 | ElevenLabs Instant Voice Clone v2.5 |
| 语音转换（重写音色） | KNN-VC 或 Diff-HierVC |
| 多说话人微调 | StyleTTS 2 + speaker adapter |
| 跨语言克隆 | XTTS v2 或 VALL-E X |
| Deepfake 检测 | Wav2Vec2-AASIST |

## 常见陷阱

- **参考转录未对齐。** F5-TTS 及类似模型要求参考文本与参考音频逐字匹配，连标点都不能差。
- **参考音频混响过重。** 回声会严重破坏克隆效果。应尽量在干声、近讲麦环境下录制。
- **情绪不匹配。** 如果训练参考音频是“欢快”的，模型往往会把所有输出都克隆成欢快语气。参考情绪应匹配目标用途。
- **语言泄漏。** 克隆一个英语说话人后再让模型讲法语，通常还是会带着原口音；这时应使用跨语言模型（XTTS、VALL-E X）。
- **没有水印。** 从 2026 年 8 月开始，这在欧盟已无法合法上线。

## 交付

将结果保存为 `outputs/skill-voice-cloner.md`。设计一个带有同意门禁 + 水印 + 质量目标的克隆或转换流水线。

## 练习

1. **简单。** 运行 `code/main.py`。它会通过计算交换前后两个“说话人”之间的余弦相似度，演示说话人嵌入替换的效果。
2. **中等。** 用 OpenVoice v2 克隆你自己的声音。测量参考音频与克隆音频之间的 SECS，再用 Whisper 测量 CER。
3. **困难。** 对 20 段克隆语音加入 SilentCipher 水印，然后进行 128 kbps MP3 编码+解码，再检测载荷。报告比特准确率。

## 关键术语

| 术语 | 人们常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Zero-shot clone | 5 秒就够了 | 预训练模型 + speaker embedding；无需额外训练。 |
| PPG | 音素后验图 | 逐帧 ASR 后验概率，可作为与语言无关的内容表示。 |
| KNN-VC | 最近邻转换 | 将每个源帧替换为目标语音池中最近的帧。 |
| Neural codec TTS | VALL-E 那一路 | 在 EnCodec/SoundStream tokens 上训练的 AR 模型。 |
| Watermark | 听不见的签名 | 嵌入到音频中的 bit，重新编码后仍能保留。 |
| SECS | 克隆保真度 | 目标说话人与克隆语音的 speaker embeddings 余弦相似度。 |
| AASIST | Deepfake 检测器 | 一种 anti-spoof 模型，用于检测合成语音。 |

## 延伸阅读

- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — 开源 SOTA zero-shot 克隆模型。
- [Baevski et al. / Microsoft (2023). VALL-E](https://arxiv.org/abs/2301.02111) and [VALL-E 2 (2024)](https://arxiv.org/abs/2406.05370) — 基于 neural codec 的 TTS。
- [Qian et al. (2019). AutoVC](https://arxiv.org/abs/1905.05879) — 基于 disentanglement 的语音转换。
- [Baas, Waubert de Puiseau, Kamper (2023). KNN-VC](https://arxiv.org/abs/2305.18975) — 基于检索的 VC。
- [SilentCipher (2024) — Audio Watermarking](https://github.com/sony/silentcipher) — 可用于生产的 32-bit 音频水印方案。
- [ASVspoof 2025 results](https://www.asvspoof.org/) — 检测器与合成器军备竞赛，更新至 2026 年。
