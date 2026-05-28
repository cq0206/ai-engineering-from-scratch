# 音频生成

> 音频（audio）是一个 16–48 kHz 的一维信号。一个 5 秒片段就有 8 万到 24 万个采样点。没有哪个 transformer 会直接对这样的序列做 attention。2026 年所有生产级音频模型的解决方案都一样：先用神经编解码器（neural codec，如 Encodec、SoundStream、DAC）把音频压缩成 50–75 Hz 的离散 token，再由 transformer 或 diffusion 模型来生成这些 token。

**类型：** Build
**语言：** Python
**先修要求：** Phase 6 · 02（Audio Features）, Phase 6 · 04（ASR）, Phase 8 · 06（DDPM）
**时长：** ~45 分钟

## 问题

三类音频生成任务：

1. **文本转语音（text-to-speech）。** 给定文本，生成语音。干净语音属于窄带分布，并且具有很强的语音学结构——用 token 上的 transformer 处理，已经解决得很好了。VALL-E（Microsoft）、NaturalSpeech 3、ElevenLabs、OpenAI TTS 都属于这一类。
2. **音乐生成（music generation）。** 给定提示（文本、旋律、和弦进行、流派），生成音乐。它的分布要宽得多。MusicGen（Meta）、Stable Audio 2.5、Suno v4、Udio、Riffusion 都在这里。
3. **音频特效 / 声音设计（audio effects / sound design）。** 给定提示，生成环境声或 Foley。AudioGen、AudioLDM 2、Stable Audio Open 都是典型代表。

这三类任务都建立在同一个底层之上：神经音频编解码器 + token 自回归（token-AR）或 diffusion 生成器。

## 概念

*音频生成：codec token + transformer 或 diffusion*

### 神经音频编解码器

Encodec（Meta，2022）、SoundStream（Google，2021）、Descript Audio Codec（DAC，2023）。一个卷积编码器（convolutional encoder）会把波形压缩成按时间步排列的向量；残差向量量化（residual vector quantization, RVQ）再把每个向量转换成由 K 个码本索引组成的级联表示。解码器再把它逆转回来。以 24 kHz 音频为例：如果用 8 个 RVQ 码本，在 75 Hz 下以 2 kbps 编码，那么每秒会产生 600 个 token。

```
waveform (16000 samples/sec)
    └─ encoder conv ─┐
                     ├─ RVQ layer 1 → indices at 75 Hz
                     ├─ RVQ layer 2 → indices at 75 Hz
                     ├─ ...
                     └─ RVQ layer 8
```

### 上层的两种生成范式

**Token 自回归（token-autoregressive）。** 把 RVQ token 展平成一个序列，然后运行 decoder-only transformer。MusicGen 使用“delayed parallel”机制，以带偏移的多流方式并行发射 K 个码本流。VALL-E 则根据文本提示 + 3 秒语音样本来生成语音 token。

**潜变量 diffusion（latent diffusion）。** 把 codec token 打包成连续潜变量，或者用 categorical diffusion 来建模它们。Stable Audio 2.5 在连续音频潜变量上使用 flow matching。AudioLDM 2 则采用 text-to-mel-to-audio diffusion。

2024–2026 年的趋势是：在音乐生成上，flow matching 正在占优（推理更快，样本更干净）；而在语音上，token-AR 依然主导，因为它天然因果、很适合流式输出。

## 生产格局

| 系统 | 任务 | 主干 | 延迟 |
|--------|------|----------|---------|
| ElevenLabs V3 | TTS | Token-AR + neural vocoder | ~300ms first token |
| OpenAI GPT-4o audio | 全双工语音 | 端到端多模态 AR | ~200ms |
| NaturalSpeech 3 | TTS | Latent flow matching | 非流式 |
| Stable Audio 2.5 | 音乐 / SFX | DiT + flow matching on audio latents | ~10s for 1-minute clip |
| Suno v4 | 完整歌曲 | 未公开；怀疑是 token-AR | ~30s per song |
| Udio v1.5 | 完整歌曲 | 未公开 | ~30s per song |
| MusicGen 3.3B | 音乐 | Encodec 32kHz 上的 token-AR | 实时 |
| AudioCraft 2 | 音乐 + SFX | Flow matching | ~5s for 5s clip |
| Riffusion v2 | 音乐 | 频谱图 diffusion | ~10s |

## 动手构建

`code/main.py` 模拟了核心思路：在合成的“音频 token”序列上训练一个很小的 next-token transformer。这些序列来自两种截然不同的“风格”：风格 A 是低 token 与高 token 交替，风格 B 是单调爬升斜坡。对风格做条件控制并进行采样。

### 第 1 步：合成音频 token

```python
def make_tokens(style, length, vocab_size, rng):
    if style == 0:  # "speech-like": alternating
        return [i % vocab_size for i in range(length)]
    # "music-like": ramp
    return [(i * 3) % vocab_size for i in range(length)]
```

### 第 2 步：训练一个小型 token 预测器

一个按风格做条件控制的 bigram 风格预测器。重点在于这个模式：codec token → cross-entropy 训练 → 自回归采样。

### 第 3 步：按条件采样

给定风格 token 和一个起始 token，从预测分布中采样下一个 token。持续生成 20–40 个 token。

## 常见陷阱

- **编解码器质量会限制最终输出质量。** 如果 codec 本身无法忠实表示某种声音，那么生成器再强也无济于事。当前开源里 DAC 是最好的。
- **RVQ 误差累积。** 每一层 RVQ 都在建模前一层的残差。第一层的误差会向后传播。对高层使用 temperature 0 采样会有帮助。
- **音乐结构。** 30 秒的 token 序列在 75 Hz 下会超过 2 万个 token。对 transformer 来说很难。MusicGen 使用滑动窗口 + prompt continuation；Stable Audio 使用更短片段 + crossfading。
- **边界伪影。** 在生成片段之间做 crossfading，需要非常仔细地处理 overlap-add。
- **对干净数据的饥渴。** 音乐生成器需要数万小时的授权音乐。Suno / Udio 与 RIAA 的诉讼（2024）把这个问题彻底暴露出来了。
- **声音克隆伦理。** 只要 3 秒样本加一个文本提示，VALL-E / XTTS / ElevenLabs 就足以克隆一段声音。每个生产模型都必须配备滥用检测和 opt-out 列表。

## 如何使用

| 任务 | 2026 年工具栈 |
|------|------------|
| 商业 TTS | ElevenLabs、OpenAI TTS 或 Azure Neural |
| 声音克隆（已验证同意） | XTTS v2（开源）或 ElevenLabs Pro |
| 快速生成背景音乐 | Stable Audio 2.5 API、Suno 或 Udio |
| 生成带歌词的音乐 | Suno v4 或 Udio v1.5 |
| 音效 / Foley | AudioCraft 2、ElevenLabs SFX 或 Stable Audio Open |
| 实时语音 Agent | GPT-4o realtime 或 Gemini Live |
| 开放权重音乐研究 | MusicGen 3.3B、Stable Audio Open 1.0、AudioLDM 2 |
| 配音 / 翻译 | HeyGen、ElevenLabs Dubbing |

## 交付

保存 `outputs/skill-audio-brief.md`。这个 Skill 接收一份音频 brief（任务、时长、风格、音色、许可证），并输出：模型 + 托管方案、提示词格式（流派标签、风格描述、结构标记）、codec + generator + vocoder 链路、seed 协议，以及评估方案（MOS / CLAP score / TTS 的 CER / 用户 A/B）。

## 练习

1. **简单。** 运行 `code/main.py` 并显式设置 style。验证生成序列是否符合该风格对应的模式。
2. **中等。** 添加 delayed parallel decoding：模拟两路 token 流，并要求它们始终保持 1 步偏移。训练一个联合预测器。
3. **困难。** 使用 HuggingFace transformers 在本地运行 MusicGen-small。用三个不同提示生成一个 10 秒片段；通过 A/B 测试比较风格遵循度。

## 关键术语

| 术语 | 人们常说什么 | 它真正的含义 |
|------|-----------------|-----------------------|
| Codec | “神经压缩（neural compression）” | 音频的编码器 / 解码器；典型输出是 50–75 Hz 的 token。 |
| RVQ | “Residual VQ” | 由 K 个量化器组成的级联；每一层都建模前一层残差。 |
| Token | “一个 codec 符号” | 码本中的离散索引；常见大小是 1024 或 2048。 |
| Delayed parallel | “带偏移的码本” | 以错位偏移发射 K 路 token 流，以缩短序列长度。 |
| Flow matching | “2024 年音频领域的赢家” | 相比 diffusion 路径更直的替代方案；采样更快。 |
| Voice prompt | “3 秒样本” | 用于引导克隆音色的 speaker embedding 或 token 前缀。 |
| Mel spectrogram | “那个可视化图” | 对数幅度的感知频谱图；很多 TTS 系统都会用。 |
| Vocoder | “Mel 转 wave” | 把 mel spectrogram 转回音频的神经组件。 |

## 生产说明：音频本质上是流式问题

音频是唯一一种用户期望*边生成边到达*的输出模态，而不是一次性全部返回。从生产角度看，这意味着 TPOT（Time Per Output Token，每个输出 token 的时间）非常重要，因为用户的收听速度才是目标吞吐，而不是阅读速度。对于以 ~75 token/秒（Encodec）进行 token 化的 16kHz 音频，服务端必须为每个用户生成 ≥75 token/秒，播放才能保持流畅。

这会带来两个架构层面的后果：

- **Flow-matching 音频模型无法轻易流式化。** Stable Audio 2.5 和 AudioCraft 2 会一次性渲染固定长度的片段。若要流式输出，你需要把片段切块并在边界处重叠——可以理解为滑动窗口 diffusion——这会比 codec AR 模型额外增加 100–300ms 的延迟开销。

如果产品是“实时语音聊天”或“实时音乐续写”，请选择 codec AR 路线。如果产品是“提交后渲染一个 30 秒片段”，那么 flow matching 会在质量和总延迟上更占优。

## 延伸阅读

- [Défossez et al. (2022). Encodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — codec 标准。
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — 第一个被广泛采用的神经音频 codec。
- [Kumar et al. (2023). High-Fidelity Audio Compression with Improved RVQGAN (DAC)](https://arxiv.org/abs/2306.06546) — DAC.
- [Wang et al. (2023). Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)](https://arxiv.org/abs/2301.02111) — VALL-E.
- [Copet et al. (2023). Simple and Controllable Music Generation (MusicGen)](https://arxiv.org/abs/2306.05284) — MusicGen.
- [Liu et al. (2023). AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734) — AudioLDM 2.
- [Stability AI (2024). Stable Audio 2.5](https://stability.ai/news/introducing-stable-audio-2-5) — 2025 年基于 flow matching 的文生音乐。
