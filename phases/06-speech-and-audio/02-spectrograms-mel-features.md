# 频谱图 (Spectrogram)、Mel 尺度 (Mel Scale) 与音频特征 (Audio Features)

> 神经网络并不擅长直接消费原始波形。它们更擅长消费频谱图；而对 Mel 频谱图 (Mel spectrogram) 的处理通常更好。到了 2026 年，几乎所有 ASR、TTS 和音频分类器的成败，都取决于这一项预处理选择。

**类型：** 构建
**语言：** Python
**前置条件：** 第 6 阶段 · 01（音频基础）
**时长：** ~45 分钟

## 问题

拿一段 10 秒、16 kHz 的音频片段来说，它包含 160,000 个浮点数，全部落在 `[-1, 1]` 范围内，与“狗叫声”或“单词 cat”这样的标签几乎没有直接相关性。原始波形确实包含信息，但它的表达形式并不便于模型提取。两个完全相同的音素，如果相隔 100 ms 发出，对应的原始采样点也会完全不同。

频谱图可以解决这个问题。它会压缩人类感知并不关注的时间细节（例如微秒级抖动），同时保留感知真正关注的结构（例如哪些频率在大约 10–25 ms 的时间窗内具有较强能量）。

Mel 频谱图还会更进一步。人类对音高的感知是对数型的：100 Hz 到 200 Hz，听起来与 1000 Hz 到 2000 Hz 的“距离感”大致相同。Mel 尺度正是通过扭曲频率轴来贴合这种感知方式。对于 2010 到 2026 年的语音 ML 而言，按 Mel 尺度表示的频谱图一直都是最重要的特征。

## 概念

*从波形 (waveform) 到 STFT，再到 Mel 频谱图，最后到 MFCC 的处理阶梯*

**STFT (Short-Time Fourier Transform，短时傅里叶变换)。** 先把波形切成彼此重叠的帧（典型配置：25 ms window、10 ms hop，在 16 kHz 下分别对应 400 samples / 160 samples）。然后对每一帧乘以窗口函数 (window function)；Hann 是默认选择，Hamming 则是在权衡上略有不同。接着对每一帧做 FFT，再把所有幅度谱堆叠成一个形状为 `(n_frames, n_freq_bins)` 的矩阵。这就是你的频谱图。

**对数幅度 (Log-magnitude)。** 原始幅度往往跨越 5 到 6 个数量级，因此通常会取 `log(|X| + 1e-6)` 或 `20 * log10(|X|)` 来压缩动态范围。所有生产级流水线都会使用对数幅度，而不是原始幅度。

**Mel 尺度 (Mel scale)。** 以 Hz 为单位的频率 `f`，会通过 `m = 2595 * log10(1 + f / 700)` 映射到 mel `m`。在 1 kHz 以下，这种映射大致线性；在 1 kHz 以上，则大致呈对数关系。覆盖 0–8 kHz 的 80 个 mel bins，是标准的 ASR 输入配置。

**Mel 滤波器组 (Mel filterbank)。** 它由一组在 Mel 尺度上等距分布的三角形滤波器组成。每个滤波器本质上都是相邻 FFT bins 的加权和。将 STFT 幅度与滤波器组矩阵相乘，就能通过一次 matmul 得到 Mel 频谱图。

**对数 Mel 频谱图 (Log-mel spectrogram)。** 其形式是 `log(mel_spec + 1e-10)`。这是 Whisper 的输入，也是 Parakeet 和 SeamlessM4T 的输入，更是 2026 年通用的音频前端表示。

**MFCCs。** 对对数 Mel 频谱图应用 DCT（type II），保留前 13 个系数，就能进一步去相关并压缩特征。在大约 2015 年之前，它一直是主流特征；之后，基于原始 log-mels 的 CNNs/Transformers 才逐渐追平并超越。即便如此，它在说话人识别中仍然常见（如 x-vectors、ECAPA）。

**分辨率权衡 (Resolution trade)。** 更大的 FFT 能带来更好的频率分辨率，但会牺牲时间分辨率。25 ms / 10 ms 是音频 ML 的默认配置；音乐任务常用 50 ms / 12.5 ms；瞬态检测（如鼓点、爆破音）则常用 5 ms / 2 ms。

## 动手构建

### 第 1 步：对波形分帧

```python
def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]
```

对于一段 10 秒、16 kHz 的音频，如果设置 `frame_len=400, hop=160`，会得到 998 帧。

### 第 2 步：Hann 窗

```python
import math

def hann(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]
```

在做 FFT 之前，先按元素逐一相乘。这样可以减小由于在非零端点处截断信号而产生的谱泄漏 (spectral leakage)。

### 第 3 步：计算 STFT 幅度

```python
def stft_magnitude(signal, frame_len=400, hop=160):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [magnitudes(dft([w * s for w, s in zip(win, f)])) for f in frames]
```

生产环境里通常会使用 `torch.stft` 或 `librosa.stft`（基于 FFT 且已向量化）。这里的循环实现只是为了教学演示；它会在 `code/main.py` 中对短音频片段运行。

### 第 4 步：构建 Mel 滤波器组

```python
def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1)

def mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mels = [hz_to_mel(fmin) + (hz_to_mel(fmax) - hz_to_mel(fmin)) * i / (n_mels + 1)
            for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    bins = [int(h * n_fft / sr) for h in hzs]
    fb = [[0.0] * (n_fft // 2 + 1) for _ in range(n_mels)]
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m][k] = (k - bins[m]) / max(1, bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m][k] = (bins[m + 2] - k) / max(1, bins[m + 2] - bins[m + 1])
    return fb
```

当 `n_fft=400` 时，覆盖 0–8 kHz 的 80 个 mels 会生成一个 `(80, 201)` 矩阵。将形状为 `(n_frames, 201)` 的 STFT 幅度矩阵与其转置相乘，就能得到形状为 `(n_frames, 80)` 的 Mel 频谱图。

### 第 5 步：计算 log-mel

```python
def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]
```

常见替代方案包括：`librosa.power_to_db`（按参考值归一化的 dB），以及 `10 * log10(power + eps)`。Whisper 使用的是更复杂的 clip + normalize 流程（见 Whisper 的 `log_mel_spectrogram`）。

### 第 6 步：计算 MFCCs

```python
def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]
```

对每一帧 log-mel 应用 DCT，并保留前 13 个系数，这就构成了 MFCC 矩阵。第一个系数通常会被丢弃，因为它编码的是整体能量。

## 如何使用

2026 年的常见配置如下：

| 任务 | 特征 |
|------|----------|
| ASR (Whisper, Parakeet, SeamlessM4T) | 80 个 log-mels，10 ms hop，25 ms window |
| TTS 声学模型 (VITS, F5-TTS, Kokoro) | 80 个 mels，5–12 ms hop，用于更细的时间控制 |
| 音频分类 (AST, PANNs, BEATs) | 128 个 log-mels，10 ms hop |
| 说话人嵌入 (ECAPA-TDNN, WavLM) | 80 个 log-mels 或原始波形 SSL |
| 音乐 (MusicGen, Stable Audio 2) | EnCodec 离散 tokens（不是 mels） |
| 关键词检测 | 面向小设备的 40 MFCCs |

经验法则：**如果你做的不是音乐任务，就先从 80 个 log-mels 开始。** 任何偏离这个默认值的选择，都应该拿出证据来证明合理性。

## 到了 2026 年仍然常见的坑

- **Mel 数量不匹配。** 训练时用 80 mels，推理时却用 128 mels。通常不会直接报错，而是静默失败。要在两端都记录特征形状。
- **上游采样率不匹配。** 在 22.05 kHz 下计算出的 mels，与 16 kHz 下的结果并不一样。必须在特征提取 *之前* 先固定 SR。
- **dB 和 log 混用。** Whisper 需要的是 log-mel，不是 dB-mel。有些 HF pipelines 会自动识别，但你自己的自定义代码不会。
- **归一化漂移。** 训练时做逐话语归一化，推理时却做全局归一化。这类生产问题可能会让 WER 直接翻倍。
- **padding 引入泄漏。** 在音频末尾补零，会让尾部几帧出现平坦频谱。更稳妥的做法是对称填充，或者复制边界内容。

## 交付

将结果保存为 `outputs/skill-feature-extractor.md`。这个技能会根据目标模型，选择合适的特征类型、mel 数量、frame/hop 和归一化方式。

## 练习

1. **简单。** 运行 `code/main.py`。它会合成一个啁啾信号 (chirp)（频率从 200 Hz 扫到 4000 Hz），并打印每一帧中 argmax 对应的 mel bin。你可以选择绘图，并确认结果与扫频过程一致。
2. **中等。** 将 `n_mels` 分别设为 `{40, 80, 128}`，再把 `frame_len` 分别设为 `{200, 400, 800}` 重新运行。测量时间轴上尖峰带宽的变化。哪种组合对啁啾信号的分辨效果最好？
3. **困难。** 实现 `power_to_db`，然后在 AudioMNIST 上训练一个小型 CNN 分类器，比较以下输入的 ASR 准确率：(a) 原始 log-mel，(b) 设置 `ref=max` 的 dB-mel，(c) MFCC-13 + delta + delta-delta。报告 top-1 准确率。

## 关键术语

| 术语 | 人们常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Frame | 一小段 | 输入到一次 FFT 的 25 ms 波形片段。 |
| Hop | 步长 | 相邻帧之间的采样点间隔；10 ms 是 ASR 默认值。 |
| Window | Hann/Hamming 那个东西 | 逐点相乘的权重函数，用来把帧边缘逐渐压到零。 |
| STFT | 频谱图生成器 | 对分帧并加窗后的信号做 FFT，得到时间 × 频率矩阵。 |
| Mel | 扭曲后的频率 | 符合对数感知的尺度；`m = 2595·log10(1 + f/700)`。 |
| Filterbank | 那个矩阵 | 将 STFT 投影到 mel bins 上的三角滤波器组。 |
| Log-mel | Whisper 的输入 | `log(mel_spec + eps)`；到 2026 年已基本标准化。 |
| MFCC | 老派特征 | 对 log-mel 做 DCT；13 个系数，且彼此去相关。 |

## 延伸阅读

- [Davis, Mermelstein (1980). Comparison of parametric representations for monosyllabic word recognition](https://ieeexplore.ieee.org/document/1163420) — MFCC 论文。
- [Stevens, Volkmann, Newman (1937). A Scale for the Measurement of the Psychological Magnitude Pitch](https://pubs.aip.org/asa/jasa/article-abstract/8/3/185/735757/) — 最早的 mel scale 论文。
- [OpenAI — Whisper source, log_mel_spectrogram](https://github.com/openai/whisper/blob/main/whisper/audio.py) — 建议直接阅读参考实现。
- [librosa feature extraction docs](https://librosa.org/doc/main/feature.html) — `mfcc`、`melspectrogram` 以及 hop/window 的参考文档。
- [NVIDIA NeMo — audio preprocessing](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_all.html#featurizers) — 适用于 Parakeet 和 Canary 模型的生产级流水线参考。
