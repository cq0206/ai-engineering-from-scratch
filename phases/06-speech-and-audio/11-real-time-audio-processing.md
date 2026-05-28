# 实时音频处理 (Real-Time Audio Processing)

> 批处理流水线处理的是一个完整文件。实时流水线处理的是“下一个 20 毫秒”，而且必须赶在再下一个 20 毫秒到来之前完成。每一个对话式 AI、广播工作室和电话机器人，最终都受制于这个延迟预算。

**类型：** 构建
**语言：** Python
**前置条件：** 第 6 阶段 · 02（频谱图），第 6 阶段 · 04（ASR），第 6 阶段 · 07（TTS）
**时长：** ~75 分钟

## 问题

你想要一个“有生命感”的语音助手。人类对话中的轮次切换延迟大约是 230 ms（从沉默到回应）。超过 500 ms 就会显得机械；超过 1500 ms 就会让人觉得系统坏了。到了 2026 年，一个完整 **听见 → 理解 → 回应 → 开口说话** 回路的预算大致如下：

| 阶段 | 预算 |
|-------|--------|
| 麦克风 → 缓冲区 | 20 ms |
| VAD | 10 ms |
| ASR（streaming） | 150 ms |
| LLM（first token） | 100 ms |
| TTS（first chunk） | 100 ms |
| 渲染 → 扬声器 | 20 ms |
| **总计** | **~400 ms** |

Moshi（Kyutai，2024）做到了 200 ms 全双工 (full-duplex)。GPT-4o-realtime（2024）大约是 320 ms。2022 年的级联流水线通常还在 2500 ms 左右。实现 10× 提升主要依赖三件事：（1）全链路流式处理 (streaming)，（2）基于部分结果的异步流水线处理，（3）可打断的生成。

## 概念

*带有环形缓冲区、VAD 门控和打断机制的流式音频流水线*

**帧 (Frame) / 数据块 (chunk) / 窗口 (window)。** 实时音频是以固定大小的数据块流动的。常见选择是 20 ms（在 16 kHz 下为 320 samples）。下游所有模块都必须跟上这个节奏。

**环形缓冲区 (Ring buffer)。** 这是一个固定大小的循环缓冲区。生产线程写入新帧，消费线程读取数据。它能避免在热点路径 (hot path) 上频繁分配内存。其大小大致等于最大延迟 × sample-rate；例如一个 2 秒、16 kHz 的 ring buffer 大约需要 32,000 samples。

**VAD (Voice Activity Detection，语音活动检测)。** 当没人说话时，它会阻断下游计算。Silero VAD 4.0（2024）在 CPU 上处理每个 30 ms frame 的耗时小于 1 ms。`webrtcvad` 是更早的替代方案。

**流式 ASR (Streaming ASR)。** 这类模型会在音频持续到达时输出部分转录结果。Parakeet-CTC-0.6B 的 streaming 模式（NeMo，2024）能在 320 ms 延迟下做到 2–5% 的 WER。Whisper-Streaming（Macháček 等，2023）则通过对 Whisper 分块，实现接近流式的效果，延迟约为 2 秒。

**打断 (Interruption)。** 当助手正在说话时，用户又开口了，你必须在 100 ms 内完成三件事：(a) 检测到插话打断 (barge-in)，(b) 停止 TTS，(c) 丢弃剩余的 LLM 输出。否则用户会觉得这个助手“听不见人说话”。

**WebRTC Opus 传输。** 常见配置是 20 ms frames、48 kHz、8–128 kbps 自适应码率。这是浏览器和移动端的标准方案。到 2026 年，LiveKit、Daily.co、Pion 是构建语音应用的主流技术栈。

**抖动缓冲区 (Jitter buffer)。** 网络包可能乱序到达，也可能延迟到达。Jitter buffer 负责重排与平滑；太小会出现可听见的断裂，太大则会增加延迟。典型值是 60–80 ms。

### 常见坑点

- **线程争用。** Python 的 GIL 加上重型模型，可能会饿死音频线程。应使用带 C 回调的音频库（如 sounddevice、PortAudio），让 Python 远离热点路径。
- **采样率转换延迟。** 在流水线内部做重采样通常会额外增加 5–20 ms。更好的做法是提前重采样，或者使用零延迟重采样器（如 PolyPhase、`soxr_hq`）。
- **TTS 预热。** 即使是 Kokoro 这种较快的 TTS，第一次请求也常常需要 100–200 ms 预热时间。应该缓存模型，并在第一次真实对话前用一次空跑提前热身。
- **回声消除。** 如果没有 AEC，TTS 输出会重新进入麦克风，并让 ASR 误识别机器自己的声音。WebRTC AEC3 是默认的开源方案。

## 动手构建

### 第 1 步：实现环形缓冲区

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)
```

容量 (capacity) 决定了最大缓冲延迟。对于 16 kHz 音频，32,000 samples 就等于 2 秒。

### 第 2 步：加入 VAD 门控

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

生产环境中建议替换为 Silero VAD：

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### 第 3 步：接入流式 ASR

```python
# Parakeet-CTC-0.6B streaming via NeMo
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### 第 4 步：实现打断处理器

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # barge-in
        # then feed to streaming ASR

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

这里的关键在于异步 I/O 和可取消的 TTS 流式输出。标准做法是在音频轨道上调用 WebRTC `peerconnection.stop()`。

## 如何使用

2026 年的常见技术栈如下：

| 层级 | 选择 |
|-------|------|
| 传输层 | LiveKit（WebRTC）或 Pion（Go） |
| VAD | Silero VAD 4.0 |
| Streaming ASR | Parakeet-CTC-0.6B 或 Whisper-Streaming |
| LLM 首 token 延迟 | Groq、Cerebras、vLLM-streaming |
| Streaming TTS | Kokoro 或 ElevenLabs Turbo v2.5 |
| 回声消除 | WebRTC AEC3 |
| 原生端到端 | OpenAI Realtime API 或 Moshi |

## 常见陷阱

- **为了稳妥先缓冲 500 ms。** 缓冲区本身就是你的延迟下限。能缩就缩。
- **没有固定线程。** 如果音频回调线程的优先级比 UI 线程还低，负载一高就会出现卡顿和爆音。
- **TTS 数据块太小。** 小于 200 ms 的数据块会让 vocoder 伪影变得明显。320 ms 通常是甜点区。
- **没有抖动缓冲区。** 真实网络一定会抖；没有平滑机制就会出现爆音和断裂。
- **一次性错误处理。** 音频流水线必须具备抗崩溃能力。一次异常就足以终止整个会话。

## 交付

将结果保存为 `outputs/skill-realtime-designer.md`。设计一个为每个阶段都给出明确延迟预算的实时音频流水线。

## 练习

1. **简单。** 运行 `code/main.py`。它会模拟一个环形缓冲区 + 能量 VAD，并打印一段虚拟 10 秒音频流各阶段的延迟。
2. **中等。** 使用 `sounddevice`，构建一个直通循环，以 20 ms 帧为单位处理你的麦克风输入，并在每一帧打印 VAD 状态。
3. **困难。** 使用 `aiortc` 构建一个全双工回声测试：浏览器 → WebRTC → Python → WebRTC → 浏览器。用一个 1 kHz 脉冲测量端到端延迟 (glass-to-glass latency)。

## 关键术语

| 术语 | 人们常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Ring buffer | 循环队列 | 面向音频帧的固定大小、无锁（或 SPSC 加锁）FIFO。 |
| VAD | 静音门 | 用模型或启发式方法判断当前是语音还是非语音。 |
| Streaming ASR | 实时 STT | 音频到达时持续输出部分文本，并受限于 lookahead。 |
| Jitter buffer | 网络平滑器 | 对乱序到达的数据包重排；典型值为 60–80 ms。 |
| AEC | 回声消除 | 消除从扬声器回流到麦克风的反馈路径。 |
| Barge-in | 用户打断 | 系统在 TTS 过程中检测到用户发言，并必须取消播放。 |
| Full duplex | 双向同时进行 | 用户和机器人可以同时说话；Moshi 就是 full duplex。 |

## 延伸阅读

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) — 分块实现、接近流式的 Whisper。
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) — 200 ms 延迟的全双工系统。
- [LiveKit Agents framework (2024)](https://docs.livekit.io/agents/) — 生产级音频 agent 编排框架。
- [Silero VAD repo](https://github.com/snakers4/silero-vad) — 小于 1 ms 的 VAD，Apache 2.0。
- [WebRTC AEC3 paper](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) — 开源回声消除方案。
