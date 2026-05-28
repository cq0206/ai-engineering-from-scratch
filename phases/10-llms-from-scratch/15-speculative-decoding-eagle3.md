# 推测解码（Speculative Decoding）与 EAGLE-3

> 第 7 阶段·第 16 课已经证明了数学基础：Leviathan 拒绝规则会精确保留验证器（verifier）的分布。本课从训练栈视角讲解 2026 年生产环境中的推测解码。EAGLE-3 把草稿模型（draft model）从廉价近似改造成了一个专门设计的小型网络：它直接用验证器自身的隐藏状态训练，并加入训练时测试（Training-Time Test, TTT）循环，让训练分布与推理分布对齐。结果是端到端 3× 到 6.5× 的加速、聊天场景中超过 0.9 的逐 token 接受率，而且没有任何分布层面的权衡。到 2026 年，所有生产级推理栈都会默认启用它。

**类型：** Build
**语言：** Python（stdlib）
**前置课程：** 第 7 阶段·16（推测解码数学），第 10 阶段·12（推理优化）
**耗时：** ~75 分钟

## 学习目标

- 用一句话陈述 Leviathan 定理，并证明推测循环产生的样本与直接从验证器采样得到的样本分布完全一致。
- 梳理从朴素推测解码（Leviathan 2023）到 EAGLE、EAGLE-2、EAGLE-3 的两年演进，并指出每一步具体消除了什么限制。
- 根据接受率 `α` 和草稿到验证器的成本比 `c` 计算期望加速比，并为不同情形选择最优的草稿长度 `N`。
- 从零实现完整的推测循环：草稿生成、验证、从残差分布（residual distribution）中进行拒绝采样、在拒绝时回滚 KV 缓存（KV cache）、在完全接受时输出额外 token。

## 问题

在 70B 模型上做自回归解码时，H100 上的速度也许只有每秒 35 个 token。GPU 远没有被打满。瓶颈是内存带宽：每个 token 都要从 HBM 里加载 700 亿参数，做一步算术，然后产出一个浮点数。计算单元大多时间都在空转。

推测解码把这件事变成了一个真正可以解决的吞吐问题。一个廉价的草稿模型会通过 `N` 次小型前向传递提出 `N` 个 token。验证器只需在“前缀 + 全部 `N` 个草稿 token”上运行一次。如果验证器在位置 `i` 的分布与草稿一致（这里的一致是统计意义上的，稍后会精确定义），我们就接受；如果不一致，就从残差分布中采样一个修正 token。这样，一次大模型前向最多可以接受 `N+1` 个 token，而不是只接受一个。

最关键的定理来自 Leviathan、Kalman、Matias（ICML 2023）：输出分布与直接从验证器采样得到的分布完全相同。不是近似，而是严格一致。这就是为什么推测解码能被生产环境接受——它只是纯粹的延迟优化，不会牺牲质量。

第 7 阶段·第 16 课给你的是数学；本课给你的是训练栈。一个优秀的草稿模型带来的加速价值，会比一个廉价草稿模型高出 2×。EAGLE、EAGLE-2 和 EAGLE-3（Li 等，2024–2025）把“草稿 = 同系列更小模型”这件事，变成了一门精确的工程学。到 2026 年，生产级推理服务器默认使用 EAGLE-3。

## 概念

### 不变量：Leviathan 拒绝采样

设 `p(t)` 是给定某个前缀时草稿模型对下一个 token 的分布，`q(t)` 是验证器的分布。从 `p` 中采样一个草稿 token `d ~ p`。以 `min(1, q(d) / p(d))` 的概率接受它。如果拒绝，就从残差分布 `(q - p)_+ / ||(q - p)_+||_1` 中采样。最终得到的样本分布就是 `q`。无论 `p` 有多差，这一点都成立——`p` 越差，只会意味着你拒绝得越频繁，但输出始终是精确的。

把这个过程连续堆叠 `N` 次，并只用一次验证器前向传递去处理 `prefix + d_1 + ... + d_N`。验证器会同时返回 `q_1, q_2, ..., q_{N+1}`。按从左到右的顺序遍历。在第一个拒绝位置 `j`，从 `residual(q_j, p_j)` 中采样并停止。如果全部接受，就从 `q_{N+1}` 中再采样一个额外 token。

### 决定加速比的因素

设 `α` 为每个草稿 token 的期望接受率（acceptance rate），设 `c = cost(draft) / cost(verifier)` 为成本比。则每次验证器前向传递所接受的 token 期望数为：

```
E[accepted] = (1 - α^(N+1)) / (1 - α)
```

每个被接受 token 的期望总墙钟时间为 `(N * c + 1) / E[accepted]`。对 `N` 做最优化，就能找到甜点区间。若 `α = 0.8, c = 0.05`：最优 `N` 大约在 5–7，加速比约为 3.2×。若 `α = 0.95, c = 0.02`：最优 `N` 大约在 8–10，加速比会逼近 5×。

最大的杠杆就是 `α`。当 `N = 5` 固定时，把 `α` 从 `0.6`（普通草稿）提升到 `0.9`（EAGLE-3），会让每次验证器前向传递的期望接受 token 数，从 2.2 提升到 4.1。对同一个验证器来说，吞吐几乎翻倍。

### 两年演进路线

**朴素推测（Vanilla speculative，Leviathan，2023）。** 草稿模型是同一家族里一个独立训练的小型 LLM。接线容易，`α ≈ 0.6`，加速最多大约 2×。

**EAGLE-1（Li 等，2024）。** 草稿是一个很小的 Transformer——通常只有一层或两层——以验证器最后一层的隐藏状态为输入，直接预测下一个 token。由于草稿能看到验证器的特征表示，它的分布会更接近验证器。`α` 上升到 0.7–0.8。

**EAGLE-2（Li 等，2024）。** 加入动态草稿树（dynamic draft tree）：不再只提出一条长度为 `N` 的序列，而是提出一棵小型候选树，用一次验证器前向传递（树注意力，tree attention）为每个候选打分，然后沿最高概率路径前进。草稿长度因此能按步自适应。按被接受路径上的 token 计算，`α` 会升到 0.85 以上。

**EAGLE-3（Li 等，2025，NeurIPS）。** 又做了两项改动。第一，完全移除特征预测损失——EAGLE-1/2 训练草稿去匹配验证器的隐藏状态，而这会限制数据规模带来的收益。EAGLE-3 改为直接训练 token 预测。第二，引入训练时测试（TTT）：在训练草稿时，让草稿在多步过程中把自己前一步的预测回灌成输入，就像它在推理时真正运行的方式一样。这让训练分布与测试分布对齐，并阻止误差累积。测得的加速：聊天场景最高可达 6.5×；在 H100 上的 SGLang、batch 64 场景里，吞吐提升 38%。

### KV 缓存回滚

验证步骤会在一次前向中把验证器的 KV 缓存扩展 `N` 个条目。如果在位置 `j` 发生拒绝，那么位置 `j-1` 之后的缓存内容就全都错了。常见实现有两种：写入暂存缓冲区并在接受后提交（vLLM、TensorRT-LLM），或者维护一个物理 KV 缓存外加一个逻辑长度，在拒绝时直接截断。无论哪种方式，回滚的成本都只是“每层每头若干字节”的量级，相比前向传递的代价几乎可以忽略。

对于 EAGLE-2 的树搜索，验证器会使用一种尊重树拓扑的非因果掩码来运行注意力。工程实现上会有点麻烦，但其计算本质上仍然是一次带自定义掩码的标准 flash-attention 调用。

### 2026 年的草稿架构

| 策略 | 草稿类型 | `α` | 加速比 | 训练成本 |
|----------|-----------|-----|---------|---------------|
| 朴素版（Vanilla） | 独立的小型 LLM | 0.55-0.70 | 1.8-2.3× | 无（复用已有小模型） |
| Medusa | 验证器上的额外 LM 头 | 0.65-0.75 | 2-3× | ~1B SFT tokens |
| EAGLE-1 | 基于隐藏状态的 1 层 Transformer | 0.70-0.80 | 2.5-3× | ~60B tokens |
| EAGLE-2 | EAGLE-1 + 动态草稿树 | 0.80-0.88 | 3-4× | ~60B tokens |
| EAGLE-3 | 多层特征融合 + TTT | 0.88-0.92 | 3.5-6.5× | ~60-200B tokens |
| 前瞻（Lookahead） | 无草稿（Jacobi iteration） | N/A | 1.3-1.6× | 无 |

在 2026 年的生产环境里：vLLM 和 SGLang 在可用时默认使用 EAGLE-3，否则使用 EAGLE-2。TensorRT-LLM 为 Meta 和 NVIDIA 的公开模型提供了最快的 Medusa 路径。llama.cpp 则为 CPU 部署提供了朴素草稿方案。

## 动手实现

见 `code/main.py`。这里实现的是完整的 Leviathan 推测循环，包含所有关键部件：长度为 N 的草稿、验证器并行前向、逐位置拒绝、残差采样、额外 token、KV 回滚，以及对输出分布与直接从 `q` 采样一致性的经验验证。

### 第 1 步：拒绝规则

```python
def accept(q_prob, p_prob, u):
    if p_prob <= 0:
        return True
    return u < min(1.0, q_prob / p_prob)
```

### 第 2 步：残差分布

```python
def residual(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    if s == 0:
        return list(q)
    return [r / s for r in raw]
```

### 第 3 步：完整的一次推测步骤

`spec_step` 函数会先从 `p` 中草拟 `N` 个 token，然后用一次并行的 `q` 求值来验证它们。对每个草稿 token，它都会应用拒绝规则；一旦第一次拒绝发生，就从残差中采样修正 token。如果一切都被接受，它就会从 `q_{N+1}` 中输出一个额外 token。

### 第 4 步：KV 回滚记账

这个模拟器为每个 worker 跟踪一个逻辑上的 `kv_length`。若接受了 `k` 个草稿，则 `kv_length += k`。若在位置 `j` 发生拒绝，缓存实际上已经写到了 `j` 之后，但逻辑长度会被设置为 `prefix_length + j + 1`——也就是修正 token 之后一个位置。后续读取会根据这个逻辑长度进行截断。

### 第 5 步：Leviathan 检查

运行 50,000 次推测步骤。统计被接受 token 的经验分布。再与从 `q` 直接采样 50,000 次的结果进行比较。卡方统计量应该明显低于临界值。这个定理在实践中同样成立。

### 第 6 步：加速比与 `α`

通过不同幅度地把 `p` 从 `q` 扰动开，来扫描草稿质量。测量 `α`，再把每次验证器调用的期望 token 数画成 `α` 与 `N` 的函数。代码会打印一张表，显示 EAGLE-3 级别的草稿质量（`α ≈ 0.9`）如何解锁每次验证器调用 4–5 个 token。

## 使用

生产级 `vllm serve` + EAGLE-3：

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --speculative-config '{
    "model": "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B",
    "num_speculative_tokens": 5,
    "method": "eagle3"
  }'
```

在 H100 上以 batch 64 运行 SGLang + EAGLE-3：根据 EAGLE-3 论文，其吞吐大约比 batch-64 的朴素解码高 1.38×。

适合使用推测解码的场景：

- 任何交互式聊天工作负载，只要 p50 延迟比峰值吞吐更重要。
- 代码生成与结构化输出（JSON、SQL）。由于目标分布高度可预测，`α` 往往高于 0.9。
- 长文本生成（数千个 token）。摊销后的加速会持续生效。

不适合的场景：

- 很小的模型（&lt; 3B）。草稿模型并不会比验证器便宜太多。
- 极小的 batch-1 CPU 部署。草稿模型带来的额外内存开销可能不值得。
- 高温度、极具创造性的采样场景，因为这会导致 `α` 崩塌。

## 交付

本课会产出 `outputs/skill-eagle3-tuner.md`。给定一个推理工作负载（模型、batch size、目标延迟、任务画像），它会推荐一套推测解码策略与调优参数（草稿家族、`N`、树深度、基于温度的切换策略）。

## 练习

1. 运行 `code/main.py`。确认 Leviathan 分布检查中的卡方统计量，在 50,000 个样本下始终低于 95% 置信水平对应的临界值。

2. 固定 `α = 0.9`、`c = 0.04`，把 `N` 从 1 扫到 10。绘制每次验证器调用的期望 token 数，以及每个 token 的实际墙钟时间。找出使墙钟时间最小的 `N`。解释这条曲线的形状。

3. 修改代码，模拟 EAGLE-2 树搜索：每一步里，草稿提出一棵形状为 `[2, 2, 2]` 的树（共八条候选路径）。验证器只运行一次，由概率最高的被接受路径胜出。计算每个叶节点的 `α` 与每次验证器调用的总 token 数。与等价计算量下的线性链式推测解码进行比较。

4. 为两个并发序列实现一个批量 KV 回滚模拟器。序列 A 的所有草稿都被接受；序列 B 在位置 2 发生拒绝。展示每个序列的正确 `kv_length` 都会被单独更新，并且没有任何工作被浪费。

5. 阅读 EAGLE-3 论文第 4 节（Training-Time Test）。用两句话解释：为什么没有 TTT 的朴素草稿训练会遭受曝光偏差（exposure bias），以及为什么在训练中把草稿自己的预测喂回去能够修复这个问题。再把它与 seq2seq 文献中的 scheduled sampling 联系起来。

## 关键术语

| 术语 | 人们常说什么 | 它真正的含义 |
|------|----------------|------------------------|
| Leviathan 规则 | “min(1, q over p)” | 以 `min(1, q(d)/p(d))` 的概率进行伯努利接受/拒绝；若拒绝则从残差中采样，因此能精确保留验证器分布 |
| 残差分布 | “(q minus p) plus, normalized” | 将 `(q - p)_+` 在零处截断后重新归一化——这才是拒绝时正确的采样分布 |
| 接受率 α | “草稿有多常猜对” | 在拒绝规则下逐 token 的期望伯努利成功概率；它决定了所有加速比数学 |
| EAGLE-1 | “hidden-state draft” | 以验证器最后一层隐藏状态为条件的微型 Transformer 草稿（Li 等，2024） |
| EAGLE-2 | “dynamic draft tree” | EAGLE-1 再加上一棵候选续写树，用树注意力在一次验证器前向中评分 |
| EAGLE-3 | “training-time test” | 去掉特征预测损失，改为直接 token 预测，并在训练中让草稿读取自己的输出 |
| 训练时测试（TTT） | “修复 exposure bias” | 在训练中让草稿自回归运行，使训练与测试时的输入分布一致——它正是 scheduled sampling 的直接对应物 |
| KV 回滚 | “撤销被拒绝的草稿” | 在发生拒绝后，把验证器的 KV 缓存重置到已接受前缀长度的记账机制 |
| 额外 token | “那个免费送的” | 当全部 `N` 个草稿都被接受时，从 `q_{N+1}` 中额外采样一个 token，且不增加验证器成本 |
| 树注意力 | “一次验证多个候选” | 使用符合草稿树拓扑的非因果掩码来做注意力；在一次前向里为树中每个节点计算 `q_i` |

## 延伸阅读

- [Leviathan, Kalman, Matias — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192, ICML 2023)](https://arxiv.org/abs/2211.17192) —— 奠基性论文与等价性定理
- [Chen et al. — Accelerating Large Language Model Decoding with Speculative Sampling (arXiv:2302.01318)](https://arxiv.org/abs/2302.01318) —— 同期独立提出的方法，证明也很清晰
- [Li et al. — EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) —— EAGLE-1，以隐藏状态为条件的草稿模型
- [Li et al. — EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) —— 动态树搜索
- [Li et al. — EAGLE-3: Scaling up Inference Acceleration via Training-Time Test (arXiv:2503.01840, NeurIPS 2025)](https://arxiv.org/abs/2503.01840) —— 2026 年的生产默认方案
- [Cai et al. — Medusa: Multiple Decoding Heads (arXiv:2401.10774)](https://arxiv.org/abs/2401.10774) —— 另一种无需草稿模型的方案
- [vLLM Speculative Decoding documentation](https://docs.vllm.ai/en/latest/features/spec_decode.html) —— 接好所有策略的规范化生产参考文档
