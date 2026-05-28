# DualPipe 并行

> DeepSeek-V3 在 2,048 张 H800 GPU 上完成训练，MoE 专家分散在不同节点中。跨节点 expert all-to-all 通信的代价，几乎达到“每 1 个 GPU 小时计算就要配 1 个 GPU 小时通信”。GPU 有一半时间处于空闲。DualPipe（DeepSeek，2024 年 12 月）是一种双向流水线（bidirectional pipeline），它把前向与反向计算，同它们触发的 all-to-all 通信重叠起来。流水线气泡（bubble）减少了，吞吐提高了，而每台设备保留两份模型参数副本（也就是名字里 “dual” 的来源）这件事，在专家并行（Expert Parallelism）本来就把专家打散到多个 rank 的前提下，其实不算贵。本课会以 Learn 风格逐步讲清：DualPipe 到底做了什么，以及为什么 Sea AI Lab 的改进版 DualPipeV 能以略微更紧的气泡代价去掉 2x 参数成本。

**类型：** Learn
**语言：** Python（stdlib，schedule simulator）
**前置课程：** 第 10 阶段·05（分布式训练、FSDP、DeepSpeed），第 10 阶段·14（开源模型架构与 MoE）
**耗时：** ~60 分钟

## 学习目标

- 说出一个 DualPipe 前向-反向 chunk 的四个组成部分，以及为什么每个部分都需要独立的重叠窗口。
- 解释大规模训练中的流水线气泡问题，并说明营销语境中的“无气泡”与技术语境中的真实含义。
- 手工跟踪一个 8 个 PP rank、16 个 micro-batch 的 DualPipe 调度，并确认正向流与反向流如何填满彼此的空闲槽位。
- 说明 DualPipeV（Sea AI Lab，2025）所做的权衡：在专家并行不活跃时，它去掉 2x 参数复制，但代价是气泡略微变大。

## 问题

在 2k 张 H800 GPU 上训练一个 671B 的 MoE 模型，会同时撞上三个不断叠加的瓶颈：

1. **内存压力。** 每张 GPU 只持有模型的一部分。对于 8k 序列、61 层、128 个头，激活内存会非常惊人。
2. **流水线气泡。** 传统流水线并行（GPipe、1F1B）会让 GPU 在等待本阶段输入或梯度时闲着。在 8 个 stage 下，即使用 1F1B 调度，大约也会有 12% 的 GPU 时间浪费在气泡上。
3. **跨节点 all-to-all。** 使用专家并行的 MoE 会把 expert 分散到多个节点。每次前向都会触发一次 all-to-all 来把 token 分发给对应 expert，还要再来一次把结果聚合回来。在 2k GPU 规模下，这很容易形成 1:1 的计算/通信比。

这三件事原本各自都有对应解法：用梯度检查点来缓解内存，用 Zero Bubble（Sea AI Lab，2023）减少流水线气泡，用 expert-parallel 通信 kernel 加速 all-to-all。DualPipe 的价值在于：它让这些办法能够协同工作。这个调度会在一个前向-反向 chunk 内同时重叠计算与通信，从流水线两端同时注入 micro-batch，再利用形成的时序，把 all-to-all 隐藏到计算窗口之内。

报告中的结果是：流水线气泡几乎被消除，在 DeepSeek-V3 这次 14.8T token 的训练中，GPU 利用率超过 95%。

## 概念

### 流水线并行回顾

把一个有 N 层的模型切到 P 台设备上。设备 `i` 持有第 `i * N/P .. (i+1) * N/P - 1` 层。一个 micro-batch 会从设备 0 依次前向流到设备 P-1，再在反向中从设备 P-1 返回设备 0。每台设备都只能在上游设备发来输出后，启动自己的前向阶段；也只能在下游设备发回上游梯度后，启动自己的反向阶段。

GPipe（Huang 等，2019）一次只调度一个 micro-batch，浪费了大部分 GPU 时间。1F1B（Narayanan 等，2021）会在多个 micro-batch 之间交织前向与反向。Zero Bubble（Qi 等，2023）则把反向拆成两部分：对输入求梯度的 backward-for-input（B）和对权重求梯度的 backward-for-weights（W），再把它们重新调度以填补气泡。经过 Zero Bubble 后，流水线已经非常紧凑。

DualPipe 是下一步。它在此基础上再加两个想法：

### 想法 1：chunk 拆解

每个前向 chunk 被拆成四部分：

- **Attention。** Q/K/V 投影、注意力、输出投影。
- **All-to-all dispatch。** 把 token 发往各自 expert 的跨节点通信。
- **MLP。** MoE expert 计算。
- **All-to-all combine。** 把 expert 输出拉回来的跨节点通信。

一个反向 chunk 也会为这四个部分分别提供对应的梯度版本。DualPipe 的调度方式是：让 all-to-all dispatch 与下一个 chunk 的 attention 计算并行执行，让 all-to-all combine 与再下一个 chunk 的 MLP 计算并行执行。

### 想法 2：双向调度

大多数流水线调度都只会从 stage 0 注入 micro-batch，让它们一路流向 stage P-1。DualPipe 会从两端同时注入 micro-batch。stage 0 会看到从左端发起的前向 micro-batch；stage P-1 也会看到从右端发起的前向 micro-batch。两股流会在中间相遇。

为了实现这一点，设备 `i` 必须同时持有“前半流水线上的第 `i` 层”以及“后半流水线上的第 `P - 1 - i` 层”。这就是 DualPipe 名字里 “dual” 的含义：每台设备都要保留自己服务所需的两份模型层（分别对应两个方向）。在 DeepSeek-V3 的规模下，这意味着 2x 的参数复制成本。不过这仍然可以接受，因为专家并行本来就已经把 MoE expert 分散得很细，额外把非 expert 层复制两遍只是小头。

关键在于：一个方向上的前向流，与另一个方向上的反向流，会恰好重叠在单向调度原本会出现气泡的位置上。于是气泡就消失了。

### 手工跟踪一个调度表

考虑 P = 4 个 rank、8 个 micro-batch，分成 4 个正向 / 4 个反向。时间从左向右推进；每一行代表一个设备 rank。

```
           Time →
rank 0:  F1 F2 F3 F4  F5R F6R F7R F8R  B1 B2 B3 B4  ...
rank 1:     F1 F2 F3  F4/F5R F6R F7R   B1 B2 ...
rank 2:        F1 F2  F3/F5R F4/F6R    B1 ...
rank 3:           F1  F2/F5R F3/F6R    ...
```

读懂 `F4/F5R` 这个记号：在同一个时间槽里，rank 1 同时执行 micro-batch 4 的前向（沿流水线从左往右）以及 micro-batch 5 的前向（沿流水线从右往左）。这就是“bidirectional”在操作层面的含义。

在 rank 2 上，两股流更早相遇；在 rank 0 和 P-1 上，它们最晚才重叠。在调度的稳定中段，每个 rank 都会同时执行“X 方向的前向”和“Y 方向的反向”。计算单元一直忙着。前向所需的 all-to-all dispatch 被藏进反向计算窗口里；all-to-all combine 被藏进前向计算窗口里。气泡被挤掉了。

### 气泡核算

标准 1F1B 流水线中的气泡（每个 rank 的浪费时间）为：

```
bubble_1F1B = (P - 1) * forward_chunk_time
```

Zero Bubble 会把它压低，但不是真正归零。DualPipe 在稳定阶段，如果 micro-batch 数量能被“2 × 流水线深度”整除，则可以做到零气泡。稳定阶段之外（预热与冷却）仍然有一些气泡，但它不会随着 micro-batch 数量增长——这是论文强调的关键性质。

用营销术语来说，它叫“bubble-free”；用技术术语来说，是“气泡不会随 micro-batch 数量增长”。Sea AI Lab 后续的分析（DualPipeV / Cut-in-half）指出：只有当专家并行不是瓶颈时，才会出现完整的零气泡；若 all-to-all 本身由 EP 驱动成为主瓶颈，那么调度上总还是要做一些妥协。

### DualPipeV：改进版

Sea AI Lab（2025）观察到：当 EP 通信重叠并不是核心目标时，2x 的参数复制会显得浪费。他们提出的 DualPipeV 调度，把双向注入折叠成一个 “V” 形时序，只需一份参数副本即可运行。它的气泡会比 DualPipe 稍微大一点，但能节省大量内存。DeepSeek 在开源 DualPipe 实现中，也把 DualPipeV 作为 EP-off 模式采用了。

权衡如下：

| 特性 | DualPipe | DualPipeV | 1F1B | Zero Bubble |
|---------|---------|-----------|------|------------|
| 每台设备的参数副本数 | 2 | 1 | 1 | 1 |
| 气泡随 micro-batch 变化 | 常数级 | 小幅增长 | 持续增长 | 持续增长 |
| 计算-通信重叠 | 完整 | 部分 | 极少 | 部分 |
| 适用场景 | EP 很重的 MoE | 稠密模型或 EP 较轻 | 基线方案 | 任意流水线 |

### 对 14.8T token 训练意味着什么

DeepSeek-V3 的预训练在 2,048 张 H800 GPU 上消耗了 14.8T token，总计约 280 万 GPU 小时。如果使用朴素的 1F1B，他们会有 12–15% 的时间浪费在流水线气泡上——也就是 34 万到 42 万 GPU 小时，足够训练完整一个 70B 模型。DualPipe 追回了其中大部分时间。由于缺少内部日志，很难精确量化它单独贡献了多少，但论文中的表述是：训练全程平均 GPU 利用率超过 95%。

对于较小规模的训练（低于 1k GPU），DualPipe 通常属于杀鸡用牛刀——相对总成本而言，流水线气泡没那么大，而稠密模型训练通常也不会碰到 all-to-all 瓶颈。可一旦来到几千 GPU 规模的前沿 MoE 训练，它几乎就是必需品。

### 它在整套栈中的位置

- 它与 **FSDP**（第 10 阶段·05）互补。FSDP 负责在 rank 之间切分模型参数；DualPipe 负责在 rank 之间调度计算。两者可以叠加。
- 它兼容 **ZeRO-3** 的梯度切分。不过双副本复制的记账逻辑，必须与 ZeRO 的梯度分片机制协同工作。
- 它需要针对具体集群拓扑调优过的 **custom all-to-all kernels**。DeepSeek 开源的 kernels 是当前参考实现。

## 使用

`code/main.py` 是一个流水线调度模拟器。它接受 `(P, n_micro_batches, schedule)`，并打印 1F1B、Zero Bubble、DualPipe 与 DualPipeV 各自在稳定阶段的利用率。它是一个教学工具——这些数字会复现论文中的定性结论，但并不等同于对生产环境真实加速比的宣称。

这个模拟器的价值在于：你可以用不同的 P 和 micro-batch 数去运行它，然后观察 1F1B 的气泡比例如何增长，而 DualPipe 不会。

真实训练运行中的集成注意事项：

- 选择一个能够整除 micro-batch 数的流水线并行深度。
- 确保你的 expert-parallel mesh 支持双向 all-to-all。DeepSeek 的 kernels 是参考实现。
- 第一次接这种调度时，准备花一周时间调 bug。这里的记账非常繁琐。
- 监控每个 rank 的 GPU 利用率，而不只是总平均值。DualPipe 的收益来自把拖后腿的 rank 收紧。

## 交付

本课会产出 `outputs/skill-dualpipe-planner.md`。给定训练集群规格（GPU 数量、拓扑、互联、模型形状），它会推荐一套流水线并行策略、应该使用的调度算法，以及目标规模下预计的气泡比例。

## 练习

1. 在 `(P=8, micro_batches=16, schedule=dualpipe)` 和 `(P=8, micro_batches=16, schedule=1f1b)` 上运行 `code/main.py`。计算 GPU 利用率差值，并把它换算成“每百万 token 训练可回收的 GPU 小时”。

2. 手工画出 `(P=4, micro_batches=8, schedule=dualpipe)` 的调度表。给每个时间槽标注 micro-batch 编号和方向。找出第一个完全没有气泡的时间槽。

3. 阅读 DeepSeek-V3 技术报告（arXiv:2412.19437）的图 5。识别 DualPipe 前向 chunk 中 all-to-all dispatch 的重叠窗口。解释计算调度如何把它隐藏起来。

4. 分别计算：对一个有 P=8 个流水线 stage 的 70B 稠密模型，以及一个有 P=16 个流水线 stage 的 671B MoE 模型，DualPipe 带来的 2x 参数开销。说明为什么 MoE 场景下这个开销在比例上更小（大部分参数是 expert，并且已经被切分到一个很大的 EP 组里）。

5. 将 DualPipe 与 Chimera（2021 年的另一个双向调度器）进行比较。参考论文第 3.4 节，指出 DualPipe 额外引入、而 Chimera 没有的两个具体性质。

## 关键术语

| 术语 | 人们常说什么 | 它真正的含义 |
|------|----------------|------------------------|
| 流水线气泡 | “每个 rank 的空闲时间” | 某个流水线 stage 因等待输入或梯度而浪费掉的 GPU 周期 |
| 1F1B | “默认流水线调度” | 一次前向 / 一次反向交错的调度；是 DualPipe 超越的基线 |
| Zero Bubble | “Sea AI Lab 2023” | 把反向拆成 B（输入梯度）和 W（权重梯度）；几乎把流水线拉紧 |
| DualPipe | “DeepSeek-V3 的调度” | 双向流水线 + 计算/通信重叠；气泡不会随 micro-batch 数增长 |
| DualPipeV | “Cut-in-half” | V 形改进版，在牺牲一点点气泡的前提下去掉 2x 参数复制 |
| Chunk | “流水线工作的单位” | 一个 micro-batch 穿过一个流水线 stage 时的前向或反向过程 |
| All-to-all dispatch | “把 token 发给 expert” | 将 token 路由到其分配的 MoE expert 的跨节点通信 |
| All-to-all combine | “把 expert 输出拿回来” | MLP 之后把 expert 输出重新聚合回来的跨节点通信 |
| 专家并行（EP） | “expert 分布在多张 GPU 上” | 把 MoE expert 切分到不同 rank 上，让不同 GPU 持有不同 expert |
| 流水线并行（PP） | “层分布在多张 GPU 上” | 把模型层切分到不同 rank；DualPipe 调度的正是这个维度 |
| 气泡比例 | “浪费掉的 GPU 时间” | `(bubble_time / total_time)`；DualPipe 试图把它压到接近零 |

## 延伸阅读

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437), Section 3.3.2 and Figure 5](https://arxiv.org/abs/2412.19437) —— DualPipe 的首要参考资料
- [DeepSeek — DualPipe GitHub repository](https://github.com/deepseek-ai/DualPipe) —— 开源参考实现，包含 DualPipeV（Cut-in-half）模式
- [Qi et al. — Zero Bubble Pipeline Parallelism (arXiv:2401.10241, Sea AI Lab 2023)](https://arxiv.org/abs/2401.10241) —— Zero Bubble 这一前身工作
- [Sea AI Lab — DualPipe could be better without the Dual](https://sail.sea.com/blog/articles/63) —— 对 DualPipeV 的分析，也是 DeepSeek EP-off 模式的灵感来源
- [Narayanan et al. — PipeDream / 1F1B (arXiv:1806.03377, 2018-2021)](https://arxiv.org/abs/1806.03377) —— DualPipe 对比的 1F1B 调度
- [Huang et al. — GPipe (arXiv:1811.06965, 2018)](https://arxiv.org/abs/1811.06965) —— 最初提出流水线并行与气泡问题的论文
