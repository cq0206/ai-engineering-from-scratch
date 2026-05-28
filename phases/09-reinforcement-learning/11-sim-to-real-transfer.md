# 仿真到现实迁移 (Sim-to-Real Transfer)

> 如果一个在模拟器里训练出的策略一上硬件就失效，那它记住的其实只是模拟器。领域随机化 (domain randomization)、领域自适应 (domain adaptation) 和系统辨识 (system identification) 是让学得的控制器跨越现实鸿沟的三种核心工具。

**类型：** Learn
**语言：** Python
**前置要求：** 第 9 阶段 · 08（PPO），第 2 阶段 · 10（偏差/方差 (Bias/Variance)）
**耗时：** ~45 分钟

## 问题

训练真实机器人很慢、很危险，也很昂贵。一个双足机器人可能需要几百万个训练 episode 才能学会走路；而真实的双足机器人哪怕只摔一次，都可能损坏硬件。仿真则提供了无限次 reset、可确定复现、并行环境，以及零物理损伤。

但模拟器是错的。真实轴承的摩擦往往比 MuJoCo 模型更大。真实相机会有模拟器里没有建模的镜头畸变。电机还会存在延迟、回程间隙和饱和，而 99% 的仿真模型都把这些跳过了。风、灰尘和变化的光照，会让在“无菌”渲染环境中训练出来的策略失灵。**现实鸿沟 (reality gap)**——即仿真分布与真实分布之间的系统性差异——是机器人 RL 落地部署时最核心的问题。

你需要的是一个*对 sim-to-real 分布偏移具有鲁棒性*的策略。历史上主要有三条路线：随机化模拟器（domain randomization）、用少量真实数据适配策略（domain adaptation / fine-tuning），或者识别真实系统参数并与之匹配（system identification）。到 2026 年，主流配方是把三者与大规模并行仿真结合起来（Isaac Sim、Isaac Lab、GPU 上的 Mujoco MJX）。

## 概念

*三种 sim-to-real 范式：领域随机化、自适应、系统辨识*

**领域随机化 (DR)。** Tobin 等（2017）、Peng 等（2018）。在训练期间，把所有可能与真实机器人不同的仿真参数都随机化：质量、摩擦系数、电机 PD 增益、传感器噪声、相机位置、光照、纹理、接触模型。策略会学到一个关于“今天自己处在哪个模拟器里”的条件分布，并在整个参数范围内泛化。只要真实机器人落在训练包络内，策略就能工作。

- **优点：** 不需要真实数据。一套方法，适配多种机器人。
- **缺点：** 随机化过头会得到一个“通用”但过于保守的策略。噪声太大 ≈ 正则化过强。

**系统辨识 (SI)。** 在训练前先把模拟器参数拟合到真实世界数据上。如果你能测出真实机器人关节的摩擦，就把它填进仿真里。然后训练一个以这些数值为预期的策略。它需要访问真实系统，但能直接缩小 reality gap。

- **优点：** 训练目标精确、噪声低。
- **缺点：** 残余模型误差对策略是不可见的；一些未被识别的小效应（例如电机死区）仍然会在部署时导致失败。

**领域自适应。** 先在仿真中训练，再用少量真实数据微调。主要有两种形式：

- **Real2Sim2Real：** 用真实 rollout 学习一个残差模拟器 `f(s, a, z) - f_sim(s, a)`，再在修正后的仿真中训练。这样无需太多真实数据就能缩小差距。
- **观测自适应：** 训练一个策略，通过学得的特征提取器把真实 obs 映射成类似仿真的 obs（例如 GAN 的 pixel-to-pixel 变换）。控制器本身仍留在仿真域。

**特权学习 / teacher-student。** Miki 等（2022，ANYmal 四足机器人）。先在仿真中训练一个 *teacher*，它可以访问特权信息（真实摩擦的 ground truth、地形高度、IMU 漂移）。再蒸馏出一个只看真实传感器观测的 *student*。student 会从历史中推断这些特权特征，并在不同物理参数下保持鲁棒。

**大规模并行仿真。** 2024–2026 年间，Isaac Lab、Mujoco MJX、Brax 都可以在一张 GPU 上同时运行成千上万个并行机器人。使用 4,096 个并行 humanoid 的 PPO，几个小时就能收集相当于数年的经验。随着训练分布变宽，“reality gap” 会缩小；当这 4,096 个环境各自采用不同随机参数时，DR 几乎就变成了免费的。

**2026 年的真实世界配方（以四足行走为例）：**

1. 使用对重力、摩擦、电机增益和负载进行领域随机化的大规模并行仿真。
2. 用特权信息（地形图、机体速度 ground truth）训练 teacher 策略。
3. 仅使用本体感觉 (proprioception)（腿部关节编码器）从 teacher 蒸馏出 student 策略。
4. 可选：在真实 IMU 上通过 autoencoder 做观测自适应。
5. 部署。在 10+ 个环境中 zero-shot 运行。如果失败，再用带安全约束的 PPO 做几分钟真实世界微调。

## 动手构建

本课代码是在一个带有*噪声*转移的 GridWorld 上，对领域随机化做的一个微型演示。我们训练一个在“sim”中经历随机滑移概率的策略，然后在“real”上用训练中从未见过的滑移水平进行评估。这个结构可以直接映射到 MuJoCo 到真实硬件的迁移问题。

### 第 1 步：参数化仿真

```python
def step(state, action, slip):
    if rng.random() < slip:
        action = random_perpendicular(action)
    ...
```

`slip` 是模拟器暴露出的一个参数。在真实机器人里，它可能对应摩擦、质量、电机增益——任何会在 sim 和 real 之间变化的因素。

### 第 2 步：使用 DR 训练

在每个 episode 开始时，采样 `slip ~ Uniform[0.0, 0.4]`。然后训练 PPO / Q-learning / 任何算法。如此持续多个 episode。

### 第 3 步：在“真实”滑移上做 zero-shot 评估

在 `slip ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}` 上评估。前四个值位于训练支持范围内；`0.5` 和 `0.7` 超出了范围。一个经 DR 训练的策略应该在支持范围内保持接近最优，并在范围外平滑退化。一个只在固定 slip 上训练的策略，则会在超出训练 slip 时变得非常脆弱。

### 第 4 步：与窄范围训练比较

再训练第二个策略，只使用 `slip = 0.0`。在相同的 `slip` 扫描上评估它。你应该会看到：只要真实 slip > 0，性能就会灾难性下跌。

## 常见陷阱

- **随机化过多。** 如果在 `slip ∈ [0, 0.9]` 上训练，你的策略会因为过于厌恶风险而从不尝试最优路径。要匹配的是*预期中的*真实世界分布，而不是“任何事情都可能发生”。
- **随机化过少。** 如果只在一小段范围内训练，策略根本无法泛化。应使用自适应课程学习，例如自动领域随机化 (Automatic Domain Randomization)，随着策略变强逐步扩大分布。
- **参数空间识别错误。** 如果随机化错了对象（比如真实差距在电机延迟，你却只随机相机色调），DR 就不会有帮助。先对真实机器人做剖析。
- **特权信息泄漏。** 如果 teacher 在决策时使用了全局状态而不只是观测，就可能得到一个 student 永远追不上的目标。必须确保 teacher 的策略在给定观测历史的条件下，对 student 来说是可实现的。
- **Sim-to-sim 迁移失败。** 如果你的策略连更难的仿真变体都不鲁棒，那它对真实世界也不可能鲁棒。部署前一定先在保留的仿真变体上测试。
- **没有真实世界安全包络。** 一个在 sim 里能跑、在 real 里“似乎也能跑”的策略，如果没有底层安全防护，仍然可能损坏硬件。要在非学习型控制器里加入速率限制、力矩限制和关节限制。

## 如何使用

2026 年的 sim-to-real 技术栈如下：

| 领域 | 技术栈 |
|------|--------|
| 足式运动（ANYmal、Spot、humanoid） | Isaac Lab + DR + 特权 teacher / student |
| 操作任务（灵巧手、抓取放置） | Isaac Lab + DR + 用于视觉的 DR-GAN |
| 自动驾驶 | CARLA / NVIDIA DRIVE Sim + DR + 真实微调 |
| 无人机竞速 | RotorS / Flightmare + DR + 在线自适应 |
| 手指 / 手内操作 | OpenAI Dactyl（前所未有规模的 DR） |
| 工业机械臂 | MuJoCo-Warp + SI + 少量真实微调 |

对于各种尺度的控制任务，工作流都很一致：尽可能把仿真拟合准确，对无法拟合的部分做随机化，训练大规模策略，蒸馏，再带着安全防护部署。

## 交付

保存为 `outputs/skill-sim2real-planner.md`：

```markdown
---
name: sim2real-planner
description: Plan a sim-to-real transfer pipeline for a given robot + task, covering DR, SI, and safety.
version: 1.0.0
phase: 9
lesson: 11
tags: [rl, sim2real, robotics, domain-randomization]
---

Given a robot platform, a task, and access to real hardware time, output:

1. Reality gap inventory. Suspected sources ranked by expected impact (contact, sensing, actuation delay, vision).
2. DR parameters. Exact list, ranges, distribution. Justify each range against real measurements.
3. SI steps. Which parameters to measure; measurement method.
4. Teacher/student split. What privileged info the teacher uses; what obs the student uses.
5. Safety envelope. Low-level limits, emergency stops, backup controller.

Refuse to deploy without (a) a zero-shot sim-variant test, (b) a safety shield, (c) a rollback plan. Flag any DR range wider than 3× measured real variability as likely over-randomized.
```

## 练习

1. **简单。** 在固定滑移的 GridWorld（slip=0.0）上训练一个 Q-learning 智能体。然后在 slip ∈ {0.0, 0.1, 0.3, 0.5} 上评估。画出 return 随 slip 变化的曲线。
2. **中等。** 训练一个 DR Q-learning 智能体，按 `slip ~ Uniform[0, 0.3]` 采样。用相同的滑移扫描进行评估。在 slip=0.5（分布外）时，DR 带来了多大提升？
3. **困难。** 实现一个课程学习：从 slip=0.0 开始，每当策略达到最优值的 90%，就扩大一次 DR 范围。测量达到 slip=0.3 的 zero-shot 表现所需的总环境步数，并与固定 DR 基线比较。

## 关键术语

| 术语 | 人们常说什么 | 它实际表示什么 |
|------|--------------|----------------|
| Reality gap | “sim-to-real 差异” | 训练与部署时物理 / 感知分布之间的偏移。 |
| Domain randomization (DR) | “在随机仿真上训练” | 训练时随机化仿真参数，从而让策略具备泛化能力。 |
| System identification (SI) | “测真实、拟合仿真” | 估计真实物理参数，并把仿真设成匹配这些参数。 |
| Domain adaptation | “用真实数据微调” | 在仿真训练后做少量真实世界微调；可以适配观测或动力学。 |
| Privileged info | “teacher 的 ground truth” | 只有仿真才有的信息；student 必须从观测历史中推断出来。 |
| Teacher/student | “把特权信息蒸馏成可观测信息” | teacher 用捷径训练；student 学会在没有捷径时模仿它。 |
| ADR | “Automatic Domain Randomization” | 随着策略变强而逐步扩大 DR 范围的课程方法。 |
| Real2Sim | “用真实数据缩小差距” | 学习一个残差，让仿真更像真实 rollout。 |

## 延伸阅读

- [Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) —— 最早的 DR 论文（面向机器人视觉）。
- [Peng et al. (2018). Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537) —— 将 DR 用于动力学和四足运动。
- [OpenAI et al. (2019). Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113) —— Dactyl，大规模 ADR 的经典案例。
- [Miki et al. (2022). Learning robust perceptive locomotion for quadrupedal robots in the wild](https://www.science.org/doi/10.1126/scirobotics.abk2822) —— ANYmal 上的 teacher-student 方法。
- [Makoviychuk et al. (2021). Isaac Gym: High Performance GPU Based Physics Simulation for Robot Learning](https://arxiv.org/abs/2108.10470) —— 驱动 2025–2026 年部署浪潮的大规模并行仿真系统。
- [Akkaya et al. (2019). Automatic Domain Randomization](https://arxiv.org/abs/1910.07113) —— ADR 课程方法。
- [Sutton & Barto (2018). Ch. 8 — Planning and Learning with Tabular Methods](http://incompleteideas.net/book/RLbook2020.pdf) —— Dyna 视角（用模型做规划和 rollout），它构成了现代 sim-to-real 流水线的基础。
- [Zhao, Queralta & Westerlund (2020). Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey](https://arxiv.org/abs/2009.13303) —— 一篇综述，总结了 sim-to-real 方法谱系及其基准结果。
