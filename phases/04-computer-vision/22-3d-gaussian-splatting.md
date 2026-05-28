# 从零开始实现三维高斯泼溅 (3D Gaussian Splatting)

> 一个场景就是由数百万个 3D 高斯组成的云。每个高斯都有位置、朝向、尺度、不透明度，以及会随观察方向变化的颜色。把它们栅格化，对栅格化过程反向传播，就完成了。

**类型：** 构建
**语言：** Python
**前置课程：** 第 4 阶段第 13 课（3D 视觉与 NeRF）、第 1 阶段第 12 课（张量操作）、第 4 阶段第 10 课（扩散基础，可选）
**时长：** ~90 分钟

## 学习目标

- 解释为什么三维高斯泼溅 (3D Gaussian Splatting) 在 2026 年取代 NeRF，成为写实三维重建的生产默认方案
- 说出每个高斯的六类参数（位置、旋转四元数、尺度、不透明度、球谐函数颜色、可选特征），以及它们各自贡献多少个浮点数
- 从零实现一个使用 `alpha` 合成的 2D Gaussian splatting 栅格器，然后说明 3D 情况如何投影到同一个循环中
- 使用 `nerfstudio`、`gsplat` 或 `SuperSplat` 从 20-50 张照片重建场景，并导出到 `KHR_gaussian_splatting` glTF 扩展或 OpenUSD 26.03 的 `UsdVolParticleField3DGaussianSplat` schema

## 问题

NeRF 将场景存储为一个 MLP 的权重。每个渲染像素都需要沿着一条光线执行数百次 MLP 查询。训练要几个小时，渲染要几秒，而且这些权重无法直接编辑——如果你想把场景里的椅子挪个位置，就得重新训练。

三维高斯泼溅 (3D Gaussian Splatting, 3DGS) 改变了这一切（Kerbl、Kopanas、Leimkühler、Drettakis，SIGGRAPH 2023）。一个场景就是一组显式的 3D 高斯。渲染是以 100+ fps 运行的 GPU 栅格化。训练只需几分钟。编辑也非常直接：平移某一部分高斯，你就把椅子挪动了。到了 2026 年，Khronos Group 已经批准了用于 Gaussian splat 的 glTF 扩展，OpenUSD 26.03 也提供了 Gaussian splat schema，Zillow 和 Apartments.com 用它们来渲染房地产场景，而大多数新的三维重建论文也都是围绕 3DGS 核心思想的变体。

它的心智模型很简单，但数学细节包含足够多的活动部件，以至于大多数入门介绍都会从栅格化开始，跳过投影和球谐函数 (spherical harmonics) 的部分。这节课会把整个流程搭起来——先做 2D 版本，再扩展到 3D。

## 概念

### 一个高斯携带什么信息

一个 3D 高斯就是空间中的参数化“团块”，具有这些属性：

```
position         mu         (3,)    centre in world coordinates
rotation         q          (4,)    unit quaternion encoding orientation
scale            s          (3,)    log-scales per axis (exponentiated at render time)
opacity          alpha      (1,)    post-sigmoid opacity [0, 1]
SH coefficients  c_lm       (3 * (L+1)^2,)   view-dependent colour
```

旋转和尺度共同构成一个 3x3 协方差 (covariance)：`Sigma = R S S^T R^T`。这就是该高斯在 3D 中的形状。球谐函数 (spherical harmonics, SH) 让颜色可以随观察方向变化——镜面高光、细微光泽、视角相关的发光——而不需要存储逐视角纹理。使用 3 阶 SH 时，每个颜色通道有 16 个系数，仅颜色一项每个高斯就需要 48 个浮点数。

一个场景通常包含 100 万到 500 万个高斯。每个高斯大约存储 60 个浮点数（3 + 4 + 3 + 1 + 48 + 杂项）。这意味着一个包含 500 万个高斯的场景大约占 240 MB——比带逐点纹理的等价点云小得多，也比重新以高分辨率渲染 NeRF 的 MLP 权重大一个数量级地更小。

### 栅格化，而不是光线步进

```mermaid
flowchart LR
    SCENE["数百万个 3D 高斯<br/>(位置、旋转、尺度、<br/>不透明度、SH 颜色)"] --> PROJ["投影到 2D<br/>(相机外参 + 内参)"]
    PROJ --> TILES["分配到瓦片<br/>(16x16 屏幕空间)"]
    TILES --> SORT["按深度排序<br/>每个瓦片内"]
    SORT --> ALPHA["Alpha 合成<br/>前到后"]
    ALPHA --> PIX["像素颜色"]

    style SCENE fill:#dbeafe,stroke:#2563eb
    style ALPHA fill:#fef3c7,stroke:#d97706
    style PIX fill:#dcfce7,stroke:#16a34a
```

五个步骤，而且都对 GPU 友好。每个像素都不再需要 MLP 查询。一张 RTX 3080 Ti 就能以 147 fps 渲染 600 万个 splat。

### 投影步骤

位于世界坐标 `mu`、具有 3D 协方差 `Sigma` 的 3D 高斯，会投影成屏幕位置 `mu'`、带有 2D 协方差 `Sigma'` 的 2D 高斯：

```
mu' = project(mu)
Sigma' = J W Sigma W^T J^T          (2 x 2)

W = viewing transform (rotation + translation of camera)
J = Jacobian of the perspective projection at mu'
```

这个 2D 高斯在屏幕上的足迹是一个椭圆，其轴由 `Sigma'` 的特征向量决定。椭圆内部的每个像素都会接收到该高斯的贡献，其权重为 `exp(-0.5 * (p - mu')^T Sigma'^-1 (p - mu'))`。

### alpha 合成规则

对于某一个像素，覆盖它的高斯会按从后到前排序（或者等价地按从前到后并使用反向公式）。颜色合成使用的是自 20 世纪 80 年代以来所有半透明栅格器都在使用的同一条公式：

```
C_pixel = sum_i alpha_i * T_i * c_i

T_i = prod_{j < i} (1 - alpha_j)       transmittance up to i
alpha_i = opacity_i * exp(-0.5 * d^T Sigma'^-1 d)   local contribution
c_i = eval_SH(SH_i, view_direction)    view-dependent colour
```

这**和 NeRF 的体渲染 (volumetric render) 方程是同一个方程**，只不过现在是在一组显式、稀疏的高斯上积分，而不是在一条光线上对稠密采样点积分。正是这种同一性，让它的渲染质量可以匹配 NeRF——因为两者积分的是同一个辐射场方程。

### 为什么它是可微的

每一步——投影、瓦片分配、alpha 合成、SH 求值——相对于高斯参数都是可微的。给定一张真实图像，计算渲染像素损失，通过栅格器反向传播，再用梯度下降更新所有 `(mu, q, s, alpha, c_lm)`。大约经过 30,000 次迭代后，这些高斯就会找到各自正确的位置、尺度和颜色。

### 加密与剪枝

固定数量的高斯不足以覆盖复杂场景。因此训练过程中会包含两种自适应机制：

- **Clone** 某个高斯：当它的梯度幅值很大，但尺度很小时，说明这里需要更多细节。
- **Split** 一个大尺度高斯为两个更小的高斯：当它的梯度很大时，说明一个大高斯太平滑，无法拟合这块区域。
- **Prune** 不透明度低于阈值的高斯：它们已经不再贡献结果。

加密 (densification) 每隔 N 次迭代运行一次。一个场景通常会从大约 10 万个初始高斯（由 SfM 点云播种）增长到训练结束时的 100 万到 500 万个。

### 用一段话理解球谐函数

视角相关颜色是单位球面上的一个函数 `c(direction)`。球谐函数就是球面上的傅里叶基。截断到 `L` 阶后，每个通道会得到 `(L+1)^2` 个基函数。为新视角求颜色时，只需把学习到的 SH 系数与在该观察方向上计算得到的基函数做点积。0 阶 = 1 个系数 = 常量颜色。3 阶 = 16 个系数 = 足以表达朗伯着色、镜面反射和轻微反光。SD Gaussian Splatting 论文默认使用 3 阶。

### 2026 年的生产栈

```
1. Capture         smartphone / DJI drone / handheld scanner
2. SfM / MVS       COLMAP or GLOMAP derives camera poses + sparse points
3. Train 3DGS      nerfstudio / gsplat / inria official / PostShot (~10-30 min on RTX 4090)
4. Edit            SuperSplat / SplatForge (clean floaters, segment)
5. Export          .ply -> glTF KHR_gaussian_splatting or .usd (OpenUSD 26.03)
6. View            Cesium / Unreal / Babylon.js / Three.js / Vision Pro
```

### 4D 与生成式变体

- **4D Gaussian Splatting** —— 高斯会随时间变化；用于体积视频（Superman 2026、A$AP Rocky 的《Helicopter》）。
- **Generative splats** —— 文生 splat 模型（如 World Labs 的 Marble），可以直接幻觉出整个场景。
- **3D Gaussian Unscented Transform** —— NVIDIA NuRec 面向自动驾驶仿真的变体。

## 动手实现

### 第 1 步：一个 2D Gaussian

我们先构建一个 2D 栅格器。3D 情况在投影后会归约成它。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def eval_2d_gaussian(means, covs, points):
    """
    means:  (G, 2)      centres
    covs:   (G, 2, 2)   covariance matrices
    points: (H, W, 2)   pixel coordinates
    returns: (G, H, W)  density at every pixel for every Gaussian
    """
    G = means.size(0)
    H, W, _ = points.shape
    flat = points.view(-1, 2)
    inv = torch.linalg.inv(covs)
    diff = flat[None, :, :] - means[:, None, :]
    d = torch.einsum("gpi,gij,gpj->gp", diff, inv, diff)
    density = torch.exp(-0.5 * d)
    return density.view(G, H, W)
```

`einsum` 为每一组（高斯、像素）对执行二次型 `diff^T Sigma^-1 diff`。

### 第 2 步：2D splatting 栅格器

按从前到后的顺序做 alpha 合成。在 2D 中深度没有实际意义，所以我们用一个学习到的每高斯标量来决定顺序。

```python
def rasterise_2d(means, covs, colours, opacities, depths, image_size):
    """
    means:     (G, 2)
    covs:      (G, 2, 2)
    colours:   (G, 3)
    opacities: (G,)     in [0, 1]
    depths:    (G,)     per-Gaussian scalar used for ordering
    image_size: (H, W)
    returns:   (H, W, 3) rendered image
    """
    H, W = image_size
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=means.device),
        torch.arange(W, dtype=torch.float32, device=means.device),
        indexing="ij",
    )
    points = torch.stack([xx, yy], dim=-1)

    densities = eval_2d_gaussian(means, covs, points)
    alphas = opacities[:, None, None] * densities
    alphas = alphas.clamp(0.0, 0.99)

    order = torch.argsort(depths)
    alphas = alphas[order]
    colours_sorted = colours[order]

    T = torch.ones(H, W, device=means.device)
    out = torch.zeros(H, W, 3, device=means.device)
    for i in range(means.size(0)):
        a = alphas[i]
        out += (T * a)[..., None] * colours_sorted[i][None, None, :]
        T = T * (1.0 - a)
    return out
```

速度并不快——真正的实现会使用基于瓦片的 CUDA kernel——但数学上完全正确，而且完全可微。

### 第 3 步：一个可训练的 2D splat 场景

```python
class Splats2D(nn.Module):
    def __init__(self, num_splats=128, image_size=64, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        H, W = image_size, image_size
        self.means = nn.Parameter(torch.rand(num_splats, 2, generator=g) * torch.tensor([W, H]))
        self.log_scale = nn.Parameter(torch.ones(num_splats, 2) * math.log(2.0))
        self.rot = nn.Parameter(torch.zeros(num_splats))  # single angle in 2D
        self.colour_logits = nn.Parameter(torch.randn(num_splats, 3, generator=g) * 0.5)
        self.opacity_logit = nn.Parameter(torch.zeros(num_splats))
        self.depth = nn.Parameter(torch.rand(num_splats, generator=g))

    def covs(self):
        s = torch.exp(self.log_scale)
        c, si = torch.cos(self.rot), torch.sin(self.rot)
        R = torch.stack([
            torch.stack([c, -si], dim=-1),
            torch.stack([si, c], dim=-1),
        ], dim=-2)
        S = torch.diag_embed(s ** 2)
        return R @ S @ R.transpose(-1, -2)

    def forward(self, image_size):
        covs = self.covs()
        colours = torch.sigmoid(self.colour_logits)
        opacities = torch.sigmoid(self.opacity_logit)
        return rasterise_2d(self.means, covs, colours, opacities, self.depth, image_size)
```

`log_scale`、`opacity_logit` 和 `colour_logits` 都是无约束参数，在渲染时通过合适的激活函数映射到目标空间。这是所有 3DGS 实现的标准模式。

### 第 4 步：让 2D 高斯拟合目标图像

```python
import math
import numpy as np

def make_target(size=64):
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    img = np.zeros((size, size, 3), dtype=np.float32)
    # Red circle
    mask = (xx - 20) ** 2 + (yy - 20) ** 2 < 10 ** 2
    img[mask] = [1.0, 0.2, 0.2]
    # Blue square
    mask = (np.abs(xx - 45) < 8) & (np.abs(yy - 40) < 8)
    img[mask] = [0.2, 0.3, 1.0]
    return torch.from_numpy(img)


target = make_target(64)
model = Splats2D(num_splats=64, image_size=64)
opt = torch.optim.Adam(model.parameters(), lr=0.05)

for step in range(200):
    pred = model((64, 64))
    loss = F.mse_loss(pred, target)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 40 == 0:
        print(f"step {step:3d}  mse {loss.item():.4f}")
```

经过 200 步训练，这 64 个高斯会收敛成那两个形状。这就是整个思想——对显式几何基元做梯度下降。

### 第 5 步：从 2D 到 3D

3D 扩展保持同一个循环，只增加这些内容：

1. 每个高斯的旋转由单个角度变成四元数 (quaternion)。
2. 协方差变为 `R S S^T R^T`，其中 `R` 由四元数构造，`S = diag(exp(log_scale))`。
3. 投影 `(mu, Sigma) -> (mu', Sigma')` 会使用相机外参以及在 `mu` 处透视投影的雅可比矩阵。
4. 颜色变成球谐函数展开；在观察方向上对其求值。
5. 深度排序不再使用学习到的标量，而是使用真实的相机空间 z 值。

所有生产级实现（`gsplat`、`inria/gaussian-splatting`、`nerfstudio`）都是在 GPU 上用基于瓦片的 CUDA kernel 来完成这件事。

### 第 6 步：球谐函数求值

截至 3 阶的 SH 基在每个通道上有 16 项。求值方式如下：

```python
def eval_sh_degree_3(sh_coeffs, dirs):
    """
    sh_coeffs: (..., 16, 3)   last dim is RGB channels
    dirs:      (..., 3)       unit vectors
    returns:   (..., 3)
    """
    C0 = 0.282094791773878
    C1 = 0.488602511902920
    C2 = [1.092548430592079, 1.092548430592079,
          0.315391565252520, 1.092548430592079,
          0.546274215296039]
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    x2, y2, z2 = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z

    result = C0 * sh_coeffs[..., 0, :]
    result = result - C1 * y[..., None] * sh_coeffs[..., 1, :]
    result = result + C1 * z[..., None] * sh_coeffs[..., 2, :]
    result = result - C1 * x[..., None] * sh_coeffs[..., 3, :]

    result = result + C2[0] * xy[..., None] * sh_coeffs[..., 4, :]
    result = result + C2[1] * yz[..., None] * sh_coeffs[..., 5, :]
    result = result + C2[2] * (2.0 * z2 - x2 - y2)[..., None] * sh_coeffs[..., 6, :]
    result = result + C2[3] * xz[..., None] * sh_coeffs[..., 7, :]
    result = result + C2[4] * (x2 - y2)[..., None] * sh_coeffs[..., 8, :]

    # degree 3 terms omitted here for brevity; full 16-coefficient version in the code file
    return result
```

学习到的 `sh_coeffs` 存的是这个高斯“在每个方向上的颜色”。渲染时，把它和当前视角方向一起求值，就得到一个 3 维 RGB 向量。

## 使用

真正做 3DGS 时，请使用 `gsplat`（Meta）或 `nerfstudio`：

```bash
pip install nerfstudio gsplat
ns-download-data example
ns-train splatfacto --data path/to/data
```

`splatfacto` 是 nerfstudio 的 3DGS 训练器。对于一个典型场景，在 RTX 4090 上运行通常需要 10-30 分钟。

2026 年真正重要的导出选项有：

- `.ply` —— 原始高斯云（可移植，但文件最大）。
- `.splat` —— PlayCanvas / SuperSplat 的量化格式。
- glTF `KHR_gaussian_splatting` —— Khronos 标准，可在不同查看器之间移植（2026 年 2 月 RC）。
- OpenUSD `UsdVolParticleField3DGaussianSplat` —— 原生 USD 格式，适用于 NVIDIA Omniverse 和 Vision Pro 流水线。

对于 4D / 动态场景，`4DGS` 和 `Deformable-3DGS` 通过随时间变化的均值和不透明度扩展了同一套机制。

## 交付

本课会产出：

- `outputs/prompt-3dgs-capture-planner.md` —— 一个用于规划采集会话的提示词（照片数量、相机路径、光照），适配给定场景类型。
- `outputs/skill-3dgs-export-router.md` —— 一个根据下游查看器或引擎来选择合适导出格式（`.ply` / `.splat` / glTF / USD）的 skill。

## 练习

1. **（简单）** 在另一张合成图像上运行上面的 2D splat 训练器。将 `num_splats` 在 `[16, 64, 256]` 中变化，并绘制每种情况下 MSE 随 step 的变化曲线。找出收益递减点。
2. **（中等）** 扩展 2D 栅格器，让每个高斯的 RGB 颜色可以通过一个 2 阶谐波依赖某个标量“视角”。在一对目标图像上训练，并验证模型能够同时重建两者。
3. **（困难）** 克隆 `nerfstudio`，并用你自己任意场景（桌面、植物、人脸、房间）的 20 张照片训练 `splatfacto`。导出为 glTF `KHR_gaussian_splatting`，并在查看器中打开（Three.js `GaussianSplats3D`、SuperSplat、Babylon.js V9）。报告训练时间、高斯数量和渲染 fps。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 3DGS | “Gaussian splats” | 由数百万个 3D 高斯构成的显式场景表示；每个高斯都有位置、旋转、尺度、不透明度和 SH 颜色 |
| Covariance | “高斯的形状” | `Sigma = R S S^T R^T`；表示单个高斯的朝向和各向异性尺度 |
| Alpha compositing | “从后到前混合” | 与 NeRF 的体渲染使用同一公式，只是现在作用于显式稀疏集合 |
| Densification | “克隆和拆分” | 在重建欠拟合的区域自适应地新增高斯 |
| Pruning | “删除低不透明度” | 移除那些在训练中塌缩到接近零不透明度的高斯 |
| Spherical harmonics | “视角相关颜色” | 球面上的傅里叶基；把颜色表示为观察方向的函数 |
| Splatfacto | “nerfstudio 的 3DGS” | 2026 年训练 3DGS 最简单的路径 |
| `KHR_gaussian_splatting` | “glTF 标准” | Khronos 在 2026 年推出的扩展，让 3DGS 能在不同查看器和引擎间移植 |

## 延伸阅读

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering (Kerbl et al., SIGGRAPH 2023)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) —— 原始论文
- [gsplat (Meta/nerfstudio)](https://github.com/nerfstudio-project/gsplat) —— 生产级 CUDA 栅格器
- [nerfstudio Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html) —— 参考训练配方
- [Khronos KHR_gaussian_splatting extension](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_gaussian_splatting/README.md) —— 2026 年的可移植格式
- [OpenUSD 26.03 release notes](https://openusd.org/release/) —— `UsdVolParticleField3DGaussianSplat` schema
- [THE FUTURE 3D State of Gaussian Splatting 2026](https://www.thefuture3d.com/blog-0/2026/4/4/state-of-gaussian-splatting-2026) —— 行业概览
