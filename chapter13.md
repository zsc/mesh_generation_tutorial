# 第13章：3D扩散模型基础

扩散模型作为当前最强大的生成模型之一，通过模拟物理扩散过程实现了高质量的3D网格生成。本章深入探讨扩散模型的数学原理及其在3D几何数据上的应用，包括前向噪声添加过程、逆向去噪生成过程、Score-based理论框架、3D特定的噪声调度策略以及条件控制机制。通过本章学习，读者将掌握扩散模型的核心理论，为理解DreamFusion、Point-E、Shap-E等前沿3D生成方法奠定坚实基础。

## 13.1 扩散过程的数学原理

### 13.1.1 前向扩散过程

扩散模型的核心思想源于非平衡热力学，通过定义一个逐步破坏数据结构的马尔可夫链，将复杂的数据分布转换为简单的高斯分布。

对于3D网格顶点坐标 $\mathbf{x}_0 \in \mathbb{R}^{N \times 3}$，前向扩散过程定义为：

$$q(\mathbf{x}_t|\mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I})$$

其中 $\beta_t \in (0,1)$ 控制第 $t$ 步的噪声强度。通过递推关系，可以推导出从 $\mathbf{x}_0$ 直接到 $\mathbf{x}_t$ 的边际分布：

$$q(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})$$

其中 $\alpha_t = 1-\beta_t$，$\bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s$。

重参数化形式：
$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$

这个性质使得我们可以在任意时间步直接采样，无需递归计算，大大提高了训练效率。

```
扩散过程可视化:
t=0      t=T/4     t=T/2     t=3T/4    t=T
 ╱╲       ╱ ╲       · ·       · ·      · ·
╱──╲     ╱ · ╲     · · ·     · · ·    · · ·
────     · · ·     · · · ·   · · · ·  · · · ·
结构清晰  轮廓模糊  形状消失  接近噪声  纯高斯噪声
```

### 13.1.2 逆向去噪过程

逆向过程通过参数化的神经网络学习条件分布：

$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

训练目标是最小化变分下界（ELBO）：

$$\mathcal{L} = \mathbb{E}_q \left[ \sum_{t>1} D_{KL}(q(x_{t-1}|x_t,x_0) || p_\theta(x_{t-1}|x_t)) + \log p_\theta(x_0|x_1) \right]$$

### 13.1.3 噪声预测参数化

实践中，通常采用噪声预测的参数化方式。神经网络 $\epsilon_\theta(x_t, t)$ 预测添加的噪声：

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t)\right)$$

训练损失简化为：

$$\mathcal{L}_{simple} = \mathbb{E}_{t,x_0,\epsilon} \left[ ||\epsilon - \epsilon_\theta(x_t, t)||^2 \right]$$

其中 $\epsilon \sim \mathcal{N}(0, I)$，$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$。

### 13.1.4 3D数据的特殊考虑

对于3D网格数据，需要考虑以下特殊性：

1. **旋转不变性**：3D物体的表示应该对旋转具有不变性或等变性
2. **尺度归一化**：不同网格的尺度差异需要预处理
3. **中心化**：将网格中心移至原点
4. **拓扑保持**：扩散过程中如何维护网格的拓扑结构

## 13.2 Score-based生成模型

### 13.2.1 Score函数与扩散的联系

Score函数定义为对数概率密度的梯度：

$$s(x, t) = \nabla_x \log p_t(x)$$

在扩散模型框架下，score函数与噪声预测网络存在直接关系：

$$s_\theta(x_t, t) = -\frac{1}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t)$$

### 13.2.2 连续时间扩散（SDE视角）

将离散时间扩散推广到连续时间，前向过程可表示为随机微分方程（SDE）：

$$dx = f(x, t)dt + g(t)dw$$

其中 $f(x, t) = -\frac{1}{2}\beta(t)x$，$g(t) = \sqrt{\beta(t)}$，$w$ 是标准布朗运动。

对应的逆向SDE为：

$$dx = [f(x, t) - g(t)^2 \nabla_x \log p_t(x)]dt + g(t)d\bar{w}$$

### 13.2.3 概率流ODE

去除随机项，得到确定性的概率流ODE：

$$dx = \left[f(x, t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)\right]dt$$

这个ODE的解与SDE具有相同的边际分布，但轨迹是确定的，便于快速采样。

### 13.2.4 Score matching训练

Score matching的目标是最小化：

$$\mathcal{L}_{score} = \mathbb{E}_{t,x_0,x_t} \left[ \lambda(t) ||s_\theta(x_t, t) - \nabla_{x_t} \log q(x_t|x_0)||^2 \right]$$

其中 $\lambda(t)$ 是时间相关的权重函数。对于3D网格，可以根据几何特征设计自适应的权重。

## 13.3 3D数据的噪声调度

### 13.3.1 线性调度与余弦调度

经典的线性调度：
$$\beta_t = \beta_{min} + \frac{t-1}{T-1}(\beta_{max} - \beta_{min})$$

余弦调度提供更平滑的信噪比变化：
$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2$$

### 13.3.2 几何感知的噪声调度

对于3D网格，可以设计几何感知的噪声调度：

1. **多尺度调度**：对不同频率的几何特征使用不同的噪声强度
   $$\beta_t^{(k)} = \beta_t \cdot w_k$$
   其中 $w_k$ 是第 $k$ 个特征频带的权重

2. **自适应调度**：基于局部曲率调整噪声强度
   $$\beta_t(v_i) = \beta_t \cdot (1 + \lambda \cdot \kappa_i)$$
   其中 $\kappa_i$ 是顶点 $v_i$ 处的平均曲率

### 13.3.3 拓扑保持的噪声策略

为保持网格拓扑，可以采用以下策略：

1. **约束噪声**：限制噪声在切空间内
   $$x_t = x_0 + P_{T(x_0)} \epsilon$$
   其中 $P_{T(x_0)}$ 是到切空间的投影算子

2. **边长保持**：通过拉格朗日乘子维护边长约束
   $$\min_x ||x - x_{noisy}||^2 + \lambda \sum_{(i,j) \in E} (||x_i - x_j|| - l_{ij})^2$$

```
噪声调度对比图:
信噪比
  ↑
1.0│●                    线性调度
   │ ●●                  
   │   ●●●               余弦调度
0.5│      ●●●●           ....
   │          ●●●●●      
   │              ●●●●●● 几何感知调度
0.0└────────────────────→
   0                    T 时间步
```

### 13.3.4 采样加速技术

1. **DDIM采样**：确定性隐式扩散模型
   $$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\epsilon_\theta(x_t, t) + \sigma_t z$$

2. **步数自适应**：根据局部复杂度动态调整采样步数

3. **预测器-校正器方法**：结合Langevin动力学进行局部细化

## 13.4 条件生成与引导

### 13.4.1 条件扩散模型

条件生成的目标是学习 $p(x|y)$，其中 $y$ 是条件信息（如类别标签、文本描述、部分几何等）。

条件score函数：
$$\nabla_x \log p(x_t|y) = \nabla_x \log p(x_t) + \nabla_x \log p(y|x_t)$$

### 13.4.2 Classifier引导

使用预训练的分类器 $p_\phi(y|x_t)$ 引导生成：

$$\tilde{s}_\theta(x_t, t, y) = s_\theta(x_t, t) + w \cdot \nabla_{x_t} \log p_\phi(y|x_t)$$

其中 $w$ 是引导强度。

### 13.4.3 Classifier-free引导

无需额外分类器，通过联合训练条件和无条件模型：

$$\tilde{\epsilon}_\theta(x_t, t, y) = (1+w) \cdot \epsilon_\theta(x_t, t, y) - w \cdot \epsilon_\theta(x_t, t, \emptyset)$$

训练时随机dropout条件信息（通常概率为10%）。

### 13.4.4 3D特定的条件模态

1. **视图条件**：给定一个或多个2D视图生成3D网格
   - 使用可微渲染计算视图一致性损失
   - 多视图聚合策略

2. **部分几何条件**：补全残缺网格
   - 掩码策略：$x_t^{masked} = m \odot x_t^{known} + (1-m) \odot x_t^{unknown}$
   - 边界条件处理

3. **语义条件**：基于语义分割或功能描述
   - 分层条件编码
   - 注意力机制整合

4. **物理约束条件**：满足特定物理属性
   - 软约束通过损失函数
   - 硬约束通过投影算子

### 13.4.5 多模态融合策略

组合多种条件信息：

$$s_{combined} = s_{unconditional} + \sum_{i} w_i \cdot \nabla_x \log p(c_i|x)$$

权重 $w_i$ 可以是固定的或学习得到的。

```
条件引导示意图:
     无条件生成              条件引导后
      · · · ·               ╱───╲
     · · · · ·             ╱     ╲
    · · · · · ·    +      │  椅子  │     =    精确的椅子形状
     · · · · ·             ╲     ╱             ╱├─╲
      · · · ·               ╲───╱             ╱ │  ╲
       噪声               条件信息            生成结果
```

## 本章小结

本章系统介绍了3D扩散模型的理论基础：

1. **扩散过程数学原理**：
   - 前向扩散：$q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$
   - 逆向去噪：$p_\theta(x_{t-1}|x_t)$ 通过神经网络参数化
   - 训练目标：最小化噪声预测误差 $||\epsilon - \epsilon_\theta(x_t, t)||^2$

2. **Score-based模型**：
   - Score函数：$s(x, t) = \nabla_x \log p_t(x)$
   - SDE/ODE统一框架
   - Score matching训练策略

3. **3D噪声调度**：
   - 几何感知调度
   - 拓扑保持策略
   - 采样加速技术（DDIM、自适应步数）

4. **条件生成机制**：
   - Classifier引导：$\tilde{s} = s + w \cdot \nabla \log p(y|x)$
   - Classifier-free引导
   - 3D特定条件（视图、部分几何、语义、物理约束）

掌握这些基础理论是理解和实现高质量3D网格生成的关键。

## 练习题

### 基础题

**练习13.1** 推导前向扩散过程的边际分布
给定前向过程 $q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$，证明：
$$q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$$

*Hint*：使用重参数化技巧和高斯分布的叠加性质。

<details>
<summary>答案</summary>

从 $x_0$ 开始，逐步应用前向过程：
- $x_1 = \sqrt{\alpha_1}x_0 + \sqrt{1-\alpha_1}\epsilon_1$
- $x_2 = \sqrt{\alpha_2}x_1 + \sqrt{1-\alpha_2}\epsilon_2 = \sqrt{\alpha_2\alpha_1}x_0 + \sqrt{\alpha_2(1-\alpha_1)}\epsilon_1 + \sqrt{1-\alpha_2}\epsilon_2$

通过归纳可得：
$$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\bar{\epsilon}$$

其中 $\bar{\epsilon} \sim \mathcal{N}(0, I)$，因为独立高斯噪声的线性组合仍是高斯分布。
方差项：$\alpha_2(1-\alpha_1) + (1-\alpha_2) = 1 - \alpha_2\alpha_1 = 1 - \bar{\alpha}_2$

因此 $q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t}I)$。
</details>

**练习13.2** Score函数计算
对于3D点云 $x \in \mathbb{R}^{N \times 3}$，如果 $p(x) \propto \exp(-||x - \mu||^2 / 2\sigma^2)$，计算score函数 $\nabla_x \log p(x)$。

*Hint*：先计算对数概率，然后求梯度。

<details>
<summary>答案</summary>

$$\log p(x) = -\frac{||x - \mu||^2}{2\sigma^2} + C$$

其中C是归一化常数（与x无关）。

求梯度：
$$\nabla_x \log p(x) = -\frac{1}{\sigma^2}(x - \mu)$$

这个结果表明，score函数指向概率密度增加最快的方向，即从当前点指向均值的方向。
对于3D点云的每个点，score是一个3维向量。
</details>

**练习13.3** DDIM采样步骤推导
给定DDIM的更新公式，证明当 $\sigma_t = 0$ 时，采样过程是确定性的。

*Hint*：分析DDIM公式中的随机项。

<details>
<summary>答案</summary>

DDIM更新公式：
$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\epsilon_\theta(x_t, t) + \sigma_t z$$

当 $\sigma_t = 0$ 时：
$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}}\epsilon_\theta(x_t, t)$$

其中 $\hat{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t, t))$

代入得完全确定性的更新规则，不含随机项z。这使得生成过程可逆，便于插值和编辑。
</details>

**练习13.4** 余弦噪声调度的信噪比
计算余弦调度在 $t = T/2$ 时的信噪比（SNR），并与线性调度比较。

*Hint*：SNR定义为 $\bar{\alpha}_t / (1-\bar{\alpha}_t)$。

<details>
<summary>答案</summary>

余弦调度：
$$\bar{\alpha}_{T/2} = \cos^2\left(\frac{\pi/4 + s\pi/2}{1 + s}\right)$$

取 $s = 0.008$（常用值）：
$$\bar{\alpha}_{T/2} \approx \cos^2(0.78) \approx 0.52$$
$$SNR_{cos} = \frac{0.52}{0.48} \approx 1.08$$

线性调度（$\beta_t \in [0.0001, 0.02]$）：
$$\bar{\alpha}_{T/2} \approx 0.98 \times 0.97 \times ... \approx 0.15$$
$$SNR_{linear} = \frac{0.15}{0.85} \approx 0.18$$

余弦调度在中间时刻保持更高的信噪比，有助于保留更多结构信息。
</details>

### 挑战题

**练习13.5** 几何感知噪声设计
设计一个基于网格局部特征（曲率、面积）的自适应噪声调度方案。要求：
1. 高曲率区域噪声较小
2. 保持全局噪声水平不变
3. 给出数学公式和归一化方法

*Hint*：考虑使用加权平均和softmax归一化。

<details>
<summary>答案</summary>

设顶点 $i$ 的高斯曲率为 $K_i$，平均曲率为 $H_i$，定义局部几何复杂度：
$$g_i = \sqrt{K_i^2 + H_i^2}$$

归一化权重：
$$w_i = \frac{\exp(-\lambda g_i)}{\sum_j \exp(-\lambda g_j)}$$

自适应噪声强度：
$$\beta_t^{(i)} = \beta_t \cdot (1 - \alpha w_i + \alpha)$$

其中 $\alpha \in [0, 1]$ 控制自适应程度。

为保持全局噪声水平，需要满足：
$$\frac{1}{N}\sum_i \beta_t^{(i)} = \beta_t$$

这可以通过调整归一化因子实现：
$$\beta_t^{(i)} = \beta_t \cdot N \cdot w_i$$

实际应用时，可以对每个局部patch计算平均权重，避免顶点级别的过度细化。
</details>

**练习13.6** Classifier-free引导的最优权重
分析classifier-free引导中权重 $w$ 对生成质量和多样性的影响，推导最优权重的理论界限。

*Hint*：考虑引导强度与KL散度的关系。

<details>
<summary>答案</summary>

Classifier-free引导的有效score：
$$\tilde{s} = (1+w)s_{cond} - ws_{uncond} = s_{uncond} + (1+w)(s_{cond} - s_{uncond})$$

这等价于从修改后的分布采样：
$$\tilde{p}(x|y) \propto p(x)p(y|x)^{1+w}$$

KL散度分析：
$$KL(\tilde{p} || p_{true}) = (1+w)E_{\tilde{p}}[\log p(y|x)] - \log Z_w$$

最优权重满足：
$$\frac{\partial KL}{\partial w} = E_{\tilde{p}}[\log p(y|x)] - \frac{\partial \log Z_w}{\partial w} = 0$$

实践中的经验范围：
- $w \in [0.5, 3.0]$：平衡质量和多样性
- $w > 3.0$：高质量但可能过拟合条件
- $w < 0.5$：保持多样性但条件遵循度降低

理论上界：$w_{max} = \frac{\log p_{max} - \log p_{mean}}{\sigma_{log p}}$，
其中 $p_{max}, p_{mean}, \sigma_{log p}$ 是条件概率的统计量。
</details>

**练习13.7** 拓扑保持的扩散采样
设计一个保持网格拓扑不变的扩散采样算法。要求：
1. 防止自相交
2. 保持亏格不变
3. 给出算法伪代码

*Hint*：使用投影方法和拓扑检查。

<details>
<summary>答案</summary>

算法：拓扑保持扩散采样

```
输入: 初始噪声 x_T, 训练好的 ε_θ, 目标拓扑 T_target
输出: 生成的网格 x_0

for t = T to 1:
    # 标准扩散步骤
    ε = ε_θ(x_t, t)
    x̂_0 = (x_t - √(1-ᾱ_t)ε) / √ᾱ_t
    x_{t-1}^{raw} = sample_step(x_t, x̂_0, t)
    
    # 拓扑修正
    x_{t-1} = x_{t-1}^{raw}
    
    # 1. 防止自相交
    intersections = detect_self_intersections(x_{t-1})
    if intersections:
        x_{t-1} = resolve_intersections(x_{t-1}, intersections)
    
    # 2. 边长约束（防止退化）
    for edge (i,j):
        if ||x_{t-1}[i] - x_{t-1}[j]|| < ε_min:
            x_{t-1} = project_to_min_edge_length(x_{t-1}, i, j)
    
    # 3. 拓扑检查
    if compute_genus(x_{t-1}) ≠ T_target.genus:
        x_{t-1} = project_to_topology_preserving_space(x_{t-1}, x_t)
    
    # 4. 平滑投影（可选）
    x_{t-1} = λ * x_{t-1} + (1-λ) * smooth_laplacian(x_{t-1})

return x_0
```

关键函数实现思路：
- `detect_self_intersections`: 使用BVH加速的三角形相交测试
- `resolve_intersections`: 通过局部顶点位移消除相交
- `project_to_topology_preserving_space`: 最小化 ||x - x_raw||² 同时保持拓扑
- `smooth_laplacian`: 应用拉普拉斯平滑但保持特征
</details>

**练习13.8** 多尺度扩散模型
设计一个多分辨率的3D网格扩散模型，能够逐步细化几何细节。

*Hint*：考虑金字塔结构和渐进式生成。

<details>
<summary>答案</summary>

多尺度扩散架构：

1. **网格层次构建**：
   - Level 0: 粗网格 (如100顶点)
   - Level 1: 中等网格 (如500顶点)  
   - Level 2: 精细网格 (如2000顶点)
   
   使用QEM简化或细分得到不同层级。

2. **分层扩散过程**：
   ```
   # 从粗到细生成
   x_0^{(0)} ~ p_θ₀(x)  # 生成粗网格
   
   for level = 1 to L:
       # 上采样
       x_init^{(level)} = upsample(x_0^{(level-1)})
       
       # 条件扩散
       x_T^{(level)} = x_init^{(level)} + noise
       x_0^{(level)} ~ p_θₗ(x | x_0^{(level-1)})
   ```

3. **条件编码**：
   $$\epsilon_\theta^{(l)}(x_t^{(l)}, t, x_0^{(l-1)}) = MLP([x_t^{(l)}, encode(x_0^{(l-1)}), t])$$

4. **损失函数**：
   $$\mathcal{L} = \sum_{l=0}^L w_l \mathbb{E}[||\epsilon - \epsilon_\theta^{(l)}(x_t^{(l)}, t, c^{(l-1)})||^2]$$
   
   其中权重 $w_l = 2^l$ 给精细层级更高权重。

5. **优势**：
   - 稳定的粗结构
   - 渐进式细节添加
   - 计算效率高（粗层级快速收敛）
   - 可控的细节程度
</details>

## 常见陷阱与错误 (Gotchas)

### 1. 数值稳定性问题

**问题**：当 $t \to T$ 时，$\bar{\alpha}_t \to 0$，导致除零或数值不稳定。

**解决方案**：
- 使用对数空间计算：$\log \bar{\alpha}_t$ 而非 $\bar{\alpha}_t$
- 设置最小值阈值：$\bar{\alpha}_t = \max(\bar{\alpha}_t, 1e-8)$
- 重参数化避免直接除法

### 2. 噪声调度不当

**问题**：线性调度在3D数据上可能过早破坏结构。

**调试技巧**：
- 可视化不同时间步的 $x_t$
- 监控信噪比曲线
- 使用余弦或sigmoid调度作为起点

### 3. 条件泄漏

**问题**：条件信息通过捷径传递，模型未真正学习条件生成。

**检查方法**：
- 测试时使用未见过的条件
- 检查无条件生成质量
- 分析中间激活的条件依赖性

### 4. 采样速度与质量权衡

**错误做法**：盲目减少采样步数。

**正确做法**：
- 使用DDIM等确定性采样器
- 实现自适应步长
- 考虑知识蒸馏到few-step模型

### 5. 3D旋转等变性

**问题**：生成的3D形状对输入方向敏感。

**解决方案**：
- 数据增强：训练时随机旋转
- 使用SO(3)等变网络架构
- 正则对齐到标准坐标系

### 6. 内存爆炸

**问题**：3D数据维度高，批量训练容易OOM。

**优化策略**：
- 梯度累积
- 混合精度训练
- 分patch处理大网格

### 7. 评估指标误导

**陷阱**：Chamfer距离低不代表视觉质量好。

**全面评估**：
- 多个指标组合
- 人工评估
- 下游任务性能

### 8. 模式坍塌

**症状**：生成样本缺乏多样性。

**诊断与修复**：
- 检查KL散度
- 调整噪声调度
- 使用更强的正则化