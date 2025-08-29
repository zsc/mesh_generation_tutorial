# 第13章：3D扩散模型基础

扩散模型作为当前最强大的生成模型之一，通过模拟物理扩散过程实现了高质量的3D网格生成。本章深入探讨扩散模型的数学原理及其在3D几何数据上的应用，包括前向噪声添加过程、逆向去噪生成过程、Score-based理论框架、3D特定的噪声调度策略以及条件控制机制。通过本章学习，读者将掌握扩散模型的核心理论，为理解DreamFusion、Point-E、Shap-E等前沿3D生成方法奠定坚实基础。

## 13.1 扩散过程的数学原理

### 13.1.1 前向扩散过程

扩散模型的核心思想源于非平衡热力学，通过定义一个逐步破坏数据结构的马尔可夫链，将复杂的数据分布转换为简单的高斯分布。这个过程模拟了物理世界中的布朗运动，粒子从有序状态逐渐扩散到无序状态。

对于3D网格顶点坐标 $\mathbf{x}_0 \in \mathbb{R}^{N \times 3}$，前向扩散过程定义为：

$$q(\mathbf{x}_t|\mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I})$$

其中 $\beta_t \in (0,1)$ 控制第 $t$ 步的噪声强度，通常满足 $\beta_1 < \beta_2 < ... < \beta_T$，保证信息逐渐丢失。整个前向过程形成马尔可夫链：

$$q(\mathbf{x}_{1:T}|\mathbf{x}_0) = \prod_{t=1}^{T} q(\mathbf{x}_t|\mathbf{x}_{t-1})$$

通过递推关系，可以推导出从 $\mathbf{x}_0$ 直接到 $\mathbf{x}_t$ 的边际分布。令 $\alpha_t = 1-\beta_t$，$\bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s$，则有：

$$q(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})$$

**推导过程**：利用高斯分布的可加性，从 $\mathbf{x}_0$ 到 $\mathbf{x}_1$：
$$\mathbf{x}_1 = \sqrt{\alpha_1}\mathbf{x}_0 + \sqrt{1-\alpha_1}\boldsymbol{\epsilon}_1$$

从 $\mathbf{x}_1$ 到 $\mathbf{x}_2$：
$$\mathbf{x}_2 = \sqrt{\alpha_2}\mathbf{x}_1 + \sqrt{1-\alpha_2}\boldsymbol{\epsilon}_2 = \sqrt{\alpha_2\alpha_1}\mathbf{x}_0 + \sqrt{\alpha_2(1-\alpha_1)}\boldsymbol{\epsilon}_1 + \sqrt{1-\alpha_2}\boldsymbol{\epsilon}_2$$

由于 $\boldsymbol{\epsilon}_1$ 和 $\boldsymbol{\epsilon}_2$ 独立，合并后的噪声仍为高斯分布，方差为：
$$\alpha_2(1-\alpha_1) + (1-\alpha_2) = 1 - \alpha_2\alpha_1 = 1 - \bar{\alpha}_2$$

重参数化形式提供了高效的采样方式：
$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$

这个性质使得我们可以在任意时间步直接采样，无需递归计算，大大提高了训练效率。对于3D网格，这意味着每个顶点独立地添加噪声，但噪声强度在全局保持一致。

**信噪比分析**：定义信噪比 $\text{SNR}(t) = \bar{\alpha}_t / (1-\bar{\alpha}_t)$，随着 $t$ 增加，SNR单调递减：
- $t=0$: $\text{SNR}(0) = \infty$（纯信号）
- $t=T$: $\text{SNR}(T) \approx 0$（纯噪声）

```
扩散过程可视化:
t=0      t=T/4     t=T/2     t=3T/4    t=T
 ╱╲       ╱ ╲       · ·       · ·      · ·
╱──╲     ╱ · ╲     · · ·     · · ·    · · ·
────     · · ·     · · · ·   · · · ·  · · · ·
结构清晰  轮廓模糊  形状消失  接近噪声  纯高斯噪声
SNR=∞    SNR≈10    SNR≈1     SNR≈0.1   SNR≈0
```

### 13.1.2 逆向去噪过程

逆向过程是扩散模型的生成阶段，目标是从噪声 $\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$ 逐步恢复到原始数据分布 $p(\mathbf{x}_0)$。理论上，如果知道真实的后验分布 $q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)$，就可以完美逆转扩散过程。

**后验分布的解析形式**：利用贝叶斯定理和高斯分布的性质，可以推导出：

$$q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t\mathbf{I})$$

其中均值和方差为：
$$\tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{x}_t$$

$$\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \cdot \beta_t$$

实际中，我们无法访问 $\mathbf{x}_0$，因此通过参数化的神经网络学习逆向分布：

$$p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_t, t), \Sigma_\theta(\mathbf{x}_t, t))$$

**变分下界（ELBO）**：训练目标是最大化对数似然的下界：

$$\log p(\mathbf{x}_0) \geq \mathbb{E}_q \left[ \log p(\mathbf{x}_T) - \sum_{t=1}^{T} D_{KL}(q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0) || p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)) \right]$$

将损失函数分解为三部分：
$$\mathcal{L} = \underbrace{D_{KL}(q(\mathbf{x}_T|\mathbf{x}_0) || p(\mathbf{x}_T))}_{L_T} + \sum_{t>1} \underbrace{D_{KL}(q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0) || p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t))}_{L_{t-1}} - \underbrace{\log p_\theta(\mathbf{x}_0|\mathbf{x}_1)}_{L_0}$$

其中 $L_T$ 在扩散过程充分长时接近零（$\mathbf{x}_T$ 接近标准高斯），$L_0$ 是重建项，$L_{t-1}$ 是去噪匹配项。

### 13.1.3 噪声预测参数化

Ho等人(2020)提出了一个关键洞察：与其直接预测 $\mu_\theta$，不如让网络预测添加的噪声 $\epsilon$，这大大简化了训练过程。

**参数化选择**：由于 $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$，可以表示：
$$\mathbf{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\epsilon)$$

将此代入后验均值公式：
$$\mu_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(\mathbf{x}_t, t)\right)$$

其中 $\epsilon_\theta(\mathbf{x}_t, t)$ 是神经网络，预测在时间步 $t$ 添加到 $\mathbf{x}_0$ 的噪声。

**简化的训练损失**：通过重参数化技巧，KL散度可以简化为：

$$\mathcal{L}_{simple} = \mathbb{E}_{t \sim \mathcal{U}(1,T), \mathbf{x}_0, \epsilon} \left[ ||\epsilon - \epsilon_\theta(\mathbf{x}_t, t)||^2 \right]$$

训练算法极其简洁：
1. 采样时间步：$t \sim \text{Uniform}(1, T)$
2. 采样噪声：$\epsilon \sim \mathcal{N}(0, \mathbf{I})$
3. 构造噪声样本：$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$
4. 优化损失：$||\epsilon - \epsilon_\theta(\mathbf{x}_t, t)||^2$

**方差的处理**：对于方差 $\Sigma_\theta(\mathbf{x}_t, t)$，常见做法：
- 固定为 $\tilde{\beta}_t\mathbf{I}$（DDPM）
- 固定为 $\beta_t\mathbf{I}$（简化版本）
- 学习插值系数：$\Sigma_\theta = \exp(v_\theta \log \beta_t + (1-v_\theta) \log \tilde{\beta}_t)$

**3D网格的特殊考虑**：
- **坐标归一化**：将顶点坐标归一化到 $[-1, 1]^3$ 内
- **批归一化**：对每个网格独立归一化，保持形状不变性
- **加权损失**：根据顶点重要性（如曲率、面积）加权：
  $$\mathcal{L}_{weighted} = \mathbb{E} \left[ \sum_{i=1}^{N} w_i ||\epsilon_i - \epsilon_\theta(\mathbf{x}_t, t)_i||^2 \right]$$

### 13.1.4 3D数据的特殊考虑

3D网格数据具有独特的几何和拓扑性质，在应用扩散模型时需要特殊处理以保证生成质量和几何合理性。

**1. 旋转不变性与等变性**

3D物体在不同方向观察应该是同一物体，因此模型需要处理旋转对称性：

- **数据增强**：训练时对网格施加随机SO(3)旋转：
  $$\mathbf{x}_0^{aug} = \mathbf{R} \cdot \mathbf{x}_0, \quad \mathbf{R} \in SO(3)$$
  
- **等变网络架构**：使用SE(3)等变的网络层，如Vector Neurons或Tensor Field Networks

- **规范化对齐**：通过PCA或其他方法将网格对齐到规范坐标系：
  $$\mathbf{C} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{v}_i \mathbf{v}_i^T$$
  主轴由 $\mathbf{C}$ 的特征向量给出

**2. 尺度与平移归一化**

不同来源的网格尺度差异巨大，需要标准化处理：

- **中心化**：将质心移至原点
  $$\mathbf{v}_i' = \mathbf{v}_i - \frac{1}{N}\sum_{j=1}^{N} \mathbf{v}_j$$

- **尺度归一化**：常见方法包括：
  - 单位球归一化：$\max_i ||\mathbf{v}_i|| = 1$
  - 单位方差归一化：$\text{Var}(\mathbf{v}) = 1$
  - 包围盒归一化：缩放到 $[-1, 1]^3$

- **保持纵横比**：使用统一缩放因子
  $$s = \frac{1}{\max(\text{bbox}_x, \text{bbox}_y, \text{bbox}_z)}$$

**3. 拓扑保持策略**

扩散过程可能破坏网格的拓扑结构，导致自相交、非流形等问题：

- **拓扑正则化损失**：
  $$\mathcal{L}_{topo} = \lambda_1 \mathcal{L}_{edge} + \lambda_2 \mathcal{L}_{angle} + \lambda_3 \mathcal{L}_{manifold}$$
  
  其中：
  - $\mathcal{L}_{edge}$：边长保持损失
  - $\mathcal{L}_{angle}$：二面角保持损失
  - $\mathcal{L}_{manifold}$：流形约束损失

- **分层扩散**：先生成粗糙拓扑，再细化几何
  $$\mathbf{x}_0^{coarse} \rightarrow \mathbf{x}_0^{medium} \rightarrow \mathbf{x}_0^{fine}$$

- **隐式表示中介**：通过SDF或占据场作为中间表示，保证水密性

**4. 网格连接性处理**

网格不仅包含顶点位置，还有面片连接信息：

- **固定拓扑**：保持面片连接不变，只改变顶点位置
  - 适用于同拓扑的形状变形
  - 可使用图神经网络编码连接信息

- **动态拓扑**：同时生成顶点和面片
  - 使用PolyGen类序列生成
  - 或通过隐式场提取

- **混合表示**：
  $$\mathbf{x}_t = [\mathbf{V}_t, \mathbf{F}_{embed}]$$
  其中 $\mathbf{V}_t$ 是扩散的顶点，$\mathbf{F}_{embed}$ 是面片的学习嵌入

**5. 采样效率优化**

3D数据维度高（$N \times 3$），需要特殊优化：

- **分块处理**：将大网格分成局部patches
- **自适应采样**：根据局部复杂度调整噪声步数
- **层级生成**：从低分辨率到高分辨率逐步生成

## 13.2 Score-based生成模型

### 13.2.1 Score函数与扩散的联系

Score-based生成模型提供了扩散模型的另一种理论视角，通过估计数据分布的score函数（对数概率的梯度）来生成样本。这个框架由Song和Ermon(2019)提出，后来与扩散模型统一。

**Score函数的定义与性质**

Score函数定义为对数概率密度的梯度：
$$\mathbf{s}(\mathbf{x}, t) = \nabla_\mathbf{x} \log p_t(\mathbf{x})$$

Score函数具有重要性质：
- **无需归一化常数**：$\nabla_\mathbf{x} \log p(\mathbf{x}) = \nabla_\mathbf{x} \log \tilde{p}(\mathbf{x})$，其中 $\tilde{p}$ 是未归一化分布
- **指向高概率方向**：score指向概率密度增加最快的方向
- **期望为零**：$\mathbb{E}_{p(\mathbf{x})}[\mathbf{s}(\mathbf{x})] = 0$（Stein恒等式）

**扩散过程的Score**

对于扩散过程 $q(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})$，其score函数为：

$$\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t|\mathbf{x}_0) = -\frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0}{1-\bar{\alpha}_t} = -\frac{\boldsymbol{\epsilon}}{\sqrt{1-\bar{\alpha}_t}}$$

这建立了score函数与噪声的直接联系。

**Score与噪声预测的等价性**

扩散模型中的噪声预测网络 $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ 与score网络 $\mathbf{s}_\theta(\mathbf{x}_t, t)$ 存在简单关系：

$$\mathbf{s}_\theta(\mathbf{x}_t, t) = -\frac{1}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$$

这意味着：
- 训练噪声预测网络等价于训练score网络
- DDPM的损失函数可以理解为score matching损失
- 两种视角可以互换使用

**Tweedie公式与去噪**

Tweedie公式提供了从噪声观测估计原始信号的方法：

$$\mathbb{E}[\mathbf{x}_0|\mathbf{x}_t] = \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t + (1-\bar{\alpha}_t)\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t))$$

将score函数代入：
$$\hat{\mathbf{x}}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t))$$

这正是扩散模型中用于预测 $\mathbf{x}_0$ 的公式，展示了两个框架的深层联系。

### 13.2.2 连续时间扩散（SDE视角）

Song等人(2021)提出了基于随机微分方程(SDE)的统一框架，将离散时间扩散推广到连续时间，提供了更灵活的理论工具。

**前向SDE**

连续时间扩散过程可以描述为：
$$d\mathbf{x} = \mathbf{f}(\mathbf{x}, t)dt + g(t)d\mathbf{w}$$

其中：
- $\mathbf{f}(\mathbf{x}, t)$：漂移系数（drift）
- $g(t)$：扩散系数（diffusion）
- $\mathbf{w}$：标准维纳过程（布朗运动）

对于variance preserving (VP) SDE：
$$\mathbf{f}(\mathbf{x}, t) = -\frac{1}{2}\beta(t)\mathbf{x}, \quad g(t) = \sqrt{\beta(t)}$$

对于variance exploding (VE) SDE：
$$\mathbf{f}(\mathbf{x}, t) = \mathbf{0}, \quad g(t) = \sqrt{\frac{d[\sigma^2(t)]}{dt}}$$

**边际分布**

VP-SDE的边际分布为：
$$p_{0t}(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \mathbf{x}_0e^{-\frac{1}{2}\int_0^t \beta(s)ds}, \mathbf{I}(1-e^{-\int_0^t \beta(s)ds}))$$

这与离散DDPM在连续极限下一致。

**逆向SDE**

Anderson(1982)证明了前向SDE存在对应的逆向SDE：
$$d\mathbf{x} = [\mathbf{f}(\mathbf{x}, t) - g(t)^2 \nabla_\mathbf{x} \log p_t(\mathbf{x})]dt + g(t)d\bar{\mathbf{w}}$$

其中 $\bar{\mathbf{w}}$ 是逆向时间的布朗运动。这个公式的关键是score函数 $\nabla_\mathbf{x} \log p_t(\mathbf{x})$。

**数值求解**

使用Euler-Maruyama方法离散化：
$$\mathbf{x}_{t-\Delta t} = \mathbf{x}_t - [\mathbf{f}(\mathbf{x}_t, t) - g(t)^2 \mathbf{s}_\theta(\mathbf{x}_t, t)]\Delta t + g(t)\sqrt{\Delta t}\mathbf{z}$$

其中 $\mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$。

**SDE的优势**

1. **统一框架**：DDPM、SMLD、DDIM等都是特例
2. **灵活性**：可设计新的SDE对
3. **理论工具**：可借用随机分析的成熟理论
4. **可控生成**：通过修改drift项实现条件生成

### 13.2.3 概率流ODE

通过去除SDE中的随机项，可以得到确定性的常微分方程(ODE)，其边际分布与原SDE相同。

**概率流ODE推导**

对于任意前向SDE，存在对应的概率流ODE：
$$\frac{d\mathbf{x}}{dt} = \mathbf{f}(\mathbf{x}, t) - \frac{1}{2}g(t)^2 \nabla_\mathbf{x} \log p_t(\mathbf{x})$$

这个ODE的特点：
- **确定性**：给定初始条件，轨迹唯一确定
- **可逆性**：可以精确重建编码过程
- **边际分布相同**：$p_t(\mathbf{x})$ 与SDE一致

**与神经ODE的联系**

概率流ODE可以看作神经ODE的特例：
$$\frac{d\mathbf{x}}{dt} = f_\theta(\mathbf{x}, t)$$

其中 $f_\theta = \mathbf{f} - \frac{1}{2}g^2 \mathbf{s}_\theta$。

**快速采样**

ODE求解器通常比SDE更高效：
- **高阶方法**：RK45、DPM-Solver等
- **自适应步长**：根据局部误差调整
- **并行化**：批量ODE求解

对于3D网格生成，ODE特别有用：
1. **插值**：在隐空间进行平滑插值
2. **编辑**：精确控制生成过程
3. **压缩**：将网格编码为隐变量

**数值求解器选择**

不同求解器的权衡：
- **Euler方法**：简单但需要小步长
- **Heun方法**：二阶精度，适中复杂度
- **RK45**：自适应高精度，但计算量大
- **DPM-Solver**：专为扩散模型设计，效率高

### 13.2.4 Score matching训练

Score matching提供了直接训练score函数的方法，无需知道归一化常数。

**Denoising Score Matching (DSM)**

Vincent(2011)提出的去噪score matching：
$$\mathcal{L}_{DSM} = \mathbb{E}_{\mathbf{x}_0 \sim p_{data}} \mathbb{E}_{\tilde{\mathbf{x}} \sim q(\tilde{\mathbf{x}}|\mathbf{x}_0)} \left[ ||\mathbf{s}_\theta(\tilde{\mathbf{x}}, \sigma) - \nabla_{\tilde{\mathbf{x}}} \log q(\tilde{\mathbf{x}}|\mathbf{x}_0)||^2 \right]$$

其中 $q(\tilde{\mathbf{x}}|\mathbf{x}_0) = \mathcal{N}(\tilde{\mathbf{x}}; \mathbf{x}_0, \sigma^2\mathbf{I})$。

**时间相关的Score Matching**

对于扩散过程，需要匹配所有时间的score：
$$\mathcal{L}_{score} = \mathbb{E}_{t \sim \mathcal{U}(0,T)} \mathbb{E}_{\mathbf{x}_0, \mathbf{x}_t} \left[ \lambda(t) ||\mathbf{s}_\theta(\mathbf{x}_t, t) - \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t|\mathbf{x}_0)||^2 \right]$$

**权重函数设计**

$\lambda(t)$ 的选择影响训练效果：

1. **常数权重**：$\lambda(t) = 1$
   - 简单但可能不平衡

2. **信噪比权重**：$\lambda(t) = \text{SNR}(t) = \bar{\alpha}_t/(1-\bar{\alpha}_t)$
   - 平衡不同噪声水平的贡献

3. **最小方差权重**：$\lambda(t) = g(t)^2$
   - 理论最优但实践中需调整

4. **几何感知权重**（3D特定）：
   $$\lambda(t, \mathbf{x}) = \lambda_{base}(t) \cdot \exp(-\gamma \cdot \text{curvature}(\mathbf{x}))$$
   - 高曲率区域给予更多关注

**Sliced Score Matching (SSM)**

对于高维3D数据，可使用切片score matching降低计算：
$$\mathcal{L}_{SSM} = \mathbb{E}_{\mathbf{v} \sim p_{\mathbf{v}}} \mathbb{E}_{\mathbf{x}} \left[ \frac{1}{2}(\mathbf{v}^T \mathbf{s}_\theta(\mathbf{x}))^2 + \mathbf{v}^T \nabla_\mathbf{x} (\mathbf{v}^T \mathbf{s}_\theta(\mathbf{x})) \right]$$

其中 $\mathbf{v}$ 是随机投影方向。

**3D网格的Score Matching优化**

1. **分层训练**：
   - 先训练粗尺度score
   - 逐步细化到精细尺度

2. **局部Score**：
   $$\mathbf{s}_{local}(\mathbf{x}_i) = \sum_{j \in \mathcal{N}(i)} w_{ij} \mathbf{s}_\theta(\mathbf{x}_j)$$
   - 利用网格局部结构

3. **等变Score网络**：
   - 保证 $\mathbf{s}_\theta(\mathbf{R}\mathbf{x}) = \mathbf{R}\mathbf{s}_\theta(\mathbf{x})$
   - 提高泛化能力

4. **多尺度损失**：
   $$\mathcal{L}_{multi} = \sum_{k=1}^{K} w_k \mathcal{L}_{score}^{(k)}$$
   - 不同分辨率同时训练

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