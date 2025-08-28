# 第3章：采样理论与重建基础

## 章节大纲

1. **开篇介绍**
   - 采样与重建的基本问题
   - 本章学习目标

2. **3.1 Nyquist-Shannon采样定理在3D中的应用**
   - 一维采样定理回顾
   - 二维图像采样扩展
   - 三维空间采样理论
   - 频域分析与带限函数
   - 采样密度与重建质量关系

3. **3.2 Voronoi图与Delaunay三角化**
   - Voronoi图的定义与性质
   - Delaunay三角化的对偶关系
   - 空圆性质与最大最小角性质
   - 增量构造与分治算法
   - 高维推广与应用

4. **3.3 点云采样策略**
   - 均匀采样与随机采样
   - 泊松盘采样
   - 蓝噪声采样
   - 自适应采样
   - 特征保持采样

5. **3.4 重建的适定性分析**
   - Hadamard适定性条件
   - 重建问题的唯一性
   - 稳定性与条件数
   - 正则化方法
   - 噪声鲁棒性

6. **本章小结**
7. **练习题**
8. **常见陷阱与错误**

---

## 开篇介绍

三维网格重建的核心挑战在于如何从离散采样点恢复连续曲面。本章深入探讨采样理论的数学基础，建立从连续到离散、再从离散到连续的理论框架。我们将学习如何确定最优采样密度、理解重建算法的理论保证，以及分析重建问题的数学性质。

**学习目标：**
- 掌握三维空间中的采样定理及其对重建质量的影响
- 理解Voronoi图与Delaunay三角化在几何重建中的核心作用
- 学会设计和评估不同的点云采样策略
- 能够分析重建问题的适定性并选择合适的正则化方法

## 3.1 Nyquist-Shannon采样定理在3D中的应用

### 3.1.1 从一维到三维的理论扩展

经典的Nyquist-Shannon采样定理告诉我们：对于带限信号$f(t)$，如果其傅里叶变换$\hat{f}(\omega)$在$|\omega| > W$时为零，则采样频率$f_s \geq 2W$（Nyquist频率）时可以完美重建原信号。

一维重建公式：
$$f(t) = \sum_{n=-\infty}^{\infty} f(nT) \cdot \text{sinc}\left(\frac{t-nT}{T}\right)$$

其中$T = 1/f_s$是采样间隔，$\text{sinc}(x) = \sin(\pi x)/(\pi x)$。

### 3.1.2 三维带限函数

在三维空间中，考虑函数$f: \mathbb{R}^3 \rightarrow \mathbb{R}$，其三维傅里叶变换为：

$$\hat{f}(\boldsymbol{\omega}) = \int_{\mathbb{R}^3} f(\mathbf{x}) e^{-i\boldsymbol{\omega} \cdot \mathbf{x}} d\mathbf{x}$$

带限条件变为：$\hat{f}(\boldsymbol{\omega}) = 0$ 当 $\|\boldsymbol{\omega}\| > W$

对于各向同性采样（采样间隔$\Delta$），Nyquist条件要求：
$$\Delta \leq \frac{\pi}{W}$$

### 3.1.3 表面采样的特殊性

对于二维流形嵌入在三维空间的情况，采样理论更加复杂：

1. **局部参数化采样**：在局部切平面上应用2D采样理论
2. **测地距离考虑**：采样密度应基于曲面测地距离而非欧氏距离
3. **曲率自适应**：高曲率区域需要更密集的采样

局部采样密度公式：
$$\rho(\mathbf{p}) = \rho_0 \cdot \max\left(1, \frac{|\kappa_1(\mathbf{p})| + |\kappa_2(\mathbf{p})|}{2\epsilon}\right)$$

其中$\kappa_1, \kappa_2$是主曲率，$\epsilon$是重建误差容限。

### 3.1.4 混叠与重建滤波器

采样不足导致的混叠在3D中表现为：
- 细节丢失
- 拓扑错误（洞的产生或消失）
- 法向翻转

理想低通滤波器在3D中：
$$H(\boldsymbol{\omega}) = \begin{cases}
1, & \|\boldsymbol{\omega}\| \leq W \\
0, & \|\boldsymbol{\omega}\| > W
\end{cases}$$

实际应用中常用的重建核：
- **三线性插值**：$K(\mathbf{x}) = \prod_{i=1}^3 \max(0, 1-|x_i|)$
- **高斯核**：$K(\mathbf{x}) = \exp(-\|\mathbf{x}\|^2/2\sigma^2)$
- **Wendland核**：紧支撑且光滑

## 3.2 Voronoi图与Delaunay三角化

### 3.2.1 Voronoi图的数学定义

给定点集$P = \{p_1, ..., p_n\} \subset \mathbb{R}^d$，点$p_i$的Voronoi区域定义为：

$$V(p_i) = \{x \in \mathbb{R}^d : \|x - p_i\| \leq \|x - p_j\|, \forall j \neq i\}$$

Voronoi图$\text{Vor}(P)$是所有Voronoi区域的集合。

**关键性质：**
1. Voronoi区域是凸多面体
2. 相邻区域共享一个$(d-1)$维面
3. Voronoi顶点等距于至少$d+1$个采样点

### 3.2.2 Delaunay三角化的对偶性

Delaunay三角化$\text{Del}(P)$是Voronoi图的对偶结构：
- 若两个Voronoi区域相邻，对应点在Delaunay中相连
- Delaunay单纯形的外接球不包含其他点（空球性质）

```
    Voronoi图                Delaunay三角化
    
    +---+---+               A-----B
    | A | B |               |\   /|
    +---+---+               | \ / |
    | C | D |               |  X  |
    +---+---+               | / \ |
                            |/   \|
                            C-----D
```

### 3.2.3 空圆性质与优化性

Delaunay三角化满足**空圆性质**：任何三角形的外接圆内部不包含其他采样点。

这导致了重要的优化性质：
1. **最大化最小角**：在所有可能的三角化中，Delaunay三角化最大化最小角
2. **最小化最大外接圆**：局部最优性

数学表述：
$$\text{Del}(P) = \arg\max_{T \in \mathcal{T}(P)} \min_{\triangle \in T} \min_{\theta \in \triangle} \theta$$

### 3.2.4 增量构造算法

Bowyer-Watson算法的关键步骤：

1. 初始化超级三角形包含所有点
2. 对每个新点$p$：
   - 找出所有外接圆包含$p$的三角形（冲突三角形）
   - 删除这些三角形，形成空腔
   - 将$p$与空腔边界连接

时间复杂度：$O(n^{\lceil d/2 \rceil})$，3D中为$O(n^2)$

### 3.2.5 受限Delaunay与曲面重建

对于曲面重建，需要**受限Delaunay三角化**：
- 保持输入的边界约束
- 在约束条件下最大化Delaunay性质

受限Delaunay的存在性条件：
$$\text{对于约束边}e, \exists \text{空球}B: e \subset \partial B$$

## 3.3 点云采样策略

### 3.3.1 均匀采样与随机采样

**均匀网格采样**：
- 优点：实现简单，采样密度可控
- 缺点：可能产生规则伪影，不适应曲面特征

均匀采样的谱特性：
$$S_{\text{uniform}}(\mathbf{k}) = \sum_{n} \delta(\mathbf{k} - 2\pi n/\Delta)$$

**随机采样**：
- 优点：避免规则伪影
- 缺点：可能产生聚集和空隙

随机采样的功率谱密度为白噪声：
$$P_{\text{random}}(\omega) = \text{const}$$

### 3.3.2 泊松盘采样（Poisson Disk Sampling）

泊松盘采样保证任意两点间距离至少为$r$（泊松盘半径）：
$$\|p_i - p_j\| \geq r, \forall i \neq j$$

**Bridson算法**（Fast Poisson Disk Sampling）：
1. 初始化背景网格，单元大小$r/\sqrt{d}$
2. 随机选择初始点，加入活跃列表
3. 从活跃列表选点，在其周围环形区域生成候选点
4. 检查候选点是否满足距离约束
5. 重复直到活跃列表为空

时间复杂度：$O(n)$，其中$n$是最终采样点数

### 3.3.3 蓝噪声采样

蓝噪声特性：低频能量少，高频能量多，径向平均功率谱：
$$P_{\text{blue}}(f) \propto f^\alpha, \alpha > 0$$

理想蓝噪声的径向分布函数：
$$g(r) = \begin{cases}
0, & r < r_{\min} \\
\text{逐渐增长到1}, & r \geq r_{\min}
\end{cases}$$

**Lloyd松弛算法**：
1. 计算Voronoi图
2. 将每个采样点移动到其Voronoi区域质心
3. 迭代直到收敛

质心计算：
$$c_i = \frac{\int_{V(p_i)} x \cdot \rho(x) dx}{\int_{V(p_i)} \rho(x) dx}$$

### 3.3.4 自适应采样

根据局部几何特征调整采样密度：

**曲率驱动采样**：
$$\rho(\mathbf{p}) = \rho_{\min} + (\rho_{\max} - \rho_{\min}) \cdot \frac{|\kappa(\mathbf{p})|}{|\kappa|_{\max}}$$

**特征敏感采样**：
1. 计算局部特征强度（如DoG、Harris角点）
2. 根据特征强度分配采样概率
3. 使用重要性采样生成点

**误差驱动采样**（迭代细化）：
```
while 重建误差 > 阈值:
    计算当前重建的Hausdorff距离
    在高误差区域增加采样点
    更新重建
```

### 3.3.5 特征保持采样

保持尖锐特征的采样策略：

**特征线采样**：
1. 检测特征边（二面角阈值）
2. 沿特征线密集采样
3. 在光滑区域稀疏采样

**角点保护**：
- 确保所有角点（度数≥3的特征点）被采样
- 在角点周围增加采样密度

特征检测准则：
$$\text{特征边}: |\theta_{ij}| > \theta_{\text{threshold}}$$
$$\text{角点}: \sum_{e \in E(v)} I[\text{特征边}(e)] \geq 3$$

## 3.4 重建的适定性分析

### 3.4.1 Hadamard适定性条件

一个数学问题称为**适定的（well-posed）**，如果满足：
1. **存在性**：解存在
2. **唯一性**：解唯一
3. **稳定性**：解连续依赖于输入数据

曲面重建问题的抽象形式：
$$\mathcal{R}: \mathcal{P} \rightarrow \mathcal{S}$$
其中$\mathcal{P}$是点云空间，$\mathcal{S}$是曲面空间。

### 3.4.2 重建唯一性的充分条件

**ε-采样定理**（Amenta & Bern）：
若点集$P$是曲面$S$的ε-采样（$\epsilon < 1$），即：
$$\forall x \in S, \exists p \in P: \|x - p\| \leq \epsilon \cdot \text{lfs}(x)$$

其中$\text{lfs}(x)$是局部特征尺寸（到中轴的距离），则：
1. Delaunay三角化包含正确的曲面三角形
2. 重建拓扑正确

### 3.4.3 稳定性与条件数分析

重建算子的条件数：
$$\kappa(\mathcal{R}) = \|\mathcal{R}\| \cdot \|\mathcal{R}^{-1}\|$$

对于线性重建问题$Ax = b$：
$$\kappa(A) = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}$$

**噪声放大因子**：
$$\frac{\|\delta x\|}{\|x\|} \leq \kappa(A) \cdot \frac{\|\delta b\|}{\|b\|}$$

### 3.4.4 正则化方法

对于病态问题，引入正则化项：

**Tikhonov正则化**：
$$\min_S \|P - S\|^2 + \lambda \cdot R(S)$$

常用正则化项：
1. **薄板样条能量**：$R(S) = \int_S (\kappa_1^2 + \kappa_2^2) dS$
2. **总变分**：$R(S) = \int_S |\nabla S| dS$
3. **拉普拉斯正则化**：$R(S) = \int_S \|\Delta S\|^2 dS$

**L-曲线方法**选择正则化参数：
绘制$\log \|P - S(\lambda)\|^2$ vs $\log R(S(\lambda))$，选择曲线拐点处的$\lambda$。

### 3.4.5 噪声鲁棒性分析

考虑带噪声的点云：$\tilde{P} = P + \mathcal{N}(0, \sigma^2 I)$

**移动最小二乘（MLS）**投影的鲁棒性：
$$f(x) = \arg\min_p \sum_{p_i \in N(x)} w(x, p_i) \cdot \|p - p_i\|^2$$

权重函数：$w(x, p_i) = \exp(-\|x - p_i\|^2/h^2)$

带宽$h$的选择：
- 过小：对噪声敏感
- 过大：过度平滑

最优带宽（基于偏差-方差权衡）：
$$h_{\text{opt}} \propto \left(\frac{\sigma^2}{n \cdot |\nabla^2 f|^2}\right)^{1/6}$$

## 本章小结

### 核心概念回顾

1. **采样定理的3D推广**：
   - Nyquist频率：$f_s \geq 2W_{\max}$
   - 曲面采样密度：$\rho \propto \max(|\kappa_1|, |\kappa_2|)$
   - 带限重建的理论保证

2. **Voronoi-Delaunay对偶性**：
   - Voronoi图定义空间分割
   - Delaunay三角化提供最优连接
   - 空球性质保证几何质量

3. **采样策略层次**：
   - 白噪声（随机）→ 蓝噪声（泊松盘）→ 自适应采样
   - 特征保持采样确保拓扑正确

4. **重建适定性**：
   - ε-采样保证拓扑正确性
   - 正则化处理病态问题
   - 噪声鲁棒性通过MLS等方法实现

### 关键公式汇总

| 概念 | 公式 | 说明 |
|------|------|------|
| 3D Nyquist条件 | $\Delta \leq \pi/W$ | 采样间隔上界 |
| 曲率自适应密度 | $\rho(\mathbf{p}) = \rho_0 \cdot \max(1, \|\kappa\|/\epsilon)$ | 局部采样密度 |
| Voronoi区域 | $V(p_i) = \{x: \|x-p_i\| \leq \|x-p_j\|, \forall j\}$ | 最近邻分割 |
| 泊松盘约束 | $\|p_i - p_j\| \geq r$ | 最小间距保证 |
| ε-采样条件 | $\|x - p\| \leq \epsilon \cdot \text{lfs}(x)$ | 拓扑保证条件 |
| Tikhonov正则化 | $\min \|P-S\|^2 + \lambda R(S)$ | 病态问题求解 |
| 条件数 | $\kappa = \sigma_{\max}/\sigma_{\min}$ | 稳定性度量 |

## 练习题

### 基础题（1-4）

**习题 3.1** 证明在二维平面上，六边形网格是最优的均匀采样模式（覆盖效率最高）。

<details>
<summary>提示（Hint）</summary>
考虑单位圆的覆盖问题，比较正方形、三角形和六边形网格的覆盖密度。
</details>

<details>
<summary>参考答案</summary>

对于半径为$r$的圆形覆盖：
- 正方形网格：每个圆覆盖面积$4r^2$，覆盖率$\pi/4 \approx 78.5\%$
- 六边形网格：每个圆覆盖面积$2\sqrt{3}r^2$，覆盖率$\pi/(2\sqrt{3}) \approx 90.7\%$

六边形排列达到平面最密堆积，因此是最优的均匀采样模式。
</details>

**习题 3.2** 给定一个带宽为$W$的2D信号，如果使用极坐标采样（径向$N_r$个采样，角向$N_\theta$个采样），推导避免混叠的采样条件。

<details>
<summary>提示（Hint）</summary>
考虑极坐标下的Jacobian变换，注意外圈的弧长采样密度。
</details>

<details>
<summary>参考答案</summary>

极坐标采样在半径$r$处的弧长间隔为$\Delta s = r \cdot \Delta\theta = r \cdot 2\pi/N_\theta$。
为避免混叠：$\Delta s \leq \pi/W$
因此：$N_\theta \geq 2rW$

对于最大半径$R$：$N_\theta \geq 2RW$
径向：$N_r \geq RW/\pi$
</details>

**习题 3.3** 实现Bridson算法时，为什么背景网格的单元大小选择为$r/\sqrt{d}$（$d$是维度）？

<details>
<summary>提示（Hint）</summary>
考虑$d$维超立方体的对角线长度。
</details>

<details>
<summary>参考答案</summary>

选择$r/\sqrt{d}$保证每个网格单元的对角线长度恰好为$r$：
$$\text{对角线} = \sqrt{d \cdot (r/\sqrt{d})^2} = r$$

这确保了：
1. 每个单元最多包含一个采样点
2. 只需检查相邻的$3^d - 1$个单元即可验证距离约束
3. 查询复杂度为$O(1)$
</details>

**习题 3.4** 证明Delaunay三角化最大化最小角的性质（在所有可能的三角化中）。

<details>
<summary>提示（Hint）</summary>
使用局部边翻转（edge flip）操作和角度单调性证明。
</details>

<details>
<summary>参考答案</summary>

考虑四点凸包的两种三角化：
- 对角线AC：角度集合$\{\alpha_1, \alpha_2, \beta_1, \beta_2\}$
- 对角线BD：角度集合$\{\gamma_1, \gamma_2, \delta_1, \delta_2\}$

Delaunay条件（空圆性）等价于：$\alpha_1 + \alpha_2 \leq \pi$

可以证明：若违反Delaunay条件，则翻转边后最小角增大。
通过归纳法，任何三角化都可通过一系列边翻转达到Delaunay三角化，且每次翻转都不减小最小角。
</details>

### 挑战题（5-8）

**习题 3.5** 设计一个算法，在给定曲面的ε-采样基础上，自动确定需要额外采样的区域以保证重建质量。考虑如何定量评估当前采样的充分性。

<details>
<summary>提示（Hint）</summary>
利用Voronoi顶点到采样点的距离估计局部特征尺寸，结合极点理论。
</details>

<details>
<summary>参考答案</summary>

算法框架：
1. 计算Voronoi图，识别极点（Voronoi顶点距离最远的两个）
2. 估计局部特征尺寸：$\widehat{\text{lfs}}(p) = \|p - v_{\text{pole}}\|$
3. 计算采样质量因子：$q(p) = \max_{x \in V(p)} \|x - p\|/\widehat{\text{lfs}}(p)$
4. 若$q(p) > \epsilon_{\text{target}}$，在Voronoi区域最远点添加新采样
5. 迭代直到所有$q(p) \leq \epsilon_{\text{target}}$

该方法保证渐进收敛到目标ε-采样密度。
</details>

**习题 3.6** 分析并比较不同采样模式（均匀、随机、泊松盘、蓝噪声）的功率谱特性。如何根据重建算法的频率响应选择最优采样策略？

<details>
<summary>提示（Hint）</summary>
考虑采样模式的径向功率谱与重建核的频率响应的卷积关系。
</details>

<details>
<summary>参考答案</summary>

功率谱分析：
- 均匀采样：$P(f) = \sum_k \delta(f - kf_s)$，离散尖峰
- 随机采样：$P(f) = \text{const}$，白噪声
- 泊松盘：$P(f) \propto f^{0.5}$（2D），缺失低频
- 蓝噪声：$P(f) \propto f$，线性增长

选择策略：
1. 若重建核是理想低通：选择蓝噪声（高频噪声被滤除）
2. 若重建包含非线性处理：选择泊松盘（避免聚集）
3. 若需要频谱分析：选择均匀采样（频谱可预测）
4. 若计算资源受限：选择随机采样（实现简单）

最优配对通过最小化重建误差的期望值确定：
$$E[\epsilon^2] = \int P_{\text{sample}}(f) \cdot |1 - H_{\text{recon}}(f)|^2 df$$
</details>

**习题 3.7** 推导在存在各向异性噪声$\mathcal{N}(0, \Sigma)$（$\Sigma$非对角）情况下的最优MLS投影参数。如何自适应调整局部坐标系？

<details>
<summary>提示（Hint）</summary>
使用主成分分析（PCA）估计噪声协方差，在变换空间中求解。
</details>

<details>
<summary>参考答案</summary>

各向异性MLS优化：

1. 估计局部噪声协方差：
$$\hat{\Sigma} = \frac{1}{|N(x)|} \sum_{p_i \in N(x)} (p_i - \bar{p})(p_i - \bar{p})^T$$

2. 特征分解：$\hat{\Sigma} = Q\Lambda Q^T$

3. 变换到主轴坐标：$\tilde{p}_i = Q^T(p_i - x)$

4. 各向异性权重：
$$w(x, p_i) = \exp\left(-\sum_{j=1}^3 \frac{\tilde{p}_{ij}^2}{2h_j^2}\right)$$

其中$h_j = \alpha\sqrt{\lambda_j}$，$\alpha$是全局平滑参数。

5. 在变换空间求解，再变换回原空间

这种方法自动适应局部噪声结构，在噪声主方向上使用更大的平滑核。
</details>

**习题 3.8** 设计一个理论框架，统一描述从不同数据源（点云、体素、图像）到网格的重建问题。定义通用的适定性条件和最优性准则。

<details>
<summary>提示（Hint）</summary>
考虑将所有输入表示为测度空间上的分布，定义统一的逼近理论。
</details>

<details>
<summary>参考答案</summary>

统一框架：

**输入空间**：测度$\mu$在$\mathbb{R}^3$上
- 点云：$\mu = \sum_i \delta_{p_i}$
- 体素：$\mu = \sum_{ijk} v_{ijk} \cdot \mathbf{1}_{V_{ijk}}$
- 图像：$\mu$由多视图反投影确定

**输出空间**：2-流形$\mathcal{M} \subset \mathbb{R}^3$

**重建泛函**：
$$\mathcal{R}[\mu] = \arg\min_{\mathcal{M}} D(\mu, \mu_{\mathcal{M}}) + \lambda \cdot R(\mathcal{M})$$

其中：
- $D$是分布距离（如Wasserstein距离）
- $\mu_{\mathcal{M}}$是曲面$\mathcal{M}$诱导的测度
- $R$是几何正则化项

**适定性条件**：
1. 测度集中性：$\text{supp}(\mu)$是$\epsilon$-Hausdorff逼近
2. 正则性：$R(\mathcal{M}^*) < \infty$
3. 唯一性：$D$在$\mathcal{M}^*$邻域严格凸

**最优性**：
$$\mathcal{M}^* = \arg\min_{\mathcal{M}} \mathbb{E}_{\mu}[\text{dist}(X, \mathcal{M})^2] + \lambda \int_{\mathcal{M}} H^2 dS$$

该框架统一了各种重建方法，提供了理论分析的通用工具。
</details>

## 常见陷阱与错误 (Gotchas)

### 1. 采样密度不足导致的拓扑错误

**问题**：采样稀疏区域可能产生错误的连接或孔洞。

**症状**：
- 薄结构断裂
- 小特征消失
- 错误的亏格（孔数）

**解决方法**：
- 使用ε-采样理论估算所需密度
- 在高曲率区域增加采样
- 后处理检查拓扑一致性

### 2. Delaunay三角化的退化情况

**问题**：共球点导致Delaunay不唯一。

**症状**：
- 数值不稳定
- 细条三角形（sliver）
- 算法崩溃

**解决方法**：
- 使用符号扰动（SoS，Simulation of Simplicity）
- 添加微小随机扰动
- 使用鲁棒的几何谓词库

### 3. 噪声放大问题

**问题**：重建算法可能放大输入噪声。

**症状**：
- 表面粗糙
- 虚假振荡
- 法向不连续

**调试技巧**：
```
1. 计算重建算子的条件数
2. 分析频率响应
3. 逐步增加正则化参数
4. 使用交叉验证选择参数
```

### 4. 边界处理不当

**问题**：开放曲面的边界重建困难。

**症状**：
- 边界收缩
- 错误填充孔洞
- 边界锯齿

**最佳实践**：
- 显式标记边界点
- 使用受限Delaunay
- 边界专用采样策略

### 5. 尺度不一致

**问题**：不同尺度特征需要不同采样密度。

**症状**：
- 大特征过采样
- 小特征欠采样
- 计算资源浪费

**多尺度策略**：
- 分层采样（octree）
- 局部特征尺寸估计
- 自适应细化

### 6. 数值精度问题

**问题**：浮点运算导致的几何谓词错误。

**症状**：
- 拓扑不一致
- 自交
- 算法死循环

**解决方案**：
- 使用精确几何计算库（CGAL、Predicates）
- 实现鲁棒的方向测试
- 避免接近退化的配置

### 调试建议清单

- [ ] 可视化采样分布（检查聚集/空隙）
- [ ] 计算并显示局部采样密度热图
- [ ] 验证Delaunay三角化的空球性质
- [ ] 检查重建曲面的法向一致性
- [ ] 测量Hausdorff距离评估重建质量
- [ ] 分析功率谱验证采样质量
- [ ] 使用已知解的合成数据测试
- [ ] 逐步增加问题复杂度调试

---

[← 第2章：几何处理基础](chapter2.md) | [目录](index.md) | [第4章：Marching Cubes与体素方法 →](chapter4.md)