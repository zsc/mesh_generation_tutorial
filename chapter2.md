# 第2章：几何处理基础

几何处理是3D网格生成与操作的数学基石。无论是经典的重建算法还是现代的深度学习方法，都需要对网格的几何性质进行精确的分析与操作。本章将系统介绍离散微分几何的核心概念，包括曲率计算、拉普拉斯算子、谱分析以及网格参数化理论。这些工具不仅是理解高级算法的必要基础，也是实际工程中解决网格质量、形状分析、纹理映射等问题的关键技术。

## 2.1 离散微分几何

离散微分几何研究如何将连续曲面的微分几何概念迁移到离散的三角网格上。这种迁移并非简单的数值近似，而是需要保持某些关键的几何性质和结构不变性。

### 2.1.1 从连续到离散的基本对应

在连续设置中，曲面 $S \subset \mathbb{R}^3$ 可以用参数化 $\mathbf{r}(u,v): \Omega \rightarrow \mathbb{R}^3$ 描述，其中 $\Omega \subset \mathbb{R}^2$ 是参数域。第一基本形式定义了曲面的内蕴几何：

$$I = E\,du^2 + 2F\,du\,dv + G\,dv^2$$

其中 $E = \langle \mathbf{r}_u, \mathbf{r}_u \rangle$，$F = \langle \mathbf{r}_u, \mathbf{r}_v \rangle$，$G = \langle \mathbf{r}_v, \mathbf{r}_v \rangle$。

在离散情况下，三角网格 $M = (V, E, F)$ 由顶点集 $V$、边集 $E$ 和面集 $F$ 组成。每个顶点 $v_i \in V$ 有位置 $\mathbf{p}_i \in \mathbb{R}^3$。离散的第一基本形式通过边长来编码：

$$l_{ij} = \|\mathbf{p}_i - \mathbf{p}_j\|$$

这组边长完全确定了网格的内蕴几何（在刚体运动等价意义下）。

### 2.1.2 离散梯度与散度

对于定义在顶点上的标量函数 $f: V \rightarrow \mathbb{R}$，其在三角形 $T = [v_i, v_j, v_k]$ 内的梯度可以通过重心坐标线性插值得到：

$$\nabla f|_T = \frac{1}{2A_T} \sum_{i \in T} f_i (\mathbf{p}_{j} - \mathbf{p}_{k})^{\perp}$$

其中 $A_T$ 是三角形面积，$(.)^{\perp}$ 表示在三角形平面内逆时针旋转90度。

离散散度通过对偶关系定义。对于定义在面上的向量场 $\mathbf{X}$，其散度在顶点 $v_i$ 处为：

$$(\text{div}\,\mathbf{X})_i = \frac{1}{A_i} \sum_{T \ni v_i} A_T \langle \mathbf{X}_T, \nabla \phi_i|_T \rangle$$

其中 $A_i$ 是顶点的Voronoi面积，$\phi_i$ 是顶点 $v_i$ 的帽函数（在 $v_i$ 处为1，其他顶点为0）。

### 2.1.3 离散测地距离

测地距离是曲面上两点间最短路径的长度。在离散网格上，精确计算测地距离是NP-hard问题，但有多种高效近似算法：

**Dijkstra算法**：将测地路径限制在网格边上，复杂度 $O(n^2 \log n)$。

**Fast Marching Method (FMM)**：求解Eikonal方程 $\|\nabla u\| = 1$，其中 $u$ 是距离函数：

$$\max\left(\frac{u - u_1}{h_1}, 0\right)^2 + \max\left(\frac{u - u_2}{h_2}, 0\right)^2 = 1$$

**Heat Method**：利用热扩散与测地距离的关系，分两步计算：
1. 求解热方程：$(I - t\Delta)u = \delta_s$
2. 归一化梯度并求解Poisson方程：$\Delta \phi = -\nabla \cdot \mathbf{X}$，其中 $\mathbf{X} = -\nabla u / \|\nabla u\|$

### 2.1.4 平行传输与连接

离散网格上的平行传输定义了如何在曲面上移动向量而保持其"方向不变"。对于相邻三角形 $T_1$ 和 $T_2$ 共享边 $e$，从 $T_1$ 到 $T_2$ 的平行传输算子为：

$$P_{12} = R_{\mathbf{n}_2}(\theta) \circ R_e(\alpha) \circ R_{\mathbf{n}_1}(-\theta)$$

其中 $\alpha$ 是两个三角形间的二面角，$\mathbf{n}_1, \mathbf{n}_2$ 是法向量，$\theta$ 是将法向量旋转到边方向的角度。

离散连接的和乐群（holonomy）刻画了网格的内蕴曲率。绕顶点一圈的平行传输累积角度即为角缺陷（angle defect）：

$$\Omega_i = 2\pi - \sum_{j \in N(i)} \theta_{ij}$$

其中 $\theta_{ij}$ 是顶点 $v_i$ 处的内角。

```
      v_k
       /\
      /  \
     /    \
    /  T   \
   /   ij   \
  /__________\
 v_i    e    v_j

三角形T_ij中的几何要素
```

## 2.2 曲率计算与法向估计

曲率是描述曲面弯曲程度的基本几何量。在离散网格上，曲率的定义和计算有多种等价但不完全相同的方法。

### 2.2.1 高斯曲率与平均曲率

**高斯曲率**（内蕴曲率）通过Gauss-Bonnet定理在离散情况下定义：

$$K_i = \frac{\Omega_i}{A_i} = \frac{2\pi - \sum_{j \in N(i)} \theta_{ij}}{A_i}$$

其中 $A_i$ 可以是：
- Barycentric面积：$A_i^{bary} = \frac{1}{3} \sum_{T \ni v_i} A_T$
- Voronoi面积：$A_i^{vor} = \frac{1}{8} \sum_{j \in N(i)} (\cot \alpha_{ij} + \cot \beta_{ij}) \|e_{ij}\|^2$
- 混合面积：对钝角三角形特殊处理

**平均曲率**通过离散拉普拉斯-贝尔特拉米算子定义：

$$H_i = \frac{1}{2} \|\mathbf{H}_i\|, \quad \mathbf{H}_i = \frac{1}{2A_i} \sum_{j \in N(i)} (\cot \alpha_{ij} + \cot \beta_{ij})(\mathbf{p}_j - \mathbf{p}_i)$$

平均曲率向量 $\mathbf{H}_i$ 指向曲率增大的方向，其模长为平均曲率的两倍。

**主曲率**通过以下关系计算：

$$\kappa_1, \kappa_2 = H \pm \sqrt{H^2 - K}$$

### 2.2.2 离散曲率估计方法

**二次曲面拟合法**：在顶点邻域内拟合二次曲面：

$$z = \frac{1}{2}(a x^2 + 2b xy + c y^2) + dx + ey + f$$

通过最小二乘拟合得到系数后，曲率张量为：

$$\mathcal{K} = \begin{pmatrix} a & b \\ b & c \end{pmatrix}$$

主曲率是 $\mathcal{K}$ 的特征值，主方向是对应的特征向量。

**法向变化法**：基于法向量的协方差矩阵：

$$\mathcal{M} = \frac{1}{A} \sum_{T \in N(v)} A_T (\mathbf{n}_T - \bar{\mathbf{n}})(\mathbf{n}_T - \bar{\mathbf{n}})^T$$

其中 $\mathbf{n}_T$ 是三角形法向量，$\bar{\mathbf{n}}$ 是加权平均法向量。

**张量投票法**：通过邻域内的法向量投票构建曲率张量：

$$\mathcal{T} = \sum_{j \in N(i)} w_{ij} \mathbf{v}_{ij} \mathbf{v}_{ij}^T$$

其中 $\mathbf{v}_{ij}$ 是从 $v_i$ 指向 $v_j$ 的单位向量，$w_{ij}$ 是基于距离和角度的权重。

### 2.2.3 法向估计算法

**面积加权平均**：

$$\mathbf{n}_i = \frac{\sum_{T \ni v_i} A_T \mathbf{n}_T}{\|\sum_{T \ni v_i} A_T \mathbf{n}_T\|}$$

**角度加权平均**：

$$\mathbf{n}_i = \frac{\sum_{T \ni v_i} \theta_i^T \mathbf{n}_T}{\|\sum_{T \ni v_i} \theta_i^T \mathbf{n}_T\|}$$

其中 $\theta_i^T$ 是三角形 $T$ 在顶点 $v_i$ 处的内角。

**主成分分析（PCA）**：对邻域点云进行PCA，最小特征值对应的特征向量即为法向估计：

$$\mathcal{C} = \frac{1}{n} \sum_{j \in N(i)} (\mathbf{p}_j - \bar{\mathbf{p}})(\mathbf{p}_j - \bar{\mathbf{p}})^T$$

### 2.2.4 法向一致性与定向

法向一致性是全局问题，需要确保相邻面片的法向指向同一侧。常用方法包括：

**最小生成树传播**：
1. 构建面片邻接图，边权重为 $w_{ij} = 1 - |\langle \mathbf{n}_i, \mathbf{n}_j \rangle|$
2. 计算最小生成树
3. 从根节点开始传播法向方向

**全局优化方法**：最小化能量函数：

$$E = \sum_{(i,j) \in E} w_{ij} (1 - \langle \mathbf{n}_i, s_j \mathbf{n}_j \rangle)$$

其中 $s_j \in \{-1, +1\}$ 是方向标记，可通过图割或整数规划求解。

## 2.3 拉普拉斯算子与谱分析

离散拉普拉斯算子是几何处理中最重要的工具之一，它编码了网格的局部和全局几何信息。

### 2.3.1 离散拉普拉斯算子构造

**组合拉普拉斯**（图拉普拉斯）：

$$L_{ij}^{comb} = \begin{cases}
-1 & \text{if } (i,j) \in E \\
d_i & \text{if } i = j \\
0 & \text{otherwise}
\end{cases}$$

其中 $d_i$ 是顶点 $v_i$ 的度数。

**余切拉普拉斯**（离散拉普拉斯-贝尔特拉米算子）：

$$L_{ij}^{cot} = \begin{cases}
-\frac{1}{2}(\cot \alpha_{ij} + \cot \beta_{ij}) & \text{if } (i,j) \in E \\
\sum_{k \in N(i)} \frac{1}{2}(\cot \alpha_{ik} + \cot \beta_{ik}) & \text{if } i = j \\
0 & \text{otherwise}
\end{cases}$$

其中 $\alpha_{ij}$ 和 $\beta_{ij}$ 是边 $e_{ij}$ 对面的两个角。

**归一化拉普拉斯**：

$$\tilde{L} = M^{-1}L$$

其中 $M$ 是质量矩阵（对角矩阵），$M_{ii} = A_i$ 是顶点面积。

### 2.3.2 余切权重公式推导

余切权重来源于有限元方法中的分片线性基函数。对于顶点 $v_i$ 的帽函数 $\phi_i$，其拉普拉斯为：

$$\Delta \phi_i = -\nabla \cdot \nabla \phi_i$$

在三角形 $T = [v_i, v_j, v_k]$ 内：

$$\int_T \nabla \phi_i \cdot \nabla \phi_j \, dA = -\frac{1}{2} \cot \theta_k$$

其中 $\theta_k$ 是顶点 $v_k$ 处的内角。累加所有包含边 $(i,j)$ 的三角形，得到余切权重。

```
        v_k
        /|\
       / | \
      /  |  \
     /α  |  β\
    /    |    \
   /____ | ____\
  v_i    h    v_j

余切权重的几何意义：
cot α = h / |v_i - m|
cot β = h / |v_j - m|
其中m是v_k在边(v_i,v_j)上的投影
```

### 2.3.3 谱分解与特征函数

拉普拉斯算子的特征值问题：

$$L\phi = \lambda M\phi$$

产生一组正交基 $\{\phi_k\}_{k=0}^{n-1}$，对应特征值 $0 = \lambda_0 \leq \lambda_1 \leq ... \leq \lambda_{n-1}$。

**Fiedler向量**（第二特征向量）用于网格分割和参数化：
- 最小化 $\sum_{(i,j) \in E} w_{ij}(\phi_i - \phi_j)^2$
- 提供网格的"最平滑"一维嵌入

**谱嵌入**：使用前 $k$ 个特征向量将网格嵌入到 $\mathbb{R}^k$：

$$\mathbf{x}_i = (\phi_1(v_i), \phi_2(v_i), ..., \phi_k(v_i))$$

这种嵌入保持了网格的内蕴几何，对等距变换不变。

**谱滤波**：任意函数 $f$ 可以在特征基下展开：

$$f = \sum_{k=0}^{n-1} \langle f, \phi_k \rangle \phi_k$$

低通滤波：$\tilde{f} = \sum_{k=0}^{K} \langle f, \phi_k \rangle \phi_k$

高通滤波：$\tilde{f} = \sum_{k=K}^{n-1} \langle f, \phi_k \rangle \phi_k$

### 2.3.4 热扩散与测地距离

热方程描述了热量在曲面上的传播：

$$\frac{\partial u}{\partial t} = \Delta u$$

离散化后：

$$\frac{u^{t+1} - u^t}{\Delta t} = -L u^t$$

隐式欧拉格式（无条件稳定）：

$$(M + \Delta t \cdot L) u^{t+1} = M u^t$$

**热核**（Heat Kernel）：

$$k_t(x, y) = \sum_{k=0}^{\infty} e^{-\lambda_k t} \phi_k(x) \phi_k(y)$$

热核与测地距离的关系（Varadhan公式）：

$$d_{geo}(x, y) = \lim_{t \to 0^+} \sqrt{-4t \log k_t(x, y)}$$

**扩散距离**：

$$d_{diff}^2(x, y) = \sum_{k=1}^{\infty} e^{-2\lambda_k t} (\phi_k(x) - \phi_k(y))^2$$

扩散距离对拓扑噪声更鲁棒，常用于形状匹配和检索。

## 2.4 网格参数化理论

网格参数化是将3D曲面映射到2D参数域的过程，在纹理映射、重网格化、形状分析等应用中至关重要。

### 2.4.1 共形映射与等距映射

**共形映射**保持角度：局部上是相似变换，保持无穷小圆映射为圆。

离散共形能量（Dirichlet能量）：

$$E_{conf} = \sum_{T} \int_T \|\nabla f\|^2 dA = \frac{1}{2} \sum_{(i,j) \in E} w_{ij} \|u_i - u_j\|^2$$

其中 $u_i \in \mathbb{R}^2$ 是参数坐标，$w_{ij}$ 是余切权重。

**等距映射**保持距离：第一基本形式不变。

离散等距条件：边长保持
$$\|u_i - u_j\| = \|\mathbf{p}_i - \mathbf{p}_j\|, \quad \forall (i,j) \in E$$

由于一般曲面不能等距展开到平面（高斯曲率非零），实践中最小化变形能量：

$$E_{iso} = \sum_{(i,j) \in E} (\|u_i - u_j\| - \|\mathbf{p}_i - \mathbf{p}_j\|)^2$$

### 2.4.2 凸组合参数化

**Tutte参数化**：将边界映射到凸多边形，内部顶点表示为邻居的凸组合：

$$u_i = \sum_{j \in N(i)} w_{ij} u_j, \quad \sum_{j \in N(i)} w_{ij} = 1, \quad w_{ij} > 0$$

保证单射（无翻转），但变形可能很大。

**均值坐标**（Mean Value Coordinates）：

$$w_{ij} = \frac{\tan(\alpha_{ij}/2) + \tan(\beta_{ij}/2)}{\|\mathbf{p}_i - \mathbf{p}_j\|}$$

保持了更好的形状特征，权重始终为正。

**调和坐标**（Harmonic Coordinates）：

最小化Dirichlet能量，等价于求解拉普拉斯方程：
$$\Delta u = 0$$

离散形式：
$$L u = 0 \quad \text{(内部顶点)}$$

### 2.4.3 LSCM与ABF方法

**最小二乘共形映射（LSCM）**：

最小化共形变形的L2范数。对于三角形 $T$，其雅可比矩阵 $J_T$ 的共形条件：

$$J_T = s_T R_T$$

其中 $s_T$ 是缩放因子，$R_T$ 是旋转矩阵。

LSCM能量：
$$E_{LSCM} = \sum_{T} A_T \|J_T - s_T R_T\|_F^2$$

通过复数表示简化：令 $z_i = u_i + iv_i$，共形映射满足Cauchy-Riemann条件：

$$\frac{\partial z}{\partial \bar{w}} = 0$$

其中 $w$ 是3D曲面的局部复坐标。

**角度基展平（ABF/ABF++）**：

直接优化角度而非位置，保证无翻转。

约束条件：
1. 三角形内角和：$\alpha_i^T + \alpha_j^T + \alpha_k^T = \pi$
2. 顶点角度和：$\sum_{T \ni v_i} \alpha_i^T = 2\pi$ （内部顶点）
3. 有效角度：$0 < \alpha_i^T < \pi$

目标函数（最小化角度变形）：

$$E_{ABF} = \sum_{T} \sum_{i \in T} (\alpha_i^T - \beta_i^T)^2 / \beta_i^T$$

其中 $\beta_i^T$ 是原始3D网格的角度。

### 2.4.4 全局参数化

**无缝参数化**：通过整数转移函数实现跨切缝的连续性：

$$u_j - u_i = R_{ij}^{k\pi/2}(u'_j - u'_i) + t_{ij}$$

其中 $R_{ij}$ 是旋转（$k \in \{0,1,2,3\}$），$t_{ij}$ 是整数平移。

**四边形主导重网格化**：

1. 计算交叉场（4-RoSy场）
2. 通过Poisson方程积分得到参数化
3. 提取整数格线作为四边形网格

交叉场的奇异点满足Poincaré-Hopf定理：

$$\sum_{i} \text{index}(s_i) = \chi(M)$$

其中 $\chi(M) = V - E + F$ 是欧拉特征数。

**多图表参数化（Atlas）**：

将网格分割成多个图表（chart），每个图表单独参数化：

1. 种子选择：基于曲率或测地距离
2. 区域生长：最小化变形能量
3. 图表参数化：LSCM或ABF
4. 图表打包：优化2D布局

## 本章小结

本章系统介绍了几何处理的核心数学工具。离散微分几何提供了从连续到离散的严格对应，包括梯度、散度、测地距离等基本算子的离散化。曲率计算是理解曲面几何性质的关键，我们讨论了高斯曲率、平均曲率和主曲率的多种估计方法，以及法向估计和一致性处理。

拉普拉斯算子是几何处理的瑞士军刀，其谱分解提供了强大的频域分析工具。通过特征值和特征函数，我们可以进行形状分析、网格分割、信号滤波等操作。热扩散方程将时间维度引入几何处理，提供了计算测地距离的优雅方法。

网格参数化建立了3D曲面与2D平面的映射关系。共形映射保持角度，等距映射保持距离，而实际应用中需要在两者间权衡。LSCM、ABF等现代方法提供了高质量的参数化结果，全局参数化技术则支持无缝纹理映射和四边形重网格化。

**核心公式汇总**：

- 离散高斯曲率：$K_i = (2\pi - \sum_j \theta_{ij}) / A_i$
- 离散平均曲率：$\mathbf{H}_i = \frac{1}{2A_i} \sum_j (\cot \alpha_{ij} + \cot \beta_{ij})(\mathbf{p}_j - \mathbf{p}_i)$
- 余切拉普拉斯：$L_{ij} = -\frac{1}{2}(\cot \alpha_{ij} + \cot \beta_{ij})$
- 热方程：$(M + \Delta t \cdot L) u^{t+1} = M u^t$
- 共形能量：$E_{conf} = \frac{1}{2} \sum_{(i,j)} w_{ij} \|u_i - u_j\|^2$

## 练习题

### 基础题

**练习2.1** 证明对于平面三角形网格，所有内部顶点的高斯曲率为零。

<details>
<summary>提示</summary>
考虑平面上顶点周围角度和的性质。
</details>

<details>
<summary>参考答案</summary>

对于平面三角形网格，任意内部顶点 $v_i$ 周围的角度和恰好等于 $2\pi$：

$$\sum_{j \in N(i)} \theta_{ij} = 2\pi$$

因此角缺陷 $\Omega_i = 2\pi - \sum_j \theta_{ij} = 0$。

根据离散高斯曲率的定义：
$$K_i = \frac{\Omega_i}{A_i} = \frac{0}{A_i} = 0$$

这与连续情况一致：平面的高斯曲率处处为零。
</details>

**练习2.2** 推导三角形面积的余切公式：$A_T = \frac{1}{4}((\cot \alpha + \cot \beta) l^2_{ab} + (\cot \beta + \cot \gamma) l^2_{bc} + (\cot \gamma + \cot \alpha) l^2_{ca})$

<details>
<summary>提示</summary>
利用三角形面积公式 $A = \frac{1}{2}ab\sin C$ 和余切的定义。
</details>

<details>
<summary>参考答案</summary>

对于三角形 $T$ 的三个顶点 $a, b, c$ 和对应角度 $\alpha, \beta, \gamma$：

从面积公式：$A = \frac{1}{2}l_{ab}h_c$，其中 $h_c$ 是从 $c$ 到边 $ab$ 的高。

由三角关系：
- $h_c = l_{ac}\sin\alpha = l_{bc}\sin\beta$
- $\cot\alpha = \cos\alpha/\sin\alpha$

利用余弦定理和正弦定理的关系，可以证明：
$$A = \frac{1}{4}\sum_{\text{edges}} (\cot\alpha + \cot\beta) l^2$$

其中求和遍历三条边，每条边对应其两个对角的余切和。
</details>

**练习2.3** 给定一个正四面体，计算其顶点的离散平均曲率向量。

<details>
<summary>提示</summary>
正四面体具有高度对称性，利用对称性简化计算。
</details>

<details>
<summary>参考答案</summary>

设正四面体边长为 $a$，顶点 $v_0$ 位于原点，其三个邻居形成等边三角形。

1. 每个二面角：$\cos\theta = 1/3$，故 $\cot\alpha = \cot\beta = \cot(\pi - \theta)/2$

2. 对每条边：$\cot\alpha + \cot\beta = 2\cot(\pi - \theta)/2 = 2/\sqrt{2}$

3. 平均曲率向量指向四面体中心：
   $$\mathbf{H}_0 = \frac{1}{2A_0} \sum_{j=1}^3 \frac{2}{\sqrt{2}}(\mathbf{p}_j - \mathbf{p}_0)$$

4. 由于对称性，$\mathbf{H}_0$ 指向形心方向，模长为常数。
</details>

### 挑战题

**练习2.4** 设计一个算法，检测网格上的脐点（umbilic points），即两个主曲率相等的点。讨论算法的数值稳定性。

<details>
<summary>提示</summary>
脐点满足 $\kappa_1 = \kappa_2$，等价于形状算子的两个特征值相等。考虑使用条件数或其他数值稳定的判据。
</details>

<details>
<summary>参考答案</summary>

脐点检测算法：

1. **计算形状算子**：在每个顶点的切平面内构造 $2 \times 2$ 形状算子 $S$

2. **特征值分析**：
   - 计算 $\lambda_1, \lambda_2$ （主曲率）
   - 计算相对差异：$\epsilon = |\lambda_1 - \lambda_2| / \max(|\lambda_1|, |\lambda_2|, \epsilon_0)$

3. **脐点判定**：当 $\epsilon < \tau$ 时标记为脐点（$\tau \approx 0.1$ 为阈值）

4. **数值稳定性考虑**：
   - 避免直接比较浮点数相等
   - 使用相对误差而非绝对误差
   - 对平坦区域（$|\lambda_1|, |\lambda_2|$ 都很小）特殊处理
   - 考虑使用形状算子的条件数作为补充判据

5. **后处理**：
   - 空间滤波去除孤立点
   - 拓扑一致性检查（脐点的指标和应满足Poincaré-Hopf定理）
</details>

**练习2.5** 证明离散调和函数（满足 $Lu = 0$ 的函数）在内部顶点达到最大值原理。这对网格参数化的单射性有什么意义？

<details>
<summary>提示</summary>
利用余切权重的正性和凸组合的性质。考虑如果内部有最大值会导致什么矛盾。
</details>

<details>
<summary>参考答案</summary>

**证明最大值原理**：

假设 $u$ 是离散调和函数，$v_i$ 是内部顶点。

由 $Lu = 0$ 在顶点 $v_i$ 处：
$$\sum_{j \in N(i)} w_{ij}(u_j - u_i) = 0$$

其中 $w_{ij} = \frac{1}{2}(\cot\alpha_{ij} + \cot\beta_{ij}) > 0$（对于良好三角化）。

重写为：
$$u_i = \frac{\sum_j w_{ij} u_j}{\sum_j w_{ij}}$$

这表明 $u_i$ 是邻居值的加权平均（凸组合）。

**矛盾论证**：若 $u_i$ 是严格最大值，则 $u_i > u_j$ 对所有邻居 $j$。但凸组合不可能严格大于所有参与值，矛盾。

因此最大值只能在边界达到。

**对参数化的意义**：

1. **单射性保证**：Tutte嵌入将边界映射到凸多边形，内部顶点满足凸组合条件（调和）

2. **无翻转保证**：由最大值原理，参数化的 $u, v$ 坐标都不会在内部达到极值，防止三角形翻转

3. **凸性传播**：边界的凸性通过调和扩展传播到内部，保证全局单射性
</details>

**练习2.6** 分析谱嵌入的等距不变性。给定两个等距的网格 $M$ 和 $M'$，证明它们的拉普拉斯特征值相同，并讨论特征函数的关系。

<details>
<summary>提示</summary>
等距变换保持边长不变，因此保持余切权重不变。考虑拉普拉斯矩阵的相似变换。
</details>

<details>
<summary>参考答案</summary>

**等距不变性证明**：

设 $M$ 和 $M'$ 等距，即存在等距映射 $\phi: M \to M'$ 保持所有边长。

1. **余切权重不变**：
   由于边长和角度都保持不变：
   $$w_{ij} = \frac{1}{2}(\cot\alpha_{ij} + \cot\beta_{ij}) = w'_{ij}$$

2. **拉普拉斯矩阵相同**：
   在适当的顶点排列下，$L = L'$ 和 $M = M'$

3. **特征值相同**：
   特征方程 $L\phi = \lambda M\phi$ 和 $L'\phi' = \lambda' M'\phi'$ 相同
   
   因此 $\lambda_k = \lambda'_k$ 对所有 $k$

4. **特征函数的关系**：
   - 特征函数通过等距映射相关联：$\phi'_k = \phi_k \circ \phi^{-1}$
   - 特征函数的节点集（零水平集）在等距变换下保持拓扑结构
   - 特征函数的梯度模长保持不变

**应用意义**：

1. **形状描述子**：拉普拉斯谱作为等距不变的形状签名
2. **对应关系**：通过特征函数匹配建立等距形状间的对应
3. **形状检索**：基于谱距离的形状相似性度量
</details>

**练习2.7** 推导并实现四边形网格上的离散拉普拉斯算子。比较与三角网格的差异，讨论各自的优缺点。

<details>
<summary>提示</summary>
四边形可以看作两个三角形，但也可以直接在四边形单元上构造基函数。考虑双线性插值。
</details>

<details>
<summary>参考答案</summary>

**四边形网格的拉普拉斯算子**：

1. **双线性基函数方法**：
   在参考四边形 $[-1,1]^2$ 上：
   $$\phi_i(\xi, \eta) = \frac{1}{4}(1 + \xi_i\xi)(1 + \eta_i\eta)$$

2. **权重公式**：
   对于邻接顶点，权重涉及四边形的几何：
   $$w_{ij}^{quad} = \sum_{Q \ni (i,j)} \frac{1}{|Q|} \int_Q \nabla\phi_i \cdot \nabla\phi_j \, dA$$

3. **均值坐标版本**：
   $$w_{ij} = \frac{\tan(\alpha_{ij}/2) + \tan(\beta_{ij}/2)}{d_{ij}}$$
   
   其中角度是四边形中的对角。

**与三角网格的比较**：

优点：
- 更规则的连接性（通常4-regular）
- 更适合各向异性特征
- 参数化更自然（张量积结构）
- 数值积分更高效

缺点：
- 非平面四边形带来歧义
- 拓扑灵活性较差
- 某些几何量（如高斯曲率）定义更复杂
- 可能出现自交或折叠

**混合方案**：
- 四边形主导网格：主要四边形，奇异点处三角形
- 多边形网格：统一处理任意多边形单元
</details>

## 常见陷阱与错误

### 1. 数值稳定性问题

**陷阱**：余切权重在接近退化三角形时趋于无穷。

**解决方案**：
- 钳制权重：$w_{ij} = \max(\epsilon, \min(w_{max}, w_{ij}^{raw}))$
- 使用混合Voronoi面积处理钝角三角形
- 预处理网格，去除退化元素

### 2. 边界处理不当

**陷阱**：在边界顶点直接应用内部顶点的公式导致错误。

**正确做法**：
- 边界顶点的Voronoi面积需要特殊计算
- 拉普拉斯算子在边界需要考虑Neumann或Dirichlet条件
- 参数化时固定边界或使用自由边界条件

### 3. 符号和定向混淆

**陷阱**：法向量方向不一致导致曲率符号错误。

**检查清单**：
- 确保所有三角形法向一致（通过传播或全局优化）
- 明确定义正向（向外或向内）
- 注意平均曲率向量vs标量平均曲率的区别

### 4. 离散化误差累积

**陷阱**：多步操作中误差累积导致结果失真。

**缓解策略**：
- 使用高阶离散化方案
- 自适应细化关键区域
- 定期重新初始化或投影回约束流形

### 5. 特征值计算效率

**陷阱**：对大规模网格计算全部特征值耗时巨大。

**优化方法**：
- 只计算前k个特征值（Lanczos/Arnoldi迭代）
- 使用稀疏矩阵数据结构
- 多重网格或谱预条件子加速

### 6. 参数化的局部最优

**陷阱**：非凸优化陷入局部最优，产生高度扭曲的参数化。

**改进策略**：
- 良好的初始化（如Tutte嵌入）
- 多尺度优化（从粗到细）
- 结合多种能量项的混合方法
- 使用凸松弛或半定规划

### 7. 浮点精度问题

**陷阱**：在接近奇异配置时的数值不稳定。

**预防措施**：
- 使用相对误差而非绝对误差
- 关键计算使用高精度算术
- 几何谓词使用精确算术或符号扰动
- 条件数监控和自适应精度控制