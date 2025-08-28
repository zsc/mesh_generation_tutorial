# 第6章：基于Delaunay的重建方法

Delaunay三角化作为计算几何的基石，在3D表面重建中扮演着核心角色。与基于隐式场的方法不同，Delaunay方法直接从点云的几何结构出发，通过组合优化原理构造网格。这类方法的理论基础深厚，能够提供拓扑和几何性质的数学保证。本章深入探讨Ball Pivoting、Alpha shapes、Power crust等经典算法，分析它们如何利用Delaunay结构的优良性质实现鲁棒的表面重建，并讨论实践中的关键技术挑战。

## 6.1 Ball Pivoting算法

Ball Pivoting Algorithm (BPA) 通过模拟一个虚拟球在点云表面滚动的过程来构造三角网格。这个直观的几何隐喻背后蕴含着深刻的采样理论和拓扑保证。

### 6.1.1 算法核心思想

算法的基本假设是点云满足 $\rho$-采样条件：对于表面上任意点 $p$，在半径 $\rho$ 的球内至少存在一个采样点。给定半径为 $\rho$ 的球 $B_\rho$，算法寻找所有能被 $B_\rho$ 支撑的三角形。

**定义**：三个点 $p_i, p_j, p_k$ 形成的三角形 $(i,j,k)$ 被半径为 $\rho$ 的球支撑，当且仅当存在一个半径为 $\rho$ 的球，使得：
1. 三点在球面上：$||c - p_i|| = ||c - p_j|| = ||c - p_k|| = \rho$
2. 球内部不包含其他采样点：$\forall p_l \in P, ||c - p_l|| \geq \rho$

其中 $c$ 是球心位置，可通过求解方程组得到：

$$c = \frac{p_i + p_j + p_k}{3} + \lambda \cdot n_{ijk}$$

其中 $n_{ijk}$ 是三角形法向量，$\lambda$ 由半径约束确定。

### 6.1.2 种子三角形选择

算法从种子三角形开始，逐步扩展构造完整网格。种子选择策略直接影响重建质量：

1. **距离准则**：选择边长接近 $2\rho\cos\theta$ 的三角形，其中 $\theta$ 是期望的最大入射角
2. **法向一致性**：验证三点法向量的夹角小于阈值 $\theta_{max}$
3. **空球性质**：确保支撑球内部为空

种子验证的数值稳定性至关重要。对于三点共线或接近共线的情况，使用条件数检测：

$$\kappa = \frac{\sigma_{max}}{\sigma_{min}} < \kappa_{threshold}$$

### 6.1.3 前沿扩展机制

```
     Active Edge Front
         p_i ---- p_j
          /        \
         /  △_old   \
        /            \
       p_k ---------- ?
           
     Ball pivots around edge (p_i, p_j)
     to find new vertex p_new
```

前沿（front）是已构造网格的边界，由活跃边组成。对每条活跃边 $(p_i, p_j)$：

1. **球心轨迹计算**：球绕边旋转时，球心轨迹是以边为轴的圆
2. **候选点搜索**：在轨迹附近搜索满足距离约束的点
3. **最优点选择**：选择使球旋转角度最小的点

球心位置参数化表示：
$$c(\theta) = m_{ij} + r \cos\theta \cdot u + r \sin\theta \cdot v$$

其中 $m_{ij}$ 是边中点，$u, v$ 是正交基，$r = \sqrt{\rho^2 - \frac{||p_j - p_i||^2}{4}}$。

### 6.1.4 多尺度策略

单一半径 $\rho$ 难以处理采样密度变化的点云。多尺度Ball Pivoting使用递增的半径序列：

$$\rho_0 < \rho_1 < ... < \rho_n$$

每个尺度的输出作为下一尺度的输入，逐步填补空洞：

- $\rho_0$：捕获细节特征
- $\rho_i$：填补中等尺度空洞
- $\rho_n$：完成大尺度闭合

半径选择基于局部特征尺度（Local Feature Size, LFS）估计：
$$\rho_i = k_i \cdot \text{median}(LFS(p)), \quad k_i \in [1.0, 2.5]$$

## 6.2 Alpha Shapes理论

Alpha shapes提供了从离散点集重建形状的系统性框架，通过参数 $\alpha$ 控制重建的"分辨率"。

### 6.2.1 Alpha复形的数学定义

给定点集 $P \subset \mathbb{R}^3$ 和参数 $\alpha \in [0, \infty]$，$\alpha$-复形定义为：

**定义 1（Alpha球）**：半径为 $\alpha$ 的球 $B$ 是 $P$ 的 $\alpha$-球，如果：
- $B \cap P \neq \emptyset$（球与点集相交）
- $B$ 的内部不包含 $P$ 中的点

**定义 2（Alpha复形）**：$P$ 的 $\alpha$-复形 $\mathcal{C}_\alpha(P)$ 是所有能被某个 $\alpha$-球包含的单形（顶点、边、三角形、四面体）的集合。

形式化表示：
$$\mathcal{C}_\alpha(P) = \{\sigma \subseteq P : \exists B_\alpha, \sigma \subseteq \partial B_\alpha \land \text{int}(B_\alpha) \cap P = \emptyset\}$$

### 6.2.2 与Delaunay三角化的关系

Alpha复形是Delaunay三角化的子复形，这个关系是算法效率的关键：

$$\mathcal{C}_0(P) \subseteq \mathcal{C}_\alpha(P) \subseteq \mathcal{C}_\beta(P) \subseteq ... \subseteq \mathcal{C}_\infty(P) = \text{DT}(P)$$

其中 $\text{DT}(P)$ 是 $P$ 的Delaunay三角化。

**筛选准则**：Delaunay三角化中的单形 $\sigma$ 属于 $\alpha$-复形，当且仅当：
$$r_\sigma \leq \alpha$$

其中 $r_\sigma$ 是 $\sigma$ 的外接球半径。

### 6.2.3 Alpha值的几何意义

```
    α = 0.5              α = 1.0              α = 2.0
     。。                。---。              。---。
    。  。              。|   |。            。|███|。
     。。                。---。              。---。
  (离散点)            (局部连接)          (全局结构)
```

Alpha参数控制重建的细节层次：
- **小 $\alpha$**：只保留紧密相邻的点之间的连接，捕获局部细节
- **中等 $\alpha$**：平衡局部和全局特征
- **大 $\alpha$**：产生更光滑、更连通的结构

最优 $\alpha$ 选择基于持续同调（Persistent Homology）分析：
$$\alpha^* = \arg\max_\alpha \{\text{Persistence}(\beta_1(\mathcal{C}_\alpha))\}$$

其中 $\beta_1$ 是第一贝蒂数，表征孔洞数量。

### 6.2.4 加权Alpha Shapes

标准Alpha shapes对所有点使用统一权重。加权版本为每个点 $p_i$ 赋予权重 $w_i$，定义加权距离：

$$d_w(p, p_i) = ||p - p_i||^2 - w_i$$

加权Alpha shapes能更好地处理：
- 采样密度变化
- 噪声鲁棒性
- 特征保持

权重设置策略：
$$w_i = k \cdot \text{LFS}(p_i)^2$$

其中 $k \in [0.8, 1.2]$ 是调节参数。

## 6.3 Power Crust与紧致曲面

Power crust算法利用Voronoi图的对偶性质和中轴变换理论，提供了理论保证的水密表面重建。

### 6.3.1 中轴变换与采样理论

**中轴（Medial Axis）** 是到表面具有多个最近点的点的集合：
$$\text{MA}(S) = \{x \in \mathbb{R}^3 : |\{p \in S : ||x-p|| = d(x,S)\}| \geq 2\}$$

Power crust的核心观察：对于密集采样的点云，Voronoi顶点近似表面的中轴。

**定理（收敛性）**：若点云 $P$ 是表面 $S$ 的 $\varepsilon$-采样（$\varepsilon < 0.1$），则：
1. Voronoi顶点到真实中轴的Hausdorff距离 $O(\varepsilon^2)$
2. 重建表面到原表面的Hausdorff距离 $O(\varepsilon^2)$

### 6.3.2 极点与局部特征尺度

对每个采样点 $p_i$，定义其**极点（poles）**：

1. **外极点** $p_i^+$：Voronoi cell $V(p_i)$ 中距离 $p_i$ 最远的顶点
2. **内极点** $p_i^-$：在 $p_i - p_i^+$ 反方向上最远的Voronoi顶点

极点近似局部中轴，提供局部特征尺度估计：
$$\text{LFS}(p_i) \approx ||p_i - p_i^+||$$

### 6.3.3 Power图构造

Power图（加权Voronoi图）定义：给定点集 $P$ 和权重 $W$，点 $x$ 属于 $p_i$ 的power cell当且仅当：

$$||x - p_i||^2 - w_i \leq ||x - p_j||^2 - w_j, \quad \forall j \neq i$$

Power crust使用极点作为加权点：
- 点集：$P \cup P^+ \cup P^-$（原始点加内外极点）
- 权重：$w_i = -||p_i - p_i^{\pm}||^2$（负的到极点距离平方）

### 6.3.4 表面提取与水密性

从Power图中提取表面：

1. **标记内外极点**：基于法向一致性和可见性
2. **选择分隔面**：Power图中分隔内外极点的面片
3. **构造Power crust**：所选面片的并集

**水密性保证**：在 $\varepsilon < 0.1$ 的采样条件下，Power crust是：
- 拓扑正确：与原表面同胚
- 几何精确：Hausdorff距离 $O(\varepsilon^2 \cdot \text{LFS})$
- 水密闭合：形成封闭流形

### 6.3.5 算法优化与实现

计算复杂度：$O(n \log n + k)$，其中 $k$ 是输出大小。

优化策略：
1. **八叉树加速**：快速定位最近邻和Voronoi顶点
2. **增量构造**：利用Delaunay三角化的局部性
3. **并行化**：极点计算和标记阶段的并行处理

数值稳定性处理：
```
IF ||p_i - p_j|| < ε_merge THEN
    合并近距离点
IF angle(n_i, p_i - pole_i) > θ_max THEN  
    使用次优极点
IF Voronoi_cell无界 THEN
    使用边界框人工极点
```

## 6.4 法向一致性处理

法向定向是基于Delaunay重建的关键预处理步骤。点云通常只有未定向的法向量（由PCA或其他方法估计），需要全局一致的定向。

### 6.4.1 法向定向问题的数学描述

给定点云 $P = \{p_1, ..., p_n\}$ 和未定向法向 $\{n_1, ..., n_n\}$，目标是为每个法向选择符号 $s_i \in \{-1, +1\}$，使得：

$$\min_{s \in \{-1,+1\}^n} \sum_{(i,j) \in E} w_{ij} \cdot (1 - s_i \cdot s_j \cdot \langle n_i, n_j \rangle)$$

其中 $E$ 是近邻图的边集，$w_{ij}$ 是权重。

这是一个NP困难的二次布尔优化问题，实践中使用贪婪算法或松弛方法。

### 6.4.2 最小生成树传播

最稳健的方法是基于Riemannian图的最小生成树（MST）：

1. **构建Riemannian图**：
   - 顶点：采样点
   - 边权：$w_{ij} = 1 - |\langle n_i, n_j \rangle|$
   
2. **计算MST**：使用Kruskal或Prim算法

3. **传播定向**：
   ```
   从种子点开始BFS遍历MST：
   FOR each edge (p_i, p_j) in MST:
       IF ⟨n_i, n_j⟩ < 0 THEN
           n_j ← -n_j
   ```

种子选择策略：
- **可见性准则**：选择到视点可见的点
- **曲率准则**：选择低曲率区域的点
- **置信度准则**：选择法向估计置信度最高的点

### 6.4.3 基于传播的优化

MST可能产生错误累积。改进方法使用加权投票：

对每个点 $p_i$，其法向由邻域投票决定：
$$n_i^{new} = \text{sign}\left(\sum_{j \in N(i)} w_{ij} \cdot \langle n_i, n_j \rangle \cdot n_j\right)$$

权重设计考虑：
- **距离衰减**：$w_{ij} = \exp(-||p_i - p_j||^2 / \sigma^2)$
- **法向相似性**：$w_{ij} = |\langle n_i, n_j \rangle|^k$
- **可见性一致**：$w_{ij} = \max(0, \langle v_i, n_j \rangle)$

迭代优化过程：
```
REPEAT until convergence:
    1. 固定高置信度点的法向
    2. 传播到邻域
    3. 解决冲突（投票）
    4. 更新置信度
```

### 6.4.4 歧义处理与鲁棒性

某些几何配置存在固有歧义：

**薄片结构**：两侧法向都合理
- 解决：使用体积化或厚度先验

**环形结构**：存在Möbius带型解
- 解决：检测并切断环路

**噪声区域**：法向估计不可靠
- 解决：使用鲁棒估计和中值滤波

**多连通组件**：组件间定向不一致
- 解决：分别处理每个连通组件

### 6.4.5 与Delaunay方法的集成

法向一致性直接影响Delaunay重建质量：

1. **Ball Pivoting**：需要一致法向来确定球的滚动方向
2. **Power Crust**：使用法向区分内外极点
3. **Alpha Shapes**：法向用于后处理和孔洞填充

集成策略：
```
Pipeline:
1. 初始法向估计（PCA/Jet fitting）
2. 法向一致性处理（本节方法）
3. Delaunay结构构建
4. 基于法向的筛选/定向
5. 网格后处理
```

## 本章小结

本章系统介绍了基于Delaunay三角化的表面重建方法。这类方法的核心优势在于：

**理论保证**：
- Ball Pivoting在满足 $\rho$-采样条件时保证流形输出
- Alpha Shapes提供多尺度拓扑控制
- Power Crust在 $\varepsilon < 0.1$ 采样下保证同胚重建

**关键技术**：
- **Delaunay/Voronoi对偶性**：提供高效的几何查询结构
- **局部特征尺度（LFS）**：自适应采样密度变化
- **法向一致性**：确保全局定向正确

**算法对比**：
| 方法 | 时间复杂度 | 空间复杂度 | 水密性 | 拓扑保证 |
|-----|----------|----------|--------|---------|
| Ball Pivoting | $O(n\log n)$ | $O(n)$ | 否 | 局部 |
| Alpha Shapes | $O(n^2)$ | $O(n^2)$ | 是 | 全局 |
| Power Crust | $O(n\log n)$ | $O(n)$ | 是 | 全局 |

**实践要点**：
1. 预处理质量（去噪、法向估计）直接决定重建质量
2. 参数选择（$\rho$, $\alpha$）需要基于采样密度分析
3. 多尺度策略能提高鲁棒性但增加计算成本
4. 法向一致性是成功重建的必要条件

这些方法为点云到网格的转换提供了坚实的理论基础和实用工具，特别适合处理扫描数据和稠密采样的几何重建任务。

## 练习题

### 基础题

**练习 6.1**：给定三个点 $p_1 = (0, 0, 0)$, $p_2 = (1, 0, 0)$, $p_3 = (0.5, \sqrt{3}/2, 0)$，计算能支撑这个三角形的最小球半径 $\rho_{min}$。

<details>
<summary>提示</summary>
考虑外接圆半径公式，这三个点构成等边三角形。
</details>

<details>
<summary>答案</summary>

等边三角形边长 $a = 1$，外接圆半径：
$$r = \frac{a}{\sqrt{3}} = \frac{1}{\sqrt{3}} = \frac{\sqrt{3}}{3} \approx 0.577$$

因此 $\rho_{min} = \frac{\sqrt{3}}{3}$。任何半径 $\rho \geq \rho_{min}$ 的球都能支撑这个三角形。
</details>

**练习 6.2**：证明对于2D点集，当 $\alpha$ 从0增长到∞时，alpha shape的连通分量数量单调递减。

<details>
<summary>提示</summary>
考虑alpha复形的嵌套性质：$\mathcal{C}_\alpha \subseteq \mathcal{C}_\beta$ 当 $\alpha < \beta$。
</details>

<details>
<summary>答案</summary>

证明：
1. 由于 $\alpha < \beta \Rightarrow \mathcal{C}_\alpha \subseteq \mathcal{C}_\beta$，增加 $\alpha$ 只会添加新的单形，不会删除
2. 新添加的边可能连接原本分离的连通分量
3. 连通分量数 $C(\alpha)$ 满足：$C(\alpha) \geq C(\beta)$ 当 $\alpha < \beta$
4. 因此连通分量数量单调递减
5. 极限情况：$C(0) = n$（n个孤立点），$C(\infty) = k$（Delaunay三角化的连通分量数）
</details>

**练习 6.3**：给定点云采样密度为 $\varepsilon = 0.05$（相对于局部特征尺度），估计Power Crust重建的Hausdorff误差上界。

<details>
<summary>提示</summary>
使用Power Crust的理论保证：误差为 $O(\varepsilon^2 \cdot \text{LFS})$。
</details>

<details>
<summary>答案</summary>

根据Power Crust收敛性定理：
- Hausdorff距离上界：$d_H \leq c \cdot \varepsilon^2 \cdot \text{LFS}$
- 其中常数 $c \approx 5$（经验值）
- 代入 $\varepsilon = 0.05$：
  $$d_H \leq 5 \times 0.05^2 \times \text{LFS} = 0.0125 \times \text{LFS}$$
- 相对误差约为1.25%的局部特征尺度
</details>

**练习 6.4**：设计一个简单的法向一致性检测算法，判断给定的法向场是否全局一致。

<details>
<summary>提示</summary>
构建近邻图，检查相邻法向的夹角。
</details>

<details>
<summary>答案</summary>

算法：
```
输入：点云P，法向N，邻域半径r
输出：是否一致（布尔值）

1. 构建k-近邻图G(V,E)
2. 对每条边(i,j)：
   计算 cos_angle = ⟨n_i, n_j⟩
   IF |cos_angle| < threshold (如0.5) THEN
      标记为不一致边
3. 计算不一致边比例：
   ratio = 不一致边数 / 总边数
4. RETURN (ratio < tolerance)
```

threshold = 0.5对应60度夹角，tolerance典型值0.05（5%容错）。
</details>

### 挑战题

**练习 6.5**：Ball Pivoting算法中，如何处理不同采样密度的点云？设计一个自适应半径选择策略。

<details>
<summary>提示</summary>
基于局部点密度估计动态调整球半径。
</details>

<details>
<summary>答案</summary>

自适应策略：

1. **局部密度估计**：
   $$\rho_i = k \cdot \sqrt[3]{\frac{\text{Volume}(N_k(p_i))}{k}}$$
   其中 $N_k(p_i)$ 是 $p_i$ 的k近邻包围盒体积

2. **多分辨率队列**：
   - 维护不同半径级别的边界队列
   - 优先处理小半径（高密度区域）
   - 逐步增大半径填补空洞

3. **过渡区处理**：
   - 在密度变化区域使用插值半径：
     $$\rho_{edge} = \frac{\rho_i + \rho_j}{2} \cdot (1 + \lambda \cdot \text{grad}(\rho))$$
   - $\text{grad}(\rho)$ 是密度梯度，$\lambda$ 控制过渡平滑度

4. **质量控制**：
   - 拒绝产生狭长三角形的配置（角度 < 15°）
   - 限制相邻三角形的尺寸比（< 3:1）
</details>

**练习 6.6**：分析Alpha shapes在处理带噪声点云时的行为。如何选择 $\alpha$ 值来平衡细节保留和噪声抑制？

<details>
<summary>提示</summary>
考虑持续同调和尺度空间分析。
</details>

<details>
<summary>答案</summary>

噪声影响分析：

1. **噪声模型**：假设噪声 $\eta \sim \mathcal{N}(0, \sigma^2)$

2. **Alpha选择原则**：
   - 下界：$\alpha > 3\sigma$（滤除噪声）
   - 上界：$\alpha < \text{min}(\text{LFS})/2$（保留特征）

3. **尺度空间优化**：
   $$\alpha^* = \arg\max_\alpha \left\{ \frac{\partial}{\partial \alpha} H_1(\mathcal{C}_\alpha) = 0 \right\}$$
   其中 $H_1$ 是一维同调群（孔洞）

4. **实用策略**：
   - 计算持续性图（persistence diagram）
   - 识别显著拓扑特征（高持续性）
   - 选择 $\alpha$ 保留这些特征，滤除短暂特征（噪声）

5. **自适应加权**：
   $$w_i = \exp\left(-\frac{\text{var}(N_k(p_i))}{\sigma_{global}^2}\right)$$
   高方差区域（噪声）赋予低权重
</details>

**练习 6.7**：Power Crust中的极点可能退化（如平面区域）。设计一个鲁棒的极点计算方法。

<details>
<summary>提示</summary>
使用主成分分析检测退化，采用替代策略。
</details>

<details>
<summary>答案</summary>

鲁棒极点计算：

1. **退化检测**：
   对Voronoi cell顶点做PCA：
   $$\lambda_1 \geq \lambda_2 \geq \lambda_3$$
   退化度：$d = \lambda_3 / \lambda_1$

2. **分类处理**：
   - **正常情况** $(d > 0.1)$：标准极点选择
   - **平面退化** $(d < 0.01)$：
     ```
     极点 = 质心 ± h·n
     h = median(LFS邻域)
     ```
   - **线性退化** $(\lambda_2/\lambda_1 < 0.01)$：
     使用主轴端点作为极点

3. **混合策略**：
   $$p^{\pm} = (1-d) \cdot p_{artificial}^{\pm} + d \cdot p_{voronoi}^{\pm}$$

4. **稳定性增强**：
   - 使用k个最远Voronoi顶点的加权平均
   - 权重：$w_k = \exp(-k/\tau)$
   - 避免数值不稳定的极远顶点
</details>

**练习 6.8**（开放问题）：设计一个统一框架，自动选择Ball Pivoting、Alpha Shapes或Power Crust中最适合给定点云的方法。

<details>
<summary>提示</summary>
分析点云特征，建立决策树。
</details>

<details>
<summary>答案</summary>

自动选择框架：

1. **特征提取**：
   - 采样均匀性：$\sigma(\text{NN distances})$
   - 噪声水平：局部平面拟合残差
   - 完整性：边界点比例
   - 规模：点数n
   - 拓扑复杂度：估计亏格

2. **决策规则**：
   ```
   IF 采样均匀 AND 低噪声 AND n < 100k:
       → Ball Pivoting（快速、简单）
   ELIF 需要拓扑控制 OR 多尺度:
       → Alpha Shapes（灵活性）
   ELIF 需要水密 AND 理论保证:
       → Power Crust（鲁棒性）
   ELSE:
       → 混合方法
   ```

3. **混合策略**：
   - 用Power Crust获得初始水密网格
   - 用Alpha Shapes细化局部区域
   - 用Ball Pivoting快速填补小洞

4. **参数自动调优**：
   - 交叉验证：留出10%点评估重建质量
   - 网格质量度量：三角形质量、曲率一致性
   - 贝叶斯优化寻找最优参数

5. **性能考虑**：
   - 预计算成本 vs 重建质量权衡
   - 用户可指定时间/质量偏好
</details>

## 常见陷阱与错误 (Gotchas)

### 1. Ball Pivoting陷阱

**问题**：算法在尖锐特征处失败
- **原因**：球无法同时接触特征两侧的点
- **解决**：使用多个小半径 + 后处理缝合

**问题**：出现自相交
- **原因**：球从多个方向访问同一区域
- **解决**：维护全局访问标记，拒绝冗余三角形

### 2. Alpha Shapes陷阱

**问题**：计算Delaunay三角化内存爆炸
- **原因**：3D Delaunay最坏情况 $O(n^2)$ 四面体
- **解决**：空间分割 + 增量构造

**问题**：Alpha值敏感性
- **原因**：固定alpha不适应密度变化
- **解决**：局部自适应alpha或加权版本

### 3. Power Crust陷阱  

**问题**：边界处极点计算错误
- **原因**：Voronoi cell无界
- **解决**：添加虚拟边界点或使用截断Voronoi

**问题**：薄片结构产生错误拓扑
- **原因**：内外极点混淆
- **解决**：最小厚度约束 + 用户指导

### 4. 法向一致性陷阱

**问题**：MST传播错误累积
- **原因**：单一路径，无冗余
- **解决**：多路径投票 + 置信度传播

**问题**：闭合曲面内外翻转
- **原因**：缺乏全局参考
- **解决**：使用包围盒射线测试确定内外

### 5. 数值稳定性陷阱

**问题**：接近共线/共面的配置
- **原因**：浮点精度限制
- **解决**：
  ```
  使用精确谓词（exact predicates）
  符号扰动（symbolic perturbation）
  区间算术（interval arithmetic）
  ```

### 6. 性能优化陷阱

**问题**：朴素近邻查询瓶颈
- **原因**：$O(n^2)$ 暴力搜索
- **解决**：空间索引（KD-tree, Octree）+ 近似算法

通过理解这些陷阱，可以在实际应用中避免常见错误，提高重建算法的鲁棒性和效率。