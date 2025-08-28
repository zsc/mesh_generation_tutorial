# 第5章：Poisson表面重建

## 本章概要

Poisson表面重建是点云到网格转换中最为优雅且数学基础扎实的方法之一。本章将深入探讨如何通过求解Poisson方程，从带法向的点云数据重建出平滑的水密曲面。我们将学习指示函数的概念、梯度场的构造、Poisson方程的离散化求解，以及八叉树加速技术。最后介绍Screened Poisson方法如何在保持全局平滑性的同时更好地保留细节特征。

本章学习目标：
- 理解指示函数与隐式曲面表示的关系
- 掌握从点云法向到梯度场的转换原理
- 深入理解Poisson方程在表面重建中的作用
- 学习八叉树自适应离散化技术
- 了解Screened Poisson的改进机制

## 5.1 指示函数与梯度场

### 5.1.1 指示函数的定义

指示函数（Indicator Function）是Poisson重建的核心概念。对于一个封闭的3D形状 $M$，其指示函数定义为：

$$\chi_M(p) = \begin{cases} 
1 & \text{if } p \in \text{interior}(M) \\
0 & \text{if } p \in \text{exterior}(M)
\end{cases}$$

这是一个理想的二值函数，在物体边界处存在不连续跳变。实际重建中，我们寻找一个平滑的近似函数 $\tilde{\chi}$，其0.5等值面对应重建的曲面。

### 5.1.2 梯度场与法向的关系

指示函数在曲面附近的关键性质是：其梯度在曲面上等于外法向。数学表达为：

$$\nabla \chi_M |_{\partial M} = \vec{n}$$

其中 $\vec{n}$ 是曲面的单位外法向。这个性质建立了指示函数与点云法向之间的桥梁。

```
        外部 (χ=0)
           ↑
      ─────┼───── 曲面 (χ=0.5)
           │
           ↓ ∇χ = n
        内部 (χ=1)
```

### 5.1.3 向量场的构造

给定带法向的点云 $\{(p_i, \vec{n}_i)\}_{i=1}^N$，我们构造一个向量场 $\vec{V}$，使其在采样点附近逼近法向场：

$$\vec{V}(x) = \sum_{i=1}^N \vec{n}_i \cdot W(x - p_i)$$

其中 $W$ 是权重函数，通常选择紧支撑的径向基函数。向量场 $\vec{V}$ 应当满足：
- 在采样点 $p_i$ 处，$\vec{V}(p_i) \approx \vec{n}_i$
- 在远离所有采样点的区域，$\vec{V} \to 0$

### 5.1.4 散度与Poisson方程

关键观察：如果 $\vec{V} = \nabla \tilde{\chi}$，则通过求解Poisson方程可以恢复指示函数：

$$\Delta \tilde{\chi} = \nabla \cdot \vec{V}$$

其中 $\Delta = \nabla^2$ 是拉普拉斯算子。这将点云重建问题转化为一个标准的偏微分方程求解问题。

## 5.2 Poisson方程的离散化

### 5.2.1 有限差分离散化

在规则网格上，拉普拉斯算子可以用中心差分离散化：

$$\Delta u(i,j,k) \approx \frac{1}{h^2}[u(i+1,j,k) + u(i-1,j,k) + u(i,j+1,k) + u(i,j-1,k) + u(i,j,k+1) + u(i,j,k-1) - 6u(i,j,k)]$$

其中 $h$ 是网格间距。

### 5.2.2 基函数展开

更一般地，我们将解表示为基函数的线性组合：

$$\tilde{\chi}(x) = \sum_{o \in \mathcal{O}} \chi_o B_o(x)$$

其中 $\mathcal{O}$ 是基函数的索引集（如八叉树节点），$B_o$ 是相应的基函数，$\chi_o$ 是待求系数。

### 5.2.3 Galerkin方法

使用Galerkin方法，将Poisson方程投影到基函数空间：

$$\langle B_o, \Delta \tilde{\chi} \rangle = \langle B_o, \nabla \cdot \vec{V} \rangle$$

利用分部积分和格林公式：

$$-\langle \nabla B_o, \nabla \tilde{\chi} \rangle = -\langle \nabla B_o, \vec{V} \rangle$$

### 5.2.4 线性系统构造

代入基函数展开，得到线性系统：

$$\sum_{o' \in \mathcal{O}} \chi_{o'} \langle \nabla B_o, \nabla B_{o'} \rangle = \langle \nabla B_o, \vec{V} \rangle$$

矩阵形式：
$$L\vec{\chi} = \vec{b}$$

其中：
- $L_{o,o'} = \langle \nabla B_o, \nabla B_{o'} \rangle$ 是刚度矩阵
- $b_o = \langle \nabla B_o, \vec{V} \rangle$ 是右端项
- $\vec{\chi}$ 是待求系数向量

## 5.3 八叉树加速结构

### 5.3.1 自适应空间剖分

八叉树提供了自适应的空间离散化：
- 在几何细节丰富的区域使用小的叶节点
- 在平坦区域使用大的叶节点

```
    ┌───────────┐
    │     R     │  根节点
    └─────┬─────┘
          │
    ┌─────┴─────┐
    ↓     ↓     ↓
  ┌───┐ ┌───┐ ┌───┐
  │ A │ │ B │ │ C │  中间层
  └───┘ └─┬─┘ └───┘
          │
      ┌───┴───┐
      ↓   ↓   ↓
    ┌─┐ ┌─┐ ┌─┐
    │1│ │2│ │3│  叶节点
    └─┘ └─┘ └─┘
```

### 5.3.2 深度确定策略

节点深度 $d$ 的确定基于：
1. **点密度**：包含点数超过阈值时细分
2. **法向变化**：法向变化剧烈时细分
3. **最大深度限制**：防止过度细分

深度函数：
$$d(p) = \min\{d_{max}, \max\{d_{density}(p), d_{normal}(p)\}\}$$

### 5.3.3 多分辨率基函数

每个八叉树节点 $o$ 对应一个三线性B样条基函数：

$$B_o(x,y,z) = b(\frac{x-c_x^o}{w_o}) \cdot b(\frac{y-c_y^o}{w_o}) \cdot b(\frac{z-c_z^o}{w_o})$$

其中：
- $(c_x^o, c_y^o, c_z^o)$ 是节点中心
- $w_o$ 是节点宽度
- $b(t)$ 是一维B样条基函数

### 5.3.4 跨层级耦合

相邻层级的基函数存在重叠，需要正确处理跨层级的相互作用：

$$\langle \nabla B_{parent}, \nabla B_{child} \rangle \neq 0$$

这导致刚度矩阵具有多分辨率结构。

## 5.4 Screened Poisson改进

### 5.4.1 原始Poisson方法的局限

传统Poisson重建过度平滑，难以保留尖锐特征。主要原因：
- 全局最小二乘优化倾向于平滑解
- 远离采样点的区域缺乏约束
- 法向信息未充分利用点的位置信息

### 5.4.2 Screened项的引入

Screened Poisson在原目标函数中加入数据项：

$$E(\tilde{\chi}) = \|\nabla \tilde{\chi} - \vec{V}\|^2 + \alpha \sum_{i=1}^N |\tilde{\chi}(p_i) - 0.5|^2$$

第二项是"screening"项，强制指示函数在采样点处接近0.5（即曲面）。

### 5.4.3 权重平衡策略

平衡参数 $\alpha$ 的选择至关重要：
- $\alpha$ 过小：退化为原始Poisson，过度平滑
- $\alpha$ 过大：过拟合噪声，失去全局一致性

自适应策略：
$$\alpha(p) = \alpha_0 \cdot \rho(p)^2$$

其中 $\rho(p)$ 是点云在 $p$ 处的局部密度。

### 5.4.4 线性系统的修正

加入screening项后，线性系统变为：

$$(L + \alpha S)\vec{\chi} = \vec{b} + \alpha \vec{s}$$

其中：
- $S$ 是screening矩阵：$S_{o,o'} = \sum_i B_o(p_i)B_{o'}(p_i)$
- $\vec{s}$ 是screening右端项：$s_o = 0.5 \sum_i B_o(p_i)$

### 5.4.5 多重网格求解

由于系统规模巨大，通常采用多重网格方法：

1. **限制**（Restriction）：将残差从细网格传递到粗网格
2. **松弛**（Relaxation）：在各层级进行局部迭代
3. **延拓**（Prolongation）：将修正从粗网格传递到细网格

V-cycle算法：
```
function V_cycle(level, x, b):
    if level == coarsest:
        return direct_solve(A[level], b)
    
    x = smooth(A[level], x, b, ν₁)  # 前平滑
    r = b - A[level] * x            # 残差
    r_c = restrict(r)                # 限制
    e_c = V_cycle(level+1, 0, r_c)   # 递归
    e = prolongate(e_c)              # 延拓
    x = x + e                        # 修正
    x = smooth(A[level], x, b, ν₂)  # 后平滑
    return x
```

## 5.5 实现细节与优化

### 5.5.1 边界条件处理

Poisson方程需要适当的边界条件：

1. **Dirichlet条件**：$\tilde{\chi}|_{\partial\Omega} = 0$（外边界设为0）
2. **Neumann条件**：$\frac{\partial\tilde{\chi}}{\partial n}|_{\partial\Omega} = 0$（自然边界）

实践中常用Neumann条件，允许解在边界自由变化。

### 5.5.2 法向一致性

输入法向可能不一致，需要预处理：

1. **传播法**：从种子点开始，通过最小生成树传播法向
2. **优化法**：最小化相邻点法向的不一致性

目标函数：
$$E_{orient} = \sum_{(i,j) \in \mathcal{E}} w_{ij}(1 + \vec{n}_i \cdot \vec{n}_j)$$

### 5.5.3 置信度加权

不同采样点的可靠性不同，引入置信度权重：

$$\vec{V}(x) = \sum_{i=1}^N c_i \vec{n}_i \cdot W(x - p_i)$$

置信度 $c_i$ 基于：
- 局部采样密度
- 法向估计质量
- 传感器噪声模型

### 5.5.4 等值面提取

从重建的指示函数提取0.5等值面：

1. **Marching Cubes**：在均匀网格上提取
2. **Adaptive extraction**：在八叉树上自适应提取
3. **Dual Contouring**：生成更好的尖锐特征

提取时的优化：
- 使用查找表加速
- 并行化处理
- 顶点位置的二次优化

## 本章小结

Poisson表面重建通过优雅的数学框架实现了从点云到网格的转换：

**核心概念**：
- 指示函数 $\chi$ 隐式表示3D形状
- 梯度场 $\nabla\chi$ 与表面法向的对应关系
- Poisson方程 $\Delta\chi = \nabla \cdot \vec{V}$ 连接法向与曲面

**关键技术**：
- 八叉树自适应离散化实现多分辨率
- Galerkin方法将PDE转化为线性系统
- Screened项平衡全局平滑与局部拟合
- 多重网格方法高效求解大规模系统

**主要公式汇总**：

1. Poisson方程：$\Delta\tilde{\chi} = \nabla \cdot \vec{V}$
2. Screened目标函数：$E = \|\nabla\tilde{\chi} - \vec{V}\|^2 + \alpha\sum_i|\tilde{\chi}(p_i) - 0.5|^2$
3. 线性系统：$(L + \alpha S)\vec{\chi} = \vec{b} + \alpha\vec{s}$
4. 刚度矩阵：$L_{o,o'} = \langle\nabla B_o, \nabla B_{o'}\rangle$

Poisson重建的优势在于全局优化带来的鲁棒性和水密性保证，而Screened改进则在保持这些优点的同时提升了细节保真度。

## 常见陷阱与错误 (Gotchas)

### 1. 法向方向不一致
**问题**：点云法向朝向混乱，导致重建失败或产生错误拓扑
**症状**：重建结果出现大量孔洞或内外翻转
**解决**：
- 使用最小生成树进行法向传播
- 检查法向与视点的一致性
- 对称物体需要特殊处理

### 2. 八叉树深度设置不当
**问题**：深度过浅丢失细节，过深导致过拟合噪声
**症状**：表面过度平滑或出现高频噪声
**解决**：
- 根据点云密度自适应设置深度
- 典型范围：8-12层
- 使用交叉验证确定最优深度

### 3. Screened权重失衡
**问题**：α参数设置不当导致欠拟合或过拟合
**症状**：表面偏离点云或产生振荡
**调试技巧**：
```
α太小 → 表面过度平滑，远离采样点
α太大 → 表面振荡，过拟合噪声
α适中 → 平衡全局平滑与局部细节
```

### 4. 边界伪影
**问题**：重建边界出现不自然的突起或凹陷
**原因**：边界条件设置不当或点云覆盖不完整
**解决**：
- 扩展包围盒，给边界留出缓冲区
- 使用Neumann边界条件
- 补充虚拟边界点

### 5. 内存溢出
**问题**：大规模点云导致内存不足
**症状**：程序崩溃或极度缓慢
**优化策略**：
- 使用out-of-core算法
- 分块处理后合并
- 稀疏矩阵存储优化

### 6. 数值稳定性
**问题**：矩阵病态导致求解不稳定
**症状**：迭代不收敛或解包含NaN
**处理方法**：
- 添加正则化项：$(L + \epsilon I)$
- 使用预条件子
- 检查基函数的条件数

### 7. 拓扑错误
**问题**：生成非流形网格或自相交
**原因**：等值面提取的二义性
**修复**：
- 使用改进的Marching Cubes变体
- 后处理拓扑修复
- 调整等值面阈值（不一定是0.5）

### 8. 性能瓶颈诊断
**常见瓶颈及优化**：
```
构造八叉树 (15-25%) → 并行化、空间哈希
构造矩阵 (20-30%) → 利用对称性、稀疏存储
求解系统 (40-50%) → 多重网格、GPU加速
等值面提取 (10-15%) → 并行Marching Cubes
```

## 练习题

### 基础题

#### 练习5.1：指示函数性质
证明对于封闭曲面 $M$，其指示函数 $\chi_M$ 的梯度在曲面上垂直于曲面。

**Hint**: 考虑指示函数的等值面族 $\chi_M(x) = c$ 以及梯度与等值面的关系。

<details>
<summary>参考答案</summary>

证明：
1. 对于曲面 $\partial M$，有 $\chi_M|_{\partial M} = 0.5$（约定）
2. 曲面可视为指示函数的0.5等值面
3. 对于任意等值面 $f(x) = c$，梯度 $\nabla f$ 垂直于等值面
4. 因此 $\nabla \chi_M$ 在 $\partial M$ 上垂直于曲面
5. 由于 $\chi_M$ 从内部（值=1）向外部（值=0）递减，梯度指向外法向方向
6. 归一化后：$\nabla \chi_M / |\nabla \chi_M| = \vec{n}$（单位外法向）

关键洞察：指示函数将3D曲面隐式编码为标量场的等值面，梯度自然编码了法向信息。
</details>

#### 练习5.2：Poisson方程推导
从向量场 $\vec{V} = \nabla \tilde{\chi}$ 出发，推导Poisson方程 $\Delta \tilde{\chi} = \nabla \cdot \vec{V}$。

**Hint**: 利用向量恒等式 $\nabla \cdot (\nabla f) = \Delta f$。

<details>
<summary>参考答案</summary>

推导过程：
1. 给定：$\vec{V} = \nabla \tilde{\chi}$
2. 两边取散度：$\nabla \cdot \vec{V} = \nabla \cdot (\nabla \tilde{\chi})$
3. 应用恒等式：$\nabla \cdot (\nabla \tilde{\chi}) = \Delta \tilde{\chi}$
4. 得到：$\Delta \tilde{\chi} = \nabla \cdot \vec{V}$

物理意义：
- 左边：指示函数的拉普拉斯（曲率信息）
- 右边：向量场的散度（源/汇分布）
- Poisson方程建立了两者的平衡关系

这是调和分析的基本结果，保证了从法向场重建出平滑的隐式曲面。
</details>

#### 练习5.3：八叉树复杂度分析
设点云有 $N$ 个点，八叉树最大深度为 $D$，分析构造八叉树的时间复杂度。

**Hint**: 考虑每个点的插入成本和树的最大节点数。

<details>
<summary>参考答案</summary>

时间复杂度分析：
1. 单点插入：从根到叶遍历，最多 $D$ 层，$O(D)$
2. $N$ 个点插入：$O(ND)$
3. 节点细分检查：每个节点最多检查一次，总节点数 $O(N)$（平均情况）
4. 总复杂度：$O(ND)$

空间复杂度：
- 最坏情况（所有点分散）：$O(N)$ 个叶节点
- 内部节点：$O(N/7)$（八叉树性质）
- 总空间：$O(N)$

实际优化：
- 使用Morton编码预排序：可将构造优化到 $O(N\log N)$
- 并行构造：使用空间哈希避免锁竞争
- 自底向上构造：先创建叶节点再合并

典型参数：$D = 8-12$，因此复杂度近似线性。
</details>

#### 练习5.4：Screened权重选择
解释为什么Screened Poisson中的权重 $\alpha$ 应该与局部点密度的平方成正比。

**Hint**: 考虑不同密度区域的相对贡献。

<details>
<summary>参考答案</summary>

理论分析：
1. 点密度 $\rho$ 表示单位体积内的点数
2. 梯度约束项贡献：$\|\nabla \tilde{\chi} - \vec{V}\|^2 \propto \rho$（更多点产生更强的梯度约束）
3. Screening项贡献：$\sum_i |\tilde{\chi}(p_i) - 0.5|^2 \propto N_{local} \propto \rho \cdot V$
4. 为平衡两项在不同密度区域的相对权重：
   - 梯度项：$\rho$
   - Screening项：$\alpha \cdot \rho$
5. 要使比例保持一致：$\alpha \propto \rho$

但实践中发现 $\alpha \propto \rho^2$ 效果更好，原因：
- 高密度区域通常包含更可靠的几何信息
- 平方关系提供更强的局部约束
- 补偿数值离散化误差

这种自适应权重避免了全局统一 $\alpha$ 导致的密度偏差。
</details>

### 挑战题

#### 练习5.5：多重网格收敛性分析
证明V-cycle多重网格方法的收敛率与网格层数无关（网格无关收敛性）。

**Hint**: 分析高频和低频误差分量的衰减率。

<details>
<summary>参考答案</summary>

收敛性证明概要：

1. **误差分解**：
   - 误差 $e = e_H + e_L$（高频+低频分量）
   - 高频：波长 $\sim h$（网格间距）
   - 低频：波长 $>> h$

2. **松弛器作用**：
   - Gauss-Seidel/Jacobi快速衰减高频：$\|e_H^{new}\| \leq \sigma\|e_H\|$，$\sigma < 1$
   - 低频分量衰减慢：$\|e_L^{new}\| \approx \|e_L\|$

3. **粗网格校正**：
   - 限制算子：将低频误差映射到粗网格（在粗网格上变成高频）
   - 粗网格求解：有效消除原低频误差
   - 延拓算子：将校正传回细网格

4. **两网格分析**：
   收敛因子 $\rho = \|(I - M_c)(I - M_s)\|$
   其中 $M_s$ 是松弛算子，$M_c$ 是粗网格校正算子

5. **递归V-cycle**：
   - 每层的收敛率类似
   - 总收敛率 $\rho_V \approx \rho^{L}$（$L$是V的层数）
   - 关键：$\rho$ 与网格大小无关！

6. **结论**：
   V-cycle收敛率主要由松弛器的平滑性质和网格转移算子的逼近性质决定，而非问题规模，实现了 $O(N)$ 的最优复杂度。
</details>

#### 练习5.6：自适应八叉树优化
设计一个基于曲率的自适应细分准则，使八叉树在几何特征丰富的区域自动加密。

**Hint**: 利用法向变化率估计局部曲率。

<details>
<summary>参考答案</summary>

自适应细分算法：

1. **局部曲率估计**：
   对于节点 $o$ 包含的点集 $\{p_i\}$：
   $$\kappa_o = \frac{1}{|P_o|} \sum_{i,j \in P_o} \|\vec{n}_i - \vec{n}_j\| / \|p_i - p_j\|$$

2. **多尺度特征检测**：
   $$F_o = \max\{\kappa_o \cdot w_o, \sigma_n, \rho_o / \rho_{avg}\}$$
   其中：
   - $w_o$：节点宽度（尺度因子）
   - $\sigma_n$：法向标准差
   - $\rho_o/\rho_{avg}$：相对密度

3. **递归细分准则**：
   ```
   if (F_o > τ(d)) and (d < d_max):
       subdivide(o)
   ```
   其中 $\tau(d) = \tau_0 \cdot 2^{-\beta d}$（深度自适应阈值）

4. **特征保持优化**：
   - 边缘检测：$E_o = \max_{i,j}|\vec{n}_i^T \vec{n}_j|$
   - 角点检测：主成分分析法向分布
   - 强制细分：$E_o > 0.9$ 时必须细分

5. **平衡策略**：
   - 相邻节点深度差 $\leq 1$（2:1平衡）
   - 防止过度细分：设置最小点数阈值

该方法确保计算资源集中在几何复杂区域，同时保持整体效率。
</details>

#### 练习5.7：边界条件影响分析
比较Dirichlet和Neumann边界条件对Poisson重建结果的影响，特别是在开放曲面的情况下。

**Hint**: 从能量最小化角度分析不同边界条件的约束作用。

<details>
<summary>参考答案</summary>

边界条件对比分析：

1. **Dirichlet条件** $\tilde{\chi}|_{\partial\Omega} = 0$：
   - **优点**：
     - 强制外边界为0，保证封闭性
     - 解唯一确定
     - 远离物体的区域自然衰减到0
   - **缺点**：
     - 边界附近可能产生不自然的"吸附"效应
     - 开放曲面被强制封闭
     - 边界位置敏感

2. **Neumann条件** $\frac{\partial\tilde{\chi}}{\partial n}|_{\partial\Omega} = 0$：
   - **优点**：
     - 允许解在边界自由浮动
     - 更好地保持开放曲面
     - 减少边界伪影
   - **缺点**：
     - 解相差一个常数（需额外约束）
     - 可能产生"漂移"
     - 数值稳定性稍差

3. **混合策略**：
   ```
   底面：Dirichlet（固定参考）
   侧面：Neumann（自由边界）
   顶面：Robin（混合条件）
   ```

4. **开放曲面特殊处理**：
   - 检测边界环
   - 局部修改权重：$\alpha_{boundary} = 0$
   - 虚拟封闭：添加虚拟点形成封闭

5. **实验对比**：
   - 封闭物体：Dirichlet更稳定
   - 地形/浮雕：Neumann更自然
   - 一般场景：混合边界最优

选择建议：根据数据特性和应用需求权衡。大多数实际系统采用Neumann配合适当的正则化。
</details>

#### 练习5.8：数值精度与稳定性
分析Poisson系统的条件数，提出改善数值稳定性的预条件子设计。

**Hint**: 考虑多分辨率基函数导致的尺度差异。

<details>
<summary>参考答案</summary>

条件数分析与预条件子设计：

1. **条件数问题来源**：
   - 八叉树多分辨率：不同层级基函数支撑差异 $2^D$ 倍
   - 刚度矩阵条件数：$\kappa(L) = O(h^{-2}) = O(4^D)$
   - 深度 $D=10$ 时：$\kappa \approx 10^6$，数值不稳定

2. **对角预条件子**：
   $$P_{diag} = diag(L)^{-1}$$
   效果有限，仅规范化不同尺度

3. **不完全Cholesky分解（ICF）**：
   $$L \approx \tilde{L}\tilde{L}^T, \quad P_{ICF} = (\tilde{L}\tilde{L}^T)^{-1}$$
   - 保持稀疏模式
   - 条件数改善：$\kappa(P_{ICF}L) = O(h^{-1})$

4. **多重网格预条件子**：
   $$P_{MG} = (I - M_L)...(I - M_1)$$
   - 利用层级结构
   - 近似最优：$\kappa(P_{MG}L) = O(1)$

5. **代数多重网格（AMG）**：
   - 自动构造粗化层级
   - 强连接检测：$|L_{ij}| > \theta\max_k|L_{ik}|$
   - 鲁棒性最好，适合不规则网格

6. **实用混合策略**：
   ```
   预处理：对角缩放 + 行平衡
   迭代求解：PCG + AMG预条件子
   容错机制：检测停滞，切换到直接法
   ```

7. **数值技巧**：
   - 使用extended precision累加器
   - Kahan求和算法减少舍入误差
   - 残差正交化（GMRES）

实践建议：AMG预条件的PCG方法提供最佳的鲁棒性/效率平衡。
</details>