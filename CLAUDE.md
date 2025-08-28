（交流可以用英文，所有文档中文，保留这句）

## 项目目标
编写一份3D mesh 生成的中文教程markdown，要包含大量的习题和参考答案（答案默认折叠）。

文件组织是 index.md + chapter1.md + ...

## Audience
verteran engineer and AI scientist

## 章节结构要求
每个章节应包含：
1. **开篇段落**：简要介绍本章内容和学习目标
2. **文字论述**：以文字论述为主，适当配上ASCII 图说明。使用公式时用 latex. 不写代码
3. **本章小结**：总结关键概念和公式
4. **练习题**：
   - 每章包含6-8道练习题
   - 50%基础题（帮助熟悉材料）
   - 50%挑战题（包括开放性思考题）
   - 每题提供提示（Hint）
   - 答案默认折叠，不包含代码
5. **常见陷阱与错误** (Gotchas)：每章包含该主题的常见错误和调试技巧

## 内容
下面把“3D 网格（mesh）生成”的主要流派，按**表示方式/训练范式**来梳理，并配上代表性论文与优缺点，便于选型。

# 1) 经典几何重建：点云/体素 → 网格

**思路**：从点云或体素场中直接抽取等值面。
**代表**：

* **Marching Cubes**：体素等值面抽取的鼻祖，用于医学与体数据，速度快、实现简单。([cs.toronto.edu][1], [ACM Digital Library][2])
* **Poisson Surface Reconstruction / Screened Poisson**：从**有法向的点云**全局求解水密曲面，鲁棒且平滑；Screened 版本更易保细节。([hhoppe.com][3], [Department of Computer Science][4])
* **Ball Pivoting / α-shapes / Delaunay**：几何/拓扑法，从点云生长或基于 Delaunay 结构重建。([mesh.brown.edu][5], [Department of Computer Science][6])
  **优缺点**：稳健、省训练，但**不能凭空“造物”**，依赖高质量采样/法向；常配合摄影测量或激光扫描。

# 2) 隐式场 → 网格（学习式

**思路**：学一个连续隐式函数（SDF/占据场），再用 MC/MT/DMTet 提取网格。
**代表**：

* **DeepSDF（SDF 连续表示）**、**Occupancy Networks（占据场）**。([arXiv][7], [CVF开放获取][8])
* **DMTet（Deep Marching Tetrahedra）**：把离散 SDF 放在**可变形四面体格**，带**可微**的 Marching Tetrahedra，训练/提取端到端。常被下游文本/图像驱动方法用作网格后端。([arXiv][9], [NeurIPS 会议录][10])
  **优缺点**：精细、可拓扑变化；需训练/优化，提取网格有阈值与后处理成本。

# 3) 直接网格生成/变形（显式）

**思路**：网络直接预测网格的**顶点/面或对模板网格进行变形**。
**代表**：

* **Pixel2Mesh**（单图像 → 网格，逐步形变椭球）([arXiv][11], [CVF开放获取][12])
* **AtlasNet**（学习一组可参数化贴片拼成曲面）([CVF开放获取][13])
* **Mesh R-CNN**（检测+网格分支，显式三角网格细化）([CVF开放获取][14])
* **PolyGen**（Transformer 自回归，直接按**顶点/面序列**生成网格）([Proceedings of Machine Learning Research][15])
  **优缺点**：输出即是“干净网格”，易于 DCC/游戏引擎；但处理拓扑/自交/全局一致性较难。

# 4) 文本/图像驱动的**优化式**生成（可微渲染 × 2D 扩散蒸馏）

**思路**：以可微渲染为桥梁，用**2D 扩散模型**给 3D 表示（NeRF/SDF/DMTet 网格）提供梯度，逐步“雕刻”出几何与材质。
**代表**：

* \*\*DreamFusion（SDS）\*\*提出*Score Distillation Sampling*范式。([dreamfusion3d.github.io][16])
* \*\*ProlificDreamer（VSD）\*\*改进 SDS 的质量/多样性，支持高保真网格微调。([NeurIPS 会议录][17])
* **Fantasia3D**（几何/外观解耦，常配 DMTet 提取网格并再纹理）([fantasia3d.github.io][18], [ar5iv][19])
  **优缺点**：开放域强、细节好；**逐样本优化**，耗时，易出现多视不一致/纹理漂移，需正则与后处理。

# 5) 生成式**前**方法（GAN/扩散/大重建模型，直接产出网格）

**思路**：端到端前向推理，几秒级得到可用网格/纹理。
**代表**：

* **GET3D**：直接生成**带纹理的显式网格**（自由拓扑，GAN+三平面/可微网格化）。([NeurIPS 会议录][20])
* **MeshDiffusion**：把网格参数化到 DMTet 上，做**3D 扩散**直接生成网格。([meshdiffusion.github.io][21])
* **InstantMesh / MeshLRM / LRM**：结合多视图扩散+大型重建模型，**单图像十秒级**生成高质量网格。([arXiv][22])
  **优缺点**：速度快、交互友好；对类别/域外分布较敏感，纹理一致性与超细节依赖训练数据与多视一致性。

# 6) 摄影测量（SfM+MVS）/ 新型可微表示 → 网格

**思路**：多视图图像 → 相机位姿/SfM → MVS 点云 → Poisson/MC 网格；或先用**3D 高斯**等表示，再转网格。
**代表**：

* **COLMAP**（业界常用 SfM+MVS 管线）([demuc.de][23], [COLMAP][24])
* **3D Gaussian Splatting（3DGS）**（新型表示，常配 **SuGaR** 做高质量**从高斯到网格**的提取/渲染）([arXiv][25], [CVF开放获取][26])
  **优缺点**：对实拍场景稳健、尺度大；但移动物体/重复纹理/无纹区域、法向估计会影响网格质量。

---

## 快速选型建议（按你的输入/目标）

* **有点云/深度数据（激光/相机）** → 优先 **Poisson/Screened Poisson**；稀疏/噪声点云可先做法向估计/滤波。([Department of Computer Science][4])
* **单/少视图图像 → 可编辑网格** → **InstantMesh/MeshLRM** 等前馈式；需更高保真时再以 **SDS/VSD** 做几分钟级细化。([arXiv][27])
* **文本到 3D** → 先选 **DreamFusion/ProlificDreamer/Fantasia3D** 这类优化式，几何用 **DMTet** 提取稳定网格，再做 UV/材质。([dreamfusion3d.github.io][16], [NeurIPS 会议录][17], [arXiv][9])
* **实拍多图场景** → **COLMAP**（SfM+MVS）→ **Poisson**；或 **3DGS** → **SuGaR** 网格化以获得更易编辑的资产。([demuc.de][23], [CVF开放获取][26])

## 数据与评测基准（常见）

* **数据集**：ShapeNet / ShapeNetPart、Objaverse / Objaverse-XL、Thingi10K（更贴近 3D 打印/真实脏网格）。([arXiv][28], [shapenet.org][29], [NeurIPS 会议录][30])
* **常用指标**：Chamfer-L2、F-score/IoU、法向一致性（不同论文实现略有差异，需对齐评测脚本）。
