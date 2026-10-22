# 11 — 驾驶世界模型（Vista / GAIA-1 / DriveDreamer / 占据栅格系）

自动驾驶是 action-conditioned 世界模型落地最积极的方向，因为「给定一条候选轨迹，预测未来会发生什么」正好是规划需要的东西。

| 方法 | 代码 | 动作形式 | 输出空间 |
|---|---|---|---|
| **Vista** (NeurIPS 2024) | ✅ `https://github.com/OpenDriveLab/Vista` | 轨迹/转角+速度/指令/目标点 | 前视视频 |
| GAIA-1 (Wayve) | ❌ | 转角 + 速度 + 文本 | 视频 |
| DriveDreamer / -2 | ✅ `https://github.com/f1yfisher/DriveDreamer` | 轨迹 + HDMap + 3D box | 多视角视频 |
| DrivingDiffusion / MagicDrive | ✅ | BEV layout + 相机参数 | 多视角视频 |
| OccWorld / OccSora | ✅ `https://github.com/wzzheng/OccWorld` | ego 轨迹 | **3D 占据栅格** |
| GenAD / OpenDV | ✅ `https://github.com/OpenDriveLab/DriveAGI` | 文本/指令 | 视频（含 1700h 数据集） |
| ReSim / WorldEngine | ✅ OpenDriveLab | 含非专家动作 | 视频 / 3DGS 仿真 |

---

## 11.1 Vista（重点，开源最完整）

论文：*Vista: A Generalizable Driving World Model with High Fidelity and Versatile Controllability*, arXiv:2405.17398

### 数据
- **OpenDV-YouTube**：1700+ 小时 YouTube 驾驶视频（比 nuScenes 大约 300 倍），**无动作标注**，用于学高保真动态和泛化
- **nuPlan**：带 ego 轨迹/速度/转角的真实驾驶数据，用于学动作条件
- 评测还用 nuScenes、Waymo、CODA 等做零样本泛化
- 分辨率 576×1024（比大多数世界模型高得多），10 Hz，25 帧片段

### 架构
基于 **Stable Video Diffusion (SVD)**，latent video diffusion：
```
图像 VAE → latent → 3D U-Net（空间注意力 + 时间注意力/卷积） → 去噪
条件：
  - 参考帧（image condition，CLIP + latent concat）
  - 动作：多种形式各自经 MLP/Fourier embedding → 统一维度 → cross-attention / 加到 timestep embedding
```
**动作支持 5 种粒度，统一处理**：
1. 高层指令（左/直/右）
2. 目标点（2D 坐标）
3. 轨迹（未来若干个 waypoint）
4. 转向角
5. 速度

训练时**随机丢弃/选择某一种动作模态**（classifier-free 风格的条件 dropout），所以推理时可以只给其中任意一种。这个设计很值得借鉴。

### 训练方法：两阶段

**阶段一：高保真未来预测（只用视频，无动作）**
- 在 OpenDV-YouTube 上微调 SVD
- 两个"动态增强"损失（本文方法贡献）：
  - **动态先验的潜空间替换 / 动态增强损失**：在 latent 上加对高频结构和时间差分敏感的加权项，抑制 SVD 原生的"过度平滑、运动幅度小"问题
  - 显式强化相邻帧差分的监督，提升大幅运动与结构细节
- 结果：学会真实的交通动态，但还不可控

**阶段二：动作条件注入（用 nuPlan 等带标注数据）**
- 冻结大部分骨干，只高效训练动作条件分支（低成本，避免大数据学到的泛化被小数据破坏）
- 这个「大数据学动态 → 小数据学控制」的解耦是驾驶世界模型的通用配方

### 长程续推：latent replacement
单次只能生成 25 帧。续推时把**上一段末尾若干帧的 latent 直接作为下一段的条件帧 latent**（而不是解码成像素再重编码，避免累积 VAE 误差），配合噪声调度，可以连续生成数秒到十几秒。

### 当奖励函数用
不需要真实动作标签：给定一个候选动作，计算世界模型对该动作条件下预测的**不确定性/似然**，作为该动作的奖励。可用于无标注地评估规划器输出。

### 代码入口
```
Vista/
├── train.py                   # 两阶段训练（配置在 configs/）
├── sample.py                  # 给定参考帧 + 动作 → 生成
├── vwm/modules/diffusionmodules/   # 动作条件注入、动态增强损失
├── docs/TRAINING.md           # 数据准备（OpenDV + nuPlan）
└── docs/INSTALL.md
```
> 训练成本高（论文用 8×A100 量级，且需要几 TB 视频数据）。实际建议：**用放出的权重做 sample / 微调动作分支**，不要从头复现阶段一。

---

## 11.2 GAIA-1（Wayve, arXiv:2309.17080）— 无代码但架构经典

- 数据：Wayve 自采 4700 小时伦敦驾驶数据
- 架构：**离散化一切**
  - 图像 → VQ token
  - 动作（转角、速度）→ 离散化成 token
  - 文本 → token
  - 一个 **autoregressive Transformer（9B）** 在 (text, image, action) 混合 token 序列上做 next-token
  - 另有一个**视频扩散 decoder** 把预测的 token 解回高清视频
- 启发：「自回归 Transformer 做动态 + 扩散模型做渲染」这个分工被后来很多工作沿用（解决了 token 保真度不足的问题）

---

## 11.3 占据栅格路线（OccWorld / OccSora / Drive-OccWorld）

不预测像素，预测 **3D occupancy grid** 的演化。

- 数据：nuScenes + Occ3D 占据标注
- 架构：占据栅格 → VQ-VAE / scene tokenizer → spatial-temporal Transformer（GPT 风格）
- **同时预测 ego 轨迹和场景占据演化**（self-supervised 的联合预测），所以它既是世界模型也是 planner
- 优点：输出结构化、可直接用于规划、不需要渲染保真度；缺点：依赖占据标注，信息量比图像少

---

## 做驾驶世界模型的实用建议

1. **动作条件别只用轨迹**。Vista 的多模态动作（指令/目标点/轨迹/角速度）+ 随机 dropout 是最实用的配方。
2. **数据分两层**：海量无标注视频学动态与泛化，少量带 ego 状态数据学控制。单独用 nuScenes（仅 ~5.5 小时）必然过拟合。
3. **长程生成用 latent 级续接**，不要像素级重编码。
4. **评测要分开看**：视觉质量（FVD）和动作一致性（给定轨迹，用感知/IDM 反推生成视频里的 ego 运动，算误差）是两件事，很多工作只报前者。
5. 工作区里已有 `DriveAgent-R1`、`DriveVLA-W0`、`OpenDriveVLA`、`LAW`、`navsim`，这条线和 VLA + 世界模型结合（用 WM 给 VLA 提供未来预测/奖励）正是当前热点，ImagiDrive / WorldEngine 可作为参考。
