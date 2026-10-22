# 12 — Dreamer 4: Training Agents Inside of Scalable World Models (2025)

- 论文：arXiv:2509.24527（Danijar Hafner et al., DeepMind）
- **官方代码未发布**。可用的实现：
  - `https://github.com/lucidrains/dreamer4`（lucidrains，PyTorch，有 action-conditioned moving-MNIST 的可跑示例，`pip install dreamer4`）
  - `https://github.com/nicklashansen/dreamer4`（非官方 PyTorch，后续被作者本人的 "Hallucination in World Models" 仓库替代，据称训练配方更好）

## 为什么重要

把 Dreamer 系列从 **online RL**（必须和环境交互）推到 **纯离线**：
> 只给一批离线视频+动作数据，训出世界模型，然后**完全在想象里**训策略，策略直接在真环境里工作。

在 Minecraft 上实现了从离线数据学会「挖钻石」这个长程任务，且不需要环境交互、不需要奖励标注的在线探索。

## 数据

- **OpenAI VPT 的 Minecraft contractor 数据**，约 2500 小时人类游玩录像（带键鼠动作标注）
- 关键设定：**offline-only**。不允许在训练过程中与 Minecraft 交互采新数据（这是和 DreamerV3 挖钻石实验的根本区别，V3 用了 1e8 级别的在线交互）
- 也在小规模可控数据（如 bouncing square）上做验证

## 架构

```
tokenizer:  视频 → latent（空间压缩，逐帧对应，不做时间压缩）
dynamics:   block-causal Transformer
            - 帧内：全注意力（同一帧的 token 互相可见）
            - 跨帧：causal（只能看过去）
            动作以 token 级条件注入每一帧的 block
生成方式:   扩散（在 latent 上），但用 shortcut forcing 做极少步采样
```

两个设计要点：

1. **Block-causal attention**：帧内双向 + 帧间因果。既保留了帧内空间全局建模能力，又支持逐帧自回归交互。这是交互式世界模型的标准注意力结构（Oasis、Matrix-Game 都类似）。

2. **Shortcut forcing**（本文核心技术）：
   - **Diffusion Forcing** 让每帧独立噪声水平，支持自回归；
   - **Shortcut models** 让扩散模型学会"一步跨过多个去噪步"（把步长也作为条件输入，训练时加自一致性约束）；
   - 两者结合 = **自回归 + 极少步（~4 步）生成**，在单 GPU 上接近实时。
   - 对比：DIAMOND 用 EDM 的 N=3 步，GameNGen 用蒸馏后的 4 步 DDIM，Dreamer 4 是把少步能力直接写进训练目标。

## 训练方法：三阶段

```
1. 世界模型预训练（大部分数据，可含无动作数据）
   - tokenizer + block-causal Transformer
   - shortcut forcing 目标（扩散损失 + 自一致性损失）

2. Agent 微调：在同一个 backbone 上接上 policy head 和 reward head
   - 用离线数据做行为克隆式的初始化（policy 预测数据里的动作）
   - 共享世界模型的表征，不另起一个网络 ← 这是"把 agent 放进世界模型里"的字面含义

3. 想象内 RL
   - 从离线数据的状态出发，用世界模型 rollout
   - actor-critic（Dreamer 风格 λ-return），**完全不碰真环境**
```

与 DreamerV3 的对照：

| | DreamerV3 | Dreamer 4 |
|---|---|---|
| 表征 | RSSM（GRU + categorical latent） | tokenizer + Transformer latent |
| 生成 | 一步预测分布 | 少步扩散（shortcut forcing） |
| 数据 | online replay，需环境交互 | **纯离线** |
| 策略 | 独立 actor-critic 网络 | 与 WM 共享 backbone 的 head |
| Minecraft 钻石 | 需 1e8 在线步 | 离线数据 + 想象 |

## 复现建议

- 官方无代码，不要指望一次复现论文结果。
- `lucidrains/dreamer4` 适合**学架构**：它有一个 action-conditioned moving MNIST / bouncing square 的最小示例，能在单卡几小时内看到"世界模型确实跟随动作"，是验证自己理解的好 sanity check。
- `nicklashansen/dreamer4` 更偏实际训练配方（更大数据和模型）。
- 想用在自己任务上，最小可行路径：
  1. 准备 (frame, action) 配对数据，动作离散化或做线性 embedding
  2. 用现成 image VAE 做 tokenizer（**确认没有时间压缩**）
  3. block-causal Transformer + diffusion forcing（独立噪声水平）先跑通多步预测
  4. 再考虑 shortcut forcing 加速和想象内 RL
