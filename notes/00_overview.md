# 00 — Action-Conditioned 世界模型总览

## 问题定义

学一个条件生成模型
$$p_\theta(o_{t+1}, r_t, d_t \mid o_{\le t}, a_{\le t})$$
动作 $a$ 是外部给定的控制输入（不是模型自己采样的），所以叫 action-conditioned。有了它可以：

1. **Planning**：在模型里 rollout 候选动作序列，选最优（PlaNet、TD-MPC2）。
2. **Imagination 训策略**：把模型当廉价模拟器，在里面跑 RL（Dreamer、SimPLe、IRIS、DIAMOND）。
3. **Neural simulator / 游戏引擎**：不训策略，只要可交互可控的高保真生成（Oasis、GameNGen、Vista）。

## 三条技术路线

### A. 低维隐空间 + 递归状态（RSSM 系）
`CNN encoder → 隐状态 (h, z) → GRU 递推 → decoder/reward head`

- 代表：Ha & Schmidhuber (2018)、PlaNet、Dreamer v1/v2/v3、DayDreamer
- 优点：rollout 极便宜（隐空间几十到上千维），适合 1e5~1e8 步 online RL
- 缺点：细节丢失（小物体、稀有事件），重建损失被大面积背景主导
- 关键演化：Gaussian latent → **categorical latent**（V2）→ symlog + free bits（V3）

### B. 离散 token + 自回归 Transformer
`VQ tokenizer → (frame tokens, action token) 交错序列 → GPT next-token`

- 代表：IRIS、Δ-IRIS、iVideoGPT、MineWorld、TWM/STORM
- 优点：动作天然就是一个 token，长上下文记忆强，能直接复用 LLM 基建
- 缺点：token 数 = 帧数 × 每帧 token 数，长序列贵；需要先训好 tokenizer
- 关键演化：降低每帧 token 数（Δ-IRIS 预测 **帧间 delta**，iVideoGPT 用 **compressive tokenization**）

### C. 扩散 / 流匹配
`past L frames + L actions 作为条件 → denoiser 预测下一帧（或下一段 latent）`

- 代表：DIAMOND（像素空间 EDM）、Oasis、GameNGen、Vista、Dreamer 4
- 优点：视觉保真度高，多模态未来建模自然
- 缺点：每帧多步去噪 → 慢；autoregressive rollout 会误差累积/漂移
- 关键演化：**少步采样**（DIAMOND N=3、shortcut forcing）、**Diffusion Forcing**（每帧独立噪声水平，天然支持自回归）

## 动作注入的几种做法（实现细节，最容易抄错的地方）

| 做法 | 适用 | 例子 |
|------|------|------|
| 向量拼接进 RNN 输入 | 低维隐空间 | PlaNet / Dreamer: `GRU([z_t, a_t], h_t)` |
| 单独做一个序列 token | 自回归 Transformer | IRIS: `[z_t^1..z_t^K, a_t, z_{t+1}^1..]` |
| broadcast 到特征图通道 | 卷积像素模型 | SimPLe、CDNA |
| 加到时间步 embedding / adaLN | DiT、U-Net | Oasis、DIAMOND |
| 经 MLP 后做 cross-attention | 视频扩散 | Vista（轨迹/角速度/指令） |

## 长期 rollout 的通病与对策

- **误差累积漂移**：训练只见 ground-truth 历史，推理见自己生成的历史（exposure bias）。
  - 对策：scheduled sampling（SimPLe）、训练时给历史帧加噪（DIAMOND）、Diffusion Forcing、多步预测损失（PlaNet latent overshooting）。
- **Model exploitation**：策略找到模型的漏洞刷分，真环境里没用。
  - 对策：想象 horizon 砍短（Dreamer H=15~20）、采样加温度（Ha 的 τ）、online 回采真实数据修补模型。
- **reward / done 预测**：别忘了这俩头，很多复现挂在 done 预测不准导致想象轨迹无限长。

## 评测维度

- 视频质量：PSNR / SSIM / LPIPS / FVD
- 控制一致性：给定动作序列的轨迹误差、action-following accuracy（MineWorld 用 IDM 反推动作判断是否跟随）
- 下游 RL：Atari 100k human-normalized score、DMC 500k return
- 长程稳定性：rollout 100+ 帧后是否崩

## 代码质量主观评价（想动手时的优先级）

1. **IRIS** — 单卡可跑，结构清晰，token-based WM 最好的教学代码
2. **DIAMOND** — 代码干净，带可交互 play 脚本，diffusion WM 入门
3. **dreamerv3-torch**（NM512）— 比官方 JAX 版好读，RSSM 必读
4. **TD-MPC2** — 工程化好，多任务，但和"生成式世界模型"距离较远
5. **iVideoGPT** — 有预训练权重，适合直接接机器人任务
6. **Vista** — 驾驶方向最完整的开源 action-conditioned 视频世界模型，但训练成本高
