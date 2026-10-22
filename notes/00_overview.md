# 00 — Action-Conditioned 世界模型总览

## 问题定义

学一个条件生成模型

$$p_\theta(o_{t+1},\, r_t,\, d_t \mid o_{\le t},\, a_{\le t})$$

动作 $a$ 是外部给定的控制输入（不是模型自己采样的），所以叫 action-conditioned。有了它可以：

1. **Planning**：在模型里 rollout 候选动作序列，选最优（PlaNet、TD-MPC2）
2. **Imagination 训策略**：把模型当廉价模拟器，在里面跑 RL（Dreamer、SimPLe、IRIS、DIAMOND）
3. **Neural simulator / 游戏引擎**：不训策略，只要可交互可控的高保真生成（Oasis、GameNGen、Vista）

## 四大类

![按「未来存在什么形式里」分出的四大类](../figures/wm-taxonomy.svg)

区分这些方法最本质的一刀是**未来被存成什么形式**：

### A. 低维隐空间 + 递归状态（RSSM 系）→ [note 02](02_ha_world_models.md) · [03](03_planet.md) · [05](05_dreamer_family.md) · [06](06_tdmpc2.md)

`CNN encoder → 隐状态 (h, z) → GRU 递推 → decoder/reward head`

- 代表：Ha & Schmidhuber (2018)、PlaNet、Dreamer v1/v2/v3、DayDreamer、TD-MPC2
- 优点：rollout 极便宜（隐空间几十到上千维），适合 1e5~1e8 步 online RL
- 缺点：细节丢失（小物体、稀有事件），重建损失被大面积背景主导
- 关键演化：Gaussian latent → **categorical latent**（V2）→ symlog + free bits（V3）→ 干脆**砍掉 decoder**（TD-MPC2）

### B. 离散 token + 自回归 Transformer → [note 07](07_iris.md) · [09](09_ivideogpt.md) · [10](10_game_world_models.md)

`VQ tokenizer → (frame tokens, action token) 交错序列 → GPT next-token`

- 代表：IRIS、Δ-IRIS、iVideoGPT、MineWorld、GAIA-1、TWM/STORM
- 优点：动作天然就是一个 token，长上下文记忆强，能直接复用 LLM 基建
- 缺点：token 数 = 帧数 × 每帧 token 数，长序列贵；需要先训好 tokenizer
- 关键演化：降低每帧 token 数（Δ-IRIS 预测 **帧间 delta**，4 token/帧；iVideoGPT 用 **compressive tokenization**，16 token/帧）

### C. 扩散 / 流匹配 → [note 08](08_diamond.md) · [10](10_game_world_models.md) · [11](11_driving_world_models.md) · [12](12_dreamer4.md)

`past L frames + L actions 作为条件 → denoiser 预测下一帧（或下一段 latent）`

- 代表：DIAMOND（像素空间 EDM）、Oasis、GameNGen、Vista、Dreamer 4
- 优点：视觉保真度高，多模态未来建模自然
- 缺点：每帧多步去噪 → 慢；autoregressive rollout 会误差累积/漂移
- 关键演化：**少步采样**（DIAMOND n=3、shortcut forcing）、**Diffusion Forcing**（每帧独立噪声水平，天然支持自回归）

### D. 动作标注从哪来 → [note 01](01_action_conditional_video_prediction.md) · [10](10_game_world_models.md) · [11](11_driving_world_models.md)

这条是数据侧的暗线，决定了前三类能有多大：

- 早期自采数据，动作标注**免费**（Oh et al.、CDNA）
- 互联网视频**没有动作标注** → 三种破局：找带标注的大数据（VPT）、有/无动作混合训练（iVideoGPT）、**自己发明动作**（Genie 的 latent action）

## 全景谱系

![全景谱系：时间纵向，技术路线横向](../figures/genealogy.svg)

实线 = 直接继承，虚线 = 影响 / 借鉴，颜色 = 所属技术路线。

## 动作注入的几种做法

实现细节里最容易抄错的地方：

| 做法 | 适用 | 例子 | 备注 |
|---|---|---|---|
| 向量拼接进 RNN 输入 | 低维隐空间 | PlaNet / Dreamer: `GRU([z_t, a_t], h_t)` | 最通用 |
| 乘法交互 $h \odot W_a a$ | 卷积 | Oh et al. 2015 | 离散动作上优于 concat |
| broadcast 到特征图通道 | 卷积像素模型 | CDNA、SimPLe | 低维动作进卷积网的标准做法 |
| **单独做一个序列 token** | 自回归 Transformer | IRIS: `[z_t^1..z_t^K, a_t, z_{t+1}^1..]` | 最优雅，但要求每帧都有标注 |
| **加到视觉 token 的 embedding 上** | 自回归 Transformer | iVideoGPT | 不改序列长度，**可与无动作数据混训** |
| 加到噪声水平 embedding / adaLN | DiT、U-Net | DIAMOND、GameNGen、Oasis | 扩散系默认做法 |
| 经 MLP 后多模态条件注入 | 视频扩散 | Vista（轨迹/角速度/指令/目标点/速度） | 支持多种粒度并存 |

IRIS 和 iVideoGPT 那组对比最值得记：一个为了优雅把动作当独立 token，一个为了能吃无标注数据牺牲优雅改成相加。**架构选择是被数据策略倒逼的**，不是纯粹的审美问题。

## 四条反复出现的技术暗线

同一个问题在不同类别里被反复解决，解法之间能互相启发。

### 暗线一：自回归漂移（exposure bias）

训练时模型看干净的真实历史，推理时看自己生成的脏历史。这个错配让长 rollout 必然崩。

```
2015  多步 curriculum / scheduled sampling        Oh et al. · CDNA · SimPLe
2019  隐空间多步预测一致性（latent overshooting）  PlaNet
2020  想象起点取真实状态 + 砍短 horizon           SimPLe · Dreamer · IRIS
2024  训练时给条件历史帧加噪                       DIAMOND · GameNGen
2024  每帧独立噪声水平（Diffusion Forcing）        Oasis · Dreamer 4
```

**五代解法，思路是收敛的**：从"训练时模仿推理的样子"（curriculum）走到"让训练目标本身就等价于推理模式"（Diffusion Forcing）。

### 暗线二：model exploitation（策略钻模型的漏洞）

只在第 ② 种用法里出现，但一旦出现就很致命。

| 年份 | 做法 | 代表 |
|---|---|---|
| 2018 | 采样时放大方差（温度 τ=1.15），让梦更不确定 | Ha & Schmidhuber |
| 2020 | 想象 horizon 砍到 15~20 步，起点取真实状态 | SimPLe · Dreamer · IRIS |
| 2020+ | online 交替采集：策略去哪钻漏洞，就在那里补真实数据 | Dreamer · IRIS · DIAMOND |
| 2024 | horizon 只留 3 步，之后交给学出来的 Q | TD-MPC2 |

核心都是**不让策略在模型里跑太远**。Dreamer 4 的纯离线设定下没法补数据，所以它更依赖世界模型本身的质量——这也是它必须把模型做大的原因。

### 暗线三：尺度对齐与"不用调参"

不同任务的奖励尺度、观测尺度差几个数量级，是世界模型长期需要逐任务调参的根本原因。

```
DreamerV2  →  离散隐变量 + KL balancing            （分布层面的尺度问题）
DreamerV3  →  symlog + two-hot + free bits
              + 回报分位数归一化                    （一套超参跑 150+ 任务）
TD-MPC2    →  SimNorm + two-hot 离散回归           （一套超参跑 104 个任务）
Dreamer 4  →  所有损失项按 RMS 运行估计归一化
```

DreamerV3 和 TD-MPC2 分属不同类别，却**不约而同地把"回归"换成了"离散分类"**（two-hot）。回归重尾分布不稳定，分类稳定——这个经验现在基本是共识。

### 暗线四：交互式世界模型的硬约束

**latent 不能做时间维压缩。** latent 必须逐帧对应原始帧，否则无法每帧接受一个新动作。而市面上绝大多数现成的视频 VAE 都带 4× 或 8× 时间压缩——拿来就用会发现模型根本不响应动作。

- Oasis 专门用了只做空间压缩的 VAE
- Vista 基于 SVD，其 VAE 本身就是逐帧空间压缩
- Dreamer 4 的 causal tokenizer 也是逐帧对应（block-causal）

这是从「视频生成」转「世界模型」时第一个会踩的坑。

## 「奖励」这个词在本仓库的三种含义

容易混，单独说明：

| 含义 | 指什么 | 出现位置 |
|---|---|---|
| ① **环境奖励 r_t / 奖励头** | 世界模型要预测的那个标量，和 obs/done 并列 | 几乎所有 note；IRIS 的是**符号三分类** |
| ② **λ-return / value** | 由 ① 沿想象 horizon 累加 + critic bootstrap 得到的回报 | note 05、07、08 的行为学习部分 |
| ③ **把世界模型当奖励函数** | 不预测环境奖励，用模型自身的预测不确定性给候选动作打分 | 仅 note 11（Vista） |

## 评测维度

- 视觉质量：PSNR / SSIM / LPIPS / FVD
- **控制一致性**：给定动作序列的轨迹误差、action-following accuracy。最靠谱的自动指标是**用 IDM 从生成视频反推动作再比对**（MineWorld 的做法，note 10）
- 下游 RL：Atari 100k human-normalized score、DMC 500k return
- 长程稳定性：rollout 100+ 帧后是否崩（DIAMOND 用"与参考轨迹的平均像素漂移"量化）

> 只报 FVD 是不够的——画得像不说明听指令。这是读这一类论文时第一个该核的点。

## 代码质量主观评价（想动手时的优先级）

1. **IRIS** — 单卡可跑，结构清晰，token-based WM 最好的教学代码
2. **DIAMOND** — 代码干净，带可交互 play 脚本，diffusion WM 入门
3. **dreamerv3-torch**（NM512）— 比官方 JAX 版好读，RSSM 必读
4. **TD-MPC2** — 工程化好，多任务，但和"生成式世界模型"距离较远
5. **iVideoGPT** — 有预训练权重，适合直接接机器人任务
6. **MineWorld** — 游戏方向唯一训练代码全开的
7. **Vista** — 驾驶方向最完整的开源 action-conditioned 视频世界模型，但训练成本高

## 三个最常见的坑

- **continue / done 预测没训好** → 想象轨迹跨过回合边界，value 直接爆炸。复现 Dreamer 系最高频的失败原因
- **拿现成视频 VAE 做交互式世界模型** → 大多带时间压缩，模型完全不响应动作（见暗线四）
- **只看 FVD/PSNR 评估** → 动作一致性要单独测
