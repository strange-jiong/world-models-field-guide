# Action-Conditioned World Models 阅读 / 复现索引

聚焦 **action-conditioned** 世界模型：给定历史观测 $o_{\le t}$ 与动作序列 $a_{\le t}$，预测 $o_{t+1}$（以及 reward / done），并用这个模型做 planning 或在"想象"里训策略。

```
world-models/
├── README.md               # 本文件：总索引 + 横向对比
├── article.md              # 科普长文：四大类 + 四条技术暗线 + 谱系图（16 张 SVG 配图）
├── article.pdf             # 上面那篇的 PDF（A4，19 页）
├── article.src.md          # 替换配图前的版本，图是 ASCII / mermaid
├── notes/                  # 每篇一个 md：数据 / 架构 / 训练方法 / 代码入口
├── figures/                # 配图：.drawio 源文件 + .svg（文档用）+ .png（预览）
│   ├── make_figures.py     # 全部配图的生成脚本（改图改这里，不要手改 svg）
│   └── patch_article.py    # 把文中代码块替换为配图引用
├── assets/pdf.css          # PDF 排版样式
├── scripts/
│   ├── clone_repos.sh      # 一键拉取所有官方仓库到 repos/
│   └── build_pdf.sh        # article.md -> article.pdf
└── repos/                  # 代码落地目录（git clone 目标）
```

## 首次 clone 后要做的一步

`notes/` 里引用了 15 张**论文原图**。这些图的著作权归各篇论文作者，本仓库不收录，需要本地一键重建（会从 arXiv 下载 PDF 到缓存，约 150MB）：

```bash
bash figures/papers/fetch_all.sh
```

不跑这一步的话，`notes/01` ~ `notes/10` 里的论文架构图会显示为断链；其余内容（自绘配图、正文、`article.pdf`）都不受影响。细节与版权说明见 [figures/papers/README.md](figures/papers/README.md)。

重新生成自绘配图与 PDF：

```bash
python3 figures/make_figures.py        # 全部重画（.drawio + .svg + .png）
python3 figures/make_figures.py rssm   # 只重画某一张
bash scripts/build_pdf.sh              # 重新导出 PDF
```

配图用 `drawio-diagram` skill 生成，依赖 `drawio` CLI + `xvfb-run`（导出图片用）。

> 想先读连贯的叙述而不是逐篇笔记，看 [article.md](article.md)：按"未来存在什么形式里"分成四大类，附谱系图与时间线。

## 笔记目录

| # | 笔记 | 方法 | 年份 | 代码 |
|---|------|------|------|------|
| 00 | [overview](notes/00_overview.md) | 分类体系、共性设计、踩坑清单 | — | — |
| 01 | [action-conditional video prediction](notes/01_action_conditional_video_prediction.md) | Oh et al. / CDNA / SV2P | 2015-2018 | ✅ |
| 02 | [World Models (Ha & Schmidhuber)](notes/02_ha_world_models.md) | VAE + MDN-RNN + CMA-ES | 2018 | ✅ |
| 03 | [PlaNet](notes/03_planet.md) | RSSM + latent CEM-MPC | 2019 | ✅ |
| 04 | [SimPLe](notes/04_simple.md) | 像素级视频模型 + PPO in model | 2020 | ✅ |
| 05 | [Dreamer v1/v2/v3](notes/05_dreamer_family.md) | RSSM + imagination actor-critic | 2020-2023 | ✅ |
| 06 | [TD-MPC / TD-MPC2](notes/06_tdmpc2.md) | decoder-free 隐式模型 + MPPI | 2022-2024 | ✅ |
| 07 | [IRIS / Δ-IRIS](notes/07_iris.md) | VQ token + autoregressive Transformer | 2023-2024 | ✅ |
| 08 | [DIAMOND](notes/08_diamond.md) | 像素空间 EDM diffusion WM | 2024 | ✅ |
| 09 | [iVideoGPT](notes/09_ivideogpt.md) | 压缩 tokenizer + 大规模预训练 | 2024 | ✅ |
| 10 | [Oasis / MineWorld / GameNGen](notes/10_game_world_models.md) | 实时可玩神经游戏引擎 | 2024-2025 | 部分 |
| 11 | [Vista / 驾驶世界模型](notes/11_driving_world_models.md) | SVD + 多模态动作控制 | 2024 | ✅ |
| 12 | [Dreamer 4](notes/12_dreamer4.md) | tokenizer + shortcut forcing，离线想象训练 | 2025 | 非官方 |

## 横向对比

### 数据

| 方法 | 环境/数据集 | 规模 | 采集方式 | 动作形式 |
|------|------------|------|---------|---------|
| Oh et al. 2015 | Atari (ALE) | ~500k-1M frames/game | DQN 策略 + ε 随机 | 离散 (18) |
| CDNA | BAIR robot pushing | 59k 推动序列 | 随机 poking | 7-DoF 末端增量 |
| Ha & Schmidhuber | CarRacing-v0 / VizDoom | 10k rollouts | **纯随机策略，一次离线采完** | 连续 3 维 / 离散 |
| PlaNet | DMControl (pixel) | ~1k episodes | **online，策略自采（replay）** | 连续 |
| SimPLe | Atari 100k | 100k 真实交互步 | online，模型内策略回采 | 离散 |
| DreamerV2/V3 | Atari 200M / DMC / Minecraft | 到 1e8 步 | online replay | 离散+连续 |
| TD-MPC2 | DMC/MetaWorld/ManiSkill 多任务 | 545 任务、10M+ 轨迹 | online + 多任务离线 | 连续（统一维度填充） |
| IRIS | Atari 100k | 100k 步 | online，交替采集/训练 | 离散 |
| DIAMOND | Atari 100k + CS:GO | 100k 步 / 95h 人类录像 | online / **纯离线** | 离散 |
| iVideoGPT | Open X-Embodiment, SSv2, RoboNet, BAIR | 百万级轨迹预训练 | **离线大规模预训练 + 下游微调** | 连续机械臂动作 |
| Oasis / MineWorld | OpenAI VPT contractor 数据 | 数千小时 Minecraft | **纯离线人类录像（带键鼠标注）** | 键盘 + 鼠标 |
| Vista | OpenDV-YouTube + nuPlan | 1700h + nuPlan | **纯离线驾驶视频** | 轨迹/角速度/速度/指令/目标点 |
| Dreamer 4 | VPT Minecraft | 2.5k 小时 | **纯离线**（强调无需环境交互） | 键鼠 |

### 架构

| 方法 | 表征 | 动态模型 | 动作注入位置 | 是否重建像素 |
|------|------|---------|------------|------------|
| Ha & Schmidhuber | CVAE, z∈R^32 | MDN-LSTM（混合高斯） | RNN 输入拼接 | ✅ |
| PlaNet | RSSM: 确定 h(GRU) + 随机 s(Gauss) | GRU + prior/posterior | GRU 输入拼接 | ✅（decoder + reward head） |
| SimPLe | 无隐状态，像素堆叠 | 卷积前馈 + 离散隐变量 | 逐通道 broadcast 到特征图 | ✅ |
| DreamerV2/V3 | RSSM，**categorical** 32×32 | GRU + 离散 prior | GRU 输入拼接 | ✅ |
| TD-MPC2 | MLP latent（**无 decoder**） | MLP 一步预测 | MLP 输入拼接 | ❌（只有 reward/Q/consistency） |
| IRIS | VQ-VAE 离散 token（16 tok/帧） | GPT，序列 = (帧 token…, 动作 token) | **作为一个序列 token** | ✅（tokenizer 解码） |
| DIAMOND | **无隐空间，直接像素** | EDM U-Net diffusion | 时间 embedding 通道 + cross/adaLN | ✅（本身就是像素） |
| iVideoGPT | 压缩 tokenizer（context 稠密 / 后续稀疏） | GPT (llama-style) | 动作 embedding 加到 token | ✅ |
| Oasis | 空间 VAE latent | DiT + Diffusion Forcing | per-frame action embedding → adaLN | ✅ |
| MineWorld | VQ token | autoregressive Transformer | 动作 token 前置于帧 token | ✅ |
| Vista | SVD latent (3D UNet) | latent video diffusion | 动作经 MLP → 交叉注意力 / 相加条件 | ✅ |
| Dreamer 4 | tokenizer latent | block-causal Transformer + shortcut 扩散 | token 级动作条件 | ✅ |

### 训练方法

| 方法 | WM 损失 | 策略学习 | 关键 trick |
|------|--------|---------|-----------|
| Ha & Schmidhuber | 分三阶段：VAE(recon+KL) → MDN-RNN(NLL) → controller | **CMA-ES 进化线性控制器** | dream 温度 τ 抑制模型利用（model exploitation） |
| PlaNet | ELBO + **latent overshooting**（多步一致） | 无策略，**CEM 在隐空间做 MPC** | action repeat；少量 seed episode |
| SimPLe | 像素 L2 + 离散隐变量 | **PPO 完全在模型内训**，多轮交替 | 迭代式 data collection；scheduled sampling |
| DreamerV1 | ELBO（recon+reward+KL） | actor-critic，**价值梯度反传穿过想象轨迹**，H=15，λ-return | — |
| DreamerV2 | 同上，离散 latent | 离散动作用 **REINFORCE**，连续用 reparam | **KL balancing**、straight-through |
| DreamerV3 | symlog recon + two-hot reward + free-bits KL | 同 V2 + **return percentile 归一化** | 一套超参跨 150+ 任务；Minecraft 挖钻石 |
| TD-MPC2 | latent consistency + reward + TD(Q)，**无重建** | MPPI planning + prior policy（MaxEnt） | SimNorm 归一化、多任务 action/obs padding |
| IRIS | tokenizer: VQ+perceptual；WM: 交叉熵（token/reward/done） | **纯想象内 actor-critic**（REINFORCE+λ） | tokenizer 与 WM 分开优化；H=20 |
| DIAMOND | EDM denoising score matching | 想象内 actor-critic | **少步采样 (N=3)**，噪声增强防漂移；CSGO 版加超分 |
| iVideoGPT | token 交叉熵（+ tokenizer VQ/GAN 损失） | 下游：MBRL（MPC / Dreamer 式）或视觉 RL | **compressive tokenization** 降 token 数；预训练→微调范式 |
| Oasis / MineWorld | diffusion / next-token | 不训策略（纯世界模型/游戏引擎） | Diffusion Forcing 独立噪声水平；对角解码加速 |
| Vista | 两阶段：高保真（含动态增强损失）→ 动作条件注入 | 不训策略，提供 **reward 用于动作评估** | latent replacement 做长程续推；多模态动作统一 |
| Dreamer 4 | shortcut forcing（few-step 扩散） | **离线想象内 RL**，无需环境交互 | 大 tokenizer + 块因果注意力；实时推理 |

## 怎么用这个目录

```bash
cd /home/shanghai/work/08_expe_code/world-models
bash scripts/clone_repos.sh            # 全部（较大，建议挑着来）
bash scripts/clone_repos.sh iris diamond tdmpc2   # 只拉指定几个
```

建议阅读顺序：`00 → 02 → 03 → 05 → 07 → 08 → 09/10/11`。
动手优先级（跑得起来 + 代码干净）：**IRIS → DIAMOND → dreamerv3-torch → TD-MPC2**。
