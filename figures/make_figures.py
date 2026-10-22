#!/usr/bin/env python3
"""
生成 article.md 的全部配图。

用法:
    python3 figures/make_figures.py            # 生成全部
    python3 figures/make_figures.py rssm iris  # 只生成指定图

每张图产出 <name>.drawio (可编辑源) + <name>.svg (文档引用) + <name>.png (肉眼检查)
"""

import os
import sys

SKILL = os.path.expanduser("~/.kiro/skills/drawio-diagram/scripts")
sys.path.insert(0, SKILL)
from drawio_builder import Diagram  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES = {}


def figure(name):
    def deco(fn):
        FIGURES[name] = fn
        return fn
    return deco


# ============================================================ §0 三种用法
@figure("wm-three-uses")
def _():
    d = Diagram(cell_w=210, cell_h=56, gap_x=50, gap_y=56)
    a = d.box("候选动作序列", 0, 0, "green")
    m = d.box("世界模型", 1, 0, "purple", bold=True)
    f = d.box("想象出的未来", 2, 0, "blue")
    d.arrow(a, m)
    d.arrow(m, f)

    u1 = d.box("① 规划\n在脑内试几千条动作，挑最好的\nPlaNet · TD-MPC2",
               0, 1.15, "blue", h=92)
    u2 = d.box("② 想象中训练\n把模型当免费环境，在里面跑 RL\nDreamer · IRIS · DIAMOND",
               1, 1.15, "orange", h=92)
    u3 = d.box("③ 神经模拟器\n不训策略，只要一个能交互的世界\nOasis · GameNGen · Vista",
               2, 1.15, "grey", h=92)
    for u, ex in ((u1, 0.15), (u2, 0.5), (u3, 0.85)):
        d.arrow(m, u, extra_style=f"exitX={ex};exitY=1;entryX=0.5;entryY=0;")
    return d


# ============================================================ §0 四大类
@figure("wm-taxonomy")
def _():
    d = Diagram(cell_w=185, cell_h=66, gap_x=66, gap_y=28)
    o = d.box("观测 o_t\n(图像)", 0, 1, "grey", h=70)

    a = d.box("① 压成小向量\n几十 ~ 上千维", 1, 0, "blue")
    b = d.box("② 切成离散 token\n每帧 16 个符号", 1, 1, "orange")
    c = d.box("③ 不压，直接画像素", 1, 2, "red")

    a2 = d.box("RSSM 系\nWorld Models · PlaNet\nDreamer · TD-MPC2",
               2, 0, "blue", w=250, h=78)
    b2 = d.box("自回归 Transformer 系\nIRIS · iVideoGPT\nMineWorld · GAIA-1",
               2, 1, "orange", w=250, h=78)
    c2 = d.box("扩散系\nDIAMOND · GameNGen\nOasis · Vista · Dreamer 4",
               2, 2, "red", w=250, h=78)

    # 必须显式指定出入口，否则 drawio 自动布线会绕成竖向双箭头
    for x in (a, b, c):
        d.arrow(o, x, extra_style="exitX=1;exitY=0.5;entryX=0;entryY=0.5;")
    for x, y in ((a, a2), (b, b2), (c, c2)):
        d.arrow(x, y, extra_style="exitX=1;exitY=0.5;entryX=0;entryY=0.5;")

    d.box("④ 动作本身从哪来？　CDNA · VPT · Genie latent action\n"
          "↑ 这是三类共同的数据侧前提，决定了能用多少数据",
          1, 3.15, "green", w=501, h=62, bold=True)
    return d


# ============================================================ §1.1 V-M-C
@figure("ha-vmc")
def _():
    d = Diagram(cell_w=150, cell_h=84, gap_x=72, gap_y=24)
    img = d.box("图像\n64×64", 0, 0, "grey", h=84)
    v = d.box("V\nVAE\n(视觉压缩)", 1, 0, "blue", bold=True)
    m = d.box("M\nMDN-RNN\n(记忆 / 动态)", 2, 0, "purple", bold=True)
    c = d.box("C\n线性层\n(控制器)", 3, 0, "orange", bold=True)
    act = d.box("动作", 4, 0, "green", h=84)

    d.arrow(img, v)
    d.arrow(v, m, "z (32 维)")
    d.arrow(m, c, "ẑ_t+1")
    d.arrow(c, act)

    d.box("① 重建损失 + KL\n训完冻结", 1, 1, "grey", h=58)
    d.box("② 预测 z_t+1 | z_t, a_t\n混合高斯 NLL，训完冻结", 2, 1, "grey", h=58)
    d.box("③ CMA-ES 进化\n仅 867 个参数", 3, 1, "grey", h=58)
    d.text("三阶段依次训练，不端到端", 1, 1.72, w=520, size=13, bold=True)
    return d


# ============================================================ §1.2 RSSM
@figure("rssm")
def _():
    d = Diagram(cell_w=120, cell_h=56, gap_x=84, gap_y=50)
    h0 = d.box("h_t−1", 0, 0, "blue")
    h1 = d.box("h_t", 1, 0, "blue", bold=True)
    h2 = d.box("h_t+1", 2, 0, "blue")
    z0 = d.box("z_t−1", 0, 1, "purple")
    z1 = d.box("z_t", 1, 1, "purple", bold=True)
    z2 = d.box("z_t+1", 2, 1, "purple")

    d.arrow(h0, h1, "GRU")
    d.arrow(h1, h2, "GRU")
    # label_x 往起点挪，否则折线中点正好落在 z 框里，标签会压住框内文字
    for zi, hi in ((z0, h1), (z1, h2)):
        d.arrow(zi, hi, "[z, a]",
                extra_style="exitX=1;exitY=0.3;entryX=0.5;entryY=1;",
                label_x=-0.75, label_y=-11)

    d.band("确定性骨架 h —— 负责长期记忆", [h0, h1, h2], "blue")
    d.band("随机状态 z —— 负责表达不确定性", [z0, z1, z2], "purple",
           label_pos="bottom")

    pri = d.box("prior　p(z | h)\n想象时用（看不到图）", 0, 2.45, "yellow",
                w=188, h=66)
    pos = d.box("posterior　q(z | h, CNN(o))\n训练时用（看得到图）", 2, 2.45,
                "green", w=205, h=66)
    d.shift(pos, dx=-17)
    d.arrow(pri, z1, extra_style="exitX=1;exitY=0;entryX=0.3;entryY=1;")
    d.arrow(pos, z1, extra_style="exitX=0;exitY=0;entryX=0.7;entryY=1;")
    d.note("两者用 KL 拉近\n这就是在学「动态」", 1, 2.45, w=126, h=66, color="orange")
    d.shift(d._cells[-1].cid, dx=14)
    return d


# ============================================================ §1.3 Dreamer 想象
@figure("dreamer-imagination")
def _():
    d = Diagram(cell_w=160, cell_h=58, gap_x=20, gap_y=46)
    b = d.box("真实 replay buffer", 0, 0, "green", span=3)
    im = d.box("用 prior + actor 想象 15 步（全程不解码图像）\n"
               "(h, z) → a → (h, z) → a → … → 15 步",
               0, 1, "purple", span=3, h=74, bold=True)
    ac = d.box("actor 最大化 λ-return　｜　critic 回归 λ-return",
               0, 2.15, "orange", span=3)
    d.arrow(b, im, "取出每个时刻的状态 (h, z) 作为想象起点")
    d.arrow(im, ac, "想象轨迹上的 reward 与 value")
    d.note("起点取自真实数据 + horizon 只有 15 步\n"
           "= 不让策略在模型里跑太远（防 model exploitation）",
           0, 3.2, w=520, h=56, color="yellow")
    return d


# ============================================================ §1.4 TD-MPC2
@figure("tdmpc2-components")
def _():
    d = Diagram(cell_w=250, cell_h=46, gap_x=70, gap_y=16)
    e = d.box("encoder　　z = h(o)", 0, 0, "blue")
    dy = d.box("dynamics　 ẑ′ = d(z, a)", 0, 1, "purple")
    r = d.box("reward　　 r̂ = R(z, a)", 0, 2, "orange")
    q = d.box("Q 集合　　 Q(z, a) × 5", 0, 3, "orange")
    p = d.box("policy prior π(z)", 0, 4, "green")
    d.band("五个组件，全是 MLP", [e, dy, r, q, p], "grey")

    loss = d.box("损失 = ‖ẑ′ − sg(h_target(o′))‖²\n"
                 "　　　+ reward 误差 + TD 误差\n\n"
                 "自监督一致性取代重建",
                 1, 1, "yellow", h=120)
    d.arrow(dy, loss, extra_style="exitX=1;exitY=0.5;entryX=0;entryY=0.3;")
    red = d.box("没有 decoder\n没有 KL\n没有随机隐变量", 1, 3, "red", h=78, bold=True)
    d.shift(red, dy=18)
    return d


# ============================================================ §2.1 IRIS 序列
@figure("iris-sequence")
def _():
    d = Diagram(cell_w=62, cell_h=44, gap_x=8, gap_y=52, font_size=12)
    img = d.box("图像", 0, 0, "grey", w=62)

    toks = [d.box(l, i, 1, "blue") for i, l in enumerate(["z¹", "z²", "…", "z¹⁶"])]
    at = d.box("a_t", 4, 1, "orange", bold=True)
    # 第二组往右挪 0.4 格，给两个 band 之间留出不重叠的空隙
    toks2 = [d.box(l, 5.4 + i, 1, "blue")
             for i, l in enumerate(["z¹", "z²", "…", "z¹⁶"])]
    at2 = d.box("a_t+1", 9.4, 1, "orange", bold=True)

    d.arrow(img, toks[0], extra_style="exitX=0.5;exitY=1;entryX=0.5;entryY=0;")
    d.text("VQ-VAE：16 token / 帧", 0.75, 0.42, w=230, size=12)
    d.band("第 t 帧", toks + [at], "blue", pad=7, label_pos="bottom")
    d.band("第 t+1 帧", toks2 + [at2], "blue", pad=7, label_pos="bottom")

    tr = d.box("causal Transformer（10 层）", 0, 2.35, "purple",
               w=694, bold=True)
    # 只画一条居中的汇入箭头，四条并进来会挤成一条横线，反而看不懂
    d.arrow(at, tr, "整条序列送入",
            extra_style="exitX=0.5;exitY=1;entryX=0.5;entryY=0;")

    h1 = d.box("下一帧 token", 1, 3.6, "blue", w=150)
    # 标明是符号三分类，避免和「环境奖励」/「λ-return」混为一谈
    h2 = d.box("奖励\n符号 3 类", 4.1, 3.6, "orange", w=125)
    h3 = d.box("是否结束", 6.5, 3.6, "red", w=130)
    for h in (h1, h2, h3):
        d.arrow(tr, h)
    d.text("动作放在该帧视觉 token 之后 = 「看完这一帧后我做了这个动作」",
           0, 4.4, w=620, size=12, bold=True)
    return d


# ============================================================ §2.2 token 预算
@figure("token-budget")
def _():
    d = Diagram(cell_w=160, cell_h=52, gap_x=26, gap_y=34, margin=195)
    d.text("朴素做法", -0.95, 0, w=150, align="right", bold=True)
    d.text("Δ-IRIS", -0.95, 1, w=150, align="right", bold=True)
    d.text("iVideoGPT", -0.95, 2, w=150, align="right", bold=True)

    for i in range(3):
        d.box(f"帧{i+1}\n256 tok", i, 0, "red")
    d.box("帧1\n256 tok", 0, 1, "red")
    d.box("帧2\n4 tok", 1, 1, "green", w=74)
    d.box("帧3\n4 tok", 2, 1, "green", w=74)
    d.box("context\n16×16 = 256", 0, 2, "red")
    d.box("动态帧\n4×4 = 16", 1, 2, "green", w=74)
    d.box("动态帧\n16", 2, 2, "green", w=74)

    d.note("相邻帧 90% 内容重复，纯浪费", 3.1, 0, w=250, h=56, color="yellow")
    d.note("只编码「给定 o_t 和 a_t 后无法推出的那部分」\n官方配置 4 token/帧，词表 1024",
           3.1, 1, w=250, h=62, color="yellow")
    d.note("后续帧经 cross-attention 看 context，稀疏编码", 3.1, 2, w=250, h=56,
           color="yellow")
    d.text("方框宽度 ∝ token 数。共同思路：只编码变化量",
           -0.95, 3, w=520, size=12, bold=True)
    return d


# ============================================================ §2.3 iVideoGPT
@figure("ivideogpt-stages")
def _():
    d = Diagram(cell_w=175, cell_h=58, gap_x=20, gap_y=44)
    s1 = d.box("阶段一　预训练\n"
               "Open X-Embodiment（有动作）+ Something-Something v2（无动作）\n"
               "150 万条轨迹混在一起训 next-token prediction",
               0, 0, "green", span=3, h=80)
    s2 = d.box("阶段二　微调\n在目标域打开动作分支 + 奖励头，少量数据即可",
               0, 1, "orange", span=3, h=62)
    s3 = d.box("阶段三　当世界模型用\n包成环境，接 MPC 或想象内 actor-critic",
               0, 2, "purple", span=3, h=62)
    d.arrow(s1, s2)
    d.arrow(s2, s3)
    d.note("关键实现细节：动作经线性层「加到」第一个视觉 token 的 embedding 上。\n"
           "不改变序列长度 → 无动作时置零即可 → 有/无标注数据可以混训。",
           0, 3.05, w=575, h=62, color="yellow")
    return d


# ============================================================ §3.1 DIAMOND
@figure("diamond-diffusion")
def _():
    d = Diagram(cell_w=175, cell_h=58, gap_x=52, gap_y=50)
    cond = d.box("条件：过去 4 帧（沿通道维拼接）+ 过去 4 个动作（embedding）",
                 0, 0, "orange", span=3, h=56)
    noise = d.box("纯噪声", 0, 1, "grey")
    unet = d.box("U-Net 去噪器\n(EDM 参数化)", 1, 1, "purple", bold=True)
    out = d.box("下一帧\n真·像素，无隐空间", 2, 1, "blue")
    d.arrow(noise, unet)
    d.arrow(unet, out)
    d.arrow(cond, unet, extra_style="exitX=0.5;exitY=1;entryX=0.5;entryY=0;")

    d.note("动作 embedding 与噪声水平 embedding 相加，\n"
           "再经 adaptive group norm 调制每个残差块",
           0, 1.88, w=330, h=60, color="yellow")
    d.note("只需 3 步 Euler 采样\n← 这是能塞进 RL 循环的前提",
           1.6, 1.88, w=330, h=60, color="green")
    return d


# ============================================================ §3.2 Diffusion Forcing
@figure("diffusion-forcing")
def _():
    d = Diagram(cell_w=110, cell_h=58, gap_x=16, gap_y=44, margin=205)
    d.text("标准视频扩散", -1.25, 0, w=150, align="right", bold=True)
    d.text("Diffusion Forcing", -1.25, 1, w=150, align="right", bold=True)

    for i in range(4):
        d.box("σ", i, 0, "grey")
    for i, (lbl, c) in enumerate([("干净", "green"), ("干净", "green"),
                                  ("半噪", "yellow"), ("纯噪声", "red")]):
        d.box(lbl, i, 1, c)

    d.note("整段共享一个噪声水平\n→ 一次生成一整段，\n没法逐帧接受新动作",
           4.1, 0, w=215, h=58, color="grey")
    d.note("每帧独立噪声水平\n→ 这正好就是自回归推理\n时的样子，模型天生就会",
           4.1, 1, w=215, h=58, color="green")
    d.text("GameNGen 的「历史帧加噪」是补丁；Diffusion Forcing 把自回归能力直接写进训练目标",
           -1.25, 2, w=640, size=12, bold=True)
    return d


# ============================================================ §3.3 Vista
@figure("vista-two-stage")
def _():
    d = Diagram(cell_w=185, cell_h=62, gap_x=66, gap_y=44)
    d1 = d.box("OpenDV-YouTube\n1700+ 小时\n无动作标注", 0, 0, "green", h=76)
    s1 = d.box("阶段一：微调 Stable Video Diffusion\n"
               "学真实交通动态与高保真细节\n（此时还不可控）",
               1, 0, "purple", w=320, h=76)
    d2 = d.box("nuPlan 等\n带 ego 动作标注\n数据量小", 0, 1, "blue", h=76)
    s2 = d.box("阶段二：冻结大部分骨干\n只训练动作条件分支\n（学听指令，不破坏泛化）",
               1, 1, "orange", w=320, h=76)
    res = d.box("可控的驾驶世界模型\n支持 指令 / 目标点 / 轨迹 / 转向角 / 速度 五种粒度",
                1, 2.1, "yellow", w=320, h=62, bold=True)
    d.arrow(d1, s1)
    d.arrow(d2, s2)
    d.arrow(s1, s2)
    d.arrow(s2, res)
    d.text("配方：海量无标注数据学动态 → 少量标注数据学控制",
           0, 3, w=560, size=12, bold=True)
    return d


# ============================================================ §4.2 三种破局
@figure("action-label-three-ways")
def _():
    d = Diagram(cell_w=190, cell_h=58, gap_x=40, gap_y=46)
    top = d.box("想用海量无标注视频训 action-conditioned 世界模型",
                0, 0, "grey", span=3, bold=True)
    w1 = d.box("① 找带标注的大数据\nOpenAI VPT 的 Minecraft\ncontractor 数据，数千小时\n带完整键鼠标注",
               0, 1, "blue", h=92)
    w2 = d.box("② 混合训练\niVideoGPT：有动作 / 无动作\n混在一起训，\n无动作时条件置零",
               1, 1, "orange", h=92)
    w3 = d.box("③ 自己发明动作\nGenie：从 (o_t, o_t+1) 反推\n一个 8 类的离散\nlatent action",
               2, 1, "green", h=92)
    for w, ex in ((w1, 0.15), (w2, 0.5), (w3, 0.85)):
        d.arrow(top, w, extra_style=f"exitX={ex};exitY=1;entryX=0.5;entryY=0;")
    r1 = d.box("Oasis · MineWorld\nDreamer 4 都吃它", 0, 2.25, "white", h=56)
    r2 = d.box("机器人操作域\n跨数据集泛化", 1, 2.25, "white", h=56)
    r3 = d.box("Genie 1/2/3\nLAPA 等 VLA 工作", 2, 2.25, "white", h=56)
    for a, b in ((w1, r1), (w2, r2), (w3, r3)):
        d.arrow(a, b)
    return d


# ============================================================ §4.2 Genie
@figure("genie-latent-action")
def _():
    d = Diagram(cell_w=150, cell_h=52, gap_x=60, gap_y=40)
    o1 = d.box("o_t", 0, 0, "grey")
    o2 = d.box("o_t+1", 0, 1, "grey")
    lam = d.box("LAM\n一个 VQ-VAE", 1, 0.5, "purple", bold=True)
    la = d.box("离散 latent action\n词表只有 8", 2, 0.5, "green", w=175, bold=True)
    d.arrow(o1, lam, extra_style="exitX=1;exitY=0.5;entryX=0;entryY=0.25;")
    d.arrow(o2, lam, extra_style="exitX=1;exitY=0.5;entryX=0;entryY=0.75;")
    d.arrow(lam, la)

    req = d.box("训练要求：用 o_t + 这个 latent action 能重建出 o_t+1",
                0, 1.9, "yellow", w=490)
    use = d.box("训完丢掉 LAM 的 encoder，让用户直接指定那 8 个 id 来控制\n"
                "←── 这就是「动作」",
                0, 2.8, "orange", w=490, h=58, bold=True)
    d.arrow(la, req, extra_style="exitX=0.5;exitY=1;entryX=0.85;entryY=0;")
    d.arrow(req, use)
    d.text("词表极小 → 模型被迫把「所有能发生的变化」归纳成几个主要控制维度，"
           "结果真的对应 上/下/左/右/跳",
           0, 3.6, w=620, size=12, bold=True)
    return d


# ============================================================ 暗线一：漂移
@figure("drift-timeline")
def _():
    d = Diagram(cell_w=300, cell_h=48, gap_x=40, gap_y=18, margin=155)
    rows = [
        ("2015", "多步 curriculum / scheduled sampling", "Oh et al. · CDNA · SimPLe", "grey"),
        ("2019", "隐空间多步预测一致性（latent overshooting）", "PlaNet", "blue"),
        ("2020", "想象起点取真实状态 + 砍短 horizon", "SimPLe · Dreamer · IRIS", "blue"),
        ("2024", "训练时给条件历史帧加噪", "DIAMOND · GameNGen", "red"),
        ("2024", "每帧独立噪声水平（Diffusion Forcing）", "Oasis · Dreamer 4", "purple"),
    ]
    prev = None
    for i, (year, tech, papers, color) in enumerate(rows):
        d.text(year, -0.42, i, w=110, align="right", bold=True, size=14)
        b = d.box(tech, 0, i, color)
        d.text(papers, 1, i, w=280, size=12)
        prev = b   # 年份列已表达先后顺序，不再画连接线（细点线在浅色主题下像杂痕）
    d.text("思路是收敛的：从「训练时模仿推理的样子」走到「让训练目标本身等价于推理模式」",
           -0.42, 5.2, w=700, size=12, bold=True)
    return d


# ============================================================ 全景谱系
@figure("genealogy")
def _():
    """时间纵向、技术路线横向。

    竖排而非横排：年份多、每个节点文字长，横排会宽到正文里缩放后看不清。
    """
    d = Diagram(cell_w=172, cell_h=58, gap_x=38, gap_y=26, margin=120,
                font_size=12)
    COLS = {"vec": 0, "tok": 1, "diff": 2, "data": 3}
    CLR = {"vec": "blue", "tok": "orange", "diff": "red", "data": "green"}

    for lane, col in COLS.items():
        d.text({"vec": "① 压成小向量", "tok": "② 离散 token",
                "diff": "③ 像素 / 扩散", "data": "④ 动作从哪来"}[lane],
               col, -0.75, w=172, align="center", bold=True, size=13)

    years = ["2015", "2016", "2018", "2019", "2020", "2021",
             "2023", "2024", "2024–25", "2025"]
    for i, y in enumerate(years):
        d.text(y, -0.52, i, w=92, align="right", bold=True, size=13)

    N = {}

    def node(key, lane, row, label, bold=False):
        N[key] = d.box(label, COLS[lane], row, CLR[lane], bold=bold)
        return N[key]

    node("oh", "tok", 0, "Oh et al. 2015\nAtari 动作条件预测")
    node("cdna", "diff", 1, "CDNA 2016\n机器人推物")
    node("ha", "vec", 2, "World Models 2018\nVAE + MDN-RNN")
    node("sv2p", "diff", 2, "SV2P 2018\n随机隐变量")
    node("planet", "vec", 3, "PlaNet 2019\nRSSM + CEM 规划")
    node("dreamer", "vec", 4, "Dreamer 2020\n想象内 actor-critic")
    node("simple", "diff", 4, "SimPLe 2020\n像素模型 + PPO\n定义 Atari 100k", )
    node("d2", "vec", 5, "DreamerV2 2021\n离散 latent")
    node("d3", "vec", 6, "DreamerV3 2023\n一套超参通吃", bold=True)
    node("iris", "tok", 6, "IRIS 2023\nVQ token + GPT", bold=True)
    node("gaia", "data", 6, "GAIA-1 2023\n全离散 + 扩散渲染")
    node("tdmpc", "vec", 7, "TD-MPC2 2024\n砍掉 decoder")
    node("diris", "tok", 7, "Δ-IRIS 2024\n只编码帧间差")
    node("diamond", "diff", 7, "DIAMOND 2024\n像素扩散，细节决定胜负", bold=True)
    node("genie", "data", 7, "Genie 2024\nlatent action\n无标注视频", bold=True)
    node("iv", "tok", 8, "iVideoGPT 2024\n大规模预训练")
    node("oasis", "diff", 8, "GameNGen → Oasis 2024\nDiffusion Forcing")
    node("vista", "data", 8, "Vista 2024\n驾驶，两阶段训练")
    node("mw", "tok", 9, "MineWorld 2025\n训练代码全开")

    d4 = d.box("Dreamer 4（2025）：纯离线 + shortcut forcing —— 三条线在这里收拢",
               0, 10.15, "purple", w=802, bold=True)

    chain = [("oh", "ha"), ("cdna", "sv2p"), ("sv2p", "simple"),
             ("ha", "planet"), ("planet", "dreamer"), ("dreamer", "d2"),
             ("d2", "d3"), ("simple", "iris"), ("iris", "diris"),
             ("diris", "iv"), ("iris", "mw"), ("diamond", "oasis"),
             ("gaia", "vista")]
    for a, b in chain:
        d.arrow(N[a], N[b])

    # 虚线 = 影响 / 借鉴。跨列的必须手定出入口，让横向段落在行间空隙里走，
    # 否则 drawio 自动布线会把线直接压过中间的方框。
    d.arrow(N["iris"], N["diamond"], style="dashed")
    d.arrow(N["genie"], N["oasis"], style="dashed",
            extra_style="exitX=0;exitY=1;entryX=1;entryY=0.5;")
    d.arrow(N["genie"], N["iv"], style="dashed",
            extra_style="exitX=0;exitY=1;entryX=1;entryY=0;")
    # 删掉 d3→TD-MPC2：TD-MPC2 并非由 DreamerV3 派生，同列同色已足够表达同路线
    # 删掉 iVideoGPT→Vista 与三条汇入底部横条的虚线：均会横穿中间方框，
    # 且底部横条的文字本身已说明「三条线在这里收拢」
    _ = d4

    d.text("实线 = 直接继承　｜　虚线 = 影响 / 借鉴　｜　颜色 = 所属技术路线",
           -0.52, 11.1, w=820, size=12, bold=True)
    return d


if __name__ == "__main__":
    want = sys.argv[1:] or list(FIGURES)
    unknown = [w for w in want if w not in FIGURES]
    if unknown:
        print("未知图名:", unknown)
        print("可选:", ", ".join(FIGURES))
        sys.exit(2)
    for name in want:
        d = FIGURES[name]()
        src = d.save(os.path.join(HERE, f"{name}.drawio"))
        svg = d.export(os.path.join(HERE, f"{name}.svg"), scale=1, border=8)
        png = d.export(os.path.join(HERE, f"{name}.png"), scale=2, border=8)
        print(f"{name:26} {'svg✓' if svg else 'svg✗'} {'png✓' if png else 'png✗'}")
