#!/usr/bin/env python3
"""把 article.md 里的 ASCII / mermaid 代码块替换为 SVG 配图引用。

幂等：重复运行无副作用（已替换过的文档里不再有对应代码块）。
原始带代码块的版本保留在 article.src.md。
"""
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "..", "article.md")
SRC = os.path.join(HERE, "..", "article.src.md")

# 按文档中 fenced block 出现顺序：(图名 or None, 图号, 图注)
FIGS = [
    ("wm-three-uses", 1, "世界模型的三种用法"),
    ("wm-taxonomy", 2, "按「未来存在什么形式里」分出的四大类"),
    ("ha-vmc", 3, "Ha & Schmidhuber 的 V-M-C 三件套，分三阶段依次训练"),
    ("rssm", 4, "RSSM：确定性骨架 h 与随机状态 z 并存，prior 与 posterior 用 KL 拉近"),
    ("dreamer-imagination", 5, "Dreamer：从真实状态出发，在隐空间想象 15 步并训练 actor-critic"),
    ("tdmpc2-components", 6, "TD-MPC2：砍掉 decoder，用隐空间一致性取代重建"),
    ("iris-sequence", 7, "IRIS：动作就是序列里的一个 token"),
    ("token-budget", 8, "token 预算的三种花法，共同思路是只编码变化量"),
    ("ivideogpt-stages", 9, "iVideoGPT：预训练 → 微调 → 当世界模型用"),
    ("diamond-diffusion", 10, "DIAMOND：像素空间的条件扩散，3 步采样"),
    ("diffusion-forcing", 11, "Diffusion Forcing：每帧独立噪声水平，训练目标等价于自回归推理"),
    ("vista-two-stage", 12, "Vista：大数据学动态，小数据学控制"),
    ("action-label-three-ways", 13, "动作标注匮乏的三种破局方式"),
    ("genie-latent-action", 14, "Genie 的 latent action：从无标注视频里自己发明动作"),
    ("drift-timeline", 15, "自回归漂移的五代解法"),
    (None, None, None),  # 第 16 块 -> 表格
    (None, None, None),  # 第 17 块 -> 表格
    ("genealogy", 16, "全景谱系：时间纵向，技术路线横向"),
]

TABLE_16 = """| 年份 | 做法 | 代表 |
|---|---|---|
| 2018 | 采样时放大方差（温度 τ=1.15），让梦更不确定 | Ha & Schmidhuber |
| 2020 | 想象 horizon 砍到 15~20 步，起点取真实状态 | SimPLe · Dreamer · IRIS |
| 2020+ | online 交替采集：策略去哪钻漏洞，就在那里补真实数据 | Dreamer · IRIS · DIAMOND |
| 2024 | horizon 只留 3 步，之后交给学出来的 Q | TD-MPC2 |"""

TABLE_17 = """| 方法 | 技术 | 覆盖范围 |
|---|---|---|
| DreamerV2 | 离散隐变量 + KL balancing | 修的是分布层面的尺度问题 |
| DreamerV3 | symlog + two-hot + free bits + 回报分位数归一化 | 一套超参跑 150+ 任务 |
| TD-MPC2 | SimNorm + two-hot 离散回归 | 一套超参跑 545 个任务 |"""

TABLES = {15: TABLE_16, 16: TABLE_17}  # 0-based index


def main() -> int:
    art = os.path.abspath(ART)
    text = open(art, encoding="utf-8").read()

    pattern = re.compile(r"^```.*?^```", re.M | re.S)
    blocks = pattern.findall(text)
    if len(blocks) != len(FIGS):
        print(f"代码块数量不符：找到 {len(blocks)}，预期 {len(FIGS)}。"
              "文档可能已被替换过或已改动，终止以免破坏内容。")
        return 1

    if not os.path.exists(os.path.abspath(SRC)):
        shutil.copy(art, os.path.abspath(SRC))
        print("已备份原文到 article.src.md")

    idx = -1

    def repl(_m):
        nonlocal idx
        idx += 1
        if idx in TABLES:
            return TABLES[idx]
        name, num, cap = FIGS[idx]
        svg = os.path.join(HERE, f"{name}.svg")
        if not os.path.exists(svg):
            raise SystemExit(f"缺少图片: {svg}（先跑 make_figures.py）")
        return (f"![图 {num}：{cap}](figures/{name}.svg)\n\n"
                f"<!-- 源文件: figures/{name}.drawio -->")

    out = pattern.sub(repl, text)
    with open(art, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"已替换 {len(blocks)} 个代码块："
          f"{len(blocks) - len(TABLES)} 张 SVG 配图 + {len(TABLES)} 个表格")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
