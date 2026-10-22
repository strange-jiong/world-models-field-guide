#!/usr/bin/env python3
"""从 arXiv 论文里抓取指定插图，供 notes/ 引用（个人学习用，带出处标注）。

两步走：
  list  — 下载 PDF，列出所有 "Figure N:" 图注及其页码，供人工挑图
  crop  — 渲染指定页并按给定裁剪框切图，同时写入出处元数据

为什么不走 arXiv HTML：架构图多为 TikZ 矢量图，HTML 版里没有 <img>，
只能从 PDF 渲染。

用法:
  python3 fetch.py list 2209.00588
  python3 fetch.py list            # 列出下面 PAPERS 里全部论文的图注
  python3 fetch.py crop 2209.00588 1 iris-arch --page 4 --box 0.08,0.12,0.95,0.42
             box = 左,上,右,下，按页面宽高的比例（0~1）
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".cache")
META = os.path.join(HERE, "sources.json")

# arxiv_id -> (短名, 论文标题, 作者, 年份)
PAPERS = {
    "1507.08750": ("oh2015", "Action-Conditional Video Prediction using Deep Networks in Atari Games", "Oh et al.", 2015),
    "1605.07157": ("cdna", "Unsupervised Learning for Physical Interaction through Video Prediction", "Finn, Goodfellow, Levine", 2016),
    "1710.11252": ("sv2p", "Stochastic Variational Video Prediction", "Babaeizadeh et al.", 2018),
    "1803.10122": ("worldmodels", "World Models", "Ha, Schmidhuber", 2018),
    "1811.04551": ("planet", "Learning Latent Dynamics for Planning from Pixels", "Hafner et al.", 2019),
    "1903.00374": ("simple", "Model-Based Reinforcement Learning for Atari", "Kaiser et al.", 2020),
    "1912.01603": ("dreamer", "Dream to Control: Learning Behaviors by Latent Imagination", "Hafner et al.", 2020),
    "2010.02193": ("dreamerv2", "Mastering Atari with Discrete World Models", "Hafner et al.", 2021),
    "2301.04104": ("dreamerv3", "Mastering Diverse Domains through World Models", "Hafner et al.", 2023),
    "2310.16828": ("tdmpc2", "TD-MPC2: Scalable, Robust World Models for Continuous Control", "Hansen et al.", 2024),
    "2209.00588": ("iris", "Transformers are Sample-Efficient World Models", "Micheli, Alonso, Fleuret", 2023),
    "2406.19320": ("deltairis", "Efficient World Models with Context-Aware Tokenization", "Micheli, Alonso, Fleuret", 2024),
    "2405.12399": ("diamond", "Diffusion for World Modeling: Visual Details Matter in Atari", "Alonso et al.", 2024),
    "2405.15223": ("ivideogpt", "iVideoGPT: Interactive VideoGPTs are Scalable World Models", "Wu et al.", 2024),
    "2504.08388": ("mineworld", "MineWorld: a Real-Time and Open-Source Interactive World Model on Minecraft", "Guo et al.", 2025),
    "2405.17398": ("vista", "Vista: A Generalizable Driving World Model", "Gao et al.", 2024),
    "2509.24527": ("dreamer4", "Training Agents Inside of Scalable World Models", "Hafner et al.", 2025),
    "2408.14837": ("gamengen", "Diffusion Models Are Real-Time Game Engines", "Valevski et al.", 2024),
    "2402.15391": ("genie", "Genie: Generative Interactive Environments", "Bruce et al.", 2024),
}


def pdf_path(aid: str) -> str:
    return os.path.join(CACHE, f"{aid}.pdf")


def download(aid: str) -> str:
    os.makedirs(CACHE, exist_ok=True)
    p = pdf_path(aid)
    if os.path.exists(p) and os.path.getsize(p) > 50_000:
        return p
    url = f"https://arxiv.org/pdf/{aid}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (notes-builder)"})
    with urllib.request.urlopen(req, timeout=90) as r, open(p, "wb") as f:
        f.write(r.read())
    return p


def captions(aid: str) -> list[tuple[int, str]]:
    """返回 [(页码, 图注首行), ...]"""
    p = download(aid)
    out = []
    n = int(re.search(r"Pages:\s+(\d+)",
                      subprocess.run(["pdfinfo", p], capture_output=True, text=True).stdout).group(1))
    for page in range(1, min(n, 30) + 1):
        txt = subprocess.run(["pdftotext", "-f", str(page), "-l", str(page), p, "-"],
                             capture_output=True, text=True).stdout
        txt = re.sub(r"\s+", " ", txt)
        for m in re.finditer(r"(Figure|Fig\.)\s*(\d+)\s*[:.]\s*(.{0,110})", txt):
            out.append((page, f"Fig {m.group(2)}: {m.group(3).strip()}"))
    return out


def words(aid: str, page: int):
    """用 pdftotext -bbox 取该页所有词的坐标（页面点坐标系）。"""
    p = download(aid)
    out = subprocess.run(["pdftotext", "-bbox", "-f", str(page), "-l", str(page), p, "-"],
                         capture_output=True, text=True).stdout
    pg = re.search(r'<page width="([\d.]+)" height="([\d.]+)"', out)
    W, H = float(pg.group(1)), float(pg.group(2))
    ws = []
    for m in re.finditer(
        r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</word>',
        out):
        x0, y0, x1, y1 = (float(m.group(i)) for i in range(1, 5))
        ws.append((x0, y0, x1, y1, m.group(5)))
    return W, H, ws


def locate(aid: str, page: int, fignum: int, cover: float = 0.55,
           x_range: tuple[float, float] = (0.0, 1.0), occurrence: str = "first"):
    """自动定位插图的上下边界。

    思路：正文行会铺满整个栏宽，插图内部的标签行不会。
    先用 "Figure N" 的坐标定出图注顶端作为下边界，再往上找连续的
    「低文字覆盖」行带，遇到成片的正文行就停。
    返回 (top, bottom) 的页面高度比例。
    """
    W, H, all_ws = words(aid, page)
    if not all_ws:
        raise SystemExit(f"{aid} p{page} 没有文字层，无法自动定位")

    # 双栏论文里的单栏插图：必须先按 x 限定到那一栏，
    # 否则另一栏的正文会把所有行都算成「高覆盖」，判据失效。
    xl, xr = x_range[0] * W, x_range[1] * W
    ws = [w for w in all_ws if xl <= (w[0] + w[2]) / 2 <= xr]
    if not ws:
        raise SystemExit(f"{aid} p{page} 指定 x 范围内没有文字")

    # 图注顶端：找 "Figure"/"Fig." 紧跟目标编号的那一处。
    # 同一页里「正文提到 Figure N」也会命中，用 occurrence 选第几个。
    cands = []
    for i, (x0, y0, x1, y1, t) in enumerate(ws):
        if t.rstrip(".") in ("Figure", "Fig") and i + 1 < len(ws):
            nxt = ws[i + 1][4].rstrip(".:")
            if nxt.isdigit() and int(nxt) == fignum:
                cands.append(y0)
    if not cands:
        raise SystemExit(f"{aid} p{page} 找不到 Figure {fignum} 的图注")
    cap_y = cands[-1] if occurrence == "last" else cands[0]

    # 每 1pt 一行，统计被词覆盖的横向宽度
    rows = [0.0] * (int(H) + 2)
    for x0, y0, x1, y1, _ in ws:
        for y in range(int(y0), min(int(y1) + 1, len(rows))):
            rows[y] = max(rows[y], x1 - x0 + rows[y] * 0)
    # 用并集长度更准：重算
    segs: dict[int, list[tuple[float, float]]] = {}
    for x0, y0, x1, y1, _ in ws:
        for y in range(int(y0), min(int(y1) + 1, len(rows))):
            segs.setdefault(y, []).append((x0, x1))
    for y, ss in segs.items():
        ss.sort()
        tot, cur_s, cur_e = 0.0, None, None
        for s, e in ss:
            if cur_s is None:
                cur_s, cur_e = s, e
            elif s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                tot += cur_e - cur_s
                cur_s, cur_e = s, e
        if cur_s is not None:
            tot += cur_e - cur_s
        rows[y] = tot

    line_w = max(rows) if rows else W
    thresh = cover * line_w

    def wide_line_at(y: int) -> int:
        """y 处若是一条宽行，返回它的高度（pt），否则 0。"""
        if rows[y] < thresh:
            return 0
        h = 0
        while y - h > 0 and rows[y - h] >= thresh:
            h += 1
        return h

    y = int(cap_y) - 2
    while y > 0 and rows[y] >= thresh:       # 掠过图注本身
        y -= 1
    bottom = y + 1

    # 往上找图的顶边。图内部的子标题 "(a) ... (b) ..." 也是宽行，但它是
    # 孤立的一条；正文则是 ~11pt 间距的连续多条。以此区分。
    top = 0
    while y > 0:
        h = wide_line_at(y)
        if h:
            if y < 0.12 * H:                 # 页眉 / 页首标题，图在它下面
                top = y + 1
                break
            y_above = y - h - 1
            has_neighbor = any(rows[yy] >= thresh
                               for yy in range(max(0, y_above - 16), y_above + 1))
            if has_neighbor:                 # 连着另一条宽行 -> 正文，停
                top = y + 1
                break
            y = y_above                      # 孤立宽行 -> 图的子标题，跳过
            continue
        y -= 1
    return max(0.0, (top - 4) / H), min(1.0, (bottom + 1) / H)


def crop(aid: str, fignum: str, name: str, page: int, box: str, dpi: int = 220) -> str:
    from PIL import Image
    p = download(aid)
    tmp = os.path.join(CACHE, f"_{aid}_p{page}")
    subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-f", str(page), "-l", str(page),
                    p, tmp], check=True)
    rendered = [os.path.join(CACHE, f) for f in sorted(os.listdir(CACHE))
                if f.startswith(f"_{aid}_p{page}") and f.endswith(".png")][0]
    im = Image.open(rendered)
    W, H = im.size
    l, t, r, b = (float(x) for x in box.split(","))
    im = im.crop((int(l * W), int(t * H), int(r * W), int(b * H)))
    im = trim(im)
    out = os.path.join(HERE, f"{name}.png")
    im.save(out, optimize=True)

    short, title, authors, year = PAPERS.get(aid, (name, "", "", ""))
    meta = json.load(open(META)) if os.path.exists(META) else {}
    meta[name] = {
        "arxiv": aid, "figure": fignum, "page": page, "box": box,
        "paper": title, "authors": authors, "year": year,
        "url": f"https://arxiv.org/abs/{aid}",
        "note": "论文插图，版权归原作者；此处为个人学习笔记引用。",
    }
    json.dump(meta, open(META, "w"), ensure_ascii=False, indent=2)
    print(f"{out}  {im.size}")
    return out


def trim(im, thresh: int = 247):
    """去掉四周空白。"""
    g = im.convert("L")
    W, H = g.size
    px = g.load()
    cols = [x for x in range(W) if any(px[x, y] < thresh for y in range(0, H, 2))]
    rows = [y for y in range(H) if any(px[x, y] < thresh for x in range(0, W, 2))]
    if not cols or not rows:
        return im
    pad = 6
    return im.crop((max(0, cols[0] - pad), max(0, rows[0] - pad),
                    min(W, cols[-1] + pad), min(H, rows[-1] + pad)))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    cmd = sys.argv[1]
    if cmd == "list":
        ids = sys.argv[2:] or list(PAPERS)
        for aid in ids:
            short = PAPERS.get(aid, ("?",))[0]
            print(f"\n{'=' * 70}\n{aid}  {short}\n{'=' * 70}")
            try:
                for page, cap in captions(aid):
                    print(f"  p{page:<3} {cap}")
            except Exception as e:
                print(f"  [失败] {e}")
    elif cmd == "crop":
        aid, fignum, name = sys.argv[2], sys.argv[3], sys.argv[4]
        args = sys.argv[5:]
        page = int(args[args.index("--page") + 1])
        if "--box" in args:
            box = args[args.index("--box") + 1]
        else:
            # 自动定位上下边界；左右按 --x 限栏（默认整页），交给 trim() 收紧
            lo = args[args.index("--x") + 1] if "--x" in args else "0,1"
            x0, x1 = lo.split(",")
            occ = args[args.index("--occurrence") + 1] if "--occurrence" in args else "first"
            t, b = locate(aid, page, int(fignum),
                          x_range=(float(x0), float(x1)), occurrence=occ)
            # 双栏页上「页眉」只有单栏宽，不会被判成宽行，自动上边界会穿到页眉之上。
            # 这种情况用 --top 显式兜底。
            if "--top" in args:
                t = max(t, float(args[args.index("--top") + 1]))
            box = f"{x0},{t:.4f},{x1},{b:.4f}"
            print(f"  auto box = {box}")
        crop(aid, fignum, name, page, box)
    else:
        print(__doc__)
        sys.exit(2)
