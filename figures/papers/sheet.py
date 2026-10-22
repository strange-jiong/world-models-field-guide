#!/usr/bin/env python3
"""把若干 (arxiv_id, 页码) 渲染成一张带标注的拼版图，用来人工定裁剪框。

拼版上叠了 10% 步长的比例网格，照着读出 --box 的 左,上,右,下 即可。

用法: python3 sheet.py out.png 2209.00588:2 1811.04551:4 ...
"""
import os
import subprocess
import sys

from PIL import Image, ImageDraw

from fetch import download  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".cache")


def render(aid: str, page: int, dpi: int = 72) -> Image.Image:
    p = download(aid)
    pre = os.path.join(CACHE, f"_s_{aid}_{page}")
    subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-f", str(page), "-l", str(page),
                    p, pre], check=True)
    f = [os.path.join(CACHE, x) for x in sorted(os.listdir(CACHE))
         if x.startswith(f"_s_{aid}_{page}") and x.endswith(".png")][0]
    return Image.open(f).convert("RGB")


def grid(im: Image.Image, label: str) -> Image.Image:
    W, H = im.size
    pad_t = 16
    out = Image.new("RGB", (W, H + pad_t), "white")
    out.paste(im, (0, pad_t))
    d = ImageDraw.Draw(out)
    d.text((3, 3), label, fill="#b00000")
    for i in range(1, 10):
        x = int(W * i / 10)
        d.line([(x, pad_t), (x, H + pad_t)], fill="#00a0ff", width=1)
        d.text((x + 1, pad_t + 1), f".{i}", fill="#0080c0")
        y = pad_t + int(H * i / 10)
        d.line([(0, y), (W, y)], fill="#ff40a0", width=1)
        d.text((1, y + 1), f".{i}", fill="#c00080")
    return out


if __name__ == "__main__":
    out_path = sys.argv[1]
    items = [a.split(":") for a in sys.argv[2:]]
    ims = [grid(render(a, int(p)), f"{a} p{p}") for a, p in items]
    cols = min(4, len(ims))
    rows = (len(ims) + cols - 1) // cols
    w = max(i.size[0] for i in ims)
    h = max(i.size[1] for i in ims)
    sheet = Image.new("RGB", (w * cols + 8 * (cols + 1), h * rows + 8 * (rows + 1)), "#808080")
    for i, im in enumerate(ims):
        sheet.paste(im, (8 + (i % cols) * (w + 8), 8 + (i // cols) * (h + 8)))
    sheet.save(out_path)
    print(out_path, sheet.size)
