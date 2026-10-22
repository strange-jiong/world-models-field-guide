# 论文原图

`notes/` 里引用的论文架构图，从各篇 arXiv PDF 自动裁剪而来。

## 版权

**这些图的著作权归各篇论文作者所有。** 此处仅作为个人学习笔记的引用，未做任何修改（除裁剪与去白边），每张图在 notes 中都标注了论文、图号、作者与 arXiv 链接。出处元数据见 `sources.json`。请勿将本目录内容用于再分发或商业用途。

## 重新生成

```bash
# 列出某篇论文的所有图注及页码，用来挑图
python3 fetch.py list 2209.00588

# 自动定位并裁剪（推荐）
python3 fetch.py crop 2209.00588 1 paper-iris-unroll --page 2

# 双栏论文里的单栏插图：先用 --x 限定到那一栏
python3 fetch.py crop 2310.16828 3 paper-tdmpc2-arch --page 3 --x 0.555,1.0

# 页内有「正文提到 Figure N」干扰时，取最后一处
python3 fetch.py crop 2405.12399 9 paper-diamond-arch --page 25 --occurrence last

# 自动上边界穿到页眉之上时兜底
python3 fetch.py crop 2010.02193 2 paper-dreamerv2-wm --page 3 --top 0.072

# 完全手动
python3 fetch.py crop <id> <图号> <输出名> --page N --box 左,上,右,下
```

PDF 缓存在 `.cache/`（已 gitignore，约 150MB）。

## 自动定位是怎么做的

架构图多为 TikZ 矢量图，arXiv 的 HTML 版里没有 `<img>`，所以只能从 PDF 渲染。关键是确定图的上下边界：

1. 用 `pdftotext -bbox` 取该页每个词的坐标。
2. 找 `Figure N` 的位置 → 图注顶端 = 图的**下边界**。
3. 往上找**上边界**：正文行会铺满整个栏宽，插图内部的标签行不会。所以逐行统计「被文字覆盖的横向宽度」，低覆盖的连续行带就是图。
4. 图内部的子标题（`(a) … (b) …`）也是铺满的宽行，但它是**孤立**一条；正文是 ~11pt 间距的连续多条。以此区分，孤立宽行跳过，连续宽行才停。
5. 左右给满，最后按像素去白边。

### 已知的三个坑

- **双栏页上的页眉只有单栏宽**，而阈值是按跨栏的最大行宽算的，所以页眉不会被判成宽行，上边界会穿到它上面 → 用 `--top` 兜底。
- **双栏论文里的单栏插图**，另一栏的正文会把所有行都算成高覆盖，判据直接失效 → 必须用 `--x` 先限定到那一栏（注意栏的左边界要按图注首词的 x 来定，不是按版面猜）。
- **同一页正文里提到「Figure N」** 会被当成图注 → 用 `--occurrence last`。

### 没有收录的

- **Oh et al. 2015**：架构图在附录 Figure 9，自动定位失败，没再手工处理；note 01 用 CDNA 的架构图代表这一代。
- **Genie**：PDF 没有文字层（41MB 全图），无法定位；note 10 用自绘的 latent action 示意图代替（`figures/genie-latent-action.svg`）。
- **Oasis**：没有 arXiv 论文。
