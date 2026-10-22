#!/usr/bin/env bash
# article.md -> PDF
#
# 路线: pandoc (md -> 自包含 HTML, SVG 内嵌为 data URI, 公式转 MathML)
#       -> headless Chrome 的 print-to-pdf
# 本机没有 xelatex，所以不走 pandoc 的 LaTeX 路径；Chrome 对 SVG 与中文的
# 支持也比 wkhtmltopdf 好。
#
# 用法: bash scripts/build_pdf.sh [输入.md] [输出.pdf]

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IN="${1:-$ROOT/article.md}"
OUT="${2:-$ROOT/article.pdf}"
HTML="$ROOT/.build/article.html"
TITLE="世界模型：从一个想法到四条路线"

command -v pandoc >/dev/null || { echo "需要 pandoc" >&2; exit 1; }
CHROME="$(command -v google-chrome || command -v chromium || true)"
[ -n "$CHROME" ] || { echo "需要 google-chrome / chromium" >&2; exit 1; }

mkdir -p "$ROOT/.build"

# 图片用相对路径引用，所以必须在 article.md 所在目录执行 pandoc
cd "$ROOT"
pandoc "$IN" \
  --from markdown+implicit_figures+pipe_tables+tex_math_dollars \
  --to html5 \
  --standalone \
  --self-contained \
  --mathml \
  --metadata pagetitle="$TITLE" \
  --css assets/pdf.css \
  -o "$HTML"

echo "HTML: $HTML ($(du -h "$HTML" | cut -f1))"

"$CHROME" \
  --headless \
  --disable-gpu \
  --no-sandbox \
  --no-pdf-header-footer \
  --virtual-time-budget=30000 \
  --run-all-compositor-stages-before-draw \
  --print-to-pdf="$OUT" \
  "file://$HTML" >/dev/null 2>&1

if [ ! -f "$OUT" ]; then
  echo "PDF 生成失败" >&2
  exit 1
fi

echo "PDF: $OUT ($(du -h "$OUT" | cut -f1))"
if command -v pdfinfo >/dev/null; then
  pdfinfo "$OUT" | grep -E "^(Pages|Page size)"
fi
