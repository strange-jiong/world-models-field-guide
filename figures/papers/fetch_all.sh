#!/usr/bin/env bash
# 重建 notes/ 引用的全部论文原图。
#
# 这些图的著作权归各篇论文作者，仓库里不收录，需要本地自行生成。
# 首次运行会从 arXiv 下载 PDF 到 .cache/（约 150MB），之后复用。
#
# 用法: bash fetch_all.sh

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

run() { python3 fetch.py crop "$@" 2>&1 | paste -sd' '; }

# --- 自动定位上下边界（整页宽）---
run 1605.07157 1 paper-cdna-arch          --page 3
run 1912.01603 3 paper-dreamer-components --page 3
run 2509.24527 2 paper-dreamer4-design    --page 4
run 2301.04104 3 paper-dreamerv3-training --page 3
run 2408.14837 3 paper-gamengen-method    --page 3
run 2209.00588 1 paper-iris-unroll        --page 2
run 2405.15223 3 paper-ivideogpt-arch     --page 4
run 2504.08388 1 paper-mineworld-arch     --page 2
run 1903.00374 2 paper-simple-arch        --page 4
run 2405.17398 3 paper-vista-pipeline     --page 4

# --- 需要 --top 兜底：双栏页的页眉只有单栏宽，不会被判成宽行 ---
run 2406.19320 1 paper-deltairis-ae       --page 2 --top 0.075
run 2010.02193 2 paper-dreamerv2-wm       --page 3 --top 0.072

# --- 需要手工 box：单栏插图或页内有 "Figure N" 正文引用干扰 ---
run 1811.04551 2 paper-planet-rssm        --page 4  --box 0.120,0.0900,0.920,0.2510
run 1803.10122 8 paper-worldmodels-agent  --page 3  --box 0.500,0.6400,0.900,0.8020
run 1803.10122 6 paper-worldmodels-mdnrnn --page 3  --box 0.000,0.4370,0.480,0.6090
run 2310.16828 3 paper-tdmpc2-arch        --page 3  --box 0.555,0.4900,1.000,0.7000
run 2405.12399 9 paper-diamond-arch       --page 25 --box 0.000,0.5650,1.000,0.6760

echo
echo "完成。共 $(ls -1 paper-*.png 2>/dev/null | wc -l) 张。"
echo "出处元数据见 sources.json；版权说明见 README.md。"
