#!/usr/bin/env bash
# 拉取 action-conditioned 世界模型相关官方仓库到 ../repos/
#
# 用法:
#   bash scripts/clone_repos.sh            # 列出所有可选项（不下载）
#   bash scripts/clone_repos.sh all        # 全部拉取
#   bash scripts/clone_repos.sh iris diamond tdmpc2
#
# 全部用 --depth 1，不拉历史；不下载权重和数据集。

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/repos"
mkdir -p "$DEST"

# key|url|说明
REPOS=(
  "world-models-pytorch|https://github.com/ctallec/world-models|Ha&Schmidhuber 的 PyTorch 复现（推荐先跑这个）"
  "WorldModelsExperiments|https://github.com/hardmaru/WorldModelsExperiments|Ha&Schmidhuber 官方 TF1"
  "planet|https://github.com/google-research/planet|PlaNet 官方 TF1"
  "planet-pytorch|https://github.com/Kaixhin/PlaNet|PlaNet PyTorch 复现"
  "dreamer-v1|https://github.com/danijar/dreamer|Dreamer v1 (TF2)"
  "dreamerv2|https://github.com/danijar/dreamerv2|DreamerV2 (TF2)"
  "dreamerv3|https://github.com/danijar/dreamerv3|DreamerV3 官方 JAX"
  "dreamerv3-torch|https://github.com/NM512/dreamerv3-torch|DreamerV3 PyTorch（推荐读）"
  "daydreamer|https://github.com/danijar/daydreamer|DayDreamer 真机"
  "tdmpc2|https://github.com/nicklashansen/tdmpc2|TD-MPC2"
  "tdmpc|https://github.com/nicklashansen/tdmpc|TD-MPC v1"
  "iris|https://github.com/eloialonso/iris|IRIS（token-based WM 首选阅读）"
  "delta-iris|https://github.com/vmicheli/delta-iris|Delta-IRIS"
  "diamond|https://github.com/eloialonso/diamond|DIAMOND 扩散世界模型"
  "ivideogpt|https://github.com/thuml/iVideoGPT|iVideoGPT（含预训练权重）"
  "vid2world|https://github.com/thuml/Vid2World|视频扩散→交互式世界模型"
  "open-oasis|https://github.com/etched-ai/open-oasis|Oasis 500M 推理"
  "mineworld|https://github.com/microsoft/mineworld|MineWorld（训练代码全开）"
  "vista|https://github.com/OpenDriveLab/Vista|Vista 驾驶世界模型"
  "occworld|https://github.com/wzzheng/OccWorld|占据栅格世界模型"
  "driveagi|https://github.com/OpenDriveLab/DriveAGI|GenAD + OpenDV 数据集"
  "dreamer4-lucidrains|https://github.com/lucidrains/dreamer4|Dreamer4 非官方（有最小可跑示例）"
  "muzero-general|https://github.com/werner-duvaud/muzero-general|MuZero 社区实现（隐式价值等价模型）"
)

list_repos() {
  printf "%-24s %s\n" "KEY" "说明"
  printf "%-24s %s\n" "---" "---"
  for item in "${REPOS[@]}"; do
    IFS='|' read -r key url desc <<< "$item"
    local mark=" "
    [ -d "$DEST/$key" ] && mark="*"
    printf "%-24s %s %s\n" "$key" "$mark" "$desc"
  done
  echo
  echo "(* 表示已存在于 repos/)"
  echo "用法: bash scripts/clone_repos.sh all | bash scripts/clone_repos.sh iris diamond"
}

clone_one() {
  local want="$1" found=0
  for item in "${REPOS[@]}"; do
    IFS='|' read -r key url desc <<< "$item"
    if [ "$key" = "$want" ]; then
      found=1
      if [ -d "$DEST/$key/.git" ]; then
        echo "[skip] $key 已存在"
      else
        echo "[clone] $key <- $url"
        git clone --depth 1 "$url" "$DEST/$key" || echo "[FAIL] $key"
      fi
    fi
  done
  [ "$found" = 0 ] && echo "[warn] 未知 key: $want"
}

if [ "$#" -eq 0 ]; then
  list_repos
  exit 0
fi

if [ "$1" = "all" ]; then
  for item in "${REPOS[@]}"; do
    IFS='|' read -r key url desc <<< "$item"
    clone_one "$key"
  done
else
  for want in "$@"; do
    clone_one "$want"
  done
fi

echo
echo "完成。目录: $DEST"
