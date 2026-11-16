#!/usr/bin/env bash
# 每個實驗結束後執行：提交並推送 lab/ 與 CLAUDE.md（登記簿、腳本、結果摘要）
# 用法：./lab/save.sh "<編號> <一句話結果>" ["署名行（選填）"]
set -e
cd "$(dirname "$0")/.."
if git ls-files --others --exclude-standard lab CLAUDE.md | grep -Ei '\.(csv|zip|parquet|pkl|gz|feather)$'; then
  echo "偵測到資料檔，這是公開 repo，已停止提交。" >&2; exit 1
fi
git add lab CLAUDE.md
git diff --cached --quiet && { echo "沒有變更"; exit 0; }
if [ -n "$2" ]; then git commit -q -m "lab: ${1:-更新}" -m "$2"; else git commit -q -m "lab: ${1:-更新}"; fi
git push -q origin HEAD:research/lab && echo "已推送 $(git rev-parse --short HEAD)"
