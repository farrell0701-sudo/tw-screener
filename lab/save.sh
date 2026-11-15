#!/usr/bin/env bash
# 每個實驗結束後執行：提交並推送 lab/（登記簿、腳本、結果摘要）
set -e
cd "$(dirname "$0")/.."
git add lab
git diff --cached --quiet && { echo "沒有變更"; exit 0; }
git commit -q -m "lab: ${1:-更新}" -m "Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012vKYvw91ja6SaKkgR6UGtw"
git push -q origin HEAD:research/lab && echo "已推送 $(git rev-parse --short HEAD)"
