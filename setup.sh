#!/usr/bin/env bash
# 在新的工作區重建回測環境（程式寫死 /home/claude 路徑）
set -e
R=$(cd "$(dirname "$0")" && pwd); D=/home/claude
mkdir -p $D/hdata $D/rawx $D/dl_csv
cd "$R"
git fetch -q origin history-data raw-data
for i in 1 2 3 4 5; do git show origin/history-data:history_part$i.zip > /tmp/h$i.zip; unzip -qo /tmp/h$i.zip -d $D/hdata; done
git show origin/raw-data:raw_run1.zip > /tmp/raw.zip; unzip -qo /tmp/raw.zip -d $D/rawx
unzip -qo data/dl_csv.zip -d $D
cp -r bt $D/
pip install -q --break-system-packages pandas numpy matplotlib 2>/dev/null || true
echo "完成：hdata $(ls $D/hdata|wc -l) 檔、rawx $(ls $D/rawx|wc -l) 檔、dl_csv $(ls $D/dl_csv|wc -l) 檔"
