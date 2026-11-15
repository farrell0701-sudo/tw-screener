# lab：T_ld 因子研究的工作區

這個資料夾是研究的唯一工作區。**不要在 repo 以外的地方放研究檔案。**

## 規則

1. 跑任何測試前，先在 `registry/registry_L.md`（組合層）或 `registry_M.md`（描述性）寫入登記：假說、定義、判準。
2. 每個實驗結束後立刻執行 `./save.sh "<編號> <一句話結果>"`：提交並推送登記簿、腳本與結果摘要。
3. 中間快取放 `cache/`（不入庫，可由 `setup.sh` 重建）。要保留的結果放 `results/`（小檔）。
4. 規則變更同步更新 claude.ai 文件「台股動能策略規則總整理」。

## 重建環境

```bash
git clone https://github.com/farrell0701-sudo/tw-screener && cd tw-screener
git checkout research/lab && ./lab/setup.sh
```

## 內容

- `ENGINE_PLAN.md`：引擎重建計畫與驗收標準。
- `registry/`：登記簿。L-096～L-108、M-076～M-096 由對話紀錄逐字還原；更早的條目原文已遺失，結論見規則總整理文件。
- `recovered/`：2026-11-14～15 的研究腳本（依賴已遺失的舊引擎，僅供對照），以及 FinLab 版完整規則實作。
- `data/`：還原價建置與檢查程式。
