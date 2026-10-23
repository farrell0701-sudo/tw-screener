# 台股因子研究（research/factors）

回測引擎、資料載入與 K 棒重疊結構研究的程式碼。研究背景與結論見 claude.ai 文件「台股因子研究專案」。

## 重建環境

```bash
git clone https://github.com/farrell0701-sudo/tw-screener && cd tw-screener
git checkout research/factors && ./setup.sh
```

會在 `/home/claude/` 下建立：

| 路徑 | 來源 | 內容 |
| --- | --- | --- |
| hdata/ | history-data 分支 | Yahoo 日線、TAIEX.csv、stock_list.csv |
| rawx/ | raw-data 分支 | FinMind 真正未還原日線＋下市股除權息／減資事件（ev_*.csv） |
| dl_csv/ | data/dl_csv.zip | 136 檔 2016 後下市股，Yahoo 格式，AdjClose 由事件推算 |
| bt/ | 本分支 | 程式 |

## 主要程式

- `bt/load.py`：`load_all()`。環境變數 `RAWSRC=finmind` 改用 FinMind 未還原價，`DELIST=1` 加入下市股。
- `bt/engine.py`：事件驅動組合回測（固定格數、整股、分級滑價、t/t+1 執行、漲跌停、ATR 停損）。
- `bt/pro_prep.py`：V3C-PRO 計分。
- `bt/finlab_*.py`、`ema12_opt.py`、`v3c_opt.py`：在 finlab 平台上執行的腳本。
- `bt/struct/scan.py`：結構轉變事件掃描與特徵；`daily.py`：逐日 overlap／上方壓力／lo_ratio 寬表。

## 注意

- 早期報告（PRO 45.4%、300萬版 38%）受 Yahoo Close 前視偏誤與存活者偏誤影響，不再使用。
- `*.pkl` 快取未上傳，第一次執行會自動重算。
