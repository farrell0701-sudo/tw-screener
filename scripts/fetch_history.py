"""下載全部上市櫃股票的長期日線,打包成 zip,給回測使用(不影響網站)。

- 股票清單:data/stock_list.csv(網站原本就有)
- 期間:2015-01-01 到今天
- 價格:原始價格(未還原)Open/High/Low/Close,另附 AdjClose(還原權息收盤)
- 成交量:單位為「股」
- 輸出:out_history/history_partN.zip,每個 zip 內含約 400 檔,每檔一個 {代號}.csv
         另含 TAIEX.csv(加權指數)與 stock_list.csv
"""
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
LIST = ROOT / "data" / "stock_list.csv"
OUT = ROOT / "out_history"
START = "2015-01-01"
BATCH = 50
PER_ZIP = 400


def ticker(code: str, market: str) -> str:
    return f"{code}.TW" if market == "TWSE" else f"{code}.TWO"


def download(tickers):
    for attempt in range(3):
        try:
            return yf.download(tickers, start=START, interval="1d", auto_adjust=False,
                               group_by="ticker", threads=True, progress=False)
        except Exception as e:  # noqa: BLE001
            print(f"下載失敗,重試 {attempt + 1}: {e}", file=sys.stderr)
            time.sleep(15)
    return pd.DataFrame()


def tidy(d: pd.DataFrame) -> pd.DataFrame:
    d = d.dropna(subset=["Close"])
    if d.empty:
        return d
    d = d[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].copy()
    d.columns = ["Open", "High", "Low", "Close", "AdjClose", "Volume"]
    d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
    d.index.name = "Date"
    d = d[d["Volume"] > 0]  # 去掉停牌日
    d[["Open", "High", "Low", "Close", "AdjClose"]] = d[["Open", "High", "Low", "Close", "AdjClose"]].round(4)
    d["Volume"] = d["Volume"].astype("int64")
    return d


def fetch_group(rows, store):
    for i in range(0, len(rows), BATCH):
        chunk = rows[i:i + BATCH]
        data = download([t for _, t in chunk])
        for code, t in chunk:
            if data.empty:
                break
            try:
                d = data[t] if isinstance(data.columns, pd.MultiIndex) else data
            except KeyError:
                continue
            d = tidy(d)
            if len(d) >= 60:
                store[code] = d.to_csv()
        print(f"{min(i + BATCH, len(rows))}/{len(rows)},已取得 {len(store)} 檔", flush=True)
        time.sleep(3)


def main() -> None:
    lst = pd.read_csv(LIST, dtype=str)
    rows = [(r.code, ticker(r.code, r.market)) for r in lst.itertuples()]
    store: dict[str, str] = {}
    fetch_group(rows, store)
    miss = [r for r in rows if r[0] not in store]
    if miss:
        print(f"第一輪缺 {len(miss)} 檔,重試一次", flush=True)
        time.sleep(30)
        fetch_group(miss, store)

    idx = tidy(download(["^TWII"]).pipe(lambda d: d["^TWII"] if isinstance(d.columns, pd.MultiIndex) else d))

    OUT.mkdir(parents=True, exist_ok=True)
    codes = sorted(store)
    for k in range(0, len(codes), PER_ZIP):
        part = k // PER_ZIP + 1
        with zipfile.ZipFile(OUT / f"history_part{part}.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for c in codes[k:k + PER_ZIP]:
                z.writestr(f"{c}.csv", store[c])
            if part == 1:
                z.writestr("TAIEX.csv", idx.to_csv())
                z.writestr("stock_list.csv", lst.to_csv(index=False))
    (OUT / "README.md").write_text(
        f"回測用歷史日線\n\n- 股票數:{len(codes)} / 清單 {len(rows)}\n- 期間:{START} 起\n"
        f"- 產生時間(UTC):{pd.Timestamp.utcnow():%Y-%m-%d %H:%M}\n", encoding="utf-8")
    print(f"完成:{len(codes)} 檔,缺 {len(rows) - len(codes)} 檔")


if __name__ == "__main__":
    main()
