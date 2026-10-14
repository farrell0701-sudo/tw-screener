"""用 yfinance 抓日K(含成交量),每檔一個檔案 data/prices/{code}.csv。

第一次執行抓近 2 年;之後每次只抓最近 10 天並合併。
上市代號加 .TW,上櫃加 .TWO。價格使用還原權息(auto_adjust=True),
避免除權息造成假的跳空。
"""
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
LIST = ROOT / "data" / "stock_list.csv"
OUT_DIR = ROOT / "data" / "prices"
BATCH = 80


def ticker(code: str, market: str) -> str:
    return f"{code}.TW" if market == "TWSE" else f"{code}.TWO"


def download(tickers: list[str], period: str) -> pd.DataFrame:
    for attempt in range(3):
        try:
            return yf.download(tickers, period=period, interval="1d", auto_adjust=True,
                               group_by="ticker", threads=True, progress=False)
        except Exception as e:  # noqa: BLE001
            print(f"[fetch_prices] 下載失敗,重試 {attempt + 1}: {e}", file=sys.stderr)
            time.sleep(10)
    return pd.DataFrame()


def main() -> None:
    lst = pd.read_csv(LIST, dtype=str)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [(r.code, ticker(r.code, r.market)) for r in lst.itertuples()]
    fresh = [r for r in rows if not (OUT_DIR / f"{r[0]}.csv").exists()]
    old = [r for r in rows if (OUT_DIR / f"{r[0]}.csv").exists()]
    for group, period in ((fresh, "2y"), (old, "10d")):
        for i in range(0, len(group), BATCH):
            chunk = group[i:i + BATCH]
            data = download([t for _, t in chunk], period)
            if data.empty:
                continue
            for code, t in chunk:
                try:
                    d = data[t] if isinstance(data.columns, pd.MultiIndex) else data
                except KeyError:
                    continue
                d = d.dropna(subset=["Close"])
                if d.empty:
                    continue
                d = d[["Open", "High", "Low", "Close", "Volume"]].copy()
                d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
                d.index.name = "Date"
                f = OUT_DIR / f"{code}.csv"
                if f.exists():
                    prev = pd.read_csv(f, index_col=0, parse_dates=True)
                    d = pd.concat([prev, d])
                    d = d[~d.index.duplicated(keep="last")]
                d.sort_index().tail(600).round(4).to_csv(f)
            time.sleep(2)
            print(f"[fetch_prices] {period}: {min(i + BATCH, len(group))}/{len(group)}")


if __name__ == "__main__":
    main()
