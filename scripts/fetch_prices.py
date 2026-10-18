"""用 yfinance 抓日K(含成交量),每檔一個檔案 data/prices/{code}.csv。

第一次執行抓近 3 年;之後每次只抓最近 10 天並合併。
上市代號加 .TW,上櫃加 .TWO。
存的是未還原的 Open/High/Low/Close(實際股價)與還原收盤 AdjClose,
build.py 用 AdjClose ÷ Close 換算還原價(避免除權息造成假的跳空),
股價與成交值門檻則用實際收盤。除權息後新抓到的還原比例會往前套用到舊資料。
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
LIST = ROOT / "data" / "stock_list.csv"
OUT_DIR = ROOT / "data" / "prices"
BATCH = 80


def ticker(code: str, market: str) -> str:
    return f"{code}.TW" if market == "TWSE" else f"{code}.TWO"


COLS = ["Open", "High", "Low", "Close", "AdjClose", "Volume"]


def download(tickers: list[str], period: str, adjust: bool = False) -> pd.DataFrame:
    for attempt in range(3):
        try:
            return yf.download(tickers, period=period, interval="1d", auto_adjust=adjust,
                               group_by="ticker", threads=True, progress=False)
        except Exception as e:  # noqa: BLE001
            print(f"[fetch_prices] 下載失敗,重試 {attempt + 1}: {e}", file=sys.stderr)
            time.sleep(10)
    return pd.DataFrame()


def current(f: Path) -> bool:
    """已是新格式(有 AdjClose)且最後一筆在 7 天內:只需補最近 10 天;否則重抓 3 年。"""
    if not f.exists():
        return False
    try:
        d = pd.read_csv(f, index_col=0, parse_dates=True)
    except Exception:  # noqa: BLE001
        return False
    return "AdjClose" in d.columns and len(d) > 0 and (pd.Timestamp.now() - d.index.max()).days <= 7


def merge(prev: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """把新抓的 10 天接到舊資料後面。新資料第一天的還原比例(AdjClose ÷ Close)
    若和舊資料同一天不同(中間有除權息),舊資料的 AdjClose 依比例一起調整。"""
    new = new.dropna(subset=["Close", "AdjClose"])
    if new.empty:
        return prev
    d0 = new.index.min()
    if d0 in prev.index and prev.at[d0, "Close"] > 0 and new.at[d0, "Close"] > 0:
        r = (new.at[d0, "AdjClose"] / new.at[d0, "Close"]) / (prev.at[d0, "AdjClose"] / prev.at[d0, "Close"])
        if np.isfinite(r) and abs(r - 1) > 1e-6:
            prev = prev.copy()
            prev["AdjClose"] *= r
    else:
        print(f"[fetch_prices] 新舊資料沒有重疊日 {d0.date()},還原比例可能不連續", file=sys.stderr)
    out = pd.concat([prev[prev.index < d0], new])
    return out[~out.index.duplicated(keep="last")]


def main() -> None:
    lst = pd.read_csv(LIST, dtype=str)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [(r.code, ticker(r.code, r.market)) for r in lst.itertuples()]
    ok = {r[0] for r in rows if current(OUT_DIR / f"{r[0]}.csv")}
    fresh = [r for r in rows if r[0] not in ok]
    old = [r for r in rows if r[0] in ok]
    for group, period in ((fresh, "3y"), (old, "10d")):
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
                d = d.rename(columns={"Adj Close": "AdjClose"})[COLS].copy()
                d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
                d.index.name = "Date"
                f = OUT_DIR / f"{code}.csv"
                if period == "10d":
                    d = merge(pd.read_csv(f, index_col=0, parse_dates=True), d)
                d.sort_index().tail(600).round(4).to_csv(f)
            time.sleep(2)
            print(f"[fetch_prices] {period}: {min(i + BATCH, len(group))}/{len(group)}")


def fetch_index() -> None:
    """加權指數(^TWII)存成 data/prices/TAIEX.csv,給大盤多空判斷用。"""
    d = download(["^TWII"], "3y", adjust=True)
    if d.empty:
        print("[fetch_prices] 加權指數下載失敗,沿用舊檔", file=sys.stderr)
        return
    if isinstance(d.columns, pd.MultiIndex):
        d = d["^TWII"]
    d = d.dropna(subset=["Close"])[["Open", "High", "Low", "Close", "Volume"]].copy()
    d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
    d.index.name = "Date"
    d.round(2).to_csv(OUT_DIR / "TAIEX.csv")
    print(f"[fetch_prices] 加權指數 {len(d)} 筆")


if __name__ == "__main__":
    main()
    fetch_index()
