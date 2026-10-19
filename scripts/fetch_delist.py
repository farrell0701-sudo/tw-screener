"""下載 2016-07 以後下市股票的日線(FinMind),補 2016-2026 回測的存活偏誤。不影響網站。

資料來源:FinMind API v4(GitHub Secret FINMIND_TOKEN)
  - TaiwanStockInfo / TaiwanStockDelisting  找出 2016-07-01 以後下市、目前不在清單上的一般股票
  - TaiwanStockPrice                         2015-01-01 起的原始日線(未還原)
每檔 1 次查詢,約 150 檔。

輸出(分支 delist-data,delist.zip):
  _delisted.csv  下市清單(date = 下市日)
  <代號>.csv     date,o,h,l,c,v,tv,ns(格式同 old-data,ns 留空)
"""
import os
import re
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out_delist"
START, CUT = "2015-01-01", "2016-07-01"
END = pd.Timestamp.today().strftime("%Y-%m-%d")
URL = "https://api.finmindtrade.com/api/v4/data"
TOKEN = os.environ.get("FINMIND_TOKEN", "").strip()
GAP = 6.2
_last = [0.0]
COLS = ["date", "o", "h", "l", "c", "v", "tv", "ns"]


def get(dataset, code=None, dated=True):
    params = {"dataset": dataset}
    if code:
        params["data_id"] = code
    if dated:
        params.update(start_date=START, end_date=END)
    for _ in range(6):
        wait = GAP - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            r = requests.get(URL, params=params, headers={"Authorization": f"Bearer {TOKEN}"}, timeout=90)
            j = r.json()
        except Exception as e:  # noqa: BLE001
            print(f"  {code} {dataset} 連線失敗({e}),重試", flush=True); time.sleep(20); continue
        if r.status_code == 200 and j.get("status") == 200:
            return pd.DataFrame(j.get("data", []))
        msg = str(j.get("msg", ""))[:150]
        if r.status_code in (402, 429) or "limit" in msg.lower():
            print(f"  達到查詢上限,暫停 10 分鐘:{msg}", flush=True); time.sleep(600); continue
        print(f"  {code} {dataset} 回應 {r.status_code}:{msg}", flush=True); time.sleep(10)
    return None


def ok_code(c):
    return bool(re.fullmatch(r"[1-9]\d{3}", str(c).strip()))


def main():
    if not TOKEN:
        sys.exit("找不到 FINMIND_TOKEN。")
    OUT.mkdir(parents=True, exist_ok=True)
    info = get("TaiwanStockInfo", dated=False)
    dl = get("TaiwanStockDelisting", dated=False)
    if info is None or dl is None or not len(dl):
        sys.exit("清單抓不到。")
    have = set(info["stock_id"].astype(str))
    dl["stock_id"] = dl["stock_id"].astype(str).str.strip()
    dl = dl[dl.stock_id.map(ok_code) & ~dl.stock_id.isin(have) & (pd.to_datetime(dl["date"]) >= CUT)]
    dl = dl.drop_duplicates("stock_id", keep="last").sort_values("stock_id")
    print(f"2016-07 後下市:{len(dl)} 檔", flush=True)
    with zipfile.ZipFile(OUT / "delist.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr("_delisted.csv", dl.to_csv(index=False))
        got = 0
        for i, code in enumerate(dl.stock_id, 1):
            df = get("TaiwanStockPrice", code)
            if df is None:
                continue
            if len(df):
                p = pd.DataFrame({"date": df["date"],
                                  "o": pd.to_numeric(df["open"], errors="coerce"),
                                  "h": pd.to_numeric(df["max"], errors="coerce"),
                                  "l": pd.to_numeric(df["min"], errors="coerce"),
                                  "c": pd.to_numeric(df["close"], errors="coerce"),
                                  "v": pd.to_numeric(df["Trading_Volume"], errors="coerce"),
                                  "tv": pd.to_numeric(df["Trading_money"], errors="coerce")})
                p = p.drop_duplicates("date", keep="last").sort_values("date")
                p = p[(p.v > 0) & (p.c > 0)]
            else:
                p = pd.DataFrame(columns=COLS)
            z.writestr(f"{code}.csv", p.reindex(columns=COLS).to_csv(index=False))
            got += 1
            if i % 25 == 0:
                print(f"{i}/{len(dl)}", flush=True)
    print(f"完成 {got}/{len(dl)} 檔")


if __name__ == "__main__":
    main()
