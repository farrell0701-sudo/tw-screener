"""下載 2015 年至今的「真正未還原」日線,修正 Yahoo Close 被配股/分割回溯調整的問題。不影響網站。

背景:Yahoo 的 Close 會把配股、股票分割回溯調整到歷史價格,不是當時真正的成交價。
回測裡「股價 < 30」「同分股價低優先」「整數股」都需要真正的未還原價,
用 Yahoo Close 會產生未來資訊偏誤(之後配股的公司在過去看起來比較便宜)。

資料來源:FinMind API v4(GitHub Secret FINMIND_TOKEN)
  - TaiwanStockPrice                          2015-01-01 起原始日線,data/stock_list.csv 全部股票
  - TaiwanStockDividendResult                 除權息結果(前收盤、參考價)
  - TaiwanStockCapitalReductionReferencePrice 減資恢復買賣參考價
  - TaiwanStockSplitPrice                     面額變更參考價
  事件資料先嘗試一次抓全市場;不支援時只替 delist-data 的下市股逐檔補抓(現存股的還原比例用 Yahoo AdjClose)。

每小時約 580 次查詢。單次最多跑 5 小時 20 分就存檔;再執行一次會自動接續。
與其他 FinMind 下載共用額度,請不要同時跑。

輸出(分支 raw-data,raw_runN.zip):
  <代號>.csv            date,o,h,l,c,v,tv(未還原)
  _ev_<資料集>.csv       全市場事件(若支援整批查詢)
  ev_<代號>.csv          逐檔事件(下市股,三個資料集合併,欄位 dataset 標示來源)
"""
import io
import os
import re
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out_raw"                       # workflow 會先把 raw-data 分支放在這裡
DELIST_ZIP = ROOT / "delist_src" / "delist.zip"
START = "2015-01-01"
END = pd.Timestamp.today().strftime("%Y-%m-%d")
URL = "https://api.finmindtrade.com/api/v4/data"
TOKEN = os.environ.get("FINMIND_TOKEN", "").strip()
GAP = 6.2
BUDGET = 5 * 3600 + 20 * 60
T0 = time.time()
_last = [0.0]
EVENTS = ["TaiwanStockDividendResult", "TaiwanStockCapitalReductionReferencePrice", "TaiwanStockSplitPrice"]
COLS = ["date", "o", "h", "l", "c", "v", "tv"]


def get(dataset, code=None, quiet=False):
    params = {"dataset": dataset, "start_date": START, "end_date": END}
    if code:
        params["data_id"] = code
    for _ in range(6):
        wait = GAP - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            r = requests.get(URL, params=params, headers={"Authorization": f"Bearer {TOKEN}"}, timeout=120)
            j = r.json()
        except Exception as e:  # noqa: BLE001
            print(f"  {code} {dataset} 連線失敗({e}),重試", flush=True); time.sleep(20); continue
        if r.status_code == 200 and j.get("status") == 200:
            return pd.DataFrame(j.get("data", []))
        msg = str(j.get("msg", ""))[:150]
        if r.status_code in (402, 429) or "limit" in msg.lower():
            print(f"  達到查詢上限,暫停 10 分鐘:{msg}", flush=True); time.sleep(600); continue
        if not quiet:
            print(f"  {code} {dataset} 回應 {r.status_code}:{msg}", flush=True)
        return None if r.status_code >= 500 else pd.DataFrame()   # 4xx(例如不支援整批)回空表
    return None


def price_csv(df):
    if df is None:
        return None
    if not len(df):
        return pd.DataFrame(columns=COLS).to_csv(index=False)
    p = pd.DataFrame({"date": df["date"],
                      "o": pd.to_numeric(df["open"], errors="coerce"),
                      "h": pd.to_numeric(df["max"], errors="coerce"),
                      "l": pd.to_numeric(df["min"], errors="coerce"),
                      "c": pd.to_numeric(df["close"], errors="coerce"),
                      "v": pd.to_numeric(df["Trading_Volume"], errors="coerce"),
                      "tv": pd.to_numeric(df["Trading_money"], errors="coerce")})
    p = p.drop_duplicates("date", keep="last").sort_values("date")
    p = p[(p.v > 0) & (p.c > 0)]
    for c in ("v", "tv"):
        p[c] = p[c].round(0).astype("Int64")
    return p.to_csv(index=False)


def main():
    if not TOKEN:
        sys.exit("找不到 FINMIND_TOKEN。請到 repo → Settings → Secrets and variables → Actions 新增。")
    OUT.mkdir(parents=True, exist_ok=True)
    runs = sorted(OUT.glob("raw_run*.zip"), key=lambda z: int(re.findall(r"\d+", z.stem)[0]))
    done = set()
    for z in runs:
        with zipfile.ZipFile(z) as f:
            done |= {n[:-4] for n in f.namelist() if n.endswith(".csv")}
    store = {}

    codes = pd.read_csv(ROOT / "data" / "stock_list.csv", dtype=str).code.tolist()
    delisted = []
    if DELIST_ZIP.exists():
        with zipfile.ZipFile(DELIST_ZIP) as f:
            delisted = sorted(n[:-4] for n in f.namelist() if n.endswith(".csv") and not n.startswith("_"))
    print(f"現存股 {len(codes)} 檔、下市股 {len(delisted)} 檔", flush=True)

    # 1) 事件資料:先試整批(全市場)
    bulk_ok = {}
    for ds in EVENTS:
        key = f"_ev_{ds}"
        if key in done:
            bulk_ok[ds] = True
            continue
        df = get(ds, quiet=True)
        ok = df is not None and len(df) > 50 and "stock_id" in df.columns and df["stock_id"].nunique() > 20
        bulk_ok[ds] = ok
        if ok:
            store[key] = df.to_csv(index=False)
        print(f"{ds} 整批查詢:{'成功 ' + str(len(df)) + ' 筆' if ok else '不支援,改逐檔(只補下市股)'}", flush=True)

    # 2) 逐檔:現存股抓價格;下市股抓事件(若整批不支援)
    todo = [("px", c) for c in codes if c not in done]
    todo += [("ev", c) for c in delisted if f"ev_{c}" not in done and not all(bulk_ok.values())]
    print(f"已完成 {len(done)} 個檔案,這次待抓 {len(todo)} 項", flush=True)
    got = 0
    for i, (kind, code) in enumerate(todo, 1):
        if time.time() - T0 > BUDGET:
            print("已達本次時間上限,先存檔。請再執行一次接續。", flush=True); break
        if kind == "px":
            csv = price_csv(get("TaiwanStockPrice", code))
            if csv is not None:
                store[code] = csv; got += 1
        else:
            parts, fail = [], False
            for ds in EVENTS:
                if bulk_ok.get(ds):
                    continue
                df = get(ds, code)
                if df is None:
                    fail = True; break
                if len(df):
                    parts.append(df.assign(dataset=ds))
            if not fail:
                store[f"ev_{code}"] = (pd.concat(parts) if parts else pd.DataFrame(columns=["date", "dataset"])).to_csv(index=False)
                got += 1
        if i % 50 == 0:
            print(f"{i}/{len(todo)}  本次已存 {got}  已用 {(time.time() - T0) / 60:.0f} 分鐘", flush=True)

    if store:
        n = len(runs) + 1
        with zipfile.ZipFile(OUT / f"raw_run{n}.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for k, s in store.items():
                z.writestr(f"{k}.csv", s)
    have = done | set(store)
    left_px = sum(c not in have for c in codes)
    left_ev = 0 if all(bulk_ok.values()) else sum(f"ev_{c}" not in have for c in delisted)
    (OUT / "README.md").write_text(
        f"回測用真正未還原日線(FinMind)\n\n- 期間:{START} ~ {END}\n- 現存股價格尚缺:{left_px} 檔\n"
        f"- 下市股事件尚缺:{left_ev} 檔\n- 事件整批查詢:{bulk_ok}\n"
        f"- 最後更新(UTC):{pd.Timestamp.utcnow():%Y-%m-%d %H:%M}\n", encoding="utf-8")
    print(f"本次存 {got} 項;現存股價格尚缺 {left_px} 檔、下市股事件尚缺 {left_ev} 檔"
          + (",請再執行一次。" if left_px or left_ev else ",全部完成!"), flush=True)


if __name__ == "__main__":
    main()
