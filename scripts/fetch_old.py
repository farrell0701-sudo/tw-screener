"""下載 2004~2016 年的舊股價(含已下市股票),給 Accum-v3C 舊資料檢驗用。不影響網站。

資料來源:FinMind API v4(GitHub Secret FINMIND_TOKEN,和籌碼下載共用額度)
  - TaiwanStockInfo        目前上市櫃清單
  - TaiwanStockDelisting   已下市清單
  - TaiwanStockPrice       日線(含 TAIEX 加權指數)
  - TaiwanStockShareholding 已發行股數(算當時的股本)
期間:2004-01-01 ~ 2016-06-30

每檔最多 2 次查詢(沒有舊股價的股票只查 1 次)。免費帳號每小時 600 次。
單次最多跑 5 小時 20 分就存檔;再按一次 Run workflow 會自動接續。
**請等籌碼下載全部完成後再跑,兩者共用每小時 600 次的額度。**

輸出(分支 old-data,old_runN.zip):
  _universe.csv  全部股票代號與來源(info / delist)
  _delisted.csv  下市清單原始欄位
  TAIEX.csv      加權指數
  <代號>.csv     date,o,h,l,c,v,tv,ns(沒有舊資料的股票只有標題列)
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
OUT = ROOT / "out_old"                         # workflow 會先把 old-data 分支放在這裡
START, END = "2004-01-01", "2016-06-30"
URL = "https://api.finmindtrade.com/api/v4/data"
TOKEN = os.environ.get("FINMIND_TOKEN", "").strip()
GAP = 6.2
BUDGET = 5 * 3600 + 20 * 60
T0 = time.time()
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
    return bool(re.fullmatch(r"[1-9]\d{3}", str(c).strip()))   # 一般股票(排除 ETF、權證、特別股)


def universe():
    info = get("TaiwanStockInfo", dated=False)
    if info is None or not len(info):
        sys.exit("TaiwanStockInfo 抓不到,請確認 FINMIND_TOKEN。")
    rows = [(c, "info") for c in info["stock_id"].astype(str).unique() if ok_code(c)]
    dl = get("TaiwanStockDelisting", dated=False)
    if dl is None or not len(dl):
        print("警告:下市清單抓不到,這次只有目前上市櫃股票(會有存活偏誤)", flush=True)
        dl = pd.DataFrame()
    else:
        col = next((c for c in dl.columns if "stock_id" in c.lower()), dl.columns[0])
        have = {c for c, _ in rows}
        rows += [(c, "delist") for c in dl[col].astype(str).str.strip().unique() if ok_code(c) and c not in have]
        print(f"下市清單欄位:{list(dl.columns)}", flush=True)
    u = pd.DataFrame(rows, columns=["code", "src"]).sort_values("code")
    print(f"股票池:目前 {sum(u.src == 'info')} 檔 + 已下市 {sum(u.src == 'delist')} 檔", flush=True)
    return u, dl


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
    return p.drop_duplicates("date", keep="last").sort_values("date")


def one(code):
    px = get("TaiwanStockPrice", code)
    if px is None:
        return None
    p = price_csv(px)
    if isinstance(p, str):                       # 這段期間沒有股價
        return p
    p = p[(p.v > 0) & (p.c > 0)]
    shr = get("TaiwanStockShareholding", code)
    if shr is None:
        return None
    if len(shr) and "NumberOfSharesIssued" in shr:
        s = shr.drop_duplicates("date", keep="last").set_index("date")["NumberOfSharesIssued"]
        p["ns"] = p["date"].map(pd.to_numeric(s, errors="coerce"))
    else:
        p["ns"] = pd.NA
    for c in ("v", "tv", "ns"):
        p[c] = pd.to_numeric(p[c], errors="coerce").round(0).astype("Int64")
    return p.reindex(columns=COLS).to_csv(index=False)


def main():
    if not TOKEN:
        sys.exit("找不到 FINMIND_TOKEN。請到 repo → Settings → Secrets and variables → Actions 新增。")
    OUT.mkdir(parents=True, exist_ok=True)
    runs = sorted(OUT.glob("old_run*.zip"), key=lambda z: int(re.findall(r"\d+", z.stem)[0]))
    done, u = set(), None
    for z in runs:
        with zipfile.ZipFile(z) as f:
            names = f.namelist()
            done |= {n[:-4] for n in names if n.endswith(".csv") and not n.startswith("_")}
            if "_universe.csv" in names:
                u = pd.read_csv(f.open("_universe.csv"), dtype=str)
    store = {}
    if u is None:
        u, dl = universe()
        store["_universe"] = u.to_csv(index=False)
        store["_delisted"] = dl.to_csv(index=False)
    if "TAIEX" not in done:
        tx = price_csv(get("TaiwanStockPrice", "TAIEX"))
        if isinstance(tx, pd.DataFrame):
            store["TAIEX"] = tx.to_csv(index=False)
            print(f"TAIEX {len(tx)} 天", flush=True)
    todo = [c for c in u.code if c not in done]
    print(f"已完成 {len(done)} 檔,這次要抓 {len(todo)} 檔", flush=True)
    got = 0
    for i, code in enumerate(todo, 1):
        if time.time() - T0 > BUDGET:
            print("已達本次時間上限,先存檔。請再按一次 Run workflow 接續。", flush=True); break
        csv = one(code)
        if csv is not None:
            store[code] = csv; got += 1
        if i % 25 == 0:
            print(f"{i}/{len(todo)}  本次已存 {got} 檔  已用 {(time.time() - T0) / 60:.0f} 分鐘", flush=True)
    if store:
        n = len(runs) + 1
        with zipfile.ZipFile(OUT / f"old_run{n}.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for c, s in store.items():
                z.writestr(f"{c}.csv", s)
    total = len([c for c in u.code if c in done or c in store])
    left = len(u) - total
    (OUT / "README.md").write_text(
        f"Accum-v3C 舊資料檢驗用股價(FinMind)\n\n- 已完成:{total} / {len(u)} 檔\n- 尚未完成:{left} 檔\n"
        f"- 期間:{START} ~ {END}\n- 最後更新(UTC):{pd.Timestamp.utcnow():%Y-%m-%d %H:%M}\n", encoding="utf-8")
    print(f"本次完成 {got} 檔;累計 {total}/{len(u)};剩 {left} 檔" + (",請再執行一次。" if left else ",全部完成!"))


if __name__ == "__main__":
    main()
