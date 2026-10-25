"""下載基本面資料(回測用,不影響網站)。資料來源:FinMind API v4(GitHub Secret FINMIND_TOKEN)
逐檔抓(現存股 data/stock_list.csv ＋ delist-data 的下市股):
  per_<代號>.csv  TaiwanStockPER                  每日本益比、股價淨值比、殖利率(2014-06 起)
  is_<代號>.csv   TaiwanStockFinancialStatements  綜合損益表(季,2013 起)
  bs_<代號>.csv   TaiwanStockBalanceSheet         資產負債表(季,2013 起)
  rev_<代號>.csv  TaiwanStockMonthRevenue         月營收(只補下市股;現存股已在 revenue-data 分支)
約 6,500 次查詢、每小時約 580 次;單次最多跑 5 小時 20 分就存檔,沒抓完 workflow 會自動再觸發接續。
注意:財報 date 是報表所屬季底,不是公布日;回測必須自行加上公布延遲(Q1→5/15、Q2→8/14、Q3→11/14、Q4→次年 3/31)。
輸出:分支 fund-data,fund_runN.zip
"""
import os, re, sys, time, zipfile
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out_fund"
DELIST_ZIP = ROOT / "delist_src" / "delist.zip"
END = pd.Timestamp.today().strftime("%Y-%m-%d")
URL = "https://api.finmindtrade.com/api/v4/data"
TOKEN = os.environ.get("FINMIND_TOKEN", "").strip()
STATUS = Path(os.environ.get("STATUS_FILE", "/tmp/fund_status.txt"))
GAP, BUDGET, T0 = 6.2, 5 * 3600 + 20 * 60, time.time()
_last = [0.0]
JOBS = [("per", "TaiwanStockPER", "2014-06-01"),
        ("is", "TaiwanStockFinancialStatements", "2013-01-01"),
        ("bs", "TaiwanStockBalanceSheet", "2013-01-01")]


def get(dataset, code, start):
    params = {"dataset": dataset, "data_id": code, "start_date": start, "end_date": END}
    for _ in range(6):
        wait = GAP - (time.time() - _last[0])
        if wait > 0: time.sleep(wait)
        _last[0] = time.time()
        try:
            r = requests.get(URL, params=params, headers={"Authorization": f"Bearer {TOKEN}"}, timeout=120); j = r.json()
        except Exception as e:  # noqa: BLE001
            print(f"  {code} {dataset} 連線失敗({e}),重試", flush=True); time.sleep(20); continue
        if r.status_code == 200 and j.get("status") == 200:
            return pd.DataFrame(j.get("data", []))
        msg = str(j.get("msg", ""))[:150]
        if r.status_code in (402, 429) or "limit" in msg.lower():
            print(f"  達到查詢上限,暫停 10 分鐘:{msg}", flush=True); time.sleep(600); continue
        print(f"  {code} {dataset} 回應 {r.status_code}:{msg}", flush=True)
        return None if r.status_code >= 500 else pd.DataFrame()
    return None


def main():
    if not TOKEN: sys.exit("找不到 FINMIND_TOKEN")
    OUT.mkdir(parents=True, exist_ok=True)
    runs = sorted(OUT.glob("fund_run*.zip"), key=lambda z: int(re.findall(r"\d+", z.stem)[0]))
    done = set()
    for z in runs:
        with zipfile.ZipFile(z) as f: done |= {n[:-4] for n in f.namelist() if n.endswith(".csv")}
    codes = pd.read_csv(ROOT / "data" / "stock_list.csv", dtype=str).code.tolist()
    delisted = []
    if DELIST_ZIP.exists():
        with zipfile.ZipFile(DELIST_ZIP) as f:
            delisted = sorted(n[:-4] for n in f.namelist() if n.endswith(".csv") and not n.startswith("_"))
    todo = [(p, ds, st, c) for c in codes + delisted for p, ds, st in JOBS if f"{p}_{c}" not in done]
    todo += [("rev", "TaiwanStockMonthRevenue", "2014-01-01", c) for c in delisted if f"rev_{c}" not in done]
    print(f"現存 {len(codes)}、下市 {len(delisted)};已完成 {len(done)},待抓 {len(todo)}", flush=True)
    store, got = {}, 0
    for i, (p, ds, st, c) in enumerate(todo, 1):
        if time.time() - T0 > BUDGET:
            print("已達本次時間上限,先存檔", flush=True); break
        df = get(ds, c, st)
        if df is not None:
            store[f"{p}_{c}"] = df.to_csv(index=False); got += 1
        if i % 100 == 0: print(f"{i}/{len(todo)} 已存 {got} 已用 {(time.time() - T0) / 60:.0f} 分", flush=True)
    if store:
        with zipfile.ZipFile(OUT / f"fund_run{len(runs) + 1}.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for k, s in store.items(): z.writestr(f"{k}.csv", s)
    left = len(todo) - got
    STATUS.write_text("MORE" if left > 0 else "DONE")
    (OUT / "README.md").write_text(f"基本面資料(FinMind)\n\n- 尚缺 {left} 項\n- 最後更新(UTC):{pd.Timestamp.utcnow():%Y-%m-%d %H:%M}\n", encoding="utf-8")
    print(f"本次存 {got} 項,尚缺 {left} 項", flush=True)


if __name__ == "__main__":
    main()
