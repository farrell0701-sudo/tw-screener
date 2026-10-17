"""下載全部上市櫃股票的歷史集保股權分散表(回測用,不影響網站)。

資料來源:FinMind API v4 TaiwanStockHoldingSharesPer(需要 GitHub Secret FINMIND_TOKEN)
期間:2015-01-01 到今天,每週一筆。每檔 1 次查詢,約 3.5 小時;沒抓完再按一次會接續。

輸出(每檔一個 CSV,欄位):
  date   集保資料日期(週)
  p100   持股 100 張以上(100,001 股以上)佔集保庫存 %
  p400   400 張以上 %
  p1000  1,000 張以上 %
  n400   400 張以上人數
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
LIST = ROOT / "data" / "stock_list.csv"
OUT = ROOT / "out_holders"                    # workflow 會先把 holders-data 分支放在這裡
START = "2015-01-01"
URL = "https://api.finmindtrade.com/api/v4/data"
TOKEN = os.environ.get("FINMIND_TOKEN", "").strip()
GAP = 6.2                                     # 每次查詢間隔秒數(600 次/小時)
BUDGET = 5 * 3600 + 20 * 60                   # 單次執行時間上限
T0 = time.time()
_last = [0.0]


def get(dataset, code):
    for attempt in range(6):
        wait = GAP - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            r = requests.get(URL, params={"dataset": dataset, "data_id": code, "start_date": START},
                             headers={"Authorization": f"Bearer {TOKEN}"}, timeout=60)
            j = r.json()
        except Exception as e:  # noqa: BLE001
            print(f"  {code} {dataset} 連線失敗({e}),重試", flush=True); time.sleep(20); continue
        if r.status_code == 200 and j.get("status") == 200:
            return pd.DataFrame(j.get("data", []))
        msg = str(j.get("msg", ""))[:120]
        if r.status_code in (402, 429) or "limit" in msg.lower():
            print(f"  達到查詢上限,暫停 10 分鐘:{msg}", flush=True); time.sleep(600); continue
        print(f"  {code} {dataset} 回應 {r.status_code}:{msg}", flush=True); time.sleep(10)
    return None


def lower_bound(level):
    """'100,001-200,000' → 100001;'more than 1,000,001' → 1000001;其他(合計、差異數調整)→ None"""
    m = re.search(r"[\d,]+", str(level))
    if not m or "total" in str(level).lower() or "差異" in str(level):
        return None
    return int(m.group().replace(",", ""))


def compact(code):
    d = get("TaiwanStockHoldingSharesPer", code)
    if d is None:
        return None
    cols = ["p100", "p400", "p1000", "n400"]
    if not len(d):
        return pd.DataFrame(columns=["date"] + cols).to_csv(index=False)
    d["lo"] = d["HoldingSharesLevel"].map(lower_bound)
    d = d.dropna(subset=["lo"])
    d["percent"] = pd.to_numeric(d["percent"], errors="coerce")
    d["people"] = pd.to_numeric(d["people"], errors="coerce")
    g = d.groupby("date")
    out = pd.DataFrame({"p100": g.apply(lambda x: x.percent[x.lo >= 100001].sum()),
                        "p400": g.apply(lambda x: x.percent[x.lo >= 400001].sum()),
                        "p1000": g.apply(lambda x: x.percent[x.lo >= 1000001].sum()),
                        "n400": g.apply(lambda x: x.people[x.lo >= 400001].sum())}).round(2)
    out.index.name = "date"
    out["n400"] = out["n400"].astype("Int64")
    return out.to_csv()


def main():
    if not TOKEN:
        sys.exit("找不到 FINMIND_TOKEN。請到 repo → Settings → Secrets and variables → Actions 新增。")
    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    runs = sorted(OUT.glob("holders_run*.zip"))
    for z in runs:
        with zipfile.ZipFile(z) as f:
            done |= {n[:-4] for n in f.namelist() if n.endswith(".csv") and n != "stock_list.csv"}
    lst = pd.read_csv(LIST, dtype=str)
    todo = [c for c in lst.code if c not in done]
    print(f"已完成 {len(done)} 檔,這次要抓 {len(todo)} 檔", flush=True)
    if not todo:
        print("全部完成,不需要再執行。"); return
    store = {}
    for i, code in enumerate(todo, 1):
        if time.time() - T0 > BUDGET:
            print("已達本次時間上限,先存檔。請再按一次 Run workflow 接續。", flush=True); break
        csv = compact(code)
        if csv is not None:
            store[code] = csv
        if i % 25 == 0:
            el = (time.time() - T0) / 60
            print(f"{i}/{len(todo)}  本次已存 {len(store)} 檔  已用 {el:.0f} 分鐘", flush=True)
    if store:
        n = len(runs) + 1
        with zipfile.ZipFile(OUT / f"holders_run{n}.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for c, s in store.items():
                z.writestr(f"{c}.csv", s)
            if n == 1:
                z.writestr("stock_list.csv", lst.to_csv(index=False))
    total = len(done) + len(store)
    left = len(lst) - total
    (OUT / "README.md").write_text(
        f"回測用集保股權分散表歷史(FinMind)\n\n- 已完成:{total} / {len(lst)} 檔\n- 尚未完成:{left} 檔\n"
        f"- 期間:{START} 起\n- 最後更新(UTC):{pd.Timestamp.utcnow():%Y-%m-%d %H:%M}\n", encoding="utf-8")
    print(f"本次完成 {len(store)} 檔;累計 {total}/{len(lst)};剩 {left} 檔" + (",請再執行一次。" if left else ",全部完成!"))


if __name__ == "__main__":
    main()
