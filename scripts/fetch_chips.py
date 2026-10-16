"""下載全部上市櫃股票的歷史籌碼資料(回測用,不影響網站)。

資料來源:FinMind API v4(需要 GitHub Secret FINMIND_TOKEN)
  - TaiwanStockInstitutionalInvestorsBuySell  三大法人買賣
  - TaiwanStockMarginPurchaseShortSale        融資融券餘額
  - TaiwanStockShareholding                   外資持股比例、發行股數
期間:2015-01-01 到今天

免費帳號每小時 600 次,每檔 3 次查詢,全部約需 10 小時。
GitHub 單次執行最多 6 小時,所以本程式最多跑 5 小時 20 分就存檔結束,
下次執行時會自動跳過已完成的股票(讀取 chips-data 分支裡已有的 zip)。

輸出(每檔一個 CSV,欄位):
  date  日期
  fi    外資買賣超(股,含外資自營商)
  it    投信買賣超(股)
  dl    自營商買賣超(股,自行買賣 + 避險)
  mb    融資餘額(張)
  sb    融券餘額(張)
  fr    外資持股比例(%)
  ns    已發行股數(股)
"""
import io
import os
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
LIST = ROOT / "data" / "stock_list.csv"
OUT = ROOT / "out_chips"                      # workflow 會先把 chips-data 分支放在這裡
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


def compact(code):
    inst = get("TaiwanStockInstitutionalInvestorsBuySell", code)
    marg = get("TaiwanStockMarginPurchaseShortSale", code)
    shr = get("TaiwanStockShareholding", code)
    if inst is None or marg is None or shr is None:
        return None
    parts = []
    if len(inst):
        inst["net"] = pd.to_numeric(inst["buy"], errors="coerce") - pd.to_numeric(inst["sell"], errors="coerce")
        p = inst.pivot_table(index="date", columns="name", values="net", aggfunc="sum")
        col = lambda names: p.reindex(columns=names).sum(axis=1, min_count=1)
        parts.append(pd.DataFrame({"fi": col(["Foreign_Investor", "Foreign_Dealer_Self"]),
                                   "it": col(["Investment_Trust"]),
                                   "dl": col(["Dealer_self", "Dealer_Hedging", "Dealer"])}))
    if len(marg):
        m = marg.set_index("date")
        parts.append(pd.DataFrame({"mb": pd.to_numeric(m["MarginPurchaseTodayBalance"], errors="coerce"),
                                   "sb": pd.to_numeric(m["ShortSaleTodayBalance"], errors="coerce")}))
    if len(shr):
        s = shr.drop_duplicates("date", keep="last").set_index("date")
        parts.append(pd.DataFrame({"fr": pd.to_numeric(s["ForeignInvestmentSharesRatio"], errors="coerce"),
                                   "ns": pd.to_numeric(s["NumberOfSharesIssued"], errors="coerce")}))
    cols = ["fi", "it", "dl", "mb", "sb", "fr", "ns"]
    if not parts:
        return pd.DataFrame(columns=["date"] + cols).to_csv(index=False)
    df = pd.concat(parts, axis=1).sort_index().reindex(columns=cols)
    df.index.name = "date"
    for c in ("fi", "it", "dl", "mb", "sb", "ns"):
        df[c] = df[c].round(0).astype("Int64")
    return df.to_csv()


def main():
    if not TOKEN:
        sys.exit("找不到 FINMIND_TOKEN。請到 repo → Settings → Secrets and variables → Actions 新增。")
    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    runs = sorted(OUT.glob("chips_run*.zip"))
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
        with zipfile.ZipFile(OUT / f"chips_run{n}.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for c, s in store.items():
                z.writestr(f"{c}.csv", s)
            if n == 1:
                z.writestr("stock_list.csv", lst.to_csv(index=False))
    total = len(done) + len(store)
    left = len(lst) - total
    (OUT / "README.md").write_text(
        f"回測用籌碼歷史(FinMind)\n\n- 已完成:{total} / {len(lst)} 檔\n- 尚未完成:{left} 檔\n"
        f"- 期間:{START} 起\n- 最後更新(UTC):{pd.Timestamp.utcnow():%Y-%m-%d %H:%M}\n", encoding="utf-8")
    print(f"本次完成 {len(store)} 檔;累計 {total}/{len(lst)};剩 {left} 檔" + (",請再執行一次。" if left else ",全部完成!"))


if __name__ == "__main__":
    main()
