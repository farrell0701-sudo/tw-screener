"""補抓 2004~2014 的除權息、減資事件,讓 old-data(FinMind 未還原 2004~2016,含下市)能完整還原。

raw-data 已有 2015 年起的事件;這裡只補 2004-01-01 ~ 2014-12-31。
面額變更(TaiwanStockSplitPrice)2014 年後才有,不抓。
股票清單 = old-data 分支 zip 裡的所有代號(約 3,000 檔,含下市)。
每檔 2 次查詢,約 6,000 次、10 小時;單次最多跑 5 小時 20 分就存檔,workflow 會自動再觸發接續。

輸出(分支 oldev-data,oldev_runN.zip):ev_<代號>.csv,欄位同 raw-data 的 ev 檔(dataset 欄標示來源)。
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
OUT = ROOT / "out_oldev"
OLD = ROOT / "old_src"
START, END = "2004-01-01", "2014-12-31"
URL = "https://api.finmindtrade.com/api/v4/data"
TOKEN = os.environ.get("FINMIND_TOKEN", "").strip()
GAP = 6.2
BUDGET = 5 * 3600 + 20 * 60
T0 = time.time()
_last = [0.0]
STATUS = Path(os.environ.get("STATUS_FILE", "/tmp/oldev_status.txt"))
EVENTS = ["TaiwanStockDividendResult", "TaiwanStockCapitalReductionReferencePrice"]


def get(dataset, code):
    params = {"dataset": dataset, "start_date": START, "end_date": END, "data_id": code}
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
        print(f"  {code} {dataset} 回應 {r.status_code}:{msg}", flush=True)
        return None if r.status_code >= 500 else pd.DataFrame()
    return None


def main():
    if not TOKEN:
        sys.exit("找不到 FINMIND_TOKEN。")
    OUT.mkdir(parents=True, exist_ok=True)
    codes = set()
    for z in OLD.glob("*.zip"):
        with zipfile.ZipFile(z) as f:
            codes |= {Path(n).stem for n in f.namelist() if n.endswith(".csv")}
    codes = sorted(c for c in codes if c[:1].isdigit())
    runs = sorted(OUT.glob("oldev_run*.zip"), key=lambda z: int(re.findall(r"\d+", z.stem)[0]))
    done = set()
    for z in runs:
        with zipfile.ZipFile(z) as f:
            done |= {n[:-4] for n in f.namelist() if n.endswith(".csv")}
    todo = [c for c in codes if f"ev_{c}" not in done]
    print(f"清單 {len(codes)} 檔,已完成 {len(done)},這次待抓 {len(todo)}", flush=True)
    store, got = {}, 0
    for i, code in enumerate(todo, 1):
        if time.time() - T0 > BUDGET:
            print("已達本次時間上限,先存檔。", flush=True); break
        parts, fail = [], False
        for ds in EVENTS:
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
        with zipfile.ZipFile(OUT / f"oldev_run{n}.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for k, s in store.items():
                z.writestr(f"{k}.csv", s)
    left = sum(f"ev_{c}" not in (done | set(store)) for c in codes)
    (OUT / "README.md").write_text(
        f"舊資料除權息/減資事件(FinMind)\n\n- 期間:{START} ~ {END}\n- 清單:{len(codes)} 檔(old-data,含下市)\n"
        f"- 尚缺:{left} 檔\n- 最後更新(UTC):{pd.Timestamp.utcnow():%Y-%m-%d %H:%M}\n", encoding="utf-8")
    STATUS.write_text("MORE" if left else "DONE")
    print(f"本次存 {got} 檔;尚缺 {left} 檔", flush=True)


if __name__ == "__main__":
    main()
