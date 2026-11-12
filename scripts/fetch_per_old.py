"""下載 2005–2014/06 的每日本益比、股價淨值比、殖利率（FinMind TaiwanStockPER，回測用，不影響網站）。
代號清單：data/codes_per_old.csv（2014/06 前就有股價的 1,852 檔，含之後下市者）。
約 1,850 次查詢；單次最多跑 5 小時 20 分就存檔，沒抓完再按一次 Run workflow 會接續。
輸出：分支 per-old-data，per_old_runN.zip（內含 per_<代號>.csv）
"""
import os, re, sys, time, zipfile
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out_per_old"
URL = "https://api.finmindtrade.com/api/v4/data"
TOKEN = os.environ.get("FINMIND_TOKEN", "").strip()
GAP, BUDGET, T0 = 6.2, 5 * 3600 + 20 * 60, time.time()
_last = [0.0]


def get(code):
    params = {"dataset": "TaiwanStockPER", "data_id": code, "start_date": "2005-01-01", "end_date": "2014-06-30"}
    for _ in range(6):
        wait = GAP - (time.time() - _last[0])
        if wait > 0: time.sleep(wait)
        _last[0] = time.time()
        try:
            r = requests.get(URL, params=params, headers={"Authorization": f"Bearer {TOKEN}"}, timeout=120); j = r.json()
        except Exception as e:  # noqa: BLE001
            print(f"  {code} 連線失敗({e})，重試", flush=True); time.sleep(20); continue
        if r.status_code == 200 and j.get("status") == 200:
            return pd.DataFrame(j.get("data", []))
        msg = str(j.get("msg", ""))[:150]
        if r.status_code in (402, 429) or "limit" in msg.lower():
            print(f"  達到查詢上限，暫停 10 分鐘：{msg}", flush=True); time.sleep(600); continue
        print(f"  {code} 回應 {r.status_code}：{msg}", flush=True)
        return None if r.status_code >= 500 else pd.DataFrame()
    return None


def main():
    if not TOKEN: sys.exit("找不到 FINMIND_TOKEN")
    OUT.mkdir(parents=True, exist_ok=True)
    runs = sorted(OUT.glob("per_old_run*.zip"), key=lambda z: int(re.findall(r"\d+", z.stem)[0]))
    done = set()
    for z in runs:
        with zipfile.ZipFile(z) as f: done |= {n[4:-4] for n in f.namelist() if n.endswith(".csv")}
    codes = pd.read_csv(ROOT / "data" / "codes_per_old.csv", dtype=str).code.tolist()
    todo = [c for c in codes if c not in done]
    print(f"共 {len(codes)} 檔，已完成 {len(done)}，待抓 {len(todo)}", flush=True)
    store, got = {}, 0
    for i, c in enumerate(todo, 1):
        if time.time() - T0 > BUDGET:
            print("已達本次時間上限，先存檔", flush=True); break
        df = get(c)
        if df is not None:
            store[f"per_{c}"] = df.to_csv(index=False); got += 1
        if i % 100 == 0: print(f"{i}/{len(todo)} 已存 {got} 已用 {(time.time() - T0) / 60:.0f} 分", flush=True)
    if store:
        with zipfile.ZipFile(OUT / f"per_old_run{len(runs) + 1}.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for k, s in store.items(): z.writestr(f"{k}.csv", s)
    left = len(todo) - got
    (OUT / "README.md").write_text(f"2005–2014/06 本益比資料（FinMind）\n\n- 尚缺 {left} 檔\n- 最後更新（UTC）：{pd.Timestamp.utcnow():%Y-%m-%d %H:%M}\n", encoding="utf-8")
    print(f"本次存 {got} 檔，尚缺 {left} 檔", flush=True)


if __name__ == "__main__":
    main()
