"""下載籌碼資料（回測用，不影響網站）。來源：FinMind API v4（GitHub Secret FINMIND_TOKEN）
逐檔抓（現存股 data/stock_list.csv ＋ delist-data 的下市股），2014-06 起，存成精簡欄位：
  inst_<代號>.csv  TaiwanStockInstitutionalInvestorsBuySell  每日法人買賣超（股）：外資、投信、自營商
  mg_<代號>.csv    TaiwanStockMarginPurchaseShortSale        每日融資融券（買賣、餘額）
  dt_<代號>.csv    TaiwanStockDayTrading                     每日當沖成交量／金額
  hold_<代號>.csv  TaiwanStockHoldingSharesPer               每週集保股權分散（各級距人數、比例）
開跑前先用 2330 試每個資料集；權限不足（贊助限定）的直接略過並寫進 README。
約 8,500 次查詢、每小時約 580 次；每次最多 5 小時 20 分，4 個工作依序接續。
輸出：分支 chip-data，chip_runN_partK.zip（每個 zip 控制在 80MB 內，避開 GitHub 100MB 上限）
"""
import os, re, sys, time, zipfile
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out_chip"
DELIST_ZIP = ROOT / "delist_src" / "delist.zip"
END = pd.Timestamp.today().strftime("%Y-%m-%d")
URL = "https://api.finmindtrade.com/api/v4/data"
TOKEN = os.environ.get("FINMIND_TOKEN", "").strip()
STATUS = Path(os.environ.get("STATUS_FILE", "/tmp/chip_status.txt"))
GAP, BUDGET, T0 = 6.2, 5 * 3600 + 20 * 60, time.time()
START = "2014-06-01"
PART_MAX = 80 * 1024 * 1024
_last = [0.0]
JOBS = [("inst", "TaiwanStockInstitutionalInvestorsBuySell"), ("mg", "TaiwanStockMarginPurchaseShortSale"),
        ("dt", "TaiwanStockDayTrading"), ("hold", "TaiwanStockHoldingSharesPer")]


def get(dataset, code, start=START, end=END):
    """回傳 (DataFrame 或 None, 錯誤訊息)"""
    params = {"dataset": dataset, "data_id": code, "start_date": start, "end_date": end}
    for _ in range(6):
        wait = GAP - (time.time() - _last[0])
        if wait > 0: time.sleep(wait)
        _last[0] = time.time()
        try:
            r = requests.get(URL, params=params, headers={"Authorization": f"Bearer {TOKEN}"}, timeout=180); j = r.json()
        except Exception as e:  # noqa: BLE001
            print(f"  {code} {dataset} 連線失敗({e})，重試", flush=True); time.sleep(20); continue
        if r.status_code == 200 and j.get("status") == 200:
            return pd.DataFrame(j.get("data", [])), ""
        msg = str(j.get("msg", ""))[:200]
        if r.status_code == 429 or "limit" in msg.lower() or "upper" in msg.lower():
            print(f"  達到查詢上限，暫停 10 分鐘：{msg}", flush=True); time.sleep(600); continue
        print(f"  {code} {dataset} 回應 {r.status_code}：{msg}", flush=True)
        return (None if r.status_code >= 500 else pd.DataFrame()), f"{r.status_code} {msg}"
    return None, "重試失敗"


def slim(p, df):
    if df is None or not len(df): return df
    if p == "inst":
        name = df["name"].astype(str)
        grp = name.map(lambda s: "foreign" if s.startswith("Foreign") else "trust" if s.startswith("Investment") else "dealer" if s.startswith("Dealer") else "other")
        df = df.assign(net=pd.to_numeric(df.buy, errors="coerce") - pd.to_numeric(df.sell, errors="coerce"), grp=grp)
        return df.pivot_table(index="date", columns="grp", values="net", aggfunc="sum").reset_index()
    if p == "mg":
        keep = [c for c in ("date", "MarginPurchaseBuy", "MarginPurchaseSell", "MarginPurchaseTodayBalance", "MarginPurchaseLimit",
                            "ShortSaleBuy", "ShortSaleSell", "ShortSaleTodayBalance") if c in df]
        return df[keep]
    if p == "dt":
        return df[[c for c in ("date", "Volume", "BuyAmount", "SellAmount") if c in df]]
    if p == "hold":
        return df[[c for c in ("date", "HoldingSharesLevel", "people", "percent", "unit") if c in df]]
    return df


class Parts:
    def __init__(self, run):
        self.run, self.k, self.z = run, 0, None
    def write(self, name, text):
        path = OUT / f"chip_run{self.run}_part{self.k}.zip"
        if self.z is None or path.stat().st_size > PART_MAX:
            if self.z: self.z.close(); self.k += 1
            self.z = zipfile.ZipFile(OUT / f"chip_run{self.run}_part{self.k}.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9)
        self.z.writestr(name, text)
    def close(self):
        if self.z: self.z.close()


def main():
    if not TOKEN: sys.exit("找不到 FINMIND_TOKEN")
    OUT.mkdir(parents=True, exist_ok=True)
    zips = list(OUT.glob("chip_run*.zip"))
    done = set()
    for z in zips:
        with zipfile.ZipFile(z) as f: done |= {n[:-4] for n in f.namelist() if n.endswith(".csv")}
    run = 1 + max([int(re.findall(r"run(\d+)", z.stem)[0]) for z in zips] or [0])
    # 權限檢查
    ok_jobs, notes = [], []
    skip_file = OUT / "skipped.txt"
    skipped = set(skip_file.read_text().split()) if skip_file.exists() else set()
    for p, ds in JOBS:
        if p in skipped: notes.append(f"{ds}：先前已判定無權限，略過"); continue
        df, err = get(ds, "2330", "2024-01-01", "2024-03-01")
        if df is not None and len(df): ok_jobs.append((p, ds))
        else:
            notes.append(f"{ds}：無法取得（{err}），略過"); skipped.add(p)
    skip_file.write_text("\n".join(sorted(skipped)))
    for n in notes: print(n, flush=True)
    codes = pd.read_csv(ROOT / "data" / "stock_list.csv", dtype=str).code.tolist()
    delisted = []
    if DELIST_ZIP.exists():
        with zipfile.ZipFile(DELIST_ZIP) as f:
            delisted = sorted(n[:-4] for n in f.namelist() if n.endswith(".csv") and not n.startswith("_"))
    todo = [(p, ds, c) for p, ds in ok_jobs for c in codes + delisted if f"{p}_{c}" not in done]
    print(f"現存 {len(codes)}、下市 {len(delisted)}；資料集 {[p for p, _ in ok_jobs]}；已完成 {len(done)}，待抓 {len(todo)}", flush=True)
    parts, got = Parts(run), 0
    for i, (p, ds, c) in enumerate(todo, 1):
        if time.time() - T0 > BUDGET:
            print("已達本次時間上限，先存檔", flush=True); break
        df, _ = get(ds, c)
        if df is not None:
            parts.write(f"{p}_{c}.csv", slim(p, df).to_csv(index=False) if len(df) else ""); got += 1
        if i % 100 == 0: print(f"{i}/{len(todo)} 已存 {got} 已用 {(time.time() - T0) / 60:.0f} 分", flush=True)
    parts.close()
    left = len(todo) - got
    STATUS.write_text("MORE" if left > 0 else "DONE")
    (OUT / "README.md").write_text("籌碼資料（FinMind）\n\n" + "\n".join(f"- {n}" for n in notes) +
                                   f"\n- 尚缺 {left} 項\n- 最後更新（UTC）：{pd.Timestamp.utcnow():%Y-%m-%d %H:%M}\n", encoding="utf-8")
    print(f"本次存 {got} 項，尚缺 {left} 項", flush=True)


if __name__ == "__main__":
    main()
