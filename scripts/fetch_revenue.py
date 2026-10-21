"""下載上市櫃歷史月營收,回測用。不影響網站。

1. 先試公開資訊觀測站(MOPS)彙總表,不需 token:
   https://mopsov.twse.com.tw/nas/t21/{sii|otc}/t21sc03_{民國年}_{月}_{0|1}.html
   GitHub 的 IP 可能被擋;前 3 個測試月都抓不到就改用 2。
2. FinMind TaiwanStockMonthRevenue(GitHub Secret FINMIND_TOKEN)
   先試整個市場一次查一個月;不行就逐檔查(約 2,000 檔,每檔 1 次,可中斷接續)。

輸出(分支 revenue-data):
  revenue.csv  month,code,rev(+ MOPS 才有的 name,market,mom_pct,yoy_pct,cum_rev... 欄)
  month = 營收所屬月份(YYYY-MM),單位:元(FinMind)/千元(MOPS),見 source.txt。
  月營收須在次月 10 日前公布,回測請用次月 11 日起才可用。
"""
import io
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out_revenue"
START = "2005-01"
T0 = time.time()
BUDGET = 320 * 60
HEAD = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}
MOPS_HOSTS = ["https://mopsov.twse.com.tw", "https://mops.twse.com.tw"]
MOPS_COLS = {
    "公司代號": "code", "公司名稱": "name", "當月營收": "rev", "上月營收": "rev_prev_month",
    "去年當月營收": "rev_last_year", "上月比較增減(%)": "mom_pct", "去年同月增減(%)": "yoy_pct",
    "當月累計營收": "cum_rev", "去年累計營收": "cum_rev_last_year", "前期比較增減(%)": "cum_yoy_pct",
    "備註": "note",
}
FM_URL = "https://api.finmindtrade.com/api/v4/data"
FM_TOKEN = os.environ.get("FINMIND_TOKEN", "").strip()
GAP = 6.2
_last = [0.0]


# ---------- MOPS ----------
def mops_get(path, verbose=False):
    for host in MOPS_HOSTS:
        try:
            r = requests.get(host + path, headers=HEAD, timeout=30)
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  {host}{path} 連線失敗:{e}", flush=True)
            continue
        if verbose:
            r.encoding = "big5"
            print(f"  {host}{path} -> {r.status_code},{len(r.content)} bytes,開頭:{r.text[:150]!r}", flush=True)
        if r.status_code == 200 and len(r.content) > 2000:
            r.encoding = "big5"
            return r.text
    return None


def mops_parse(html):
    out = []
    for t in pd.read_html(io.StringIO(html)):
        if isinstance(t.columns, pd.MultiIndex):
            t.columns = t.columns.get_level_values(-1)
        t.columns = [str(c).strip() for c in t.columns]
        if "公司代號" not in t.columns or "當月營收" not in t.columns:
            continue
        t = t.rename(columns=MOPS_COLS)
        t = t[[c for c in MOPS_COLS.values() if c in t.columns]]
        out.append(t[t["code"].astype(str).str.fullmatch(r"\d{4,6}")])
    return pd.concat(out) if out else None


def mops_month(p, verbose=False):
    roc, frames = p.year - 1911, []
    for market in ("sii", "otc"):
        for s in (["_0", "_1"] if roc > 98 else [""]):
            html = mops_get(f"/nas/t21/{market}/t21sc03_{roc}_{p.month}{s}.html", verbose)
            df = mops_parse(html) if html else None
            if df is not None:
                df.insert(0, "month", str(p)); df.insert(3, "market", market)
                frames.append(df)
        time.sleep(1.5)
    return pd.concat(frames) if frames else None


def try_mops(months):
    print("== 測試 MOPS ==", flush=True)
    for p in (months[-1], months[len(months) // 2], months[0]):
        df = mops_month(p, verbose=True)
        print(f"  {p}: {0 if df is None else len(df)} 筆", flush=True)
        if df is None:
            return None
    frames = []
    for p in months:
        df = mops_month(p)
        print(f"{p}: {0 if df is None else len(df)} 筆", flush=True)
        if df is not None:
            frames.append(df)
    all_ = pd.concat(frames, ignore_index=True)
    for c in ["rev", "rev_prev_month", "rev_last_year", "mom_pct", "yoy_pct", "cum_rev", "cum_rev_last_year", "cum_yoy_pct"]:
        if c in all_:
            all_[c] = pd.to_numeric(all_[c].astype(str).str.replace(",", ""), errors="coerce")
    return all_


# ---------- FinMind ----------
def fm_get(params):
    for _ in range(6):
        wait = GAP - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            r = requests.get(FM_URL, params={"dataset": "TaiwanStockMonthRevenue", **params},
                             headers={"Authorization": f"Bearer {FM_TOKEN}"}, timeout=90)
            j = r.json()
        except Exception as e:  # noqa: BLE001
            print(f"  連線失敗({e}),重試", flush=True); time.sleep(20); continue
        if r.status_code == 200 and j.get("status") == 200:
            return pd.DataFrame(j.get("data", []))
        msg = str(j.get("msg", ""))[:150]
        if r.status_code in (402, 429) or "limit" in msg.lower():
            print(f"  達到查詢上限,暫停 10 分鐘:{msg}", flush=True); time.sleep(600); continue
        print(f"  {params} 回應 {r.status_code}:{msg}", flush=True)
        return None
    return None


def fm_norm(df):
    df = df.rename(columns={"stock_id": "code", "revenue": "rev"})
    df["month"] = df["revenue_year"].astype(int).astype(str) + "-" + df["revenue_month"].astype(int).map("{:02d}".format)
    return df[["month", "code", "rev", "date"]].rename(columns={"date": "fm_date"})


def try_finmind():
    if not FM_TOKEN:
        sys.exit("MOPS 抓不到,也找不到 FINMIND_TOKEN。")
    print("== 改用 FinMind ==", flush=True)
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    test = fm_get({"start_date": "2024-02-01", "end_date": "2024-02-29"})
    if test is not None and test["stock_id"].nunique() > 500:
        print(f"  整個市場一次查可用({test['stock_id'].nunique()} 檔/月)", flush=True)
        frames = []
        for d in pd.date_range("2005-01-01", end, freq="MS"):
            df = fm_get({"start_date": d.strftime("%Y-%m-%d"), "end_date": (d + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")})
            if df is not None and len(df):
                frames.append(df)
            print(f"{d:%Y-%m}: {0 if df is None else len(df)} 筆", flush=True)
        return fm_norm(pd.concat(frames, ignore_index=True)), True
    print("  整個市場查不行,改逐檔", flush=True)
    part = OUT / "parts"
    part.mkdir(parents=True, exist_ok=True)
    codes = pd.read_csv(ROOT / "data" / "stock_list.csv", dtype=str).code.tolist()
    dl = ROOT / "out_delist_list.csv"
    if dl.exists():
        codes += pd.read_csv(dl, dtype=str).stock_id.tolist()
    todo = [c for c in dict.fromkeys(codes) if not (part / f"{c}.csv").exists()]
    print(f"  共 {len(codes)} 檔,尚未抓 {len(todo)} 檔", flush=True)
    for i, c in enumerate(todo, 1):
        if time.time() - T0 > BUDGET:
            print("  時間到,下次再按一次會接續", flush=True)
            break
        df = fm_get({"data_id": c, "start_date": "2004-12-01", "end_date": end})
        if df is not None:
            (fm_norm(df) if len(df) else pd.DataFrame(columns=["month", "code", "rev", "fm_date"])).to_csv(part / f"{c}.csv", index=False)
        if i % 100 == 0:
            print(f"  {i}/{len(todo)}", flush=True)
    files = list(part.glob("*.csv"))
    done = len(files) >= len(dict.fromkeys(codes))
    return pd.concat([pd.read_csv(f, dtype={"code": str}) for f in files], ignore_index=True), done


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    months = pd.period_range(START, (pd.Timestamp.today() - pd.DateOffset(months=1)).strftime("%Y-%m"), freq="M")
    df = None if (OUT / "parts").exists() else try_mops(months)
    if df is not None:
        src, done = "MOPS(單位千元)", True
    else:
        df, done = try_finmind()
        src = "FinMind TaiwanStockMonthRevenue(單位元)"
    df["code"] = df["code"].astype(str)
    df = df.drop_duplicates(["month", "code"], keep="last").sort_values(["month", "code"])
    df.to_csv(OUT / "revenue.csv", index=False, encoding="utf-8-sig")
    (OUT / "source.txt").write_text(f"{src}\n{'完成' if done else '未完成,請再執行一次'}\n", encoding="utf-8")
    print(f"{src}:{len(df)} 筆,{df.month.nunique()} 個月,{df.code.nunique()} 檔,{'完成' if done else '未完成'}", flush=True)


if __name__ == "__main__":
    main()
