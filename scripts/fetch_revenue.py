"""下載上市櫃月營收(公開資訊觀測站 MOPS 彙總表),回測用。不影響網站。

不需要任何 token。來源同 FinLab 月營收教學:
  https://mopsov.twse.com.tw/nas/t21/{sii|otc}/t21sc03_{民國年}_{月}_{0|1}.html
  sii=上市、otc=上櫃;_0=國內公司、_1=國外(KY)公司;民國 98 年(含)以前沒有 _0/_1。

輸出(分支 revenue-data):
  revenue.csv  month,code,name,market,rev,rev_prev_month,rev_last_year,mom_pct,yoy_pct,
               cum_rev,cum_rev_last_year,cum_yoy_pct,note
  month = 營收所屬月份(YYYY-MM)。月營收須在次月 10 日前公布,回測請用次月 11 日起才可用。
  單位:千元
"""
import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out_revenue"
START = (2005, 1)
HOSTS = ["https://mopsov.twse.com.tw", "https://mops.twse.com.tw"]
HEAD = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}
COLS = {
    "公司代號": "code", "公司名稱": "name", "當月營收": "rev", "上月營收": "rev_prev_month",
    "去年當月營收": "rev_last_year", "上月比較增減(%)": "mom_pct", "去年同月增減(%)": "yoy_pct",
    "當月累計營收": "cum_rev", "去年累計營收": "cum_rev_last_year", "前期比較增減(%)": "cum_yoy_pct",
    "備註": "note",
}


def fetch(path):
    for host in HOSTS:
        for _ in range(3):
            try:
                r = requests.get(host + path, headers=HEAD, timeout=60)
            except Exception as e:  # noqa: BLE001
                print(f"  {host}{path} 連線失敗({e})", flush=True); time.sleep(5); continue
            if r.status_code == 404:
                break
            if r.status_code == 200 and len(r.content) > 2000:
                r.encoding = "big5"
                return r.text
            time.sleep(5)
    return None


def parse(html):
    out = []
    for t in pd.read_html(io.StringIO(html)):
        if isinstance(t.columns, pd.MultiIndex):
            t.columns = t.columns.get_level_values(-1)
        t.columns = [str(c).strip() for c in t.columns]
        if "公司代號" not in t.columns or "當月營收" not in t.columns:
            continue
        t = t.rename(columns=COLS)
        t = t[[c for c in COLS.values() if c in t.columns]]
        t = t[t["code"].astype(str).str.fullmatch(r"\d{4,6}")]
        out.append(t)
    return pd.concat(out) if out else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.today()
    months = pd.period_range(f"{START[0]}-{START[1]:02d}", (today - pd.DateOffset(months=1)).strftime("%Y-%m"), freq="M")
    frames, miss = [], []
    for p in months:
        roc = p.year - 1911
        got = 0
        for market in ("sii", "otc"):
            suffixes = ["_0", "_1"] if roc > 98 else [""]
            for s in suffixes:
                html = fetch(f"/nas/t21/{market}/t21sc03_{roc}_{p.month}{s}.html")
                if html is None:
                    continue
                df = parse(html)
                if df is None:
                    continue
                df.insert(0, "month", str(p))
                df.insert(3, "market", market)
                frames.append(df)
                got += len(df)
            time.sleep(1.5)
        print(f"{p}: {got} 筆", flush=True)
        if not got:
            miss.append(str(p))
    if not frames:
        sys.exit("一個月都抓不到,MOPS 可能擋了 GitHub 的 IP。")
    all_ = pd.concat(frames, ignore_index=True)
    for c in ["rev", "rev_prev_month", "rev_last_year", "mom_pct", "yoy_pct", "cum_rev", "cum_rev_last_year", "cum_yoy_pct"]:
        if c in all_:
            all_[c] = pd.to_numeric(all_[c].astype(str).str.replace(",", ""), errors="coerce")
    all_["code"] = all_["code"].astype(str)
    all_ = all_.drop_duplicates(["month", "code"], keep="last").sort_values(["month", "code"])
    all_.to_csv(OUT / "revenue.csv", index=False, encoding="utf-8-sig")
    (OUT / "missing_months.txt").write_text("\n".join(miss), encoding="utf-8")
    print(f"完成:{len(all_)} 筆,{all_.month.nunique()} 個月,{all_.code.nunique()} 檔;缺 {len(miss)} 個月", flush=True)


if __name__ == "__main__":
    main()
