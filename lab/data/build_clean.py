"""用 FinMind 未還原日線 + 除權息/減資/面額變更事件，自算 2004~2026 還原價。
輸入資料夾: old/(2004~2016 未還原, 含下市) raw/(2015~ 現存股未還原 + ev_*) dl/(2015~ 下市股未還原) oldev/(2004~2014 ev_*)
輸出: out_clean/<代號>.csv  欄位 date,o,h,l,c,v,tv,af   (o~c 為未還原價, af=還原因子, 還原價 = 價格 × af)
"""
import glob, os, sys
import numpy as np, pandas as pd

B = sys.argv[1]; OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
COLS = ["date", "o", "h", "l", "c", "v", "tv"]

def rd(p):
    if not os.path.exists(p): return None
    d = pd.read_csv(p, parse_dates=["date"])
    return d if len(d) else None

def events(code):
    out = []
    for p in [f"{B}/oldev/ev_{code}.csv", f"{B}/raw/ev_{code}.csv"]:
        if not os.path.exists(p): continue
        e = pd.read_csv(p)
        if "stock_id" not in e or not len(e): continue
        for c in ["reference_price", "PostReductionReferencePrice", "after_price"]:
            if c not in e: e[c] = np.nan
        # 用交易所實際參考價(reference_price);現增除權若認購價高於市價,參考價不調整,after_price 只是理論值
        e["after"] = pd.to_numeric(e.reference_price, errors="coerce") \
            .fillna(pd.to_numeric(e.PostReductionReferencePrice, errors="coerce")) \
            .fillna(pd.to_numeric(e.after_price, errors="coerce"))
        e["date"] = pd.to_datetime(e.date)
        out.append(e[["date", "after", "dataset"]].dropna(subset=["after"]))
    if not out: return pd.DataFrame(columns=["date", "after", "dataset"])
    e = pd.concat(out)
    return e[e.after > 0].sort_values("date")

codes = sorted({os.path.basename(p)[:-4] for d in ["old", "raw", "dl"] for p in glob.glob(f"{B}/{d}/[0-9]*.csv")})
log = []; fb = []
for code in codes:
    old, new = rd(f"{B}/old/{code}.csv"), rd(f"{B}/raw/{code}.csv")
    if new is None: new = rd(f"{B}/dl/{code}.csv")
    parts = []
    if old is not None:
        parts.append(old[old.date < new.date.min()] if new is not None else old)
    if new is not None: parts.append(new)
    if not parts: continue
    p = pd.concat(parts)[COLS].drop_duplicates("date", keep="last").sort_values("date")
    p = p[(p.c > 0) & (p.v > 0)].dropna(subset=["o", "h", "l", "c"]).reset_index(drop=True)
    if len(p) == 0: continue
    for col in ["o", "h", "l"]:                      # 少數興櫃列開高低為 0,以收盤補
        p.loc[p[col] <= 0, col] = p.c
    af = np.ones(len(p)); dates = p.date.values; c = p.c.values
    nev = 0
    for _, r in events(code).iterrows():
        i = np.searchsorted(dates, np.datetime64(r.date))     # 第一個 >= 事件日的位置
        if i == 0 or i > len(p): continue
        f = r.after / c[i - 1]
        if not (0.02 < f < 50): log.append((code, r.date, f, "skip")); continue
        if abs(f - 1) < 1e-9: continue
        af[:i] *= f; nev += 1
    # 補漏:2015 前 FinMind 漏記的除權/減資。開盤就跳空超過當時漲跌停(上市櫃股不可能),且前後 5 個交易日內沒有事件,
    # 視為漏記事件,以 開盤/前收 當比例往前還原(同舊 load_old 做法),並記錄在 _fallback.csv
    evd = set()
    for d0 in events(code).date.values:
        j = np.searchsorted(dates, np.datetime64(d0)); evd |= set(range(j - 5, j + 6))
    o = p.o.values
    # 興櫃(無漲跌幅)期間不補漏:FinMind 興櫃價常不在升降單位上,用近 20 日「四價都在升降單位上」比例 ≥95% 判定為上市櫃
    def tick(x): return np.select([x < 10, x < 50, x < 100, x < 500, x < 1000], [0.01, 0.05, 0.1, 0.5, 1], 5)
    def ongrid(x): t = tick(x); return np.abs(x / t - np.round(x / t)) < 1e-6
    g = ongrid(p.o.values) & ongrid(p.h.values) & ongrid(p.l.values) & ongrid(c)
    listed = pd.Series(g.astype(float)).rolling(20, min_periods=20).mean().values >= 0.95
    # 另外要求前 60 個交易日內沒有任何一天漲幅超過漲跌停(興櫃常見;上市櫃不可能)
    up = pd.Series(np.r_[0, c[1:] / c[:-1] - 1] > 0.075).rolling(60, min_periods=1).max().values > 0
    listed &= ~up
    for i in range(1, len(p)):
        if dates[i] >= np.datetime64("2015-01-01"): break
        lim = 0.075
        if 0.3 < o[i] / c[i - 1] < 1 - lim and i not in evd and (dates[i] - dates[i - 1]) <= np.timedelta64(10, "D") and i >= 20 and listed[i - 1]:
            f = o[i] / c[i - 1]
            af[:i] *= f; fb.append((code, str(dates[i])[:10], round(f, 4)))
    p["af"] = af
    p.to_csv(f"{OUT}/{code}.csv", index=False, float_format="%.10g")
pd.DataFrame(log, columns=["code", "date", "f", "note"]).to_csv(f"{OUT}/_skipped_events.csv", index=False)
pd.DataFrame(fb, columns=["code", "date", "f"]).to_csv(f"{OUT}/_fallback.csv", index=False)
print(len(codes), "檔完成; 異常事件略過", len(log), "; 補漏", len(fb))
