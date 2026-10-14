"""整合集保股權分散 + 日K,計算籌碼指標與技術型態,輸出 site/data/screen.json。"""
import json
import math
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "site" / "data" / "screen.json"
WEEKS = 13
SIGNAL_LOOKBACK = 20  # 回報最近幾根內的訊號


# ───────────────────────── 籌碼 ─────────────────────────
def load_tdcc():
    files = sorted((DATA / "tdcc").glob("*.csv"))[-WEEKS:]
    weeks, recs = [], {}
    for f in files:
        d = f.stem
        weeks.append(f"{d[:4]}-{d[4:6]}-{d[6:]}")
        df = pd.read_csv(f, dtype={"code": str})
        df["level"] = pd.to_numeric(df["level"], errors="coerce")
        df["pct"] = pd.to_numeric(df["pct"], errors="coerce")
        df["holders"] = pd.to_numeric(df["holders"], errors="coerce")
        df["shares"] = pd.to_numeric(df["shares"], errors="coerce")
        g = df.groupby("code")
        p400 = df[df.level.between(12, 15)].groupby("code").pct.sum()
        p1000 = df[df.level == 15].groupby("code").pct.sum()
        tot = df[df.level == 17].set_index("code")
        for code in g.groups:
            r = recs.setdefault(code, {"p400": [], "p1000": [], "holders": [], "shares": None})
            r["p400"].append(round(float(p400.get(code, np.nan)), 2))
            r["p1000"].append(round(float(p1000.get(code, np.nan)), 2))
            r["holders"].append(int(tot.holders.get(code, 0)) if code in tot.index else None)
            if code in tot.index:
                r["shares"] = float(tot.shares.get(code))
    n = len(weeks)
    for r in recs.values():  # 補齊長度(新上市股票)
        for k in ("p400", "p1000", "holders"):
            r[k] = [None] * (n - len(r[k])) + r[k]
    return weeks, recs


# ───────────────────────── 技術面 ─────────────────────────
def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().values


def true_range(h, l, c):
    pc = np.r_[c[0], c[:-1]]
    return np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])


def ab_signals(o, h, l, c, P):
    """A→B 突破 v2:回傳每根是否為訊號(布林陣列)。"""
    n = len(c)
    bt = np.maximum(o, c)
    tr = true_range(h, l, c)
    tr_s = pd.Series(tr)
    vc = (tr_s.rolling(P["vcN"]).mean().shift(1) / tr_s.rolling(P["vcM"]).mean().shift(1)).values
    e150 = ema(c, P["slopeLen"])
    slope = np.full(n, np.nan)
    lb = P["slopeLb"]
    slope[lb:] = (e150[lb:] / e150[:-lb] - 1) * 100 / lb
    hiPrev = pd.Series(bt).rolling(P["hiLen"]).max().shift(1).values
    sig = np.zeros(n, bool)
    start = max(P["lenH"], P["awin"]) + 2
    for t in range(max(start, n - SIGNAL_LOOKBACK - 1), n):
        w = bt[t - P["lenH"]:t]
        H = w.max()
        if not c[t] > H:
            continue
        idxH = t - P["lenH"] + len(w) - 1 - int(np.argmax(w[::-1]))
        bn = t - 1 - idxH
        if bn < P["minB"]:
            continue
        lo0 = max(0, idxH - P["awin"])
        seg = c[lo0:idxH + 1]
        idxL = lo0 + len(seg) - 1 - int(np.argmin(seg[::-1]))
        if idxH - idxL < P["minALen"] or (H / seg.min() - 1) * 100 < P["minAGain"]:
            continue
        if not c[t] > hiPrev[t]:
            continue
        rg = h[t] - l[t]
        clpos = (c[t] - l[t]) / rg if rg > 0 else 0.5
        if not (slope[t] >= P["slopeMin"] and clpos >= P["clMin"] and vc[t] <= P["vcK"]):
            continue
        sig[t] = True
    return sig


def vcp_signals(o, h, l, c, v, P):
    """放量推升 → 量縮收斂 → 突破(推升 ≥ push%)。"""
    n = len(c)
    bt, bb = np.maximum(o, c), np.minimum(o, c)
    tr = true_range(h, l, c)
    a60 = pd.Series(tr).rolling(60).mean().shift(1).values
    v60 = pd.Series(v).rolling(60).mean().shift(1).values
    cmax = pd.Series(c).rolling(10).max().shift(1).values
    up = np.r_[False, c[1:] > c[:-1]]
    surge = up & (c >= cmax) & ((tr >= 2 * a60) | (v >= 2.5 * v60))
    atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().values
    sig = np.zeros(n, bool)
    used = set()
    for t in range(max(90, n - SIGNAL_LOOKBACK - 1), n):
        js = [i for i in range(t - 4, t - 31, -1) if i > 70 and surge[i]]
        if not js or js[0] in used:
            continue
        j = js[0]
        seg = bt[j:min(j + 5, t - 4) + 1]
        s = j + len(seg) - 1 - int(np.argmax(seg[::-1]))
        Lb = t - s
        if not 4 <= Lb <= 25:
            continue
        top, bot = bt[s:t].max(), bb[s:t].min()
        if not c[t] > top or (Lb > 1 and c[t - 1] > bt[s:t - 1].max()):
            continue
        if (c[s] / l[j - 10:j + 1].min() - 1) * 100 < P["push"]:
            continue
        k = min(5, Lb - 1)
        if tr[t - k:t].mean() / tr[max(j - 2, 0):j + 1].mean() > 0.6:
            continue
        if (top - bot) / atr[t - 1] > 3.5:
            continue
        used.add(j)
        sig[t] = True
    return sig


AB_P = dict(lenH=60, minB=5, awin=30, minAGain=25, minALen=2, hiLen=20,
            slopeLen=150, slopeLb=10, slopeMin=0.2, clMin=0.75, vcN=5, vcM=60, vcK=1.0)
VCP_P = dict(push=20)


def days_since(sig):
    idx = np.where(sig)[0]
    return int(len(sig) - 1 - idx[-1]) if len(idx) else None


def tech(df):
    o, h, l, c = (df[k].values.astype(float) for k in ("Open", "High", "Low", "Close"))
    v = df["Volume"].fillna(0).values.astype(float)
    n = len(c)
    if n < 160:
        return None
    tr = true_range(h, l, c)
    e20, e60, e150 = ema(c, 20), ema(c, 60), ema(c, 150)
    atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().values
    out = {
        "date": df.index[-1].strftime("%Y-%m-%d"),
        "close": round(c[-1], 2),
        "chg": round((c[-1] / c[-2] - 1) * 100, 2),
        "vol": int(v[-1] / 1000),  # 張
        "volR": round(v[-1] / v[-21:-1].mean(), 2) if v[-21:-1].mean() > 0 else None,
        "aboveE20": bool(c[-1] > e20[-1]),
        "aboveE60": bool(c[-1] > e60[-1]),
        "slope150": round((e150[-1] / e150[-11] - 1) * 10, 3),
        "atrp": round(atr[-1] / c[-1] * 100, 2),
        "vc": round(tr[-5:].mean() / tr[-65:-5].mean(), 2),
        "res120": round((h[-121:-1].max() / c[-1] - 1) * 100, 1),
        "ret20": round((c[-1] / c[-21] - 1) * 100, 1),
        "ab": days_since(ab_signals(o, h, l, c, AB_P)),
        "vcp": days_since(vcp_signals(o, h, l, c, v, VCP_P)),
        "spark": [round(x, 2) for x in c[-60:]],
    }
    return out


def clean(x):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, list):
        return [clean(y) for y in x]
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    return x


def main():
    lst = pd.read_csv(DATA / "stock_list.csv", dtype=str).set_index("code")
    weeks, hold = load_tdcc()
    rows = []
    for code, info in lst.iterrows():
        f = DATA / "prices" / f"{code}.csv"
        t = None
        if f.exists():
            try:
                t = tech(pd.read_csv(f, index_col=0, parse_dates=True).dropna(subset=["Close"]))
            except Exception as e:  # noqa: BLE001
                print(f"[build] {code} 技術面失敗: {e}", file=sys.stderr)
        hrec = hold.get(code)
        if t is None and hrec is None:
            continue
        r = {"code": code, "name": info["name"], "market": info["market"], "industry": info["industry"]}
        if hrec:
            r.update(p400=hrec["p400"], p1000=hrec["p1000"], holders=hrec["holders"])
            if t and hrec["shares"]:
                r["mcap"] = round(t["close"] * hrec["shares"] / 1e8, 1)  # 億元
        if t:
            r.update(t)
        rows.append(r)
    caps = sorted([r for r in rows if r.get("mcap")], key=lambda r: -r["mcap"])
    for i, r in enumerate(caps, 1):
        r["mcapRank"] = i
    tz = timezone(timedelta(hours=8))
    payload = {"updated": datetime.now(tz).strftime("%Y-%m-%d %H:%M"), "weeks": weeks,
               "abParams": AB_P, "vcpParams": VCP_P, "stocks": rows}
    import os
    if os.environ.get("DEMO_NOTE"):
        payload["demo"] = os.environ["DEMO_NOTE"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(clean(payload), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"[build] {len(rows)} 檔, {len(weeks)} 週集保 → {OUT}")


if __name__ == "__main__":
    main()
