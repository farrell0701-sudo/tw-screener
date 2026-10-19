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
WEEKS = 12   # 網站只保留最新 12 週集保資料
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
        p100 = df[df.level.between(10, 15)].groupby("code").pct.sum()   # 100,001 股以上 = 超過 100 張
        tot = df[df.level == 17].set_index("code")
        for code in g.groups:
            r = recs.setdefault(code, {"p100": [], "p400": [], "p1000": [], "holders": [], "shares": None})
            r["p100"].append(round(float(p100.get(code, np.nan)), 2))
            r["p400"].append(round(float(p400.get(code, np.nan)), 2))
            r["p1000"].append(round(float(p1000.get(code, np.nan)), 2))
            r["holders"].append(int(tot.holders.get(code, 0)) if code in tot.index else None)
            if code in tot.index:
                r["shares"] = float(tot.shares.get(code))
    n = len(weeks)
    for r in recs.values():  # 補齊長度(新上市股票)
        for k in ("p100", "p400", "p1000", "holders"):
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


BULL = {}    # 日期 -> 大盤 EMA60 是否上彎(EMA60 今 ÷ 20 日前 − 1 > 0,v3c 的大盤條件)
ABOVE = {}   # 日期 -> 加權收盤是否 > EMA60(舊版 v3 另外要求,網頁可勾選)


def load_bull():
    f = DATA / "prices" / "TAIEX.csv"
    if not f.exists():
        return {}, {}
    d = pd.read_csv(f, index_col=0, parse_dates=True).dropna(subset=["Close"])
    c = d["Close"].values.astype(float)
    e = ema(c, 60)
    s = np.full(len(e), np.nan)
    s[20:] = e[20:] / e[:-20] - 1
    idx = [(i, dt) for i, dt in enumerate(d.index) if i >= 80]
    return {dt: bool(s[i] > 0) for i, dt in idx}, {dt: bool(c[i] > e[i]) for i, dt in idx}


SG_W = SIGNAL_LOOKBACK + 1   # 輸出最近幾根的條件數值,讓網頁端依開關即時計算


def ac_parts(o, h, l, c, v, dates, rc):
    """Accum V3C-PRO(囤貨後被發現)最近 SG_W 根的各條件與評分數值。
    網頁端依使用者的勾選與門檻組合出訊號與分數,所以這裡只輸出原始數值。
    o/h/l/c 是還原價(技術指標、停損用),rc 是實際收盤(股價、成交值門檻用)。"""
    n = len(c)
    cs, hs, ls, vs = pd.Series(c), pd.Series(h), pd.Series(l), pd.Series(v)
    e200 = ema(c, 200)
    s200 = np.full(n, np.nan)
    s200[10:] = (e200[10:] / e200[:-10] - 1) * 10                    # EMA200 十日斜率 %/日
    e150 = ema(c, 150)
    s150 = np.full(n, np.nan)
    s150[10:] = (e150[10:] / e150[:-10] - 1) * 10                    # 模型特徵仍用 EMA150(訓練時的定義)
    m20 = cs.rolling(20).mean()
    bbu = (m20 + 2 * cs.rolling(20).std(ddof=0)).values
    ma5 = cs.rolling(5).mean().values
    bbg = np.abs(bbu - ma5) / c * 100
    bbs = np.full(n, np.nan)
    bbs[5:] = (bbu[5:] / bbu[:-5] - 1) * 100 / 5                      # 上軌 5 日斜率 %/日
    vr = (vs / vs.rolling(20).mean()).values
    spk = pd.Series(vr).rolling(5).max().shift(1).values              # 前 5 日(不含今日)最大量比
    pre = ((cs.shift(1) / cs.shift(6) - 1) * 100).values              # 爆量組:爆量前 5 日漲幅
    n5 = ((cs / cs.shift(5) - 1) * 100).values                        # 非爆量組:近 5 日漲幅(含今日)
    bl = np.array([BULL.get(d, False) for d in dates]) if BULL else np.ones(n, bool)
    ba = np.array([ABOVE.get(d, False) for d in dates]) if ABOVE else np.ones(n, bool)
    pc = np.r_[np.nan, c[:-1]]
    atr = pd.Series(true_range(h, l, c)).ewm(alpha=1 / 14, adjust=False).mean().values
    sl = slice(max(0, n - SG_W), n)
    f = lambda a, d=2: [None if not np.isfinite(x) else round(float(x), d) for x in a[sl]]
    b = lambda a: [1 if x else 0 for x in a[sl]]
    out = {"s200": f(s200, 4), "bbg": f(bbg), "bbs": f(bbs, 3), "spk": f(spk), "vr": f(vr),
           "pre": f(pre, 1), "n5": f(n5, 1), "bl": b(bl), "ba": b(ba), "c": f(c, 4), "rc": f(rc, 2)}
    for N in (5, 10, 20):
        out[f"nh{N}"] = b(c > cs.rolling(N).max().shift(1).values)
    # 評分用(ATR%、距 250 日高;成交值 tv 在下方、股價用實際收盤 rc)。小數位要夠,門檻邊界才不會被四捨五入改變
    out["atr"] = f(atr / c * 100, 4)
    out["ddh"] = f((1 - c / hs.rolling(250).max().values) * 100, 4)   # 距 250 日高點(正數=低於高點的 %)
    out["rng"] = f((hs.rolling(20).max().values - ls.rolling(20).min().values) / c * 100, 2)
    out["d60"] = f((c / ema(c, 60) - 1) * 100, 2)
    out["r1"] = f((c / pc - 1) * 100, 4)
    tv = (vs.rolling(20).mean() * cs).values                        # 20 日均成交值(元,還原價,模型特徵用)
    # 模型特徵(不輸出到網頁,只在 Python 端預測用)
    sl_arr = lambda a: np.asarray(a, float)[sl]
    out["_x"] = {"atrp": sl_arr(atr / c * 100), "dd250": sl_arr((c / hs.rolling(250).max().values - 1) * 100),
                 "rng20": sl_arr((hs.rolling(20).max().values - ls.rolling(20).min().values) / c * 100),
                 "d60": sl_arr((c / ema(c, 60) - 1) * 100), "ret1": sl_arr((c / pc - 1) * 100), "price": sl_arr(c),
                 "tv": sl_arr(tv), "vr": sl_arr(vr), "ret5": sl_arr(n5), "ret20": sl_arr((cs / cs.shift(20) - 1).values * 100),
                 "s150": sl_arr(s150)}
    return out


# ───────────────────────── 機器學習排序 ─────────────────────────
# 用 2016~2021 年 Accum-v3 樣本外交易(scripts/model_train.csv,6,621 筆)訓練,預測單筆報酬。
# 同一天訊號多時,網頁依「模型預測」由高到低排序。回測:2022~2026 年化約 37%(分數優先約 26%)。
ML_FEATS = ["atrp", "dd250", "rng20", "d60", "ret1", "cap", "price", "tv", "vr", "ret5", "ret20", "s150"]


def train_model():
    f = ROOT / "scripts" / "model_train.csv"
    if not f.exists():
        return None
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError:
        print("[build] 沒有安裝 scikit-learn,略過模型預測", file=sys.stderr)
        return None
    d = pd.read_csv(f)
    m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=80, random_state=0)
    return m.fit(d[ML_FEATS], d["r"].clip(-0.3, 1.0))


def add_ml(rows, model):
    X, owner = [], []
    for r in rows:
        x = r.get("ac", {}).pop("_x", None)
        if x is None or model is None:
            continue
        n = len(x["atrp"])
        for i in range(n):
            X.append([x[k][i] if k != "cap" else (r.get("cap") if r.get("cap") is not None else np.nan) for k in ML_FEATS])
        owner.append((r, n))
    if not X:
        return
    pred = model.predict(np.array(X, float)) * 100
    j = 0
    for r, n in owner:
        r["ac"]["ml"] = [round(float(v), 1) for v in pred[j:j + n]]
        j += n


# ── 機器學習排序模型(2016~2021 樣本外交易訓練,預測單筆報酬;樹狀結構存在 scripts/ml_model.json)
_MP = Path(__file__).resolve().parent / "ml_model.json"
MODEL = json.loads(_MP.read_text(encoding="utf-8")) if _MP.exists() else None
TREES = None
if MODEL:
    TREES = [tuple(np.array(col, dtype=(float if j in (1, 6) else int)) for j, col in enumerate(zip(*t)))
             for t in MODEL["trees"]]


def ml_predict(X):
    """X: (N, 12) 特徵矩陣,欄位順序同 MODEL['feats'];回傳預測報酬(小數)。"""
    out = np.full(len(X), MODEL["base"])
    rows = np.arange(len(X))
    for f, th, mg, lft, rgt, leaf, val in TREES:
        i = np.zeros(len(X), int)
        for _ in range(20):
            go = leaf[i] == 0
            if not go.any():
                break
            v = X[rows, f[i]]
            left = np.where(np.isnan(v), mg[i] == 1, v <= th[i])
            i = np.where(go, np.where(left, lft[i], rgt[i]), i)
        out += val[i]
    return out


def ml_feats(h, l, c, v):
    """最近 SG_W 根的模型特徵(股本先填 NaN,之後在 main 補上)。"""
    cs, hs, ls, vs = pd.Series(c), pd.Series(h), pd.Series(l), pd.Series(v)
    atr = pd.Series(true_range(h, l, c)).ewm(alpha=1 / 14, adjust=False).mean().values
    vma = vs.rolling(20).mean().values
    e150 = ema(c, 150)
    s150 = np.full(len(c), np.nan)
    s150[10:] = (e150[10:] / e150[:-10] - 1) * 10
    pc = np.r_[np.nan, c[:-1]]
    F = {"atrp": atr / c * 100, "dd250": (c / hs.rolling(250).max().values - 1) * 100,
         "rng20": (hs.rolling(20).max().values - ls.rolling(20).min().values) / c * 100,
         "d60": (c / ema(c, 60) - 1) * 100, "ret1": (c / pc - 1) * 100, "cap": np.full(len(c), np.nan),
         "price": c, "tv": vma * c, "vr": v / vma, "ret5": (cs / cs.shift(5) - 1).values * 100,
         "ret20": (cs / cs.shift(20) - 1).values * 100, "s150": s150}
    sl = slice(max(0, len(c) - SG_W), len(c))
    return np.column_stack([F[k][sl] for k in MODEL["feats"]]).astype(float)


AB_P = dict(lenH=60, minB=5, awin=30, minAGain=25, minALen=2, hiLen=20,
            slopeLen=150, slopeLb=10, slopeMin=0.2, clMin=0.75, vcN=5, vcM=60, vcK=1.0)
VCP_P = dict(push=20)


def days_since(sig):
    idx = np.where(sig)[0]
    return int(len(sig) - 1 - idx[-1]) if len(idx) else None


def sig_count(sig):
    """訊號出現次數(ab/vcp 只檢查最近 SIGNAL_LOOKBACK 根左右,範圍同 days_since)。"""
    return int(sig.sum())


def tech(df):
    o, h, l, c = (df[k].values.astype(float) for k in ("Open", "High", "Low", "Close"))
    rc = c.copy()                       # 實際收盤
    if "AdjClose" in df.columns:        # 新格式:OHLC 是實際價,依 AdjClose ÷ Close 換成還原價
        k = df["AdjClose"].values.astype(float) / c
        o, h, l, c = o * k, h * k, l * k, df["AdjClose"].values.astype(float)
    v = df["Volume"].fillna(0).values.astype(float)
    n = len(c)
    if n < 160:
        return None
    tr = true_range(h, l, c)
    e20, e60, e200 = ema(c, 20), ema(c, 60), ema(c, 200)
    atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().values
    out = {
        "date": df.index[-1].strftime("%Y-%m-%d"),
        "close": round(rc[-1], 2),   # 實際收盤
        "chg": round((c[-1] / c[-2] - 1) * 100, 2),
        "vol": int(v[-1] / 1000),  # 張
        "volR": round(v[-1] / v[-21:-1].mean(), 2) if v[-21:-1].mean() > 0 else None,
        "aboveE20": bool(c[-1] > e20[-1]),
        "aboveE60": bool(c[-1] > e60[-1]),
        "slope200": round((e200[-1] / e200[-11] - 1) * 10, 3),
        "atrp": round(atr[-1] / c[-1] * 100, 2),
        "vc": round(tr[-5:].mean() / tr[-65:-5].mean(), 2),
        "res120": round((h[-121:-1].max() / c[-1] - 1) * 100, 1),
        "ret20": round((c[-1] / c[-21] - 1) * 100, 1),
    }
    ab, vcp = ab_signals(o, h, l, c, AB_P), vcp_signals(o, h, l, c, v, VCP_P)
    out.update({
        "ab": days_since(ab), "abN": sig_count(ab),
        "vcp": days_since(vcp), "vcpN": sig_count(vcp),
        "spark": [round(x, 2) for x in c[-60:]],
    })
    out["ac"] = ac_parts(o, h, l, c, v, df.index, rc)
    tvs = pd.Series(v).rolling(20).mean().values * rc / 1e8      # 20 日均量 × 實際收盤(億元),V3C-PRO 的成交值門檻
    out["ac"]["tv"] = [None if not np.isfinite(x) else round(float(x), 6) for x in tvs[max(0, n - SG_W):]]
    if MODEL:
        out["_mlX"] = ml_feats(h, l, c, v)
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
    global BULL, ABOVE
    BULL, ABOVE = load_bull()
    lst = pd.read_csv(DATA / "stock_list.csv", dtype=str).set_index("code")
    weeks, hold = load_tdcc()
    rows = []
    for code, info in lst.iterrows():
        f = DATA / "prices" / f"{code}.csv"
        t = None
        if f.exists():
            try:
                d = pd.read_csv(f, index_col=0, parse_dates=True).dropna(subset=["Close"])
                t = tech(d[(d["Volume"] > 0) & (d["Close"] > 0)])   # 同回測:略過沒有成交的日子
            except Exception as e:  # noqa: BLE001
                print(f"[build] {code} 技術面失敗: {e}", file=sys.stderr)
        hrec = hold.get(code)
        if t is None and hrec is None:
            continue
        r = {"code": code, "name": info["name"], "market": info["market"], "industry": info["industry"]}
        if hrec:
            r.update(p100=hrec["p100"], p400=hrec["p400"], p1000=hrec["p1000"], holders=hrec["holders"])
            if hrec["shares"]:
                r["cap"] = round(hrec["shares"] * 10 / 1e8, 1)   # 股本(億元,面額 10 元)
            if t and hrec["shares"]:
                r["mcap"] = round(t["close"] * hrec["shares"] / 1e8, 1)  # 億元
        if t:
            r.update(t)
        rows.append(r)
    if MODEL:   # 一次算完全部股票、最近 21 根的模型分數(預測報酬 %)
        mats, owners = [], []
        for r in rows:
            X = r.pop("_mlX", None)
            if X is None:
                continue
            X[:, MODEL["feats"].index("cap")] = r.get("cap", np.nan) if r.get("cap") is not None else np.nan
            mats.append(X)
            owners.append(r)
        if mats:
            P = ml_predict(np.vstack(mats)) * 100
            k = 0
            for r, X in zip(owners, mats):
                r["ac"]["ml"] = [round(float(x), 2) for x in P[k:k + len(X)]]
                k += len(X)
    model = train_model()
    add_ml(rows, model)
    for r in rows:
        if "ac" in r:
            r["ac"].pop("_x", None)
    caps = sorted([r for r in rows if r.get("mcap")], key=lambda r: -r["mcap"])
    for i, r in enumerate(caps, 1):
        r["mcapRank"] = i
    tz = timezone(timedelta(hours=8))
    payload = {"updated": datetime.now(tz).strftime("%Y-%m-%d %H:%M"), "weeks": weeks,
               "abParams": AB_P, "vcpParams": VCP_P,
               "bull": (BULL[max(BULL)] if BULL else None), "bullDate": (max(BULL).strftime("%Y-%m-%d") if BULL else None), "stocks": rows}
    import os
    if os.environ.get("DEMO_NOTE"):
        payload["demo"] = os.environ["DEMO_NOTE"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(clean(payload), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"[build] {len(rows)} 檔, {len(weeks)} 週集保 → {OUT}")


if __name__ == "__main__":
    main()
