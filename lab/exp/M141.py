# M-141：Rolling Candle Profile（Zeiierman）Price Pressure 模式移植，0050 週 K 訊號 → 加權指數後續報酬。定義見 registry_M.md M-141。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from chips import twdata_path
from M110 import wrsi, triggers

TICK = 0.01; BANDS = 12; ALEN = 21; PXW = 0.01; WICK = 0.85; DOM = 0.25; MINB = 7; GAP = 6


def weekly_0050():
    fs = sorted((twdata_path() / 'derived' / 'price_adj').glob('*.parquet'))
    d = pd.concat([pd.read_parquet(f, columns=['security_key', 'date', 'open', 'high', 'low', 'close', 'af', 'price_status']).query("security_key == '0050'") for f in fs])
    d = d[(d.close > 0) & (d.price_status == 'traded')].sort_values('date'); d['date'] = pd.to_datetime(d.date)
    for c in ('open', 'high', 'low', 'close'): d[c] = d[c] * d.af
    d = d.set_index('date'); g = d.groupby(d.index.to_period('W-FRI'))
    w = pd.DataFrame({'o': g.open.first(), 'h': g.high.max(), 'l': g.low.min(), 'c': g.close.last(), 'end': g.apply(lambda x: x.index.max())})
    return w.set_index('end')


def rcp(w):
    o, h, l, c = (w[k].to_numpy(float) for k in 'ohlc'); n = len(c)
    clamp = lambda x, a, b: np.minimum(np.maximum(x, a), b)
    r = np.maximum(h - l, TICK); pc = np.r_[o[0], c[:-1]]
    dirn = clamp((c - o) / r, -1, 1); travel = clamp((c - pc) / r, -1.5, 1.5); cl = clamp((2 * c - h - l) / r, -1, 1)
    resp = clamp(0.55 * travel + 0.25 * dirn + 0.20 * cl, -1.5, 1.5); force = np.abs(resp)
    prior = np.r_[0, 0, c[1:-1] - c[:-2]]
    cont = np.where(prior * resp > 0, 1.45, np.where(prior * resp < 0, 0.55, 1.0))
    buy = np.maximum(resp, 0) * (1 + force * 1.25) * cont * 2; sell = np.maximum(-resp, 0) * (1 + force * 1.25) * cont * 2
    lowW = np.maximum(0, np.minimum(o, c) - l) / r; highW = np.maximum(0, h - np.maximum(o, c)) / r
    lrej = lowW * (0.35 + 1.20 * np.maximum(resp, 0)); hrej = highW * (0.35 + 1.20 * np.maximum(-resp, 0))
    stall = (1 - clamp(force, 0, 1)) * 0.20
    hlc3 = (h + l + c) / 3
    bctr = np.where(buy > 0, 0.7 * c + 0.3 * h, hlc3); sctr = np.where(sell > 0, 0.7 * c + 0.3 * l, hlc3)
    lctr = np.where(lrej > 0, (l + np.minimum(o, c)) / 2, l); hctr = np.where(hrej > 0, (h + np.maximum(o, c)) / 2, h)
    total = buy + sell + lrej + hrej + stall
    base = pd.Series(total).ewm(span=55, adjust=False).mean().to_numpy()
    fs = np.where(total > 0, np.minimum(1.0, np.maximum(base, 1e-6) * 20 / total), 1.0)
    buy, sell, lrej, hrej, stall = buy * fs, sell * fs, lrej * fs, hrej * fs, stall * fs
    anchor = pd.Series(c).ewm(span=ALEN, adjust=False).mean().to_numpy()
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc))); atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()
    bl, bh = np.minimum(o, c), np.maximum(o, c)
    edge = np.where(anchor < bl, bl, np.where(anchor > bh, bh, np.nan))
    baseW = np.maximum(h - l, TICK) * PXW; baseWW = baseW * 0.6
    kern = lambda here, ctr, invw: (lambda q: np.where(q <= 0, 0.0, q * q * (3 - 2 * q)))(1 - np.abs(here - ctr) * invw)
    age = 0.5 ** (np.arange(60) / 3)
    buyAl = np.zeros(n, bool); sellAl = np.zeros(n, bool)
    for i in range(n):
        if i < max(ALEN + 2, 62) or not np.isfinite(edge[i]) or abs(edge[i] - anchor[i]) <= 2 * TICK: continue
        dist = (edge[i] - anchor[i]) / BANDS; ks = np.arange(i, i - 60, -1)
        invW = 1 / np.maximum(np.maximum(baseW[ks], abs(dist) * 1.15), TICK); invWW = 1 / np.maximum(np.maximum(baseWW[ks], abs(dist) * 1.00), TICK)
        pb = ps = 0.0; vis = bb = sb = 0
        for b in range(BANDS):
            here = anchor[i] + dist * (b + 0.5)
            bf = np.sum(buy[ks] * kern(here, bctr[ks], invW) * age) + WICK * np.sum(lrej[ks] * kern(here, lctr[ks], invWW) * age)
            sf = np.sum(sell[ks] * kern(here, sctr[ks], invW) * age) + WICK * np.sum(hrej[ks] * kern(here, hctr[ks], invWW) * age)
            tot = bf + sf; imb = np.clip((bf - sf) / tot * 3, -1, 1) if tot > 0 else 0.0
            act = np.clip(tot / (max(1e-6, base[i]) * 0.80), 0, 1)
            if act >= 0.08 and abs(imb) >= 0.14:
                pb += bf; ps += sf; vis += 1
                if imb > 0.03: bb += 1
                elif imb < -0.03: sb += 1
        ptot = pb + ps; pbias = (pb - ps) / ptot if ptot > 0 else 0.0
        cb, cs = buy[i] + WICK * lrej[i], sell[i] + WICK * hrej[i]; cbias = (cb - cs) / (cb + cs) if cb + cs > 0 else 0.0
        hasP = vis >= MINB and ptot > 0; ab = pbias if hasP else cbias
        bp = ab >= DOM and (bb > sb if hasP else (ptot <= 0 or pbias > -DOM)); sp = ab <= -DOM and (sb > bb if hasP else (ptot <= 0 or pbias < DOM))
        rise, fall = anchor[i] > anchor[i - 1], anchor[i] < anchor[i - 1]
        buyAl[i] = hasP and c[i] > anchor[i] and rise and bp; sellAl[i] = hasP and c[i] < anchor[i] and fall and sp
    return buyAl, sellAl, anchor


def first_with_gap(al):
    sig = np.zeros(len(al), bool); last = -10 ** 9
    for i in range(1, len(al)):
        if al[i] and not al[i - 1] and i - last >= GAP: sig[i] = True; last = i
    return sig


def main():
    w = weekly_0050(); bA, sA, anc = rcp(w)
    c = w.c.to_numpy(); rise = np.r_[False, anc[1:] > anc[:-1]]; fall = np.r_[False, anc[1:] < anc[:-1]]
    eB, eS = (c > anc) & rise, (c < anc) & fall
    sig = {'RCP 買': first_with_gap(bA), 'RCP 賣': first_with_gap(sA), 'EMA21 買（對照）': first_with_gap(eB), 'EMA21 賣（對照）': first_with_gap(eS)}
    P = load_panel(); ix = P['index']['TAIEX_P'].dropna()
    wk = ix.groupby(ix.index.to_period('W-FRI')).last(); wk.index = [ix.index[ix.index.to_period('W-FRI') == p].max() for p in wk.index]
    tx = wk.reindex(w.index, method='ffill').to_numpy(float)
    rB = wrsi(pd.Series(tx, index=w.index), 14); bt = set(triggers(rB.dropna(), 30, 40))
    sig['週RSI B 買（對照）'] = np.array([d in bt for d in w.index])
    n = len(tx); idx = w.index; start = 62
    def fwd(i, k, arr): return arr[i + k] / arr[i] - 1 if i + k < n else np.nan
    def mdd(i, k, arr): seg = arr[i:i + k + 1]; return (seg.min() / arr[i] - 1, seg.max() / arr[i] - 1) if i + k < n else (np.nan, np.nan)
    L = ['# M-141 結果：Rolling Candle Profile 在大盤週 K（0050 週 K 算訊號、加權指數衡量）\n',
         f"週 K {len(w)} 根（{idx[0].date()}–{idx[-1].date()}）；訊號自第 63 根起有效。報酬為加權價格指數（不含息）。\n"]
    eras = (('全期', '2003', '2026-12-31'), ('2003–14', '2003', '2014-12-31'), ('2015–26', '2015', '2026-12-31'))
    for era, lo, hi in eras:
        L += [f'## {era}\n', '| 組 | 次數 | 後4週 | 後13週 | 後26週 | 13週勝率 | 13週內最大跌 | 13週內最大漲 |', '|---|---|---|---|---|---|---|---|']
        msk = np.array([(pd.Timestamp(lo) <= d <= pd.Timestamp(hi)) for d in idx]) & (np.arange(n) >= start)
        groups = [('全部週', msk)] + [(k, v & msk) for k, v in sig.items()]
        for k, m in groups:
            ii = np.nonzero(m)[0]
            if len(ii) == 0: continue
            f4 = np.nanmean([fwd(i, 4, tx) for i in ii]); f13 = np.array([fwd(i, 13, tx) for i in ii]); f26 = np.nanmean([fwd(i, 26, tx) for i in ii])
            dd = np.array([mdd(i, 13, tx) for i in ii])
            L.append(f"| {k} | {len(ii)} | {f4:+.1%} | {np.nanmean(f13):+.1%} | {f26:+.1%} | {np.nanmean(f13[np.isfinite(f13)] > 0):.0%} | {np.nanmean(dd[:, 0]):.1%} | {np.nanmean(dd[:, 1]):+.1%} |")
        st = np.where(bA, '買方對齊', np.where(sA, '賣方對齊', '都不是'))
        L += ['', '| 狀態（每週） | 週數 | 後13週 | 13週內最大跌 |', '|---|---|---|---|']
        for s_ in ('買方對齊', '賣方對齊', '都不是'):
            ii = np.nonzero((st == s_) & msk)[0]
            f13 = np.array([fwd(i, 13, tx) for i in ii]); dd = np.array([mdd(i, 13, tx)[0] for i in ii])
            L.append(f"| {s_} | {len(ii)} | {np.nanmean(f13):+.1%} | {np.nanmean(dd):.1%} |")
        L.append('')
    L += ['## 訊號清單（RCP）\n'] + [f"- {k}：" + '、'.join(str(idx[i].date()) for i in np.nonzero(sig[k])[0] if i >= start) for k in ('RCP 買', 'RCP 賣')]
    return dict(summary='\n'.join(L), headline='RCP 大盤週 K 描述完成', n_variants=0)
