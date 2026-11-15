import sys, numpy as np, pandas as pd, warnings, gc; warnings.filterwarnings('ignore')
exec(open('/home/claude/llab/l032.py').read().split("VARS = {")[0])
from numpy.lib.stride_tricks import sliding_window_view as swv
Hv, Lv, Cv = H0.values.astype(np.float64), L0.values.astype(np.float64), C0.values.astype(np.float64); T_, N_ = Cv.shape
Vv = D['V'].reindex(index=dates, columns=C0.columns).values.astype(np.float64); ATR = F['atr'].values
ret1 = Cv[1:] / Cv[:-1] - 1; ret1[~U[:-1]] = np.nan; IDX = np.concatenate([[1.0], np.cumprod(1 + np.nan_to_num(np.nanmean(ret1, 1)))]); del ret1
lo_, hi_ = ('2005-01-01', '2015-12-31') if which == 'cpold' else ('2016-01-01', '2026-12-31')
tm = (dates >= pd.Timestamp(lo_)) & (dates <= pd.Timestamp(hi_)); tm[:300] = False; tm[T_ - 64:] = False
PAT = ['ENG', 'GAP', 'SPR', 'VREC', 'HAM']
def outcome(t, j, SL):
    e = t + 1; px = Ov[e, j]
    if not np.isfinite(px): return None
    c = Cv[e:e + 62, j]; o = Ov[e:e + 63, j]; out = {}
    ap = ATR[t, j] / Cv[t, j] * 4; d = 0.04 if not np.isfinite(ap) else min(max(ap, 0.04), 0.15)
    def xit(k):
        k = min(k, 60); p = o[k + 1] if np.isfinite(o[k + 1]) else c[k]
        return p / px - 1, IDX[e + k] / IDX[t] - 1
    stop = None; trl = None; peak = -np.inf
    for k in range(61):
        ck = c[k]
        if not np.isfinite(ck): continue
        peak = max(peak, ck)
        if stop is None and SL is not None and ck < SL: stop = k
        if trl is None and (ck < peak * (1 - d) or (peak >= px * 1.06 and ck < px * 1.005)): trl = k
    for h in (20, 40, 60):
        k = h - 1; ce = c[k] if np.isfinite(c[k]) else np.nan
        out[f'F{h}'] = (ce / px - 1) - (IDX[e + k] / IDX[t] - 1)
        if SL is not None:
            if stop is not None and stop < k: r, m = xit(stop); out[f'F{h}S'] = r - m
            else: out[f'F{h}S'] = out[f'F{h}']
    kt = 59 if trl is None else trl; r, m = xit(kt); out['TR'] = r - m; out['TRd'] = kt + 1
    if SL is not None:
        ks = kt if stop is None else min(kt, stop); r, m = xit(ks); out['TRS'] = r - m; out['stopped'] = stop is not None and stop <= 59
    return out
rows = []; base = []; rng = np.random.default_rng(0)
poolw = pd.DataFrame(pool).rolling(10, min_periods=1).max().values > 0
for j in range(N_):
    if np.isfinite(Cv[:, j]).sum() < 320: continue
    H, L, C, O, V = (pd.Series(x[:, j]) for x in (Hv, Lv, Cv, Ov, Vv)); a = ATR[:, j]
    BH = H.rolling(20, min_periods=15).max().shift(4).values; BL = L.rolling(20, min_periods=15).min().shift(4).values
    TL = L.rolling(4, min_periods=1).min().values
    h = np.where(np.isfinite(Hv[:, j]), Hv[:, j], -np.inf); l = np.where(np.isfinite(Lv[:, j]), Lv[:, j], np.inf)
    pw = swv(h, 60).argmax(1); tt = np.arange(59, T_); p = tt - 59 + pw; lw = swv(l, 61).argmin(1); okp = p >= 60
    s_ = np.full(len(p), -1); s_[okp] = p[okp] - 60 + lw[p[okp] - 60]
    HP = np.full(T_, np.nan); LS = np.full(T_, np.nan); ok = okp & (p - s_ >= 5)
    hp = Hv[p[ok], j]; ls = Lv[s_[ok], j]; g = hp / ls - 1 >= 0.15; HP[tt[ok][g]] = hp[g]; LS[tt[ok][g]] = ls[g]
    HP = pd.Series(HP).shift(4).values; LS = pd.Series(LS).shift(4).values
    c, o, hh, ll, v = C.values, O.values, H.values, L.values, V.values
    c1, o1, h1 = C.shift(1).values, O.shift(1).values, H.shift(1).values; va = V.rolling(20, min_periods=15).mean().shift(1).values
    touch = (TL <= BL + 0.25 * a) & (TL >= BL - 2 * a) & ((BH - BL) / BH <= 0.20) & ((HP - BL) / (HP - LS) <= 0.5) & poolw[:, j] & U[:, j] & tm
    if not touch.any(): continue
    red = c > o; up = c >= BL; brk = TL < BL; big = v >= 1.5 * va
    cmin3 = C.shift(1).rolling(3, min_periods=1).min().values
    P = {'ENG': red & (c1 < o1) & (o <= c1) & (c > o1) & up, 'GAP': (ll > h1) & (c >= o) & up, 'SPR': (cmin3 < BL) & red & (c > BL),
         'VREC': brk & red & (c > BL) & big, 'HAM': (ll < BL) & (c > BL) & ((np.minimum(o, c) - ll) >= 2 * np.abs(c - o)) & ((np.minimum(o, c) - ll) >= 0.5 * (hh - ll)) & big}
    anyp = np.zeros(T_, bool)
    for k in PAT: anyp |= np.nan_to_num(P[k]).astype(bool)
    last = {True: -99, False: -99}
    for t in np.where(touch)[0]:
        ip = bool(anyp[t])
        if t - last[ip] < 10: continue
        w = Lv[t - 23:t - 3, j]
        if not np.isfinite(w).any(): continue
        pos = int(np.nanargmin(w))
        if pos > 14: continue
        seg = Hv[t - 23 + pos + 1:t - 3, j]
        if not (np.nanmax(seg) >= BL[t] + 0.5 * (BH[t] - BL[t])): continue
        oc = outcome(t, j, TL[t])
        if oc is None: continue
        last[ip] = t; oc.update(t=t, j=j, brk=bool(brk[t]), anyp=ip, pool_t=bool(pool[t, j]), **{k: bool(np.nan_to_num(P[k][t])) for k in PAT}); rows.append(oc)
    cand = np.where(pool[:, j] & tm)[0][::5]
    for t in cand[rng.random(len(cand)) < 0.25]:
        oc = outcome(t, j, None)
        if oc is not None: oc.update(t=t, j=j); base.append(oc)
E = pd.DataFrame(rows); B = pd.DataFrame(base); E['date'] = dates[E.t.values]; B['date'] = dates[B.t.values]
pd.to_pickle({'E': E, 'B': B}, f'/home/claude/old/M078_{which}.pkl'); print('saved', which, len(E), int(E.anyp.sum()), len(B), flush=True)
