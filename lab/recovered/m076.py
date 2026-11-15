import sys, numpy as np, pandas as pd, warnings, gc; warnings.filterwarnings('ignore')
exec(open('/home/claude/llab/l032.py').read().split("VARS = {")[0])
from numpy.lib.stride_tricks import sliding_window_view as swv
Hv, Lv, Cv = H0.values.astype(np.float64), L0.values.astype(np.float64), C0.values.astype(np.float64); T_, N_ = Cv.shape; NW = 60
RR = np.full((T_, N_), np.nan, np.float32)
for j in range(N_):
    if np.isfinite(Cv[:, j]).sum() < 200: continue
    h = np.where(np.isfinite(Hv[:, j]), Hv[:, j], -np.inf); l = np.where(np.isfinite(Lv[:, j]), Lv[:, j], np.inf)
    pw = swv(h, NW).argmax(1); tt = np.arange(NW - 1, T_); p = tt - NW + 1 + pw
    lw = swv(l, NW + 1).argmin(1); okp = p >= NW
    s_ = np.full(len(p), -1); s_[okp] = p[okp] - NW + lw[p[okp] - NW]
    ok = okp & (p - s_ >= 5) & np.isfinite(Cv[tt, j]) & (p < tt)
    hp = Hv[p[ok], j]; ls = Lv[s_[ok], j]; g = hp / ls - 1 >= 0.15
    RR[tt[ok][g], j] = ((hp - Cv[tt[ok], j]) / (hp - ls))[g]
TL = pd.DataFrame(np.asarray(Tr, dtype=np.float64))
O1 = np.roll(Ov, -1, 0); r20 = np.full_like(Cv, np.nan); r20[:-21] = Cv[21:] / O1[:-21] - 1; r20[~U] = np.nan; mu = np.nanmean(r20, 1)
Sx = pool & ~hot
prk = lambda X, M: pd.DataFrame(np.where(M, X, np.nan)).rank(axis=1, pct=True).values
SCx = 0.5 * prk(TL.values, pool) + 0.5 * prk((TL - TL.shift(10)).values, pool)
lo_, hi_ = ('2005-01-01', '2015-12-31') if which == 'cpold' else ('2016-01-01', '2026-12-31')
tm = (dates >= pd.Timestamp(lo_)) & (dates <= pd.Timestamp(hi_)); idx = np.where(tm)[0][::5]; idx = idx[(idx >= 260) & (idx < T_ - 22)]
rows = []
for t in idx:
    m = Sx[t] & np.isfinite(r20[t]) & np.isfinite(SCx[t])
    if m.sum() < 20: continue
    sc = SCx[t][m]; rank_in = pd.Series(sc).rank(ascending=False).values
    rows.append(pd.DataFrame({'t': t, 'ex': r20[t][m] - mu[t], 'sc': sc, 'r': RR[t][m], 'top6': rank_in <= 6}))
Df = pd.concat(rows, ignore_index=True); Df['date'] = dates[Df.t.values]
Df.to_pickle(f'/home/claude/old/M076_{which}.pkl'); print('saved', which, len(Df), flush=True)
