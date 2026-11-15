import sys, os, numpy as np, pandas as pd, warnings, gc; warnings.filterwarnings('ignore')
exec(open('/home/claude/llab/l032.py').read().split("VARS = {")[0])
exec(open('/home/claude/llab/valfac.py').read())
from numpy.lib.stride_tricks import sliding_window_view as swv
Cv = C0.values.astype(np.float64); Hv = H0.values.astype(np.float64); Lv = L0.values.astype(np.float64); T_, N_ = Cv.shape
O1 = np.roll(Ov, -1, 0); R21 = np.full((T_, N_), np.nan); R21[:-22] = Cv[22:] / O1[:-22] - 1; R21[~U] = np.nan
IX = np.asarray(ixd, dtype=np.float64); MK = np.full(T_, np.nan); MK[:-22] = IX[22:] / IX[:-22] - 1
TL = pd.DataFrame(np.asarray(Tr, dtype=np.float64)); C = pd.DataFrame(Cv)
prk = lambda X, M: pd.DataFrame(np.where(M, X, np.nan)).rank(axis=1, pct=True).values
dT = (TL - TL.shift(10)).values; X1c = 0.5 * prk(TL.values, pool) + 0.5 * prk(dT, pool)
S85 = pd.read_pickle(f'/home/claude/old/M085_{which}.pkl')['S'].astype(np.float64)
up = pool & (np.nan_to_num(S85, nan=-1) >= 0.05); steep = up & (np.nan_to_num(prk(S85, up), nan=0) > 0.8)
M84 = pd.read_pickle(f'/home/claude/old/M084_{which}.pkl'); idio = (np.nan_to_num(M84['DS'], nan=0) >= 0.4) & (np.nan_to_num(M84['DM'], nan=1) < 0.05); del M84
RR = np.full((T_, N_), np.nan, np.float32)
for j in range(N_):
    if np.isfinite(Cv[:, j]).sum() < 200: continue
    h = np.where(np.isfinite(Hv[:, j]), Hv[:, j], -np.inf); l = np.where(np.isfinite(Lv[:, j]), Lv[:, j], np.inf)
    pw = swv(h, 60).argmax(1); tt = np.arange(59, T_); p = tt - 59 + pw; lw = swv(l, 61).argmin(1); okp = p >= 60
    s_ = np.full(len(p), -1); s_[okp] = p[okp] - 60 + lw[p[okp] - 60]; ok = okp & (p - s_ >= 5) & np.isfinite(Cv[tt, j])
    hp = Hv[p[ok], j]; ls = Lv[s_[ok], j]; g = hp / ls - 1 >= 0.15; RR[tt[ok][g], j] = ((hp - Cv[tt[ok], j]) / (hp - ls))[g]
deep = pool & (np.nan_to_num(RR, nan=0) >= 0.236)
VAL = build_val(); gc.collect()
ER = ((C - C.shift(20)) / C.diff().abs().rolling(20).sum()).values
ATRP = F['atr'].values / Cv
V = D['V'].reindex(index=dates, columns=C0.columns).values.astype(np.float64); FN = np.full((T_, N_), np.nan)
for k, c in enumerate([str(x) for x in C0.columns]):
    f = f'/home/claude/chip/inst_{c}.csv'
    if os.path.exists(f) and os.path.getsize(f) > 30:
        d = pd.read_csv(f, parse_dates=['date']).set_index('date').reindex(dates); FN[:, k] = d.foreign.values
FP = pd.DataFrame(np.abs(FN)).rolling(20, min_periods=16).sum().values / pd.DataFrame(V).rolling(20, min_periods=16).sum().values; del FN, V
RANK = {'T_ld': (TL.values, U), 'ΔT_ld10': (dT, pool), 'X1c': (X1c, pool), '斜率平緩': (-S85, pool), '淺回撤': (-RR.astype(np.float64), pool),
        '價值': (VAL.astype(np.float64), pool), '低波動': (-ATRP, pool), 'ER20': (ER, pool), '外資參與度': (FP, pool)}
EXCL = {'過熱': (hot, -(C / C.shift(20) - 1).values), '最陡斜率': (steep, -S85), '深回撤': (deep, -RR.astype(np.float64)), '個股獨跌': (idio, None), '營收衰退': (neg & pool, FAC['營收年增'])}
lo_, hi_ = ('2005-01-01', '2015-12-31') if which == 'cpold' else ('2016-01-01', '2026-12-31')
tm = (dates >= pd.Timestamp(lo_)) & (dates <= pd.Timestamp(hi_)); idx = np.where(tm)[0]; idx = idx[(idx >= 260) & (idx < T_ - 23)][::21]
rows = []
for t in idx:
    r = R21[t]; uok = U[t] & np.isfinite(r); um = np.nanmean(r[uok]); pok = pool[t] & np.isfinite(r)
    if pok.sum() < 20: continue
    row = dict(t=t, date=dates[t], MKT=MK[t], U=um, POOL=np.nanmean(r[pok]))
    for nm, (X, M) in RANK.items():
        m = M[t] & np.isfinite(r) & np.isfinite(X[t])
        if m.sum() < 25: row[nm + '_ic'] = np.nan; row[nm + '_p'] = np.nan; continue
        x = X[t][m]; e = r[m] - um
        row[nm + '_ic'] = pd.Series(x).rank().corr(pd.Series(e).rank())
        q = x >= np.nanpercentile(x, 80); row[nm + '_p'] = np.nanmean(r[m][q])
    for nm, (FL, X) in EXCL.items():
        fl = FL[t] & pok; kp = pok & ~FL[t]
        row[nm + '_p'] = np.nanmean(r[kp]) if kp.sum() else np.nan; row[nm + '_f'] = np.nanmean(r[fl]) if fl.sum() >= 3 else np.nan; row[nm + '_n'] = fl.sum() / pok.sum()
        if X is not None:
            m = pok & np.isfinite(X[t]); row[nm + '_ic'] = pd.Series(X[t][m]).rank().corr(pd.Series(r[m] - um).rank()) if m.sum() >= 25 else np.nan
        else:
            row[nm + '_ic'] = pd.Series((~fl[pok]).astype(float)).rank().corr(pd.Series(r[pok] - um).rank()) if fl.sum() >= 3 else np.nan
    rows.append(row)
pd.DataFrame(rows).to_pickle(f'/home/claude/old/M095_{which}.pkl'); print('saved', which, len(rows), flush=True)
