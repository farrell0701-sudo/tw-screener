import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('/home/claude/llab/l032.py').read().split("VARS = {")[0])
E = pd.read_pickle(f'/home/claude/old/bell_ev_{which}.pkl')
C = pd.DataFrame(C0.values.astype(np.float64)); Cv = C.values; Hv = H0.values.astype(np.float64); Lv = L0.values.astype(np.float64)
ER = ((C - C.shift(20)) / C.diff().abs().rolling(20).sum()).values
h20 = pd.DataFrame(Hv).rolling(20).max().shift(1).values
ti = dates.get_indexer(pd.to_datetime(E.date)); cix = {str(c): k for k, c in enumerate(C0.columns)}
ji = np.array([cix.get(str(c), -1) for c in E.code]); ok = (ti >= 130) & (ji >= 0) & (ti + 6 < len(dates))
E = E[ok].copy(); ti, ji = ti[ok], ji[ok]
rr = np.full(len(E), np.nan)
for k, (t, j) in enumerate(zip(ti, ji)):
    h = Hv[t - 59:t + 1, j]
    if not np.isfinite(h).any(): continue
    p = t - 59 + int(np.nanargmax(h)); l = Lv[p - 60:p + 1, j]
    if not np.isfinite(l).any(): continue
    s = p - 60 + int(np.nanargmin(l)); hp, ls = Hv[p, j], Lv[s, j]
    if p - s >= 5 and hp / ls - 1 >= 0.15: rr[k] = (hp - Cv[t, j]) / (hp - ls)
E['r'] = rr; E['ER'] = ER[ti, ji]
E['fail'] = [np.nanmin(Cv[t + 1:t + 6, j]) < h20[t, j] for t, j in zip(ti, ji)]
E[['date', 'code', 'x20', 'T_ld', 'ER', 'r', 'fail']].to_pickle(f'/home/claude/old/M077_{which}.pkl'); print('saved', which, len(E))
