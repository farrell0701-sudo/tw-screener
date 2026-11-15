import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('/home/claude/llab/l032.py').read().split("VARS = {")[0])
Cv = C0.values.astype(np.float64); O1 = np.roll(Ov, -1, 0); T_, N_ = Cv.shape
lo_, hi_ = ('2005-01-01', '2015-12-31') if which == 'cpold' else ('2016-01-01', '2026-12-31')
tm = (dates >= pd.Timestamp(lo_)) & (dates <= pd.Timestamp(hi_)); rows = []
TL = np.asarray(Tr, dtype=np.float64)
for h, step in ((21, 21), (63, 63)):
    r = np.full((T_, N_), np.nan); r[:-h-1] = Cv[h + 1:] / O1[:-h-1] - 1; r[~U] = np.nan
    idx = np.where(tm)[0]; idx = idx[(idx >= 260) & (idx < T_ - h - 2)][::step]
    for t in idx:
        m = U[t] & np.isfinite(r[t]) & np.isfinite(TL[t])
        if m.sum() < 100: continue
        x = TL[t][m]; e = r[t][m] - np.nanmean(r[t][m]); q = pd.qcut(pd.Series(x).rank(method='first'), 10, labels=False).values + 1
        rows.append(pd.DataFrame({'h': h, 't': t, 'q': q, 'ex': e}))
D = pd.concat(rows); D['date'] = dates[D.t.values]; D.to_pickle(f'/home/claude/old/M082_{which}.pkl'); print('saved', which, flush=True)
