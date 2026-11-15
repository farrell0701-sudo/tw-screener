import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('/home/claude/llab/l032.py').read().split("VARS = {")[0])
Cv = C0.values.astype(np.float64); O1 = np.roll(Ov, -1, 0); T_, N_ = Cv.shape
r = np.full((T_, N_), np.nan); r[:-21] = Cv[21:] / O1[:-21] - 1; r[~U] = np.nan; e20 = r - np.nanmean(r, 1, keepdims=True)
E2 = F['e200'].values.astype(np.float64); S = np.full((T_, N_), np.nan); S[60:] = E2[60:] / E2[:-60] - 1
up = pool & (np.nan_to_num(S, nan=-1) >= 0.05); sp = pd.DataFrame(np.where(up, S, np.nan)).rank(axis=1, pct=True).values
steep = up & (np.nan_to_num(sp, nan=0) > 0.8)
R20 = pd.DataFrame(Cv).pct_change(20).values
lo_, hi_ = ('2005-01-01', '2015-12-31') if which == 'cpold' else ('2016-01-01', '2026-12-31')
tm = (dates >= pd.Timestamp(lo_)) & (dates <= pd.Timestamp(hi_)); idx = np.where(tm)[0]; idx = idx[(idx >= 260) & (idx < T_ - 22)][::5]
rows = []
for t in idx:
    m = pool[t] & np.isfinite(e20[t])
    if m.sum() < 20: continue
    rows.append(pd.DataFrame({'t': t, 'hot': hot[t][m], 'steep': steep[t][m], 'S': S[t][m], 'r20': R20[t][m], 'e20': e20[t][m]}))
D = pd.concat(rows); D['date'] = dates[D.t.values]; D.to_pickle(f'/home/claude/old/M087_{which}.pkl'); print('saved', which, flush=True)
