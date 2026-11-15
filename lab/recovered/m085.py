import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('/home/claude/llab/l032.py').read().split("VARS = {")[0])
Cv = C0.values.astype(np.float64); O1 = np.roll(Ov, -1, 0); T_, N_ = Cv.shape
def fwd(h):
    r = np.full((T_, N_), np.nan); r[:-h-1] = Cv[h + 1:] / O1[:-h-1] - 1; r[~U] = np.nan; return r - np.nanmean(r, 1, keepdims=True)
e20, e60 = fwd(20), fwd(60)
E2 = F['e200'].values.astype(np.float64); S = np.full((T_, N_), np.nan); S[60:] = E2[60:] / E2[:-60] - 1
TL = pd.DataFrame(np.asarray(Tr, dtype=np.float64))
prk = lambda X, M: pd.DataFrame(np.where(M, X, np.nan)).rank(axis=1, pct=True).values
tp = prk(TL.values, pool); SCx = 0.5 * prk(TL.values, pool) + 0.5 * prk((TL - TL.shift(10)).values, pool)
r60 = pd.DataFrame(Cv).pct_change(60).values
lo_, hi_ = ('2005-01-01', '2015-12-31') if which == 'cpold' else ('2016-01-01', '2026-12-31')
tm = (dates >= pd.Timestamp(lo_)) & (dates <= pd.Timestamp(hi_)); idx = np.where(tm)[0]; idx = idx[(idx >= 260) & (idx < T_ - 62)][::5]
rows = []
for t in idx:
    m = pool[t] & np.isfinite(e20[t]) & np.isfinite(S[t])
    if m.sum() < 20: continue
    rows.append(pd.DataFrame({'t': t, 'j': np.where(m)[0], 'S': S[t][m], 'tp': tp[t][m], 'sc': SCx[t][m], 'x1': (~hot[t])[m], 'e20': e20[t][m], 'e60': e60[t][m], 'r60': r60[t][m]}))
D = pd.concat(rows); D['date'] = dates[D.t.values]
pd.to_pickle({'D': D, 'S': S.astype(np.float32)}, f'/home/claude/old/M085_{which}.pkl'); print('saved', which, len(D), flush=True)
