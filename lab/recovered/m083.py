import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('/home/claude/llab/l032.py').read().split("VARS = {")[0])
from numpy.lib.stride_tricks import sliding_window_view as swv
Cv = C0.values.astype(np.float64); Hv = H0.values.astype(np.float64); Lv = L0.values.astype(np.float64); O1 = np.roll(Ov, -1, 0); T_, N_ = Cv.shape
def fwd(h):
    r = np.full((T_, N_), np.nan); r[:-h-1] = Cv[h + 1:] / O1[:-h-1] - 1; r[~U] = np.nan; return r - np.nanmean(r, 1, keepdims=True)
e20, e60 = fwd(20), fwd(60)
DD = np.full((T_, N_), np.nan, np.float32); BB = np.full((T_, N_), np.nan, np.float32)
for j in range(N_):
    l = np.where(np.isfinite(Lv[:, j]), Lv[:, j], np.inf); h = np.where(np.isfinite(Hv[:, j]), Hv[:, j], -np.inf)
    if np.isfinite(Cv[:, j]).sum() < 400: continue
    sw = swv(l, 121).argmin(1); tt = np.arange(120, T_); s = tt - 120 + sw
    ok = s >= 250; hw = np.full(len(s), np.nan)
    hm = swv(h, 250).max(1)  # hm[k] = max h[k..k+249]
    hw[ok] = hm[s[ok] - 250]
    Ls = Lv[s, j]; DD[tt, j] = np.where(ok & np.isfinite(hw) & (hw > 0), 1 - Ls / hw, np.nan); BB[tt, j] = Cv[tt, j] / Ls - 1
lo_, hi_ = ('2005-01-01', '2015-12-31') if which == 'cpold' else ('2016-01-01', '2026-12-31')
tm = (dates >= pd.Timestamp(lo_)) & (dates <= pd.Timestamp(hi_)); tm[:400] = False; tm[T_ - 62:] = False
was = pd.DataFrame(pool).rolling(20, min_periods=1).max().shift(1).fillna(0).values > 0
ent = pool & ~was & tm[:, None]
ti, ji = np.where(ent)
E = pd.DataFrame({'t': ti, 'j': ji, 'dd': DD[ti, ji], 'bb': BB[ti, ji], 'e20': e20[ti, ji], 'e60': e60[ti, ji]}); E['date'] = dates[E.t.values]
idx = np.where(tm)[0][::5]; pi, pj = np.where(pool[idx]); P = pd.DataFrame({'t': idx[pi], 'dd': DD[idx[pi], pj], 'e20': e20[idx[pi], pj]}); P['date'] = dates[P.t.values]
pd.to_pickle({'E': E, 'P': P, 'DD': DD, 'BB': BB}, f'/home/claude/old/M083_{which}.pkl'); print('saved', which, len(E), len(P), flush=True)
