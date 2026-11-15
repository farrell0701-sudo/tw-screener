import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('/home/claude/llab/l032.py').read().split("VARS = {")[0])
from numpy.lib.stride_tricks import sliding_window_view as swv
Cv = C0.values.astype(np.float64); Hv = H0.values.astype(np.float64); Lv = L0.values.astype(np.float64); O1 = np.roll(Ov, -1, 0); T_, N_ = Cv.shape
IX = np.asarray(ixd, dtype=np.float64)
def fwd(h):
    r = np.full((T_, N_), np.nan); r[:-h-1] = Cv[h + 1:] / O1[:-h-1] - 1; r[~U] = np.nan; return r - np.nanmean(r, 1, keepdims=True)
e20, e60 = fwd(20), fwd(60)
DS = np.full((T_, N_), np.nan, np.float32); DM = np.full((T_, N_), np.nan, np.float32)
for j in range(N_):
    if np.isfinite(Cv[:, j]).sum() < 400: continue
    l = np.where(np.isfinite(Lv[:, j]), Lv[:, j], np.inf); h = np.where(np.isfinite(Hv[:, j]), Hv[:, j], -np.inf)
    tt = np.arange(120, T_); s = tt - 120 + swv(l, 121).argmin(1); ok = s >= 250
    ha = swv(h, 250).argmax(1)  # index offset within window starting at k
    hidx = np.full(len(s), -1); hidx[ok] = s[ok] - 250 + ha[s[ok] - 250]
    Hh = np.where(ok, Hv[np.maximum(hidx, 0), j], np.nan); Ls = Lv[s, j]
    DS[tt, j] = np.where(ok & (Hh > 0), 1 - Ls / Hh, np.nan)
    DM[tt, j] = np.where(ok, 1 - IX[s] / IX[np.maximum(hidx, 0)], np.nan)
lo_, hi_ = ('2005-01-01', '2015-12-31') if which == 'cpold' else ('2016-01-01', '2026-12-31')
tm = (dates >= pd.Timestamp(lo_)) & (dates <= pd.Timestamp(hi_)); tm[:400] = False; tm[T_ - 62:] = False
was = pd.DataFrame(pool).rolling(20, min_periods=1).max().shift(1).fillna(0).values > 0
ti, ji = np.where(pool & ~was & tm[:, None])
E = pd.DataFrame({'t': ti, 'j': ji, 'ds': DS[ti, ji], 'dm': DM[ti, ji], 'e20': e20[ti, ji], 'e60': e60[ti, ji]}); E['date'] = dates[E.t.values]
pd.to_pickle({'E': E, 'DS': DS, 'DM': DM}, f'/home/claude/old/M084_{which}.pkl'); print('saved', which, len(E), flush=True)
