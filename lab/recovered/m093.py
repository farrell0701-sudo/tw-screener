import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('/home/claude/llab/l032.py').read().split("VARS = {")[0])
Cv = C0.values.astype(np.float64); r1 = np.full(Cv.shape, np.nan); r1[1:] = Cv[1:] / Cv[:-1] - 1
pm = np.r_[np.zeros((1, Cv.shape[1]), bool), pool[:-1]]
rp = np.where(pm, r1, np.nan); s = pd.Series(np.nan_to_num(np.nanmean(rp, 1)), index=dates)
lo_, hi_ = ('2005-01-01', '2015-12-31') if which == 'cpold' else ('2016-01-01', '2026-12-31')
s.loc[lo_:hi_].to_pickle(f'/home/claude/old/M093_{which}.pkl'); print('saved', which)
