import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('/home/claude/llab/l032.py').read().split("VARS = {")[0])
Cv = C0.values.astype(np.float64); r1 = Cv[1:] / Cv[:-1] - 1; r1[~U[:-1]] = np.nan
EW = pd.Series(np.r_[0.0, np.nan_to_num(np.nanmean(r1, 1))], index=dates); IX = pd.Series(np.asarray(ixd, dtype=np.float64), index=dates)
pd.to_pickle({'EW': EW, 'IX': IX}, f'/home/claude/old/M091_{which}.pkl'); print('saved', which)
