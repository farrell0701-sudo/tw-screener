import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('/home/claude/llab/l032.py').read().split("VARS = {")[0])
Cv = C0.values.astype(np.float64); T_, N_ = Cv.shape
r1 = Cv[1:] / Cv[:-1] - 1; r1[~U[:-1]] = np.nan; EW = pd.Series(np.r_[1.0, np.cumprod(1 + np.nan_to_num(np.nanmean(r1, 1)))], index=dates)
IX = pd.Series(np.asarray(ixd, dtype=np.float64), index=dates)
CONC = IX.pct_change(250) - EW.pct_change(250)
C = pd.DataFrame(Cv); R60 = (C / C.shift(60) - 1).values; R250 = (C / C.shift(250) - 1).values; ix250 = IX.pct_change(250).values
DISP = pd.Series([np.nanstd(R60[t][U[t]]) for t in range(T_)], index=dates)
BEAT = pd.Series([np.nanmean((R250[t] > ix250[t])[U[t] & np.isfinite(R250[t])]) if np.isfinite(ix250[t]) else np.nan for t in range(T_)], index=dates)
lo_, hi_ = ('2005-01-01', '2015-12-31') if which == 'cpold' else ('2016-01-01', '2026-12-31')
D = pd.DataFrame({'CONC': CONC, 'DISP': DISP, 'BEAT': BEAT}).loc[lo_:hi_]
D.to_pickle(f'/home/claude/old/M090_{which}.pkl'); print('saved', which, flush=True)
