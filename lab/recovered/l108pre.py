import sys, numpy as np, pandas as pd, warnings, gc; warnings.filterwarnings('ignore')
exec(open('/home/claude/llab/l032.py').read().split("VARS = {")[0])
from numpy.lib.stride_tricks import sliding_window_view as swv
T_, N_ = C0.shape; Cv_ = C0.values.astype(np.float64)
pp = lambda X: pd.DataFrame(np.where(pool, X, np.nan)).rank(axis=1, pct=True).values.astype(np.float32)
Cp = pd.DataFrame(Cv_).ffill().shift(1).values; Ov_ = Ov.astype(np.float64)
ON = (pd.DataFrame(np.log(Ov_ / Cp)).rolling(20, min_periods=15).sum() - pd.DataFrame(np.log(Cv_ / Ov_)).rolling(20, min_periods=15).sum()).values
del Cp, Ov_; gc.collect()
P = pp(ON); np.save(f'/home/claude/old/L108_ON_P_{which}.npy', P); np.save(f'/home/claude/old/L108_ON_F_{which}.npy', pool & (np.nan_to_num(P, nan=0) > 0.8)); del ON, P; gc.collect()
rr1 = pd.DataFrame(Cv_).pct_change(); Cdf = pd.DataFrame(Cv_)
ER = ((Cdf - Cdf.shift(20)) / Cdf.diff().abs().rolling(20).sum()).values; MAX20 = rr1.rolling(20, min_periods=15).max().values
SPK = pp(MAX20) - pp(ER); del ER, MAX20, Cdf; gc.collect()
P = pp(SPK); np.save(f'/home/claude/old/L108_SPK_P_{which}.npy', P); np.save(f'/home/claude/old/L108_SPK_F_{which}.npy', pool & (np.nan_to_num(P, nan=0) > 0.8)); del SPK, P; gc.collect()
DVS = np.sqrt((rr1.where(rr1 < 0, 0.0) ** 2).rolling(60, min_periods=45).sum() / (rr1 ** 2).rolling(60, min_periods=45).sum()).values; del rr1; gc.collect()
P = pp(DVS); np.save(f'/home/claude/old/L108_DVS_P_{which}.npy', P); np.save(f'/home/claude/old/L108_DVS_F_{which}.npy', pool & (np.nan_to_num(P, nan=0) > 0.8)); del DVS, P; gc.collect()
Hv_ = H0.values.astype(np.float64); Lv_ = L0.values.astype(np.float64); RR = np.zeros((T_, N_), np.float32)
for j in range(N_):
    if np.isfinite(Cv_[:, j]).sum() < 200: continue
    h = np.where(np.isfinite(Hv_[:, j]), Hv_[:, j], -np.inf); l = np.where(np.isfinite(Lv_[:, j]), Lv_[:, j], np.inf)
    pw = swv(h, 60).argmax(1); tt = np.arange(59, T_); p = tt - 59 + pw; lw = swv(l, 61).argmin(1); okp = p >= 60
    s_ = np.full(len(p), -1); s_[okp] = p[okp] - 60 + lw[p[okp] - 60]; ok = okp & (p - s_ >= 5) & np.isfinite(Cv_[tt, j])
    hp = Hv_[p[ok], j]; ls = Lv_[s_[ok], j]; g = hp / ls - 1 >= 0.15; RR[tt[ok][g], j] = ((hp - Cv_[tt[ok], j]) / (hp - ls))[g]
P = pp(RR); np.save(f'/home/claude/old/L108_RR_P_{which}.npy', P); np.save(f'/home/claude/old/L108_RR_F_{which}.npy', pool & (RR >= 0.236))
print('saved pre', which, flush=True)
