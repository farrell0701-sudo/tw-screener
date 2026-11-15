import sys, numpy as np, pandas as pd, warnings, gc; warnings.filterwarnings('ignore')
exec(open('/home/claude/llab/l032.py').read().split("VARS = {")[0])
Cv = C0.values.astype(np.float64); T_, N_ = Cv.shape
E20 = C0.ewm(span=20, adjust=False, ignore_na=True).mean().where(C0.notna()).values
E60 = C0.ewm(span=60, adjust=False, ignore_na=True).mean().where(C0.notna()).values
def lagself(X, k):
    out = np.full(X.shape, np.nan)
    for j in range(N_):
        ii = np.where(np.isfinite(X[:, j]))[0]
        if len(ii) > k: out[ii[k:], j] = X[ii[:-k], j]
    return out
S20 = E20 / lagself(E20, 10) - 1; S60 = E60 / lagself(E60, 20) - 1; del E20, E60; gc.collect()
S200 = pd.read_pickle(f'/home/claude/old/M085_{which}.pkl')['S'].astype(np.float64)
prk = lambda X, M: pd.DataFrame(np.where(M, X, np.nan)).rank(axis=1, pct=True).values
def steepf(S, th):
    up_ = pool & (np.nan_to_num(S, nan=-1) > th); return up_ & (np.nan_to_num(prk(S, up_), nan=0) > 0.8)
st200 = pool & (np.nan_to_num(S200, nan=-1) >= 0.05) & (np.nan_to_num(prk(S200, pool & (np.nan_to_num(S200, nan=-1) >= 0.05)), nan=0) > 0.8)
st20 = steepf(S20, 0); st60 = steepf(S60, 0)
O1 = np.roll(Ov, -1, 0); R21 = np.full((T_, N_), np.nan); R21[:-22] = Cv[22:] / O1[:-22] - 1; R21[~U] = np.nan
IX = np.asarray(ixd, dtype=np.float64); MK = np.full(T_, np.nan); MK[:-22] = IX[22:] / IX[:-22] - 1
TL = pd.DataFrame(np.asarray(Tr, dtype=np.float64)); X1c = 0.5 * prk(TL.values, pool) + 0.5 * prk((TL - TL.shift(10)).values, pool)
r20 = pd.DataFrame(Cv).pct_change(20).values
cand = pool & ~hot & ~st200
lo_, hi_ = ('2005-01-01', '2015-12-31') if which == 'cpold' else ('2016-01-01', '2026-12-31')
tm = (dates >= pd.Timestamp(lo_)) & (dates <= pd.Timestamp(hi_)); idx = np.where(tm)[0]; idx = idx[(idx >= 260) & (idx < T_ - 23)]
ov = []; rows = []
for t in idx[::5]:
    m = pool[t] & np.isfinite(S20[t]) & np.isfinite(S60[t]) & np.isfinite(S200[t]) & np.isfinite(r20[t])
    if m.sum() < 30: continue
    df = pd.DataFrame({'S20': S20[t][m], 'S60': S60[t][m], 'S200': S200[t][m], 'r20': r20[t][m]}).rank()
    c = df.corr()
    ov.append(dict(c20r=c.loc['S20', 'r20'], c60r=c.loc['S60', 'r20'], c20_200=c.loc['S20', 'S200'], c60_200=c.loc['S60', 'S200'], c20_60=c.loc['S20', 'S60'],
                   n20=st20[t][pool[t]].mean(), n60=st60[t][pool[t]].mean(),
                   h20=(st20[t] & hot[t]).sum() / max(st20[t].sum(), 1), h60=(st60[t] & hot[t]).sum() / max(st60[t].sum(), 1),
                   s20=(st20[t] & st200[t]).sum() / max(st20[t].sum(), 1), s60=(st60[t] & st200[t]).sum() / max(st60[t].sum(), 1),
                   c20=(st20[t] & cand[t]).sum() / max(cand[t].sum(), 1), c60=(st60[t] & cand[t]).sum() / max(cand[t].sum(), 1)))
for t in idx[::21]:
    r = R21[t]; um = np.nanmean(r[U[t] & np.isfinite(r)]); pok = pool[t] & np.isfinite(r); cm = cand[t] & np.isfinite(r)
    if cm.sum() < 20: continue
    row = dict(date=dates[t], MKT=MK[t], U=um, POOL=np.nanmean(r[pok]), CAND=np.nanmean(r[cm]))
    for nm, FL, S in (('S20', st20, S20), ('S60', st60, S60)):
        fl = cm & FL[t]; kp = cm & ~FL[t]
        row[nm + '_f'] = np.nanmean(r[fl]) if fl.sum() >= 3 else np.nan; row[nm + '_k'] = np.nanmean(r[kp]); row[nm + '_n'] = fl.sum() / cm.sum()
        mm = cm & np.isfinite(S[t]) & np.isfinite(X1c[t])
        a = pd.Series(-S[t][mm]).rank().values; e = pd.Series(r[mm] - um).rank().values; x = pd.Series(X1c[t][mm]).rank().values
        res = lambda v: v - np.polyval(np.polyfit(x, v, 1), x)
        row[nm + '_pic'] = np.corrcoef(res(a), res(e))[0, 1] if mm.sum() >= 25 else np.nan
    rows.append(row)
pd.to_pickle({'ov': pd.DataFrame(ov), 'rows': pd.DataFrame(rows)}, f'/home/claude/old/M096_{which}.pkl'); print('saved', which, flush=True)
