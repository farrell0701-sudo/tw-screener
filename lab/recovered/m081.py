import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('/home/claude/llab/l032.py').read().split("VARS = {")[0])
cols_s = [str(c) for c in C0.columns]; T_, N_ = C0.shape
RC = D['RC'].reindex(index=dates, columns=C0.columns).values.astype(np.float64)
V = D['V'].reindex(index=dates, columns=C0.columns).values.astype(np.float64)
FN = np.full((T_, N_), np.nan); TN = np.full((T_, N_), np.nan); SH = np.full((T_, N_), np.nan)
for k, c in enumerate(cols_s):
    f = f'/home/claude/chip/inst_{c}.csv'
    if os.path.exists(f):
        d = pd.read_csv(f, parse_dates=['date']).set_index('date').reindex(dates)
        FN[:, k] = d.foreign.values; TN[:, k] = d.trust.values
    g = f'/home/claude/fund/bs_{c}.csv'
    if os.path.exists(g):
        b = pd.read_csv(g, usecols=['date', 'type', 'value']); b = b[b.type == 'OrdinaryShare']
        if len(b):
            s = pd.Series(b.value.values / 10, index=pd.to_datetime(b.date) + pd.Timedelta(days=90)).sort_index()
            s = s[~s.index.duplicated(keep='last')]
            SH[:, k] = s.reindex(s.index.union(dates)).ffill().reindex(dates).values
MC = RC * SH
lo_, hi_ = ('2014-07-01', '2015-12-31') if which == 'cpold' else ('2016-01-01', '2026-12-31')
tm = (dates >= pd.Timestamp(lo_)) & (dates <= pd.Timestamp(hi_))
print('單位檢查 外資|淨買|/成交量 中位', np.nanmedian(np.abs(FN[tm]) / V[tm]), '有外資資料比例', np.mean(np.isfinite(FN[tm]) & U[tm]) / np.mean(U[tm]), '有市值比例', np.mean(np.isfinite(MC[tm]) & U[tm]) / np.mean(U[tm]), flush=True)
rs = lambda X, n: pd.DataFrame(X).rolling(n, min_periods=int(n * 0.8)).sum().values
Vs5, Vs20 = rs(V, 5), rs(V, 20)
FAC = {'FNB5': rs(FN, 5) / Vs5, 'FNB20': rs(FN, 20) / Vs20, 'FPART20': rs(np.abs(FN), 20) / Vs20, 'TNB20': rs(TN, 20) / Vs20}
Cv = C0.values.astype(np.float64); O1 = np.roll(Ov, -1, 0)
r21 = np.full((T_, N_), np.nan); r21[:-22] = Cv[22:] / O1[:-22] - 1; r21[~U] = np.nan
ex = r21 - np.nanmean(r21, 1, keepdims=True)
C = pd.DataFrame(Cv); R20 = (C / C.shift(20) - 1).values; R60 = (C / C.shift(60) - 1).values
idx = np.where(tm)[0]; idx = idx[(idx >= 60) & (idx < T_ - 23)][::21]
rows = []
for t in idx:
    m0 = U[t] & np.isfinite(ex[t]) & np.isfinite(MC[t])
    if m0.sum() < 100: continue
    big = np.zeros(N_, bool); thr = np.nanpercentile(MC[t][m0], 70); big[m0 & (MC[t] >= thr)] = True
    for k, X in FAC.items():
        m = m0 & np.isfinite(X[t]) & np.isfinite(R20[t]) & np.isfinite(R60[t])
        for grp, mm in (('全體', m), ('大型', m & big)):
            if mm.sum() < 50: continue
            rows.append(pd.DataFrame({'t': t, 'fac': k, 'grp': grp, 'x': X[t][mm], 'ex': ex[t][mm], 'r20': R20[t][mm], 'r60': R60[t][mm], 'mc': MC[t][mm]}))
Df = pd.concat(rows, ignore_index=True); Df['date'] = dates[Df.t.values]
Df.to_pickle(f'/home/claude/old/M081_{which}.pkl'); print('saved', which, len(Df), Df.t.nunique(), flush=True)
