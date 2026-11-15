import sys, numpy as np, pandas as pd, warnings, gc; warnings.filterwarnings('ignore')
exec(open('/home/claude/llab/l032.py').read().split("VARS = {")[0])
Cv = C0.values.astype(np.float64); Hv = H0.values.astype(np.float64); Lv = L0.values.astype(np.float64); T_, N_ = Cv.shape
Vv = D['V'].reindex(index=dates, columns=C0.columns).values.astype(np.float64)
r1 = Cv[1:] / Cv[:-1] - 1; r1[~U[:-1]] = np.nan; IDX = np.concatenate([[1.0], np.cumprod(1 + np.nan_to_num(np.nanmean(r1, 1)))]); del r1
C = pd.DataFrame(Cv); rk = lambda X: pd.DataFrame(np.where(U, X, np.nan)).rank(axis=1, pct=True).values.astype(np.float32)
TL = pd.DataFrame(np.asarray(Tr, dtype=np.float64))
FAC = {'T_ld百分位': rk(TL.values), 'ΔT_ld10百分位': rk((TL - TL.shift(10)).values),
       '20日漲幅': (C / C.shift(20) - 1).values, '60日漲幅': (C / C.shift(60) - 1).values,
       'ATR%': F['atr'].values / Cv, 'ER20': ((C - C.shift(20)) / C.diff().abs().rolling(20).sum()).values,
       '距250日高/ATR': ((pd.DataFrame(Hv).rolling(250, min_periods=120).max() - C).values / F['atr'].values),
       '成交額(log)': np.log(F['amt20'].values.astype(np.float64)),
       '量比5/20': (pd.DataFrame(Vv).rolling(5).mean() / pd.DataFrame(Vv).rolling(20).mean()).values}
H_, L_ = pd.DataFrame(Hv), pd.DataFrame(Lv); hp, lp = H_.rolling(5).max().shift(1), L_.rolling(5).min().shift(1)
w = H_ - L_; inter = (np.minimum(H_, hp) - np.maximum(L_, lp)).clip(lower=0); ins = ((H_ <= hp) & (L_ >= lp)).astype(float)
ov = (inter / w.where(w > 0)).where(w > 0, ins).where(hp.notna()); FAC['重疊度短線'] = ov.ewm(span=5, adjust=False).mean().values; del H_, L_, hp, lp, w, inter, ins, ov; gc.collect()
mkt20 = np.full(T_, np.nan); mkt20[20:] = IDX[20:] / IDX[:-20] - 1
ma200 = pd.Series(IDX).rolling(200).mean().values
def rr_at(t, j):
    h = Hv[t - 59:t + 1, j]
    if t < 130 or not np.isfinite(h).any(): return np.nan
    p = t - 59 + int(np.nanargmax(h)); l = Lv[p - 60:p + 1, j]
    if not np.isfinite(l).any(): return np.nan
    s = p - 60 + int(np.nanargmin(l)); hp_, ls = Hv[p, j], Lv[s, j]
    return (hp_ - Cv[t, j]) / (hp_ - ls) if (p - s >= 5 and hp_ / ls - 1 >= 0.15) else np.nan
out = {}
for nm in ['X', 'V']:
    T = pd.read_pickle(f'/home/claude/old/M080{nm}_{which}.pkl')
    npath = T.path.nunique()
    g = T.groupby(['j', 't0']).agg(net=('net', 'mean'), hold=('hold', 'mean'), mfe=('mfe', 'mean'), rs=('rs', 'first'), n=('net', 'size')).reset_index()
    s = g.t0.values - 1; j = g.j.values
    for k, X in FAC.items(): g[k] = X[s, j]
    g['回撤比'] = [rr_at(a, b) for a, b in zip(s, j)]
    g['大盤20日'] = mkt20[s]; g['大盤>200MA'] = (IDX[s] > ma200[s]).astype(float)
    te = np.minimum(g.t0.values + np.round(g.hold.values).astype(int), T_ - 1)
    g['持有期大盤'] = IDX[te] / IDX[s] - 1
    g['date'] = dates[g.t0.values]; g['npath'] = npath
    out[nm] = g; print(nm, which, '原始', len(T), '去重', len(g), flush=True)
# 範圍限縮：交易樣本 vs 可交易池（每 5 日抽樣）的分散度
idx = np.where((dates >= pd.Timestamp('2005-01-01')))[0][260::5]
uni = {k: np.concatenate([X[t][U[t]] for t in idx]) for k, X in FAC.items()}
out['_uni_iqr'] = {k: float(np.nanpercentile(v, 75) - np.nanpercentile(v, 25)) for k, v in uni.items()}
pd.to_pickle(out, f'/home/claude/old/M080F_{which}.pkl'); print('saved F', which, flush=True)
