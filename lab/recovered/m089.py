import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('/home/claude/llab/l032.py').read().split("VARS = {")[0])
Cv = C0.values.astype(np.float64); T_, N_ = Cv.shape
E2 = F['e200'].values.astype(np.float64); S = np.full(E2.shape, np.nan); S[60:] = E2[60:] / E2[:-60] - 1
up = pool & (np.nan_to_num(S, nan=-1) >= 0.05); sp = pd.DataFrame(np.where(up, S, np.nan)).rank(axis=1, pct=True).values
steep = up & (np.nan_to_num(sp, nan=0) > 0.8); TL = np.asarray(Tr, dtype=np.float64)
O = Ov.astype(np.float64); r_oo = np.full((T_, N_), np.nan); r_oo[:-1] = O[1:] / O[:-1] - 1   # 開盤到次日開盤
ix = np.asarray(ixd, dtype=np.float64); ixr = np.r_[ix[1:] / ix[:-1] - 1, 0]
lo_, hi_ = ('2005-01-01', '2015-12-31') if which == 'cpold' else ('2016-01-01', '2026-12-31')
tm = (dates >= pd.Timestamp(lo_)) & (dates <= pd.Timestamp(hi_)); tm[:260] = False
COST = 0.0077
def sim(selfn):
    ret = np.zeros(T_); held = np.zeros(N_, bool); age = 0; inpos = np.zeros(T_, bool)
    for t in range(1, T_ - 1):
        if not tm[t]: held[:] = False; continue
        bear = bool(below[t - 1])
        c = 0.0
        if bear and (not held.any() or age >= 20):
            new = selfn(t - 1); w_old = held.astype(float) / max(held.sum(), 1); w_new = new.astype(float) / max(new.sum(), 1)
            c = 0.5 * np.abs(w_new - w_old).sum() * COST; held = new; age = 0
        elif not bear and held.any():
            c = 0.5 * COST; held = np.zeros(N_, bool)
        if held.any():
            rr = r_oo[t][held]; ret[t] = np.nanmean(np.where(np.isfinite(rr), rr, 0)) - c; inpos[t] = True; age += 1
        else: ret[t] = -c
    return ret, inpos
selA = lambda t: steep[t] & np.isfinite(O[t + 1])
def selB(t):
    m = steep[t] & np.isfinite(O[t + 1]); k = np.where(m)[0]
    out = np.zeros(N_, bool); out[k[np.argsort(-TL[t, k])][:6]] = True; return out
selC = lambda t: pool[t] & np.isfinite(O[t + 1])
R = {}
for nm, f in (('A 最陡全部', selA), ('B 最陡T_ld前6', selB), ('C 強勢池全部', selC)):
    R[nm], inp = sim(f)
R['D 加權指數'] = np.where(inp, ixr, 0.0)
pd.to_pickle({'R': R, 'inpos': inp, 'dates': dates, 'below': below, 'tm': tm}, f'/home/claude/old/M089_{which}.pkl'); print('saved', which, flush=True)
