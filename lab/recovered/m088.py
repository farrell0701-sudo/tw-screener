import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('/home/claude/llab/l032.py').read().split("VARS = {")[0])
from numpy.lib.stride_tricks import sliding_window_view as swv
E2 = F['e200'].values.astype(np.float64); S = np.full(E2.shape, np.nan); S[60:] = E2[60:] / E2[:-60] - 1
up = pool & (np.nan_to_num(S, nan=-1) >= 0.05); sp = pd.DataFrame(np.where(up, S, np.nan)).rank(axis=1, pct=True).values
steep = up & (np.nan_to_num(sp, nan=0) > 0.8)
E = pd.read_pickle(f'/home/claude/old/bell_ev_{which}.pkl'); ti = dates.get_indexer(pd.to_datetime(E.date)); cix = {str(c): k for k, c in enumerate(C0.columns)}
ji = np.array([cix.get(str(c), -1) for c in E.code]); ok = (ti >= 0) & (ji >= 0); E = E[ok].copy(); ti, ji = ti[ok], ji[ok]
Hv, Lv, Cv = H0.values.astype(np.float64), L0.values.astype(np.float64), C0.values.astype(np.float64)
def rr_at(t, j):
    h = Hv[t - 59:t + 1, j]
    if t < 130 or not np.isfinite(h).any(): return np.nan
    p = t - 59 + int(np.nanargmax(h)); l = Lv[p - 60:p + 1, j]
    if not np.isfinite(l).any(): return np.nan
    s = p - 60 + int(np.nanargmin(l)); hp, ls = Hv[p, j], Lv[s, j]
    return (hp - Cv[t, j]) / (hp - ls) if (p - s >= 5 and hp / ls - 1 >= 0.15) else np.nan
E['rr'] = [rr_at(a, b) for a, b in zip(ti, ji)]; E = E[~(E.rr >= 0.236)]
ti = dates.get_indexer(pd.to_datetime(E.date)); ji = np.array([cix[str(c)] for c in E.code])
E['steep'] = steep[ti, ji]; E['bear'] = below[ti]; E['S'] = S[ti, ji]
pd.to_pickle(E[['date', 'code', 'x20', 'steep', 'bear', 'S']], f'/home/claude/old/M088_{which}.pkl')
idx = np.arange(260, len(dates), 5)
R = pd.DataFrame({'date': dates[idx], 'bear': below[idx], 'up_share': [up[t][pool[t]].mean() if pool[t].any() else np.nan for t in idx], 'pos_share': [(np.nan_to_num(S[t], nan=-1) > 0)[pool[t]].mean() if pool[t].any() else np.nan for t in idx], 'U_pos': [(np.nan_to_num(S[t], nan=-1) > 0)[U[t]].mean() for t in idx]})
R.to_pickle(f'/home/claude/old/M088R_{which}.pkl'); print('saved', which, len(E), flush=True)
