# M-172：營收池內依 X1 分數排序——偶數代碼確認集（只測一次）。定義見 registry_M.md M-172。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals
from signals import prk
from M122 import fin_factors
from M107 import nw_t
import M165

PARITY = 0


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    keys = P['meta']['security_key'].astype(str).to_numpy()
    par = (P['meta']['stock_id'].astype(str).str[-1].astype(int) % 2 == PARITY).to_numpy()
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; U, pool, Tr = dbg['U'], dbg['pool'], dbg['Tr']; cols = dbg['cols']; del S
    if Tr.shape[1] != N: o = np.full((T, N), np.nan); o[:, cols] = Tr; Tr = o
    dT = Tr - np.vstack([np.full((10, N), np.nan), Tr[:-10]])
    VALID = np.asarray(F['VALID']); CA = np.asarray(F['CA'], np.float64)
    Cf = pd.DataFrame(np.where(VALID, CA, np.nan)).ffill().to_numpy()
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CX = np.cumsum(R - ew[:, None], axis=0)
    A = M165.layers(U, pool, dates, keys)[0][1]
    Q5 = fin_factors(dates)['Q5 毛利率創高']; qj = pd.Index(Q5.columns.astype(str)).get_indexer(keys)
    rows = []
    for m, (t, _) in M165.samp_prev(dates).items():
        if dates[t] < pd.Timestamp('2005-01-01') or t + 121 >= T or not A[t].any(): continue
        mem = np.nonzero(A[t])[0]
        q = np.where(qj[mem] >= 0, Q5.loc[m].to_numpy(float)[np.clip(qj[mem], 0, None)], np.nan) if m in Q5.index else np.full(len(mem), np.nan)
        fin = np.isfinite(q); drop = np.zeros(len(mem), bool)
        if fin.sum() >= 20: drop[fin] = q[fin] <= np.nanpercentile(q[fin], 20)
        s = mem[~drop & U[t, mem]]
        if len(s) < 20: continue
        sc = 0.5 * pd.Series(Tr[t, s]).rank(pct=True).fillna(0).to_numpy() + 0.5 * pd.Series(dT[t, s]).rank(pct=True).fillna(0).to_numpy()
        jj = s[par[s]]; scj = sc[par[s]]
        if len(jj) < 10: continue
        order = jj[np.argsort(-scj, kind='stable')]
        era = '2005–15' if dates[t] <= pd.Timestamp('2015-12-31') else '2016–26'
        e60u = (CX[t + 60] - CX[t])[U[t]]; q10, q90 = np.nanpercentile(e60u, [10, 90])
        groups = {'全池': jj, '前 6': order[:6], '前 10（參考）': order[:10], '前 20%（參考）': order[:max(1, int(len(order) * 0.2))]}
        for g, sel in groups.items():
            for j in sel:
                rows.append((str(m), era, g, CX[t + 20, j] - CX[t, j], CX[t + 60, j] - CX[t, j], CX[t + 120, j] - CX[t, j], q10, q90))
    D = pd.DataFrame(rows, columns=['m', 'era', 'g', 'e20', 'e60', 'e120', 'q10', 'q90'])
    D['L'] = D.e60 <= D.q10; D['Rt'] = D.e60 >= D.q90
    L = ['# M-172 結果：營收池內 X1 分數排序｜偶數代碼確認集\n',
         '| 組 | 期間 | 筆數 | 20 日 | 60 日 | 120 日 | 60 日中位數 | 勝率 | 左尾 | 右尾 | 60 日 − 全池（NW t） | 120 日 − 全池 |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
    res = {}
    for era in ('2005–15', '2016–26'):
        base = D[(D.era == era) & (D.g == '全池')]; bm = base.groupby('m').e60.mean(); bm120 = base.groupby('m').e120.mean()
        for g in ('全池', '前 6', '前 10（參考）', '前 20%（參考）'):
            x = D[(D.era == era) & (D.g == g)]
            dif = (x.groupby('m').e60.mean() - bm).dropna(); d120 = (x.groupby('m').e120.mean() - bm120).dropna()
            tt = nw_t(dif.to_numpy(), 3) if g != '全池' else np.nan
            res[(era, g)] = dict(d=dif.mean(), t=tt, d120=d120.mean(), L=x.L.mean(), R=x.Rt.mean())
            L.append(f"| {g} | {era} | {len(x)} | {x.e20.mean():+.2%} | {x.e60.mean():+.2%} | {x.e120.mean():+.2%} | {x.e60.median():+.2%} | {(x.e60 > 0).mean():.0%} | {x.L.mean():.0%} | {x.Rt.mean():.0%} | "
                     + ('—' if g == '全池' else f"{dif.mean():+.2%}（{tt:.1f}）") + ' | ' + ('—' if g == '全池' else f'{d120.mean():+.2%}') + ' |')
    a, b = res[('2005–15', '前 6')], res[('2016–26', '前 6')]; pa, pb = res[('2005–15', '全池')], res[('2016–26', '全池')]
    c1 = a['d'] > 0 and b['d'] > 0 and max(a['t'], b['t']) >= 2.0 and min(a['t'], b['t']) >= 1.0
    c2 = a['d120'] > 0 and b['d120'] > 0
    c3 = a['R'] > pa['R'] and b['R'] > pb['R']
    c4 = a['L'] - pa['L'] <= 0.02 and b['L'] - pb['L'] <= 0.02
    ok = c1 and c2 and c3 and c4; mk = lambda f: '✓' if f else '✗'
    L.insert(1, f"**{'確認通過' if ok else '確認不通過'}**（判準 1／2／3／4：{mk(c1)}／{mk(c2)}／{mk(c3)}／{mk(c4)}）\n")
    return dict(summary='\n'.join(L), headline=f"營收池 X1 分數排序確認集{'通過' if ok else '不通過'}", n_variants=0)
