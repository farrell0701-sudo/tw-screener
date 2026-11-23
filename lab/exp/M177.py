# M-177：個股多空點能否當營收池的進場確認（奇數代碼探索集）。定義見 registry_M.md M-177。
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

PARITY = 1


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    keys = P['meta']['security_key'].astype(str).to_numpy()
    par = (P['meta']['stock_id'].astype(str).str[-1].astype(int) % 2 == PARITY).to_numpy()
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; U, pool, Tr = dbg['U'], dbg['pool'], dbg['Tr']; cols = dbg['cols']; del S
    if Tr.shape[1] != N: o = np.full((T, N), np.nan); o[:, cols] = Tr; Tr = o
    dT = Tr - np.vstack([np.full((10, N), np.nan), Tr[:-10]])
    VALID = np.asarray(F['VALID']); CA = np.asarray(F['CA'], np.float64); LA = np.asarray(F['LA'], np.float64)
    CAv = np.where(VALID, CA, np.nan)
    Cf = pd.DataFrame(np.where(VALID, CA, np.nan)).ffill().to_numpy()
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CX = np.cumsum(R - ew[:, None], axis=0)
    A = M165.layers(U, pool, dates, keys)[0][1]
    Q5 = fin_factors(dates)['Q5 毛利率創高']; qj = pd.Index(Q5.columns.astype(str)).get_indexer(keys)
    rows = []; SHARE = []
    for m, (t, _) in M165.samp_prev(dates).items():
        if dates[t] < pd.Timestamp('2005-01-01') or t + 121 >= T or not A[t].any(): continue
        mem = np.nonzero(A[t])[0]
        q = np.where(qj[mem] >= 0, Q5.loc[m].to_numpy(float)[np.clip(qj[mem], 0, None)], np.nan) if m in Q5.index else np.full(len(mem), np.nan)
        fin = np.isfinite(q); drop = np.zeros(len(mem), bool)
        if fin.sum() >= 20: drop[fin] = q[fin] <= np.nanpercentile(q[fin], 20)
        s = mem[~drop & U[t, mem]]
        if len(s) < 20: continue
        sc = 0.5 * pd.Series(Tr[t, s]).rank(pct=True).fillna(0).to_numpy() + 0.5 * pd.Series(dT[t, s]).rank(pct=True).fillna(0).to_numpy()
        s0 = dates.searchsorted((dates[t - 1].to_period('M') - 1).start_time); w = CAv[s0:t]
        with np.errstate(all='ignore'): hh, ll = np.nanmax(w[:, s], 0), np.nanmin(w[:, s], 0)
        cc = pd.DataFrame(CAv[max(0, t - 6):t, s]).ffill().to_numpy()[-1]
        up = np.isfinite(hh) & np.isfinite(cc) & (cc > (hh + ll) / 2)
        jj = s[par[s]]; scj = sc[par[s]]; upj = up[par[s]]
        if len(jj) < 10: continue
        order = jj[np.argsort(-scj, kind='stable')]
        cj = jj[upj]; orderc = cj[np.argsort(-scj[upj], kind='stable')]
        era = '2005–15' if dates[t] <= pd.Timestamp('2015-12-31') else '2016–26'
        e60u = (CX[t + 60] - CX[t])[U[t]]; q10, q90 = np.nanpercentile(e60u, [10, 90])
        top6 = order[:6]; SHARE.append((era, float(np.isin(top6, cj).mean())))
        groups = {'G1 全池': jj, 'G2 已確認': cj, 'G3 未確認': jj[~upj], 'G4 X1 前 6': top6, 'G5 確認後 X1 前 6': orderc[:6]}
        for g, sel in groups.items():
            for j in sel:
                with np.errstate(all='ignore'): mdd = np.nanmin(LA[t + 1:t + 61, j]) / Cf[t, j] - 1
                rows.append((str(m), era, g, CX[t + 20, j] - CX[t, j], CX[t + 60, j] - CX[t, j], CX[t + 120, j] - CX[t, j], q10, q90, mdd))
    D = pd.DataFrame(rows, columns=['m', 'era', 'g', 'e20', 'e60', 'e120', 'q10', 'q90', 'mdd'])
    D['L'] = D.e60 <= D.q10; D['Rt'] = D.e60 >= D.q90
    L = ['# M-177 結果：個股多空點當營收池進場確認（奇數代碼）\n',
         '| 組 | 期間 | 筆數 | 20 日 | 60 日 | 120 日 | 60 日中位數 | 勝率 | 左尾 | 右尾 | 60 日最大跌幅 | 曾跌破 −15% |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
    st = {}
    for era in ('2005–15', '2016–26'):
        for g in ('G1 全池', 'G2 已確認', 'G3 未確認', 'G4 X1 前 6', 'G5 確認後 X1 前 6'):
            x = D[(D.era == era) & (D.g == g)]
            st[(era, g)] = x
            L.append(f"| {g} | {era} | {len(x)} | {x.e20.mean():+.2%} | {x.e60.mean():+.2%} | {x.e120.mean():+.2%} | {x.e60.median():+.2%} | {(x.e60 > 0).mean():.0%} | {x.L.mean():.0%} | {x.Rt.mean():.0%} | {x.mdd.mean():+.1%} | {(x.mdd <= -0.15).mean():.0%} |")
    def diff(era, a, b):
        ga = st[(era, a)].groupby('m').e60.mean(); gb = st[(era, b)].groupby('m').e60.mean(); d = (ga - gb).dropna()
        return d.mean(), nw_t(d.to_numpy(), 3)
    res = {}
    L += ['', '| 比較 | 期間 | 60 日差（NW t） | 左尾 a／b | 右尾 a／b |', '|---|---|---|---|---|']
    for nm, a, b in (('池子層 G2 − G1', 'G2 已確認', 'G1 全池'), ('子策略層 G5 − G4', 'G5 確認後 X1 前 6', 'G4 X1 前 6')):
        for era in ('2005–15', '2016–26'):
            dm, tt = diff(era, a, b); xa, xb = st[(era, a)], st[(era, b)]
            res[(nm, era)] = (dm, tt, xa.L.mean(), xb.L.mean(), xa.Rt.mean(), xb.Rt.mean())
            L.append(f"| {nm} | {era} | {dm:+.2%}（{tt:.1f}） | {xa.L.mean():.0%}／{xb.L.mean():.0%} | {xa.Rt.mean():.0%}／{xb.Rt.mean():.0%} |")
    def judge(nm, ltol, rtol):
        a, b = res[(nm, '2005–15')], res[(nm, '2016–26')]
        c1 = a[0] > 0 and b[0] > 0 and max(a[1], b[1]) >= 2 and min(a[1], b[1]) >= 1
        c2 = a[2] <= a[3] + ltol and b[2] <= b[3] + ltol
        c3 = a[4] >= a[5] + rtol and b[4] >= b[5] + rtol
        return c1 and c2 and c3, (c1, c2, c3)
    ok1, f1 = judge('池子層 G2 − G1', 0.01, 0.0); ok2, f2 = judge('子策略層 G5 − G4', 0.01, -0.01)
    sh = pd.DataFrame(SHARE, columns=['era', 's']).groupby('era').s.mean()
    mk = lambda f: '／'.join('✓' if x else '✗' for x in f)
    L.insert(1, f"**判準 1 池子層：{'通過' if ok1 else '不通過'}**（{mk(f1)}）；**判準 2 子策略層：{'通過' if ok2 else '不通過'}**（{mk(f2)}）；X1 前 6 中已確認的比例：" + '、'.join(f'{e} {v:.0%}' for e, v in sh.items()) + '\n')
    return dict(summary='\n'.join(L), headline=f"多空點進場確認：池子層{'過' if ok1 else '不過'}、子策略層{'過' if ok2 else '不過'}", n_variants=0)
