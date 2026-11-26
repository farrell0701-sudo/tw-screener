# M-245：強勢池三支合成（R6＋散戶減少，DH 剔除）訊號層。定義見 registry_M.md M-245。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, M180
from L123 import daily_from_monthly
from M195 import HOLDOUT
from M107 import nw_t

ERAS = (('2010–15', '2010-01-01', '2015-12-31'), ('2016–24', '2016-01-01', '2024-11-24'))


def panel(parity):
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    signals.build('fix', mode='fix', P=P, F=F)
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; cols = dbg['cols']
    def full(a, fill):
        a = np.asarray(a)
        if a.shape[1] == N: return a
        o = np.full((T, N), fill, a.dtype); o[:, cols] = a; return o
    U = full(dbg['U'], False); pool = full(dbg['pool'], False); del S
    RKX = np.asarray(signals.load_sig('fix')['RKX'])
    VALID = np.asarray(F['VALID'])
    C = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill(); H = pd.DataFrame(np.asarray(F['HA'], np.float64)).ffill(); Cf = C.to_numpy()
    DH = (C / H.rolling(250, min_periods=200).max() - 1).to_numpy(); del H
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CX = np.cumsum(R - ew[:, None], axis=0); del R
    keys = P['meta']['security_key'].astype(str).to_numpy()
    par = (P['meta']['stock_id'].astype(str).str[-1].astype(int) % 2 == parity).to_numpy()
    r6 = daily_from_monthly(M180.r6_factors(dates)['R6 營收短長比'], dates, keys)
    B4 = np.load(LAB / 'cache' / 'M244' / 'B4.npy', mmap_mode='r')   # 先跑 lab/exp/M244_prep.py
    t0 = dates.searchsorted(pd.Timestamp('2010-01-01')); t1 = min(dates.searchsorted(HOLDOUT), T - 122); rows = []
    for t in range(t0, t1, 5):
        jj = np.nonzero(pool[t])[0]
        if len(jj) < 20: continue
        d = pd.DataFrame(dict(j=jj, r6=r6[t, jj], b4=np.asarray(B4[t, jj], np.float64), dh=DH[t, jj], x6=np.isfinite(RKX[t, jj]) & (RKX[t, jj] <= 6)))
        for c in ('r6', 'b4', 'dh'): d[c] = d[c].rank(pct=True)
        d['nb4'] = 1 - d.b4; d['Craw'] = (d.r6 + d.nb4) / 2
        d['C'] = np.where(d.dh < 0.2, np.nan, d.Craw)
        for h in (20, 60, 120): d[f'e{h}'] = CX[t + h, jj] - CX[t, jj]
        d['t'] = t; d['par'] = par[jj]; rows.append(d)
    D = pd.concat(rows, ignore_index=True); D['date'] = dates[D.t.to_numpy()]
    return D[D.par].drop(columns='par')


def main():
    D = panel(1); L = ['# M-245 結果：強勢池三支合成（R6＋散戶減少，DH 後 20% 剔除）（奇數代碼）\n']; res = {}
    for era, lo, hi in ERAS:
        E = D[(D.date >= lo) & (D.date <= hi)]
        L += [f'## {era}（{len(E):,} 筆、{E.t.nunique()} 期）\n', '| 因子 | IC20（t） | IC60（t） | IC120（t） |', '|---|---|---|---|']
        r = {}
        for f in ('C', 'Craw', 'r6', 'nb4'):
            o = {}
            for h, lag in ((20, 4), (60, 12), (120, 24)):
                s = E.dropna(subset=[f]).groupby('t').apply(lambda g: g[f].corr(g[f'e{h}'], method='spearman') if len(g) >= 10 else np.nan).dropna()
                o[h] = (s.mean(), nw_t(s, lag))
            r[f] = o
            L.append(f"| {f} | " + ' | '.join(f'{o[h][0]:+.3f}（{o[h][1]:.1f}）' for h in (20, 60, 120)) + ' |')
        X = E.dropna(subset=['C']).copy()
        X['q'] = X.groupby('t').C.transform(lambda s: pd.qcut(s.rank(method='first'), 5, labels=False) + 1 if len(s) >= 5 else np.nan)
        L += ['', '| C 五分位（Q5＝最高） | 60日超額 | 120日超額 | 60日右尾率（≥+30%） | 60日左尾率（≤−20%） | 與 X1 前 6 名重疊 |', '|---|---|---|---|---|---|']
        for q, g in X.dropna(subset=['q']).groupby('q'):
            L.append(f"| Q{int(q)} | {g.e60.mean():+.2%} | {g.e120.mean():+.2%} | {(g.e60 >= 0.3).mean():.1%} | {(g.e60 <= -0.2).mean():.1%} | {g.x6.mean():.0%} |")
        L.append(f"| 全體（剔除後） | {X.e60.mean():+.2%} | {X.e120.mean():+.2%} | {(X.e60 >= 0.3).mean():.1%} | {(X.e60 <= -0.2).mean():.1%} | {X.x6.mean():.0%} |")
        dh = E[E.dh < 0.2]; L.append(f"| 被 DH 剔除者 | {dh.e60.mean():+.2%} | {dh.e120.mean():+.2%} | {(dh.e60 >= 0.3).mean():.1%} | {(dh.e60 <= -0.2).mean():.1%} | {dh.x6.mean():.0%} |\n")
        q5 = X[X.q == 5]
        r['top'] = q5.e60.mean() - X.e60.mean(); r['rt'] = ((q5.e60 >= 0.3).mean(), (X.e60 >= 0.3).mean()); res[era] = r
    v = list(res.values())
    c1 = all(x['C'][60][0] > 0 for x in v) and sorted(abs(x['C'][60][1]) for x in v)[1] >= 2 and sorted(abs(x['C'][60][1]) for x in v)[0] >= 1
    c2 = all(x['top'] > 0 for x in v)
    c3 = any(x['C'][60][0] >= max(x['r6'][60][0], x['nb4'][60][0]) for x in v)
    c4 = all(x['rt'][0] >= x['rt'][1] - 0.01 for x in v)
    nom = c1 and c2 and c3 and c4
    head = f"三支合成：{'提名' if nom else '不提名'}（IC60 {'✓' if c1 else '✗'}、頭組超額 {'✓' if c2 else '✗'}、合成加值 {'✓' if c3 else '✗'}、右尾 {'✓' if c4 else '✗'}）"
    return dict(summary='\n'.join(L[:1] + [f'**{head}**\n'] + L[1:]), headline=head, n_variants=0)
