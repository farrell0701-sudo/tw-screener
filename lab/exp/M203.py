# M-203：第三個帳戶——營收子策略與 X1、V4.1 的相關。定義見 registry_M.md M-203。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
import sim, evalu as E
from L143 import mix3

NPATH = 100; END = '2024-11-13'
SEGS = (('2005–15', '2005-01-01', '2015-12-31'), ('2016–20', '2016-01-01', '2020-12-31'), ('2021–24/11', '2021-01-01', END), ('16–24/11', '2016-01-01', END))


def med(paths, lo, hi): return np.nanmedian(np.array([E.met(p.loc[lo:hi]) for p in paths]), axis=0)


def main():
    P = load_panel(); dates = P['dates']; end = int(dates.searchsorted(pd.Timestamp(END), side='right'))
    R = sim.run_paths('fix', NPATH, sig='fix', x_kw=dict(end=end), v_kw=dict(end=end))
    X = [R['X'][i][0].loc[:END] for i in range(NPATH)]; V = [R['V'][i][0].loc[:END] for i in range(NPATH)]
    RR = sim.run_paths('fix', NPATH, sig='L131_rev', x_kw=dict(end=end, n=6), v_kw=dict(end=end))
    G = [RR['X'][i][0].loc[:END] for i in range(NPATH)]
    mx, mg, mv = E.path_mean(X), E.path_mean(G), E.path_mean(V)
    L = ['# M-203 結果：第三個帳戶——營收子策略（L-131 規則）與 X1、V4.1（100 路徑；到 2024-11-13）\n', '## 各帳戶單獨（CAGR／最大回撤／Calmar，路徑中位數）\n',
         '| 帳戶 | ' + ' | '.join(s[0] for s in SEGS) + ' |', '|---|---|---|---|---|']
    for nm, Z in (('X1', X), ('營收子策略', G), ('V4.1', V)):
        L.append(f'| {nm} | ' + ' | '.join('{:+.1%}／{:.1%}／{:.2f}'.format(*med(Z, lo, hi)[:3]) for _, lo, hi in SEGS) + ' |')
    L += ['', '## 相關（路徑平均日報酬｜月報酬）\n', '| 配對 | 2005–15 | 2016–24/11 |', '|---|---|---|']
    for a, b, nm in ((mx, mg, 'X1 × 營收'), (mx, mv, 'X1 × V4.1'), (mg, mv, '營收 × V4.1')):
        c = []
        for lo, hi in (('2005', '2015'), ('2016', END)):
            x, y = a.loc[lo:hi], b.loc[lo:hi]; xm, ym = (1 + x).resample('ME').prod() - 1, (1 + y).resample('ME').prod() - 1
            c.append(f'{x.corr(y):.2f}｜{xm.corr(ym):.2f}')
        L.append(f'| {nm} | {c[0]} | {c[1]} |')
    base = sim.mix(mx, mv); bm = (1 + base).resample('ME').prod() - 1; gm = (1 + mg).resample('ME').prod() - 1
    w20 = bm.nsmallest(20).index
    L += ['', f"基準（X1 60／V 40）最差 20 個月中，營收子策略同月也虧損的有 {(gm.loc[w20] < 0).sum()} 個；這 20 個月基準平均 {bm.loc[w20].mean():+.1%}、營收子策略平均 {gm.loc[w20].mean():+.1%}\n",
          '## 三帳戶（X1／營收／V4.1，每月再平衡；路徑中位數 CAGR／最大回撤／Calmar）\n', '| 組合 | ' + ' | '.join(s[0] for s in SEGS) + ' |', '|---|---|---|---|---|']
    for wx, wg in ((0.6, 0.0), (0.5, 0.1), (0.4, 0.2), (0.3, 0.3), (0.4, 0.3)):
        B = [mix3(X[i], G[i], V[i], wx, wg) for i in range(NPATH)]
        L.append(f'| {wx:.0%}／{wg:.0%}／{1 - wx - wg:.0%} | ' + ' | '.join('{:+.1%}／{:.1%}／{:.2f}'.format(*med(B, lo, hi)[:3]) for _, lo, hi in SEGS) + ' |')
    return dict(summary='\n'.join(L), headline='第三帳戶相關描述', n_variants=0)
