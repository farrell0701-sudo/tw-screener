# M-199：M-197B 擬合模型（係數凍結）套用到偶數代碼。定義見 registry_M.md M-199。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'exp'))
import M197
from M195 import ERAS
from M107 import nw_t

FE = ['DEPTH', 'DUR', 'RETR', 'VOLPB', 'ATRC', 'DTLD', 'EMA20D', 'EMA60D', 'CLV']
BETA = dict(c=-0.0027, DEPTH=-0.0070, DUR=0.0000, RETR=0.0066, VOLPB=-0.0068, ATRC=-0.0034, DTLD=-0.0096, EMA20D=-0.0106, EMA60D=0.0236, CLV=-0.0013, MKT=-0.0013)


def main():
    D, A0 = M197.build(parity=0, a0=True)
    for f in FE: D[f + '_p'] = D.groupby('t')[f].rank(pct=True)
    D = D.dropna(subset=[f + '_p' for f in FE]).copy()
    D['p'] = BETA['c'] + sum(BETA[f] * (D[f + '_p'] - 0.5) for f in FE) + BETA['MKT'] * D.MKT
    L = ['# M-199 結果：M-197B 凍結模型套用到偶數代碼\n']
    res = {}
    for era, lo, hi in ERAS:
        E = D[(D.date >= lo) & (D.date <= hi)].copy(); a0 = A0[(A0.date >= lo) & (A0.date <= hi)].e20.mean()
        ic = E.groupby('t').apply(lambda g: g.p.rank().corr(g.e20.rank()) if len(g) >= 8 else np.nan).dropna()
        E['q'] = E.groupby('t').p.transform(lambda s: pd.qcut(s.rank(method='first'), 5, labels=False) + 1 if len(s) >= 5 else np.nan)
        N1 = E.sort_values('t').drop_duplicates(['j', 'h'], keep='first')
        top = N1[N1.q == 5]
        res[era] = dict(ic=ic.mean(), t=nw_t(ic, 7), d=top.e20.mean() - a0, lt=E[E.q == 5].left.mean(), lp=E.left.mean())
        L += [f"## {era}（{len(E):,} 筆；不重疊 {len(N1):,} 段；A0 20 日超額 {a0:+.2%}）\n", f"IC {ic.mean():+.3f}（t {nw_t(ic, 7):.1f}）\n",
              '| 預測組 | 重疊：20日超額 | 重疊：淨勝率 | 不重疊：筆數 | 不重疊：20日超額 | 不重疊：淨 平均／中位／勝率 | 左尾 |', '|---|---|---|---|---|---|---|']
        for qq in range(1, 6):
            y = E[E.q == qq]; z = N1[N1.q == qq]
            L.append(f"| Q{qq} | {y.e20.mean():+.2%} | {(y.n20 > 0).mean():.0%} | {len(z):,} | {z.e20.mean():+.2%} | {z.n20.mean():+.2%}／{z.n20.median():+.2%}／{(z.n20 > 0).mean():.0%} | {y.left.mean():.1%} |")
        L.append('')
    a = list(res.values()); ts = sorted(x['t'] for x in a)
    c1 = all(x['ic'] > 0 for x in a) and ts[1] >= 2 and ts[0] >= 1; c2 = all(x['d'] > 0.0077 for x in a); c3 = all(x['lt'] <= x['lp'] for x in a)
    vt = (f"判準：①IC {'✓' if c1 else '✗'}（" + '｜'.join(f"{x['ic']:+.3f} t {x['t']:.1f}" for x in a) + f"）；②不重疊頭組 − A0 > +0.77% {'✓' if c2 else '✗'}（" + '｜'.join(f"{x['d']:+.2%}" for x in a)
          + f"）；③頭組左尾 ≤ 母體 {'✓' if c3 else '✗'}（" + '｜'.join(f"{x['lt']:.1%} vs {x['lp']:.1%}" for x in a) + f"）→ {'通過確認集' if c1 and c2 and c3 else '確認集不通過'}\n")
    return dict(summary='\n'.join(L[:1] + [vt] + L[1:]), headline=vt.split('→')[-1].strip(), n_variants=0)
