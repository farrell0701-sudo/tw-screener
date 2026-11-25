# M-198：DTLD 確認集（偶數代碼、不重疊、殘差化、超過成本）。定義見 registry_M.md M-198。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'exp'))
import M197
from M195 import ERAS
from M107 import nw_t


def main():
    D, A0 = M197.build(parity=0, a0=True)
    D = D.sort_values('t').drop_duplicates(['j', 'h'], keep='first').dropna(subset=['DTLD', 'DEPTH', 'EMA20D', 'R5', 'R20', 'TLD']).copy()
    D['m'] = D.date.dt.to_period('M')
    def resid(g):
        if len(g) < 30: return pd.Series(np.nan, index=g.index)
        y = g.DTLD.rank(pct=True).to_numpy(); X = np.column_stack([np.ones(len(g))] + [g[c].rank(pct=True) for c in ('DEPTH', 'EMA20D', 'R5', 'R20', 'TLD')])
        b = np.linalg.lstsq(X, y, rcond=None)[0]; return pd.Series(y - X @ b, index=g.index)
    D['RES'] = D.groupby('m', group_keys=False).apply(resid)
    L = ['# M-198 結果：DTLD 確認集（偶數代碼、每段回檔只取第一筆、殘差化）\n',
         '殘差＝每月橫斷面 DTLD 百分位對 DEPTH、EMA20D、R5、R20、T_ld 水準百分位回歸後的殘差。\n']
    res = {}
    for era, lo, hi in ERAS:
        E = D[(D.date >= lo) & (D.date <= hi)]; a0 = A0[(A0.date >= lo) & (A0.date <= hi)].e20.mean()
        L += [f"## {era}（{len(E):,} 段回檔；A0 20 日超額 {a0:+.2%}；回檔母體 20 日超額 {E.e20.mean():+.2%}、淨勝率 {(E.n20 > 0).mean():.0%}、左尾 {E.left.mean():.1%}）\n",
              '| 變數 | IC20（t） | Q1 筆數 | Q1 20日超額 | Q1 − A0 | Q1 淨 平均／中位／勝率 | Q1 左尾 | Q5 20日超額 | Q5 淨勝率 | Q5 左尾 |', '|' + '---|' * 10]
        for v in ('RES', 'DTLD'):
            x = E.dropna(subset=[v]).copy()
            ic = x.groupby('m').apply(lambda g: g[v].rank().corr(g.e20.rank()) if len(g) >= 20 else np.nan).dropna()
            x['q'] = x.groupby('m')[v].transform(lambda s: pd.qcut(s.rank(method='first'), 5, labels=False) + 1 if len(s) >= 10 else np.nan)
            q1, q5 = x[x.q == 1], x[x.q == 5]
            res[(v, era)] = dict(ic=ic.mean(), t=nw_t(ic, 1), d=q1.e20.mean() - a0, l1=q1.left.mean(), lp=E.left.mean())
            L.append(f"| {'殘差 DTLD' if v == 'RES' else '原始 DTLD'} | {ic.mean():+.3f}（{nw_t(ic, 1):.1f}） | {len(q1):,} | {q1.e20.mean():+.2%} | {q1.e20.mean() - a0:+.2%} | "
                     f"{q1.n20.mean():+.2%}／{q1.n20.median():+.2%}／{(q1.n20 > 0).mean():.0%} | {q1.left.mean():.1%} | {q5.e20.mean():+.2%} | {(q5.n20 > 0).mean():.0%} | {q5.left.mean():.1%} |")
        L.append('')
    a = [res[('RES', e)] for e, _, _ in ERAS]; ts = sorted(x['t'] for x in a)
    c1 = all(x['ic'] < 0 for x in a) and ts[0] <= -2 and ts[1] <= -1
    c2 = all(x['d'] > 0.0077 for x in a); c3 = all(x['l1'] <= x['lp'] for x in a)
    vt = (f"判準：①殘差 IC20 兩期負、t ≤ −2／≤ −1 {'✓' if c1 else '✗'}（" + '｜'.join(f"{x['ic']:+.3f} t {x['t']:.1f}" for x in a) + f"）；②最低組 − A0 > +0.77% {'✓' if c2 else '✗'}（"
          + '｜'.join(f"{x['d']:+.2%}" for x in a) + f"）；③最低組左尾 ≤ 母體 {'✓' if c3 else '✗'}（" + '｜'.join(f"{x['l1']:.1%} vs {x['lp']:.1%}" for x in a) + f"）→ {'通過確認集' if c1 and c2 and c3 else '確認集不通過，結案'}\n")
    return dict(summary='\n'.join(L[:2] + [vt] + L[2:]), headline=vt.split('→')[-1].strip(), n_variants=0)
