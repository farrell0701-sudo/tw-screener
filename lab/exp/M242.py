# M-242：X1 候選內 QUIET 殘差。定義見 registry_M.md M-242。
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import M200, M232
from M236 import resid_ic
from M195 import ERAS

FAC = ['T_ld', 'R6', 'VC', 'VX', 'DH250']


def extra(P, F, Tr, U):
    return {'DH250': M232.extra(P, F, Tr, U)['DH250']}


def main():
    D = M200.panel(1, extra=extra, fac=FAC)['X1 候選'].dropna(subset=['VC', 'VX', 'DH250']).copy()
    D['QUIET'] = -(D.groupby('t').VC.rank(pct=True) + D.groupby('t').VX.rank(pct=True)) / 2
    L = ['# M-242 結果：X1 候選內安靜度（QUIET）殘差（奇數代碼）\n']; cells = []; rt = []
    for era, lo, hi in ERAS:
        E = D[(D.date >= lo) & (D.date <= hi)].copy(); E6 = E.dropna(subset=['R6'])
        a = resid_ic(E, 'QUIET', ['S']); b = resid_ic(E6, 'QUIET', ['S', 'DH250', 'R6'])
        cells.append(b)
        L += [f'## {era}（{len(E):,} 筆；有 R6 {len(E6):,}）\n', '| 控制 | IC20（t） | IC60（t） |', '|---|---|---|',
              f"| S | {a['e20'][0]:+.3f}（{a['e20'][1]:.1f}） | {a['e60'][0]:+.3f}（{a['e60'][1]:.1f}） |",
              f"| S＋DH250＋R6 | {b['e20'][0]:+.3f}（{b['e20'][1]:.1f}） | {b['e60'][0]:+.3f}（{b['e60'][1]:.1f}） |", '']
        E['q'] = E.groupby('t').QUIET.transform(lambda s: pd.qcut(s.rank(method='first'), 5, labels=False) + 1 if len(s) >= 5 else np.nan)
        allr = (E.n20 >= 0.20).mean(); L += ['| 組（Q5＝最安靜） | 20日超額 | 60日超額 | 20日淨勝率 | 左尾率 | 右尾率 |', '|---|---|---|---|---|---|']
        for q, g in E.dropna(subset=['q']).groupby('q'):
            L.append(f"| Q{int(q)} | {g.e20.mean():+.2%} | {g.e60.mean():+.2%} | {(g.n20 > 0).mean():.0%} | {(g.n20 <= -0.10).mean():.1%} | {(g.n20 >= 0.20).mean():.1%} |")
        L.append(f"| 全體 | {E.e20.mean():+.2%} | {E.e60.mean():+.2%} | {(E.n20 > 0).mean():.0%} | {(E.n20 <= -0.10).mean():.1%} | {allr:.1%} |\n")
        rt.append((E[E.q == 5].n20 >= 0.20).mean() >= allr - 0.01)
    ok = False; why = []
    for h in ('e20', 'e60'):
        m = [c[h] for c in cells]; ts = sorted(abs(t) for _, t in m)
        if np.sign(m[0][0]) == np.sign(m[1][0]) and ts[1] >= 2 and ts[0] >= 1: ok = True; why.append(h[1:])
    nom = ok and all(rt)
    head = f"QUIET 殘差：{'提名' if nom else '不提名'}（IC 條件 {'✓ ' + '、'.join(why) + ' 日' if ok else '✗'}；右尾條件 {'✓' if all(rt) else '✗'}）"
    return dict(summary='\n'.join(L[:1] + [f'**{head}**\n'] + L[1:]), headline=head, n_variants=0)
