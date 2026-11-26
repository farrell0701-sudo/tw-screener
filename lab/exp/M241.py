# M-241：M-240 兩項發現的偶數代碼確認。定義見 registry_M.md M-241。
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import M240
from M195 import ERAS
from M107 import nw_t


def main():
    D = M240.build(0); L = ['# M-241 結果：M-240 兩項發現的偶數代碼確認\n']; ra, rb = [], []
    msr = M240.msr
    for era, lo, hi in ERAS:
        E = D[(D.date >= lo) & (D.date <= hi)].copy(); ny = (pd.Timestamp(hi) - pd.Timestamp(lo)).days / 365.25
        E['lft'] = (E.n20 <= -0.10).astype(float); E['rgt'] = (E.n20 >= 0.20).astype(float)
        A, C, Dd = E, E[E.qt], E[E.hi & E.qt]
        G = {'A 全部突破': A, 'C 突破＋量縮': C, 'D 突破＋貼高＋量縮': Dd, 'A｜營收低': A[A.rv == '低'], 'A｜營收高': A[A.rv == '高'], 'D｜營收低': Dd[Dd.rv == '低'], 'D｜營收高': Dd[Dd.rv == '高']}
        L += [f'## {era}\n', '| 組 | 每年事件 | 20日 | 60日 | 120日 | 20日淨勝率 | 左尾率 | 右尾率 | 高峰中位日 |', '|---|---|---|---|---|---|---|---|---|']
        for nm, g in G.items():
            L.append(f"| {nm} | {len(g) / ny:.0f} | {g.e20.mean():+.2%} | {g.e60.mean():+.2%} | {g.e120.mean():+.2%} | {(g.n20 > 0).mean():.0%} | {g.lft.mean():.1%} | {g.rgt.mean():.1%} | {g.pk.median():.0f} |")
        dl = (msr(A, 'lft') - msr(C, 'lft')).dropna()
        ra.append(dict(dl=A.lft.mean() - C.lft.mean(), t=nw_t(dl, 1), dr=C.rgt.mean() - A.rgt.mean(), de=C.e20.mean() - A.e20.mean()))
        o = {}
        for h, lag in ((60, 3), (120, 6)):
            x = (msr(G['A｜營收高'], f'e{h}') - msr(G['A｜營收低'], f'e{h}')).dropna(); o[h] = (x.mean(), nw_t(x, lag))
        y = {h: (msr(G['D｜營收高'], f'e{h}') - msr(G['D｜營收低'], f'e{h}')).dropna() for h in (20, 60, 120)}
        rb.append(o)
        L += ['', f"(a) 左尾率 A − C {ra[-1]['dl']:+.1%}（月序列 t {ra[-1]['t']:.1f}）；右尾率 C − A {ra[-1]['dr']:+.1%}；20 日超額 C − A {ra[-1]['de']:+.2%}",
              '(b) A 內營收高 − 低：' + '；'.join(f"{h} 日 {o[h][0]:+.2%}（t {o[h][1]:.1f}）" for h in (60, 120)),
              'D 內營收高 − 低：' + '；'.join(f"{h} 日 {y[h].mean():+.2%}（t {nw_t(y[h], {20: 1, 60: 3, 120: 6}[h]):.1f}）" for h in (20, 60, 120)), '']
    ts = sorted(abs(r['t']) for r in ra)
    pa = all(r['dl'] >= 0.02 for r in ra) and all(r['t'] > 0 for r in ra) and ts[1] >= 2 and ts[0] >= 1 and all(r['dr'] >= -0.01 for r in ra) and all(r['de'] >= -0.003 for r in ra)
    def okh(h):
        m = [r[h] for r in rb]; tt = sorted(abs(t) for _, t in m)
        return all(a > 0 for a, _ in m) and tt[1] >= 2 and tt[0] >= 1
    pb = okh(60) or okh(120)
    head = f"(a) 量縮降左尾不降右尾：{'通過' if pa else '不通過'}；(b) 突破＋營收高延續：{'通過' if pb else '不通過'}（60 日 {'✓' if okh(60) else '✗'}、120 日 {'✓' if okh(120) else '✗'}）"
    return dict(summary='\n'.join(L[:1] + [f'**{head}**\n'] + L[1:]), headline=head, n_variants=0)
