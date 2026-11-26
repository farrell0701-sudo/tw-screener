# M-236：X1 候選內控制 S 與 DH250 後的 VWG 殘差 IC。定義見 registry_M.md M-236。
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import M200, M234
from M195 import ERAS
from M107 import nw_t


def resid_ic(D, y, ctrl, grp='t'):
    def f(g):
        if len(g) < 10: return np.nan
        a = g[y].rank(pct=True).to_numpy(); X = np.column_stack([np.ones(len(g))] + [g[c].rank(pct=True).to_numpy() for c in ctrl])
        r = a - X @ np.linalg.lstsq(X, a, rcond=None)[0]
        return pd.Series(r, index=g.index)
    D = D.dropna(subset=[y] + ctrl).copy(); D['r'] = D.groupby(grp, group_keys=False).apply(f)
    o = {}
    for h, lag in (('e20', 4), ('e60', 12)):
        s = D.groupby(grp).apply(lambda g: g.r.rank().corr(g[h].rank()) if len(g) >= 10 else np.nan).dropna()
        o[h] = (s.mean(), nw_t(s, lag))
    return o


def main():
    L = ['# M-236 結果：X1 候選內控制 S 與 DH250 後的 VWG 殘差 IC\n', '| 代碼 | 期間 | 因子 | 控制 | IC20（t） | IC60（t） |', '|---|---|---|---|---|---|']
    cells = []
    for par, pn in ((1, '奇數'), (0, '偶數')):
        D = M200.panel(par, extra=M234.extra, fac=M234.FAC)['X1 候選']
        for era, lo, hi in ERAS:
            E = D[(D.date >= lo) & (D.date <= hi)]
            for y, ctrl in (('VWG', ['S', 'DH250']), ('VWG|DH', ['S', 'DH250']), ('DH250', ['S', 'VWG']), ('VWG', ['S'])):
                o = resid_ic(E, y, ctrl)
                L.append(f"| {pn} | {era} | {y} | {'＋'.join(ctrl)} | {o['e20'][0]:+.3f}（{o['e20'][1]:.1f}） | {o['e60'][0]:+.3f}（{o['e60'][1]:.1f}） |")
                if y == 'VWG' and ctrl == ['S', 'DH250']: cells.append(o['e60'])
    neg1 = sum(1 for m, t in cells if m < 0 and abs(t) >= 1); neg2 = sum(1 for m, t in cells if m < 0 and abs(t) >= 2)
    ok = neg1 >= 3 and neg2 >= 2
    head = f"VWG 控制 DH 後：{'有獨立資訊' if ok else '主要是 DH 反影'}（負且 |t|≥1：{neg1}/4，|t|≥2：{neg2}/4）"
    return dict(summary='\n'.join(L[:1] + [f'**{head}**\n'] + L[1:]), headline=head, n_variants=0)
