# M-238：急漲後回落合成 vs DH250。定義見 registry_M.md M-238。
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import M200, M234, M237
from M236 import resid_ic
from M195 import ERAS
from feat import load_feat

FAC = ['T_ld', 'DH250', 'VWG', 'SL250', 'S60']


def extra(P, F, Tr, U):
    a = M237.extra(P, F, Tr, U); C = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill(); A = pd.DataFrame(np.asarray(P['AMT'], np.float64)).fillna(0.0)
    vwap = A.rolling(250, min_periods=200).sum() / (A / C).rolling(250, min_periods=200).sum()
    return {'DH250': a['DH250'], 'SL250': a['SL250'], 'VWG': (C / vwap - 1).to_numpy(), 'S60': np.asarray(F['S60'], np.float64)}


def tails(E, col):
    E = E.copy(); E['q'] = E.groupby('t')[col].transform(lambda s: pd.qcut(s.rank(method='first'), 5, labels=False) + 1 if len(s) >= 5 else np.nan)
    rows = []
    for q, g in E.dropna(subset=['q']).groupby('q'):
        rows.append(f"| Q{int(q)} | {g.e60.mean():+.2%} | {(g.n20 > 0).mean():.0%} | {(g.n20 <= -0.10).mean():.1%} | {(g.n20 >= 0.20).mean():.1%} |")
    return rows


def main():
    D = M200.panel(1, extra=extra, fac=FAC)['X1 候選'].dropna(subset=['DH250', 'VWG', 'SL250', 'S60']).copy()
    rk = lambda c: D.groupby('t')[c].rank(pct=True)
    D['nDH'] = -D.DH250; D['CMP'] = ((1 - rk('DH250')) + rk('VWG') + rk('SL250') + rk('S60')) / 4
    L = ['# M-238 結果：急漲後回落合成（CMP）vs 單用 DH250（X1 候選、奇數代碼）\n']; V = []
    for era, lo, hi in ERAS:
        E = D[(D.date >= lo) & (D.date <= hi)]
        cm = E[['nDH', 'VWG', 'SL250', 'S60', 'CMP']].rank().corr()
        L += [f'## {era}（{len(E):,} 筆）\n', '候選內 Spearman：' + '；'.join(f'{a}–{b} {cm.loc[a, b]:+.2f}' for i, a in enumerate(['nDH', 'VWG', 'SL250', 'S60', 'CMP']) for b in ['nDH', 'VWG', 'SL250', 'S60', 'CMP'][i + 1:]) + '\n',
              '| 因子 | 控制 | IC20（t） | IC60（t） |', '|---|---|---|---|']
        r = {}
        for y, ctrl in (('nDH', ['S']), ('CMP', ['S']), ('CMP', ['S', 'DH250']), ('VWG', ['S', 'DH250']), ('SL250', ['S', 'DH250']), ('S60', ['S', 'DH250']), ('S60', ['S'])):
            o = resid_ic(E, y, ctrl); r[(y, tuple(ctrl))] = o
            L.append(f"| {y} | {'＋'.join(ctrl)} | {o['e20'][0]:+.3f}（{o['e20'][1]:.1f}） | {o['e60'][0]:+.3f}（{o['e60'][1]:.1f}） |")
        V.append(dict(era=era, dh=r[('nDH', ('S',))]['e60'][0], cmp=r[('CMP', ('S',))]['e60'][0], res=r[('CMP', ('S', 'DH250'))]['e60']))
        for nm, col in (('CMP（Q5＝最急漲回落）', 'CMP'), ('−DH250（Q5＝離高點最遠）', 'nDH')):
            L += ['', f'### 尾端：{nm}\n', '| 組 | 60 日超額 | 20 日淨勝率 | 左尾率 | 右尾率 |', '|---|---|---|---|---|'] + tails(E, col)
        L.append('')
    V = pd.DataFrame(V)
    c1 = all(abs(V.cmp) >= 1.1 * abs(V.dh)); rs = V.res.tolist()
    c2 = np.sign(rs[0][0]) == np.sign(rs[1][0]) and sorted(abs(x[1]) for x in rs)[1] >= 2 and sorted(abs(x[1]) for x in rs)[0] >= 1
    head = ('整合有價值' if c1 and c2 else '單用 DH250') + '（CMP/DH IC60：' + '；'.join(f"{r.era} {r.cmp:+.3f}/{r.dh:+.3f}" for r in V.itertuples()) + f"；CMP 扣 DH 殘差 t {rs[0][1]:.1f}／{rs[1][1]:.1f}）"
    return dict(summary='\n'.join(L[:1] + [f'**{head}**\n'] + L[1:]), headline=head, n_variants=0)
