# M-237：離一年高點遠 vs 長期均線下彎。定義見 registry_M.md M-237。
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import M200
from M236 import resid_ic
from M195 import ERAS

FAC = ['T_ld', 'DH250', 'SL60', 'SL120', 'SL250']


def extra(P, F, Tr, U):
    C = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill(); H = pd.DataFrame(np.asarray(F['HA'], np.float64)).ffill()
    o = {'DH250': (C / H.rolling(250, min_periods=200).max() - 1).to_numpy()}
    for k in (60, 120, 250):
        ma = C.rolling(k, min_periods=int(k * 0.8)).mean(); o[f'SL{k}'] = (ma / ma.shift(20) - 1).to_numpy()
    return o


def main():
    D = M200.panel(1, extra=extra, fac=FAC)['X1 候選']
    L = ['# M-237 結果：離一年高點遠 vs 長期均線下彎（X1 候選、奇數代碼）\n']
    verdict = []
    for era, lo, hi in ERAS:
        E = D[(D.date >= lo) & (D.date <= hi)].dropna(subset=['DH250', 'SL60', 'SL120', 'SL250']).copy()
        E['q'] = E.groupby('t').DH250.transform(lambda s: pd.qcut(s.rank(method='first'), 5, labels=False) + 1 if len(s) >= 5 else np.nan)
        E = E.dropna(subset=['q'])
        cm = E[['DH250', 'SL60', 'SL120', 'SL250']].rank().corr()
        L += [f'## {era}（{len(E):,} 筆）\n', 'Spearman（DH250 與）：' + '；'.join(f'{k} {cm.loc["DH250", k]:+.2f}' for k in ('SL60', 'SL120', 'SL250')) + '\n',
              '### DH250 五分位的均線下彎比例（Q1＝離高點最遠）\n', '| DH 組 | 筆數 | DH250 中位 | SL60 下彎 | SL120 下彎 | SL250 下彎 | 60 日超額 |', '|---|---|---|---|---|---|---|']
        for q, g in E.groupby('q'):
            L.append(f"| Q{int(q)} | {len(g):,} | {g.DH250.median():+.1%} | {(g.SL60 < 0).mean():.0%} | {(g.SL120 < 0).mean():.0%} | {(g.SL250 < 0).mean():.0%} | {g.e60.mean():+.2%} |")
        L += ['', '### 增量 IC（控制後殘差）\n', '| 因子 | 控制 | IC20（t） | IC60（t） |', '|---|---|---|---|']
        base = resid_ic(E, 'DH250', ['S'])
        L.append(f"| DH250 | S | {base['e20'][0]:+.3f}（{base['e20'][1]:.1f}） | {base['e60'][0]:+.3f}（{base['e60'][1]:.1f}） |")
        for k in ('SL60', 'SL120', 'SL250'):
            a = resid_ic(E, 'DH250', ['S', k]); b = resid_ic(E, k, ['S', 'DH250']); c = resid_ic(E, k, ['S'])
            L += [f"| DH250 | S＋{k} | {a['e20'][0]:+.3f}（{a['e20'][1]:.1f}） | {a['e60'][0]:+.3f}（{a['e60'][1]:.1f}）＝保留 {a['e60'][0] / base['e60'][0]:.0%} |",
                  f"| {k} | S | {c['e20'][0]:+.3f}（{c['e20'][1]:.1f}） | {c['e60'][0]:+.3f}（{c['e60'][1]:.1f}） |",
                  f"| {k} | S＋DH250 | {b['e20'][0]:+.3f}（{b['e20'][1]:.1f}） | {b['e60'][0]:+.3f}（{b['e60'][1]:.1f}） |"]
            verdict.append(dict(era=era, k=k, keep=a['e60'][0] / base['e60'][0], sl=b['e60']))
        L += ['', '### 2×2：離高點遠（DH 後 20%）× SL120 下彎\n', '| 組 | 筆數占比 | 60 日超額 | 20 日淨勝率 | 左尾率 |', '|---|---|---|---|---|']
        for nm, m in (('遠＋下彎', (E.q == 1) & (E.SL120 < 0)), ('遠＋上彎', (E.q == 1) & (E.SL120 >= 0)), ('近＋下彎', (E.q > 1) & (E.SL120 < 0)), ('近＋上彎', (E.q > 1) & (E.SL120 >= 0))):
            g = E[m]; L.append(f"| {nm} | {len(g) / len(E):.0%} | {g.e60.mean():+.2%} | {(g.n20 > 0).mean():.0%} | {(g.n20 <= -0.10).mean():.1%} |")
        L.append('')
    V = pd.DataFrame(verdict)
    keep_all = all(V.groupby('k').keep.min() >= 0.7); mostly = any(V.groupby('k').keep.max() < 0.4)
    nom = [k for k, g in V.groupby('k') if np.sign(g.sl.iloc[0][0]) == np.sign(g.sl.iloc[1][0]) and sorted(abs(x[1]) for x in g.sl)[1] >= 2 and sorted(abs(x[1]) for x in g.sl)[0] >= 1]
    head = ('離高點遠不是均線下彎的代稱' if keep_all else ('主要是均線下彎' if mostly else '部分重疊')) + f"；均線斜率提名：{'、'.join(nom) or '無'}"
    return dict(summary='\n'.join(L[:1] + [f'**{head}**（DH 保留比例：' + '；'.join(f"{r.era} {r.k} {r.keep:.0%}" for r in V.itertuples()) + '）\n'] + L[1:]), headline=head, n_variants=0)
