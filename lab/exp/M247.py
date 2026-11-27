# M-247：上／下行 ATR 當進場因子（X1 候選）。定義見 registry_M.md M-247。
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import M200, M232
from M236 import resid_ic
from M195 import ERAS

FAC = ['T_ld', 'R6', 'VC', 'EXT', 'DH250', 'R20', 'S60', 'UDS', 'DATRp']


def updn(F):
    C = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill(); H = pd.DataFrame(np.asarray(F['HA'], np.float64)).ffill(); L = pd.DataFrame(np.asarray(F['LA'], np.float64)).ffill()
    pc = C.shift(1); up = (np.maximum(H, pc) - pc).clip(lower=0); dn = (pc - np.minimum(L, pc)).clip(lower=0)
    w = lambda x: x.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    return w(up).to_numpy(), w(dn).to_numpy(), C.to_numpy()


def extra(P, F, Tr, U):
    a = M232.extra(P, F, Tr, U); ua, da, c = updn(F)
    with np.errstate(all='ignore'): uds = ua / (ua + da); dp = da / c
    return {'DH250': a['DH250'].astype(np.float32), 'R20': a['R20'].astype(np.float32), 'S60': np.asarray(F['S60'], np.float32), 'UDS': uds.astype(np.float32), 'DATRp': dp.astype(np.float32)}


def main():
    D = M200.panel(1, extra=extra, fac=FAC)['X1 候選'].dropna(subset=['UDS', 'DATRp', 'DH250']).copy()
    L = ['# M-247 結果：上／下行 ATR 當進場因子（X1 候選，奇數代碼）\n']; res = {}
    for era, lo, hi in ERAS:
        E = D[(D.date >= lo) & (D.date <= hi)].copy(); E6 = E.dropna(subset=['R6'])
        cm = np.nanmean(np.stack([g[['UDS', 'DATRp', 'R20', 'EXT', 'S60', 'VC', 'T_ld']].rank().corr().to_numpy() for _, g in E.groupby('t') if len(g) >= 15]), axis=0)
        cm = pd.DataFrame(cm, index=['UDS', 'DATRp', 'R20', 'EXT', 'S60', 'VC', 'T_ld'], columns=['UDS', 'DATRp', 'R20', 'EXT', 'S60', 'VC', 'T_ld'])
        L += [f'## {era}（{len(E):,} 筆）\n', '候選內相關：' + '；'.join(f'UDS–{k} {cm.loc["UDS", k]:+.2f}' for k in ('DATRp', 'R20', 'EXT', 'S60', 'VC', 'T_ld')) + '；' + '；'.join(f'DATRp–{k} {cm.loc["DATRp", k]:+.2f}' for k in ('R20', 'VC', 'S60')) + '\n',
              '| 因子 | 控制 | IC20（t） | IC60（t） |', '|---|---|---|---|']
        r = {}
        for f in ('UDS', 'DATRp'):
            a = resid_ic(E, f, ['S']); b = resid_ic(E6, f, ['S', 'DH250', 'R6']); r[f] = b
            L += [f"| {f} | S | {a['e20'][0]:+.3f}（{a['e20'][1]:.1f}） | {a['e60'][0]:+.3f}（{a['e60'][1]:.1f}） |",
                  f"| {f} | S＋DH250＋R6 | {b['e20'][0]:+.3f}（{b['e20'][1]:.1f}） | {b['e60'][0]:+.3f}（{b['e60'][1]:.1f}） |"]
        allr = (E.n20 >= 0.20).mean()
        for f in ('UDS', 'DATRp'):
            E['q'] = E.groupby('t')[f].transform(lambda s: pd.qcut(s.rank(method='first'), 5, labels=False) + 1 if len(s) >= 5 else np.nan)
            L += ['', f'| {f} 五分位（Q5＝最高） | 20日超額 | 60日超額 | 20日淨勝率 | 左尾率 | 右尾率 |', '|---|---|---|---|---|---|']
            for q, g in E.dropna(subset=['q']).groupby('q'):
                L.append(f"| Q{int(q)} | {g.e20.mean():+.2%} | {g.e60.mean():+.2%} | {(g.n20 > 0).mean():.0%} | {(g.n20 <= -0.10).mean():.1%} | {(g.n20 >= 0.20).mean():.1%} |")
            sgn = np.sign(r[f]['e20'][0] if abs(r[f]['e20'][1]) >= abs(r[f]['e60'][1]) else r[f]['e60'][0])
            good = E[E.q == (5 if sgn >= 0 else 1)]; r[f + '_rt'] = (good.n20 >= 0.20).mean() >= allr - 0.01
        L.append(f"\n全體右尾率 {allr:.1%}\n"); res[era] = r
    heads = []
    for f in ('UDS', 'DATRp'):
        ok = False
        for h in ('e20', 'e60'):
            m = [res[e][f][h] for e in res]; ts = sorted(abs(t) for _, t in m)
            if np.sign(m[0][0]) == np.sign(m[1][0]) and ts[1] >= 2 and ts[0] >= 1: ok = True
        rt = all(res[e][f + '_rt'] for e in res)
        heads.append(f"{f} {'提名' if ok and rt else '不提名'}（IC {'✓' if ok else '✗'}、右尾 {'✓' if rt else '✗'}）")
    head = '；'.join(heads)
    return dict(summary='\n'.join(L[:1] + [f'**{head}**\n'] + L[1:]), headline=head, n_variants=0)
