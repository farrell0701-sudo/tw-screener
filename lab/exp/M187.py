# M-187：③ 爆量長黑跌破低點當保護獲利出場，偶數代碼確認集（只測一次）。定義見 registry_M.md M-187。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'exp')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core'))
import M182, M185
from panel import load as load_panel
from feat import load_feat


def main():
    P = load_panel(); F = load_feat('AF')
    CA = np.asarray(F['CA'], np.float64); OA = np.asarray(F['OA'], np.float64)
    HA = np.asarray(F['HA'], np.float64); LA = np.asarray(F['LA'], np.float64); VALID = np.asarray(F['VALID'])
    AMT = np.asarray(P['AMT'], np.float64)
    D = M182.build(parity=0); D = D[D.q >= 2].reset_index(drop=True)
    cache = {}; rows = []
    for r in D.itertuples():
        S = cache.get(r.j) or cache.setdefault(r.j, M185.Stock(r.j, CA, OA, HA, LA, AMT, VALID))
        o = M185.simulate(S, r.t)
        if o is None: continue
        for nm, x in o.items():
            if nm.startswith('④'): continue
            rows.append(dict(date=r.date, v=nm, net=x['net'], sig=x.get('sig', False), base=o['S15']['net']))
    R = pd.DataFrame(rows)
    VS = ['S15', '③ k=1', '③ k=2', '③ k=3']
    L = ['# M-187 結果（偶數代碼確認集）：③ 保護獲利出場（R6 頭部剔除 Q1）\n']
    res = {}
    for era, lo, hi in M182.ERAS:
        E = R[(R.date >= lo) & (R.date <= hi)]
        L += [f'## {era}（{(E.v == "S15").sum():,} 筆）\n', '| 版本 | 觸發率 | 準確率 | 平均 | 中位數 | 勝率 | P5 | P95 |', '|---|---|---|---|---|---|---|---|']
        for nm in VS:
            g = E[E.v == nm]; sg = g[g.sig]
            res[(nm, era)] = dict(mean=g.net.mean(), med=g.net.median(), win=(g.net > 0).mean(), p5=g.net.quantile(.05))
            L.append(f"| {nm} | {g.sig.mean():.1%} | {'' if nm == 'S15' else f'{(sg.base <= 0).mean():.0%}'} | {g.net.mean():+.2%} | {g.net.median():+.2%} | {(g.net > 0).mean():.1%} | {g.net.quantile(.05):+.1%} | {g.net.quantile(.95):+.1%} |")
        L.append('')
    ok = {}
    for k in (1, 2, 3):
        nm = f'③ k={k}'
        ok[k] = all(res[(nm, e)]['med'] > res[('S15', e)]['med'] and res[(nm, e)]['win'] > res[('S15', e)]['win'] and
                    res[(nm, e)]['mean'] >= res[('S15', e)]['mean'] - 0.03 and res[(nm, e)]['p5'] >= res[('S15', e)]['p5'] - 0.005 for e, _, _ in M182.ERAS)
    passed = ok[2] and sum(ok.values()) >= 2
    vt = '判準（兩期；中位數↑、勝率↑、平均降 ≤ 3pt、P5 不深於 −0.5pt）：' + '、'.join(f"k={k} {'✓' if ok[k] else '✗'}" for k in ok) + f" → {'通過確認集（主版本 k=2）' if passed else '確認集不通過，結案'}\n"
    return dict(summary='\n'.join(L[:1] + [vt] + L[1:]), headline=vt.split('→')[-1].strip(), n_variants=0)
