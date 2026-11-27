# M-250：X1 閘門 200MA 擋、100／150MA 放（描述性）。定義見 registry_M.md M-250。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim
from M110 import wrsi, triggers, stats

NP = 100; WIN = 60


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T = len(dates)
    signals.build('fix', mode='fix', P=P, F=F)
    ix = P['index']['TAIEX_P'].reindex(dates).ffill(); below = (ix < ix.rolling(200).mean()).to_numpy()
    pm = lambda ps: pd.concat(ps, axis=1).mean(axis=1)
    B = sim.run_paths('fix', NP, sig='fix'); rV = [B['V'][i][0] for i in range(NP)]
    res = {'基準': (pm([B['X'][i][0] for i in range(NP)]), pm([sim.mix(B['X'][i][0], rV[i]) for i in range(NP)]), None)}
    wk = pd.Series(dates, index=dates).groupby(dates.to_period('W-FRI')).max(); w = ix.loc[wk.values]
    trig = [d for d in triggers(wrsi(w, 14), 30, 40) if below[dates.get_loc(d)]]; unlockB = np.zeros(T, bool)
    for d in trig: s = dates.get_loc(d); unlockB[s:s + WIN] = True
    V = {'K100（200 擋、100 放）': dict(gate=below & (ix < ix.rolling(100).mean()).to_numpy()),
         'K150（200 擋、150 放）': dict(gate=below & (ix < ix.rolling(150).mean()).to_numpy()),
         'M-110 B（週 RSI 回升）': dict(unlock=unlockB)}
    for nm, kw in V.items():
        U = sim.run_paths('fix', NP, sig='fix', x_kw=kw); seen = {}
        for i in range(NP):
            for (code, d0, d1, wt, ret, *_r) in U['X'][i][1]: seen.setdefault((code, d0), dict(d0=d0, ret=ret))
        X = pd.DataFrame(seen.values()); X = X[below[dates.get_indexer(X.d0) - 1]]
        res[nm] = (pm([U['X'][i][0] for i in range(NP)]), pm([sim.mix(U['X'][i][0], rV[i]) for i in range(NP)]), X)
    f = lambda t: f"{t[0]:+.1%} / {t[1]:.1%} / {t[2]:.2f}"
    L = ['# M-250 結果：X1 閘門 200MA 擋、100／150MA 放（描述性、不能提名）\n', '## X1 子帳戶與合併版（100 路徑平均日報酬；年化 / 回撤 / Calmar）\n',
         '| 版本 | X1 2005–15 | X1 2016–26 | 合併 2005–15 | 合併 2016–26 |', '|---|---|---|---|---|']
    for nm, (rx, rc, X) in res.items():
        L.append(f"| {nm} | {f(stats(rx, '2005', '2015'))} | {f(stats(rx, '2016', '2026'))} | {f(stats(rc, '2005', '2015'))} | {f(stats(rc, '2016', '2026'))} |")
    L += ['\n## 額外交易（開倉決策日指數 < MA200；不重複計）\n', '| 版本 | 05–15 筆數 | 平均 | 勝率 | 16–26 筆數 | 平均 | 勝率 |', '|---|---|---|---|---|---|---|']
    for nm, (_, _, X) in res.items():
        if X is None: continue
        a, b = X[X.d0 < '2016'], X[X.d0 >= '2016']
        L.append(f"| {nm} | {len(a)} | {a.ret.mean():+.1%} | {(a.ret > 0).mean():.0%} | {len(b)} | {b.ret.mean():+.1%} | {(b.ret > 0).mean():.0%} |")
    ys = [2008, 2011, 2015, 2018, 2020, 2022, 2025]
    L += ['\n## 關鍵年份（合併版，路徑平均）\n', '| 版本 | ' + ' | '.join(map(str, ys)) + ' |', '|' + '---|' * (len(ys) + 1)]
    for nm, (_, rc, _) in res.items():
        yy = rc.groupby(rc.index.year).apply(lambda x: (1 + x).prod() - 1); L.append(f'| {nm} | ' + ' | '.join(f'{yy.get(y, np.nan):+.0%}' for y in ys) + ' |')
    k100 = stats(res['K100（200 擋、100 放）'][1], '2005', '2015'); b0 = stats(res['基準'][1], '2005', '2015')
    head = f"K100 合併 05–15 Calmar {k100[2]:.2f}（基準 {b0[2]:.2f}）；描述性、不能提名"
    return dict(summary='\n'.join(L), headline=head, n_variants=0)
