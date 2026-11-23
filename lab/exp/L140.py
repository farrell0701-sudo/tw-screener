# L-140：產業分類更正（上市電子股子類、類字尾合併）對定版合併版的影響。判準見 registry_L.md L-140。
import sys, copy
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim, evalu as E
from M120 import TWD

NPATH = 100
SEGS = (('2005–15', '2005-01-01', '2015-12-31'), ('2016–20', '2016-01-01', '2020-12-31'), ('2021–26', '2021-01-01', '2026-12-31'), ('2016–26', '2016-01-01', '2026-12-31'))


def fixed_industry(meta):
    info = pd.read_parquet(TWD / 'staged' / 'TaiwanStockInfo' / 'all.parquet')
    sub = info[~info.industry_category.isin(['電子工業', '創新板股票'])].groupby('stock_id').industry_category.first()
    ind = meta['industry'].astype(object).copy(); sid = meta['stock_id'].astype(str)
    m1 = (ind == '電子工業') & sid.isin(sub.index)
    ind[m1] = sid[m1].map(sub)
    names = set(ind.dropna().unique())
    mp = {n: n[:-1] for n in names if isinstance(n, str) and n.endswith('類') and n[:-1] in names}
    mp['其他電子類'] = '其他電子業'
    m2 = ind.isin(list(mp)); ind[m2] = ind[m2].map(mp)
    return ind, int(m1.sum()), int(m2.sum()), mp


def med(paths, lo, hi): return np.nanmedian(np.array([E.met(p.loc[lo:hi]) for p in paths]), axis=0)


def main():
    P = load_panel(); F = load_feat('AF')
    ind, n1, n2, mp = fixed_industry(P['meta'])
    P2 = dict(P); P2['meta'] = P['meta'].copy(); P2['meta']['industry'] = ind
    signals.build('fix', mode='fix', P=P, F=F)
    signals.build('L140_ind', mode='fix', P=P2, F=F)
    d0 = signals.compute('fix', P=P, F=F)['daily']; d1 = signals.compute('fix', P=P2, F=F)['daily']
    OUT = {}
    for nm, sg in (('Z0 新基準', 'fix'), ('Z1 產業更正', 'L140_ind')):
        R = sim.run_paths('fix', NPATH, sig=sg)
        OUT[nm] = ([sim.mix(R['X'][i][0], R['V'][i][0]) for i in range(NPATH)], [R['V'][i][0] for i in range(NPATH)], [R['V'][i][1] for i in range(NPATH)])
    L = ['# L-140 結果：產業分類更正對定版合併版的影響（100 路徑中位數）\n',
         f'更正：上市「電子工業」改子類 {n1} 檔；「類」字尾合併 {n2} 檔（{"、".join(f"{a}→{b}" for a, b in mp.items())}）\n',
         '| 版本 | ' + ' | '.join(s[0] for s in SEGS) + ' |', '|---|---|---|---|---|']
    for nm, (B, V, _) in OUT.items():
        L.append(f'| {nm} 合併版 | ' + ' | '.join('{:+.1%}／{:.1%}／{:.2f}'.format(*med(B, lo, hi)[:3]) for _, lo, hi in SEGS) + ' |')
        L.append(f'| {nm} V4.1 單獨 | ' + ' | '.join('{:+.1%}／{:.1%}／{:.2f}'.format(*med(V, lo, hi)[:3]) for _, lo, hi in SEGS) + ' |')
    b0, b1 = OUT['Z0 新基準'][0], OUT['Z1 產業更正'][0]
    dC = [med(b1, lo, hi)[2] - med(b0, lo, hi)[2] for _, lo, hi in SEGS]
    pb = E.boot_p(E.path_mean(b1), E.path_mean(b0))
    L += ['', f"合併版 Calmar 變化（05–15｜16–20｜21–26｜16–26）：{'｜'.join(f'{x:+.2f}' for x in dC)}；bootstrap P（16–26，Z1 優於 Z0）：{pb:.2f}", '']
    y0 = d0['n_signal'].groupby(d0.index.year).sum(); y1 = d1['n_signal'].groupby(d1.index.year).sum()
    L += ['## V4.1 每年訊號數（全部代碼）\n', '| 年 | 新基準 | 產業更正 | 變化 |', '|---|---|---|---|']
    for y in y0.index:
        if y < 2005: continue
        L.append(f'| {y} | {y0[y]} | {y1[y]} | {(y1[y] / y0[y] - 1) if y0[y] else 0:+.0%} |')
    L.append(f'| 合計 | {y0.loc[2005:].sum()} | {y1.loc[2005:].sum()} | {y1.loc[2005:].sum() / y0.loc[2005:].sum() - 1:+.1%} |')
    return dict(summary='\n'.join(L), headline=f"產業更正：合併 Calmar Δ {'｜'.join(f'{x:+.2f}' for x in dC[:3])}", n_variants=1, series={'Z0': E.path_mean(b0), 'Z1': E.path_mean(b1)})
