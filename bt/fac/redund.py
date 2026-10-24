"""因子間相關（每 20 日抽樣的橫斷面 Spearman 平均）與增量 IC（對 T_ld＋控制中性化後）"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
import numpy as np, pandas as pd
from panel import load_panel
from evaluate import prepare, neutralize, _row_spearman, _nw_t
from factors import build
P = load_panel(); E = prepare(P)
D = pd.read_pickle('/home/claude/bt/struct/daily.pkl')
D = {k: v.reindex(index=P['C'].index, columns=P['C'].columns) for k, v in D.items()}
F = build(P, D); d = E['dates']
G = {k: (X * s).where(E['univ']).loc[d] for k, (X, s, _) in F.items()}
keys = ['T_ld', 'T_slope200', 'T_align', 'T_dhi250', 'M_r60va', 'M_r120', 'M_12_1', 'M_r60ind', 'S_ovh', 'S_far5', 'S_loratio', 'X_atrpct']
smp = d[::20]
M = pd.DataFrame(index=keys, columns=keys, dtype=float)
for i, a in enumerate(keys):
    for b in keys[i:]:
        v = np.nanmean([_row_spearman(G[a].loc[[t]], G[b].loc[[t]]).iloc[0] for t in smp])
        M.loc[a, b] = M.loc[b, a] = v
pd.set_option('display.width', 250); print(M.round(2).to_string())
ctrl = [G['X_atrpct'], F['X_amt'][0].loc[d], G['T_ld']]
print('\n增量 IC（中性化：波動、成交額、T_ld）')
for k in ['T_slope200', 'T_align', 'T_dhi250', 'M_r60va', 'M_r120', 'M_12_1', 'M_r60ind', 'S_ovh', 'S_far5', 'S_loratio']:
    Fn = neutralize(G[k], ctrl)
    r = [(_row_spearman(Fn, E['fwd'][h]['va'].loc[d])) for h in (20, 60)]
    print(f'{k:12s} ICn20 {r[0].mean():+.3f} t {_nw_t(r[0],20):+.1f} | ICn60 {r[1].mean():+.3f} t {_nw_t(r[1],60):+.1f}', flush=True)
M.to_csv('corr_batch1.csv', encoding='utf-8-sig')
