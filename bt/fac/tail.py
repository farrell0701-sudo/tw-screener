"""因子在「最頂端」的表現：每日前 20／50／100／280 檔的未來 60 日超額報酬（相對股票池平均，t+1 收盤進場）"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
import numpy as np, pandas as pd
from panel import load_panel
from evaluate import prepare
from factors2 import build2
P = load_panel(); E = prepare(P); U = E['univ']; d = E['dates'][::5]
F2 = build2(P, grid=False)
pct = lambda X: X.where(U).rank(axis=1, pct=True)
ld, ab, r2, es = pct(F2['T_ld'][0]), pct(F2['Q_above20'][0]), pct(F2['Q_r2'][0]), pct(F2['E_pct'][0])
S = {'T_ld': ld, 'E_pct（百分比距離）': es, 'Q_above20': ab, 'Q_r2': r2, 'T_ld+above20': (ld + ab) / 2, 'T_ld+r2': (ld + r2) / 2,
   }
for h in (20, 60):
    R = E['fwd'][h]['raw']; X = R.where(U).sub(R.where(U).mean(1), axis=0).loc[d]
    print(f'\n未來 {h} 日超額報酬（%）')
    for name, sc in S.items():
        s = sc.loc[d]; rk = s.rank(axis=1, ascending=False)
        out = [X.where(rk <= n).mean(1).mean() * 100 for n in (20, 50, 100, 280)]
        tie = (s.eq(s.max(axis=1), axis=0)).sum(1).median()
        print(f'{name:22s} 前20 {out[0]:+.2f}  前50 {out[1]:+.2f}  前100 {out[2]:+.2f}  前280 {out[3]:+.2f}   最高分同分檔數 {tie:.0f}')
