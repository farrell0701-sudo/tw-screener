"""相對強度候選的組合檢驗：T_ldf 為主，兩種結合法；多4空3 停損，20 個起始日"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('/home/claude/bt/fac/port2.py').read().split("if __name__")[0])
from factors3 import build3
import numpy as np, pandas as pd
F3 = build3(P)
pct = lambda X: X.where(U).rank(axis=1, pct=True)
ld = pct(F3['T_ldf'][0])
S = {'T_ldf': ld}
for k in ('R_rshigh', 'R_indld', 'R_consist', 'R_upday'):
    x = pct(F3[k][0])
    S[f'平均 {k}'] = (ld + x) / 2
    S[f'過濾 {k}≥中位'] = ld.where(x >= 0.5)
rows = []
for n in (10, 20):
    globals()['N'] = n
    for name, sc in S.items():
        globals()['Sv'] = sc.values
        st = pd.DataFrame([stats(run(off, False, 4, 3)[0]) for off in range(20)]); m = st.median()
        rows.append(dict(持股=n, 分數=name, 年化=round(m.CAGR * 100, 1), 年化最低=round(st.CAGR.min() * 100, 1),
                         回撤=round(m.MDD * 100, 1), Sharpe=round(m.Sharpe, 2)))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv('/home/claude/bt/fac/port_rs.csv', index=False, encoding='utf-8-sig')
