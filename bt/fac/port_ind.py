"""族群趨勢過濾的穩健性：無產業資料（下市股）一律視為通過，避免用到「之後會下市」的資訊"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('/home/claude/bt/fac/port2.py').read().split("if __name__")[0])
from factors3 import tld_floor
import numpy as np, pandas as pd
ldf = tld_floor(P).where(U)
indv = P['industry'].reindex(C.columns)
known = indv.notna().values
g = ldf.loc[:, known].T.groupby(indv[known].values).transform('median').T.reindex(columns=C.columns)
gp = g.where(U).rank(axis=1, pct=True)                  # 族群趨勢的橫斷面百分位
ld = ldf.rank(axis=1, pct=True)
mid = dates[dates.get_indexer([pd.Timestamp('2021-01-04')], method='bfill')[0]]
rows = []
for n in (10, 20):
    globals()['N'] = n
    for th in (0, 0.3, 0.4, 0.5, 0.6, 0.7):
        sc = ld.where((gp >= th) | gp.isna()) if th else ld
        globals()['Sv'] = sc.values
        res = [run(off, False, 4, 3)[0] for off in range(20)]
        st = pd.DataFrame([dict(**stats(s), 前半=stats(s[:mid])['CAGR'], 後半=stats(s[mid:])['CAGR']) for s in res]); m = st.median()
        rows.append(dict(持股=n, 族群門檻=th, 年化=round(m.CAGR * 100, 1), 最低=round(st.CAGR.min() * 100, 1), 回撤=round(m.MDD * 100, 1),
                         Sharpe=round(m.Sharpe, 2), 前半=round(m.前半 * 100, 1), 後半=round(m.後半 * 100, 1)))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv('/home/claude/bt/fac/port_ind.csv', index=False, encoding='utf-8-sig')
