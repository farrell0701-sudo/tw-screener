"""趨勢合成分數的組合檢驗（多4空3 停損、無濾網），20 個起始日"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
src = open('/home/claude/bt/fac/port2.py').read().split("if __name__")[0]
exec(src)
from factors2 import build2
from evaluate import _row_spearman
import numpy as np, pandas as pd
F2 = build2(P, grid=False)
pct = lambda X: X.where(U).rank(axis=1, pct=True)
ld, ab, r2 = pct(F2['T_ld'][0]), pct(F2['Q_above20'][0]), pct(F2['Q_r2'][0])
d = dates[d0::20]
print('與 T_ld 的橫斷面相關：above20 %.2f  r2 %.2f' % (np.nanmean([_row_spearman(ld.loc[[t]], ab.loc[[t]]).iloc[0] for t in d]),
                                                    np.nanmean([_row_spearman(ld.loc[[t]], r2.loc[[t]]).iloc[0] for t in d])))
S = {'T_ld': ld, 'T_ld+above20': (ld + ab) / 2, 'T_ld+r2': (ld + r2) / 2, '三者平均': (ld + ab + r2) / 3}
mid = dates[dates.get_indexer([pd.Timestamp('2021-01-04')], method='bfill')[0]]
for name, sc in S.items():
    globals()['Sv'] = sc.values
    res = [run(off, False, 4, 3)[0] for off in range(20)]
    st = pd.DataFrame([dict(**stats(s), 前半=stats(s[:mid])['CAGR'], 後半=stats(s[mid:])['CAGR']) for s in res])
    m = st.median()
    print(f'{name:14s} 年化 {m.CAGR:.1%}（{st.CAGR.min():.1%}～{st.CAGR.max():.1%}） 回撤 {m.MDD:.1%} Sharpe {m.Sharpe:.2f} 前半 {m.前半:.1%} 後半 {m.後半:.1%}', flush=True)
