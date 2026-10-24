"""持股數敏感度：T_ld 排序、多4空3 停損、無濾網，20 個起始日"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('/home/claude/bt/fac/port2.py').read().split("if __name__")[0])
import numpy as np, pandas as pd
rows = []
for n in (3, 5, 7, 10, 15, 20):
    globals()['N'] = n
    st = []
    for off in range(20):
        s = run(off, False, 4, 3)[0]
        y = s.resample('YE').last(); yr = y.pct_change(); yr.iloc[0] = y.iloc[0] / s.iloc[0] - 1
        st.append(dict(**stats(s), 最差年=yr.min(), 虧損年=(yr < 0).sum()))
    st = pd.DataFrame(st); m = st.median()
    rows.append(dict(持股=n, 年化中位=m.CAGR, 年化最低=st.CAGR.min(), 年化最高=st.CAGR.max(),
                     回撤中位=m.MDD, 回撤最差=st.MDD.min(), Sharpe=m.Sharpe, 最差年中位=m.最差年, 虧損年中位=m.虧損年))
    print(rows[-1], flush=True)
R = pd.DataFrame(rows); R.to_csv('/home/claude/bt/fac/port_n.csv', index=False, encoding='utf-8-sig')
