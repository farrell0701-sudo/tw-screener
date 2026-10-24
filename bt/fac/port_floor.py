"""ATR 下限：分母 = max(ATR, 當日全市場 ATR% 的 q 分位 × 股價)，避免收購案等「價格被釘住」的股票衝到第一名"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('/home/claude/bt/fac/port2.py').read().split("if __name__")[0])
import numpy as np, pandas as pd
B = base(P); e12 = P['C'].ewm(span=12, adjust=False).mean(); e200 = P['C'].ewm(span=200, adjust=False).mean()
atrp = (B['atr'] / P['C']).where(U)
rows = []
for q in (0, 0.05, 0.10, 0.20):
    fl = atrp.quantile(q, axis=1) if q else pd.Series(0, index=dates)
    den = np.maximum(B['atr'], P['C'].mul(fl, axis=0))
    globals()['Sv'] = ((e12 - e200) / den).where(U).values
    for n in (5, 7, 10, 20):
        globals()['N'] = n
        st = pd.DataFrame([stats(run(off, False, 4, 3)[0]) for off in range(20)])
        m = st.median()
        rows.append(dict(下限分位=q, 持股=n, 年化中位=round(m.CAGR * 100, 1), 年化範圍=f'{st.CAGR.min():.1%}～{st.CAGR.max():.1%}',
                         回撤=round(m.MDD * 100, 1), Sharpe=round(m.Sharpe, 2)))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv('/home/claude/bt/fac/port_floor.csv', index=False, encoding='utf-8-sig')
