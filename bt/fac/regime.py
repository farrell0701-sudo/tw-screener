"""多空判斷方式比較（停損固定：多頭 4 ATR、空頭 k_bear ATR；無濾網、無停利）"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('/home/claude/bt/fac/port2.py').read().split("if __name__")[0])
import numpy as np, pandas as pd
Cc = P['C']; Uc = U
br = lambda span: (Cc > Cc.ewm(span=span, adjust=False).mean()).where(Uc).astype(float).sum(1) / Uc.sum(1).astype(float).replace(0, np.nan)
b200, b60 = br(200), br(60)
e200x = tx.ewm(span=200, adjust=False).mean()
R = {
    '指數 EMA60 斜率>0（現行）': BULL,
    '指數 > EMA200': (tx > e200x).values,
    '站上EMA200比例 > 50%': (b200 > 0.5).values,
    '站上EMA200比例 > 自身250日中位': (b200 > b200.rolling(250, min_periods=120).median()).values,
    '站上EMA200比例 20日上升': (b200.rolling(5).mean() > b200.rolling(5).mean().shift(20)).values,
    '站上EMA60比例 > 50%': (b60 > 0.5).values,
    '站上EMA60比例 > 自身250日中位': (b60 > b60.rolling(250, min_periods=120).median()).values,
}
d = dates[d0:]
rows = []
for name, arr in R.items():
    arr = np.asarray(arr, bool)
    a = pd.Series(arr, index=dates).loc[d]
    # 判斷品質：空頭期間的加權指數未來 20 日報酬 vs 多頭期間
    f20 = (tx.shift(-20) / tx - 1).loc[d]
    rec = dict(判斷=name, 空頭天數比=(~a).mean(), 每年切換次數=(a != a.shift()).sum() / (len(a) / 252),
               多頭後20日指數=f20[a].mean() * 100, 空頭後20日指數=f20[~a].mean() * 100)
    for kr in (3, 2):
        res = [run(off, False, 4, kr, None, bull_arr=arr)[0] for off in range(0, 20, 4)]
        st = pd.DataFrame([stats(s) for s in res]).median()
        rec.update({f'k空{kr} 年化': st.CAGR * 100, f'k空{kr} 回撤': st.MDD * 100, f'k空{kr} Sharpe': st.Sharpe})
    rows.append(rec); print(rec, flush=True)
base = pd.DataFrame([stats(run(off, False, 4, 4, None)[0]) for off in range(0, 20, 4)]).median()
print('對照：不分多空，全程 4 ATR', (base * [100, 100, 1]).round(2).to_dict())
X = pd.DataFrame(rows); X.to_csv('/home/claude/bt/fac/regime.csv', index=False, encoding='utf-8-sig')
pd.set_option('display.width', 250); print(X.round(2).to_string(index=False))
