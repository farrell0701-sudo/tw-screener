import sys; sys.path.insert(0, '.')
exec(open('port.py').read().split("if __name__")[0])
import numpy as np, pandas as pd
AMT20 = P['AMT'].rolling(20).mean().values
def trades(S, N, off):
    Sv = S.values; reb = list(range(d0 + off, len(dates) - 2, REB)); T = []
    for j, t in enumerate(reb):
        e = t + 1; x = reb[j + 1] + 1 if j + 1 < len(reb) else len(dates) - 1
        sc = np.where(BUY[t] & U.values[t], Sv[t], np.nan); ok = np.where(~np.isnan(sc))[0]
        pick = ok[np.argsort(-sc[ok])[:N]]
        ex = np.where(np.isnan(Ov[x, pick]), Cfv_[x, pick], Ov[x, pick])
        for k, p in enumerate(pick):
            T.append(dict(date=dates[t], code=C.columns[p], ret=ex[k] / Ov[e, p] - 1, amt20=AMT20[t, p], raw=P['RC'].values[t, p]))
    return pd.DataFrame(T)
T = trades(SC['T_ld'], 20, 0)
print('筆數', len(T), '平均報酬 %.2f%%' % (T.ret.mean() * 100), '中位 %.2f%%' % (T.ret.median() * 100), '勝率 %.1f%%' % ((T.ret > 0).mean() * 100))
print('成交額中位（萬）', round(T.amt20.median() / 1e4), ' 10 分位', round(T.amt20.quantile(.1) / 1e4))
print('去掉最好 1%% 筆後平均 %.2f%%' % (T.ret[T.ret < T.ret.quantile(.99)].mean() * 100))
print(T.nlargest(8, 'ret').to_string(index=False))
# 單日還原價跳動 >25% 檢查（資料錯誤）
jump = (C / C.shift(1) - 1).abs() > 0.25
print('單日漲跌 >25% 的筆數（全市場）', int(jump.sum().sum()))
# 股票池流動性收緊到 50 分位
amt20 = P['AMT'].rolling(20, min_periods=15).mean()
U50 = U & amt20.ge(amt20.quantile(0.5, axis=1), axis=0)
globals()['U'] = U50
for name in ['T_ld', '合成']:
    r = [stats(backtest(SC[name].where(U50), 20, off)[0])['CAGR'] for off in range(0, 20, 4)]
    print(name, 'N20 流動性 50 分位以上：CAGR', np.round(r, 3))
