import sys; sys.path.insert(0, '.')
exec(open('port.py').read().split("if __name__")[0])
import numpy as np, pandas as pd
tx = P['taiex']
res = {}
for name, N in [('T_ld', 20), ('合成', 20), ('S_ovh', 20), (None, 0)]:
    Y = []; H = []
    for off in range(REB):
        s, h = backtest(SC[name] if name else None, N, off, cost=bool(name))
        Y.append(s.resample('YE').last().pct_change().fillna(s.resample('YE').last().iloc[0] / s.iloc[0] - 1))
        if name: H.append(h.set_index('date').top_ind_share)
    res[name or '等權股票池'] = pd.concat(Y, axis=1).median(1)
    if name == 'T_ld': ind_share = pd.concat(H).sort_index()
t = tx.loc['2016-02-01':]; res['加權指數(價格)'] = t.resample('YE').last().pct_change().fillna(t.resample('YE').last().iloc[0] / t.iloc[0] - 1)
R = pd.DataFrame(res); R.index = R.index.year
R['T_ld超額'] = R['T_ld'] - R['等權股票池']; R['合成超額'] = R['合成'] - R['等權股票池']
print((R * 100).round(1).to_string())
print('T_ld 前20：單一產業占比 平均 %.0f%%，最高 %.0f%%（%s）' % (ind_share.mean() * 100, ind_share.max() * 100, ind_share.idxmax().date()))
print('單一產業占比 >50%% 的換股次數比例 %.1f%%' % ((ind_share > 0.5).mean() * 100))
R.to_csv('port_yearly.csv', encoding='utf-8-sig')
# 每年持股最多的產業
s, h = None, None
Sv = SC['T_ld'].values; reb = list(range(d0, len(dates) - 2, REB)); rows = []
for t in reb:
    sc = np.where(BUY[t] & U.values[t], Sv[t], np.nan); ok = np.where(~np.isnan(sc))[0]
    for p in ok[np.argsort(-sc[ok])[:20]]: rows.append((dates[t].year, ind[p]))
D2 = pd.DataFrame(rows, columns=['y', 'ind'])
print(D2.groupby('y').ind.agg(lambda x: ', '.join(f'{k} {v / len(x):.0%}' for k, v in x.value_counts().head(3).items())).to_string())
