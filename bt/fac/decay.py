import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('/home/claude/bt/fac/ignite.py').read().split("lu_next = ")[0])
lu_next = O.shift(-1) / C.ffill() - 1 >= 0.095; ent = O.shift(-1)
d0 = pd.Timestamp('2016-02-01')
base_m = (strong & ~lu_next & ent.notna()).loc[d0:]
anyev = (EV['布林：收斂後首次突破上軌'] | EV['布林：收斂後帶寬張開 + 收上軌'] | EV['整理後突破 20 日高']).loc[d0:].fillna(False)
m = base_m & anyev
# 多空：加權指數 EMA60 的 20 日斜率
tx = P['taiex']; e60 = tx.ewm(span=60, adjust=False).mean(); bull = (e60 / e60.shift(20) - 1 > 0).loc[d0:]
Cf = C.ffill().loc[:]
rows = {}
for h in (1, 3, 5, 10, 15, 20, 25, 30, 40, 50, 60):
    R = (Cf.shift(-h) / ent - 1).loc[d0:]
    ex = R.sub(R.where(base_m).mean(1), axis=0).where(m)
    e = ex.stack().dropna(); dts = e.index.get_level_values(0)
    yrs = e.groupby(dts.year).mean()
    rows[h] = {'累積超額': e.mean() * 100, '多頭期': e[bull.reindex(dts).values].mean() * 100, '空頭期': e[~bull.reindex(dts).values].mean() * 100,
               '每日邊際': np.nan, '為正年數': f'{(yrs > 0).sum()}/{len(yrs)}'}
D = pd.DataFrame(rows).T
D['每日邊際'] = D['累積超額'].astype(float).diff() / pd.Series(D.index, index=D.index).diff()
pd.set_option('display.width', 200); print(D.round(2).to_string())
