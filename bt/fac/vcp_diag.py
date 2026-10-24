"""波動擴張過濾在 finlab 沒有重現：檢查是否來自自建資料的價格跳動問題"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('dense_ctrl.py').read().split("rows = []")[0])
rows = []
exec(open('dense_ctrl.py').read().split("rows = []")[1].split("for K in (30,):")[0])
from panel import build
raw = build()
r = raw['C'] / raw['C'].ffill().shift(1) - 1
age = raw['C'].notna().cumsum()
badcnt = ((r.abs() > 0.11) & (age > 5)).sum()
clean_codes = badcnt[badcnt == 0].index
print('曾有超過漲跌幅跳動的股票', int((badcnt > 0).sum()), '檔，乾淨的', len(clean_codes), '檔')
F4 = pd.read_pickle('/home/claude/bt/fac/f4.pkl'); vpall = F4['B_vcp'][0]
for lab, Umask in (('全部股票', U), ('只用沒有跳動問題的股票', U & pd.DataFrame(np.isin(C.columns, clean_codes)[None, :].repeat(len(dates), 0), index=dates, columns=C.columns))):
    ldf2 = ldf.where(Umask)
    g2 = ldf2.loc[:, known].T.groupby(indv[known].values).transform('median').T.reindex(columns=C.columns)
    gp2 = g2.where(Umask).rank(axis=1, pct=True)
    cur2 = ldf2.where((gp2 >= 0.5) | gp2.isna())
    vp2 = vpall.where(Umask).rank(axis=1, pct=True)
    for vl, c2 in (('無波動過濾', cur2), ('排除波動擴張20%', cur2.where(vp2 <= 0.8))):
        globals()['RK'] = c2.where(pd.DataFrame(BUY, index=dates, columns=C.columns)).rank(axis=1, ascending=False).values
        globals()['RKall'] = c2.rank(axis=1, ascending=False).values
        go(Umask, f'{lab}｜{vl}', 30, offs=range(0, 20, 4))
