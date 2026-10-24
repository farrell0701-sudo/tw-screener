"""強中選抗跌：先用 T_ldf（＋族群過濾）取前 K 檔候選，再用第二個分數挑 10 檔
對照：候選中依 T_ldf 挑、隨機挑（看第二分數是否真的比亂挑好）"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('/home/claude/bt/fac/port2.py').read().split("if __name__")[0])
from factors3 import tld_floor, build3
import numpy as np, pandas as pd
F3 = build3(P)
ldf = tld_floor(P).where(U)
indv = P['industry'].reindex(C.columns); known = indv.notna().values
g = ldf.loc[:, known].T.groupby(indv[known].values).transform('median').T.reindex(columns=C.columns)
gp = g.where(U).rank(axis=1, pct=True)
base_sc = ldf.where((gp >= 0.5) | gp.isna())
rk = base_sc.where(pd.DataFrame(BUY, index=dates, columns=C.columns)).rank(axis=1, ascending=False)
sec = {'T_ldf（原排序）': None, '抗跌高': F3['R_downday'][0], '抗跌低': -F3['R_downday'][0],
       '上漲日強': F3['R_upday'][0], '低 beta': -F3['R_beta'][0]}
globals()['N'] = 10
mid = dates[dates.get_indexer([pd.Timestamp('2021-01-04')], method='bfill')[0]]
rows = []
def go(sc_df, label, K, seeds=(None,)):
    st = []
    for sd in seeds:
        for off in range(20):
            globals()['Sv'] = sc_df(sd).values
            s = run(off, False, 4, 3)[0]
            st.append(dict(**stats(s), 前半=stats(s[:mid])['CAGR'], 後半=stats(s[mid:])['CAGR']))
    st = pd.DataFrame(st); m = st.median()
    rows.append(dict(候選K=K, 第二分數=label, 年化=round(m.CAGR * 100, 1), 最低=round(st.CAGR.min() * 100, 1), 回撤=round(m.MDD * 100, 1),
                     Sharpe=round(m.Sharpe, 2), 前半=round(m.前半 * 100, 1), 後半=round(m.後半 * 100, 1)))
    print(rows[-1], flush=True)
for K in (20, 30, 40):
    pool = rk <= K
    for label, x in sec.items():
        if x is None:
            go(lambda sd: base_sc.where(pool), label, K)
        else:
            xp = x.where(U).rank(axis=1, pct=True)
            go(lambda sd, xp=xp: (xp + base_sc.rank(axis=1, pct=True) * 1e-6).where(pool), label, K)
    rng = lambda sd: pd.DataFrame(np.random.default_rng(sd).random(C.shape), index=dates, columns=C.columns).where(pool & base_sc.notna())
    go(rng, '隨機（5 次）', K, seeds=range(5))
pd.DataFrame(rows).to_csv('/home/claude/bt/fac/port_dd.csv', index=False, encoding='utf-8-sig')
