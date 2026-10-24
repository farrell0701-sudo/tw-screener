"""結構因子當過濾：現行版本（T_ldf 排序＋族群趨勢≥中位＋多4空3 停損）再加一道「結構因子百分位 ≥ 門檻」
用法：python3 port_struct.py 因子1 因子2 ..."""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('/home/claude/bt/fac/port2.py').read().split("if __name__")[0])
from factors3 import tld_floor
import numpy as np, pandas as pd
ldf = tld_floor(P).where(U)
indv = P['industry'].reindex(C.columns); known = indv.notna().values
g = ldf.loc[:, known].T.groupby(indv[known].values).transform('median').T.reindex(columns=C.columns)
gp = g.where(U).rank(axis=1, pct=True)
cur = ldf.where((gp >= 0.5) | gp.isna())
mid = dates[dates.get_indexer([pd.Timestamp('2021-01-04')], method='bfill')[0]]


def test(sc, label, n):
    globals()['N'] = n; globals()['Sv'] = sc.values
    res = [run(off, False, 4, 3)[0] for off in range(20)]
    st = pd.DataFrame([dict(**stats(s), 前半=stats(s[:mid])['CAGR'], 後半=stats(s[mid:])['CAGR']) for s in res]); m = st.median()
    r = dict(持股=n, 版本=label, 年化=round(m.CAGR * 100, 1), 最低=round(st.CAGR.min() * 100, 1), 回撤=round(m.MDD * 100, 1),
             回撤最差=round(st.MDD.min() * 100, 1), Sharpe=round(m.Sharpe, 2), 前半=round(m.前半 * 100, 1), 後半=round(m.後半 * 100, 1))
    print(r, flush=True); return r


if __name__ == '__main__':
    F4 = pd.read_pickle('/home/claude/bt/fac/f4.pkl')
    rows = []
    for n in (10, 20):
        rows.append(test(cur, '現行', n))
        for k in sys.argv[1:]:
            X, s, _ = F4[k]
            xp = (X * s).where(U).rank(axis=1, pct=True)
            for th in (0.3, 0.5, 0.7):
                rows.append(test(cur.where(xp >= th), f'{k}≥{th}', n))
    pd.DataFrame(rows).to_csv('/home/claude/bt/fac/port_struct_' + '_'.join(sys.argv[1:])[:60] + '.csv', index=False, encoding='utf-8-sig')
