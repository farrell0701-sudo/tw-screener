"""族群共振：同產業近 5 日內有多少檔強勢股出現收斂突破事件"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('/home/claude/bt/fac/sleeve.py').read().split("\nif __name__")[0])
ind = P['industry'].reindex(codes)
evs = (strong & anyev).astype(float)
cnt5 = evs.rolling(5, min_periods=1).sum().clip(upper=1)                      # 每檔近 5 日有沒有事件
known = ind.notna().values
grp_cnt = cnt5.loc[:, known].T.groupby(ind[known].values).transform('sum').T.reindex(columns=codes)   # 同產業近 5 日有事件的檔數（含自己）
grp_n = strong.astype(float).loc[:, known].T.groupby(ind[known].values).transform('sum').T.reindex(columns=codes)
others = grp_cnt - cnt5                                                         # 扣掉自己
ind_size = pd.Series(1, index=codes)[known].groupby(ind[known].values).transform('sum').reindex(codes)
pd.to_pickle(dict(others=others, grp_n=grp_n), '/home/claude/bt/fac/resonance_ind.pkl')
if __name__ == '__main__':
    ent = O.shift(-1); Cf = C.ffill(); dd0 = pd.Timestamp('2016-02-01'); lu_next = (O.shift(-1) / Cf - 1 >= 0.095)
    base_m = (strong & ~lu_next & ent.notna()).loc[dd0:]; evm = base_m & anyev.loc[dd0:]
    oth = others.loc[dd0:]
    rows = []
    for lab, m_ in (('單獨突破（同產業近 5 日沒有其他事件）', oth == 0), ('同產業另有 1 檔', oth == 1), ('同產業另有 2 檔', oth == 2),
                    ('同產業另有 ≥3 檔', oth >= 3), ('無產業分類', oth.isna())):
        o = {'族群共振': lab}
        for h in (5, 20, 60):
            Rr = (Cf.shift(-h) / ent - 1).loc[dd0:]; m = evm & m_ & Rr.notna()
            ex = Rr.sub(Rr.where(base_m).mean(1), axis=0).where(m).stack().dropna(); rr = Rr.where(m).stack().dropna()
            yr = ex.groupby(ex.index.get_level_values(0).year).mean()
            o[f'{h}日超額'] = f'{ex.mean():+.2%}'; o[f'{h}日正年'] = f'{(yr > 0).sum()}/{len(yr)}'
            if h == 60: o['60日跌超20'] = f'{(rr < -.2).mean():.1%}'
        o['占事件'] = f'{m.sum().sum() / evm.sum().sum():.0%}'; rows.append(o)
    pd.set_option('display.width', 250); print(pd.DataFrame(rows).to_string(index=False), flush=True)
    tp = cur.rank(axis=1, pct=True); oo = others.reindex_like(C)
    sleeve(anyev, '基準：收斂突破')
    sleeve(anyev & (oo >= 1), '只買有族群共振的（同產業另有 ≥1 檔）')
    sleeve(anyev, '有族群共振的優先買', prio=tp + (oo >= 1).astype(float))
    sleeve(anyev, '族群共振越多越優先', prio=tp + oo.fillna(0).clip(upper=3))
    pd.set_option('display.width', 260); print(pd.DataFrame(RES).to_string(index=False))
