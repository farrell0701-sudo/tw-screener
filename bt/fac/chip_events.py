"""籌碼事件與籌碼修飾（法人、融資、當沖；當天收盤後可用）"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('/home/claude/bt/fac/sleeve.py').read().split("\nif __name__")[0])
CH = pd.read_pickle('/home/claude/bt/fac/chip.pkl')
V = P['V']; tr, fo, mb, dtv = CH['trust'], CH['foreign'], CH['mbal'], CH['dtv']
def rsum(x, n): return x.rolling(n, min_periods=max(1, n - 1)).sum()
t_first = (tr > 0) & (tr / V >= 0.01) & (rsum((tr > 0).astype(float).where(tr.notna()), 60).shift(1) == 0)
t_run3 = (tr > 0) & (tr.shift(1) > 0) & (tr.shift(2) > 0) & (rsum(tr, 3) / rsum(V, 3) >= 0.02)
t_run3 = t_run3 & ~(t_run3.astype(float).rolling(20, min_periods=1).max().shift(1) > 0)
f_run5 = sum((fo.shift(k) > 0).astype(int) for k in range(5)) == 5
f_run5 = f_run5 & ~(f_run5.astype(float).rolling(20, min_periods=1).max().shift(1) > 0)
m_drop = (mb / mb.shift(5) - 1 <= -0.10) & (C > C.shift(5)) & (mb.shift(5) > 0.02 * V.rolling(20).mean() * 20)
CEV = {'C1 投信首買（60 日未買、≥1% 成交量）': t_first, 'C2 投信連買 3 日（≥2% 成交量）': t_run3,
       'C3 外資連買 5 日': f_run5, 'C4 融資 5 日減 ≥10%、股價上漲（籌碼沉澱）': m_drop}
MOD = {'外資 5 日買超／成交量': rsum(fo, 5) / rsum(V, 5), '投信 5 日買超／成交量': rsum(tr, 5) / rsum(V, 5),
       '法人合計 20 日買超／成交量': rsum(fo + tr.fillna(0), 20) / rsum(V, 20), '融資 20 日變化率': mb / mb.shift(20) - 1,
       '當沖比例 20 日': rsum(dtv, 20) / rsum(V, 20)}
pd.to_pickle(dict(CEV=CEV, MOD=MOD), '/home/claude/bt/fac/chip_events.pkl')
if __name__ == '__main__':
    from evaluate import prepare
    univ = prepare(P)['univ']
    ent = O.shift(-1); Cf = C.ffill(); dd0 = pd.Timestamp('2016-02-01'); lu_next = (O.shift(-1) / Cf - 1 >= 0.095)
    rows = []
    for grp, pool in (('強勢股', strong), ('全市場', univ)):
        base_m = (pool & ~lu_next & ent.notna()).loc[dd0:]
        for k, M in CEV.items():
            o = {'群體': grp, '事件': k}
            for h in (5, 20, 60):
                Rr = (Cf.shift(-h) / ent - 1).loc[dd0:]; m = base_m & M.loc[dd0:].fillna(False) & Rr.notna()
                ex = Rr.sub(Rr.where(base_m).mean(1), axis=0).where(m).stack().dropna()
                yr = ex.groupby(ex.index.get_level_values(0).year).mean()
                o[f'{h}日超額'] = f'{ex.mean():+.2%}'; o[f'{h}日正年'] = f'{(yr > 0).sum()}/{len(yr)}'
            o['每年次數'] = int(m.sum().sum() / 10.7); rows.append(o)
    pd.set_option('display.width', 260); print('══ 籌碼事件'); print(pd.DataFrame(rows).to_string(index=False), flush=True)
    # 修飾：收斂突破事件發生時的籌碼狀態（三等分，全期門檻）
    base_m = (strong & ~lu_next & ent.notna()).loc[dd0:]; evm = base_m & anyev.loc[dd0:]
    rows = []
    for k, X in MOD.items():
        Xs = X.loc[dd0:]
        for h in (20, 60):
            Rr = (Cf.shift(-h) / ent - 1).loc[dd0:]; g = evm & Xs.notna() & Rr.notna()
            v = Xs.where(g).stack().dropna(); a, b = v.quantile([1 / 3, 2 / 3])
            ex = Rr.sub(Rr.where(base_m).mean(1), axis=0)
            o = dict(修飾=k, 持有=h, 切點=f'{a:+.3f} / {b:+.3f}')
            for nm, mm in (('低', Xs <= a), ('中', (Xs > a) & (Xs <= b)), ('高', Xs > b)):
                e = ex.where(g & mm).stack().dropna(); yr = e.groupby(e.index.get_level_values(0).year).mean()
                o[f'{nm}：超額'] = f'{e.mean():+.2%}'; o[f'{nm}：正年'] = f'{(yr > 0).sum()}/{len(yr)}'
            rows.append(o)
    print('\n══ 收斂突破事件 × 事件當下籌碼（三等分）'); print(pd.DataFrame(rows).to_string(index=False))
