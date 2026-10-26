"""月營收公布事件：可用日 = 公布月份 11 號起第一個交易日（保守）"""
import sys, glob; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('/home/claude/bt/fac/sleeve.py').read().split("\nif __name__")[0])
r = pd.read_csv('/tmp/claude-0/rev.csv', dtype={'code': str}); r.columns = [c.strip('﻿') for c in r.columns]
r = r.rename(columns={'rev': 'revenue'}); r['ym'] = pd.PeriodIndex(r['month'], freq='M'); r['fm'] = pd.to_datetime(r['fm_date'])
r = r[['code', 'ym', 'revenue', 'fm']]
dl = []
for f in glob.glob('/home/claude/fund/rev_*.csv'):
    try: x = pd.read_csv(f, dtype={'stock_id': str})
    except Exception: continue
    if not len(x) or 'revenue' not in x: continue
    x['ym'] = pd.PeriodIndex(pd.to_datetime(dict(year=x.revenue_year, month=x.revenue_month, day=1)), freq='M')
    dl.append(pd.DataFrame({'code': x.stock_id, 'ym': x.ym, 'revenue': x.revenue, 'fm': pd.to_datetime(x.date)}))
R = pd.concat([r] + dl).drop_duplicates(['code', 'ym'], keep='first').sort_values(['code', 'ym'])
R = R[R.code.isin(set(codes))]
g = R.groupby('code')['revenue']
R['ly'] = g.shift(12); R['yoy1'] = R.revenue / R.ly - 1
R['r3'] = g.transform(lambda s: s.rolling(3, min_periods=3).sum()); R['r3ly'] = R.groupby('code')['r3'].shift(12)
R['yoy3'] = R.r3 / R.r3ly - 1; R['yoy3_prev'] = R.groupby('code')['yoy3'].shift(1)
R['r12'] = g.transform(lambda s: s.rolling(12, min_periods=12).sum()); R['mom'] = R.r3 / (R.r12 / 4)
R['hi12'] = g.transform(lambda s: s.shift(1).rolling(11, min_periods=11).max()); R['n'] = R.groupby('code').cumcount()
R['hiall'] = g.transform(lambda s: s.shift(1).cummax())
R['avail'] = R.fm + pd.Timedelta(days=10)
pos_ = dates.searchsorted(R.avail.values); R = R[pos_ < len(dates)].copy(); R['t'] = dates.searchsorted(R.avail.values)
R = R[R.t > 0]
R['mom_pct'] = R.groupby('ym')['mom'].rank(pct=True)
ev_defs = {'R1 單月營收創 12 月新高': (R.revenue > R.hi12),
           'R2 單月營收創歷史新高（≥36 月）': (R.revenue > R.hiall) & (R.n >= 36),
           'R3 單月年增 ≥30%': R.yoy1 >= 0.3,
           'R4 近 3 月年增由負轉正': (R.yoy3 > 0) & (R.yoy3_prev <= 0),
           'R5 營收加速前 20%（排除前 2%）': (R.mom_pct >= 0.8) & (R.mom_pct < 0.98)}
REVEV = {}
for k, m in ev_defs.items():
    M = np.zeros((len(dates), len(codes)), dtype=bool); ci = pd.Index(codes).get_indexer(R.code[m])
    M[R.t[m].values, ci] = True; REVEV[k] = pd.DataFrame(M, index=dates, columns=codes)
pd.to_pickle(REVEV, '/home/claude/bt/fac/rev_events.pkl')
if __name__ == '__main__':
    U_ = P['eligible'].reindex_like(C).fillna(False) if 'eligible' in P else C.notna()
    from evaluate import prepare
    E = prepare(P); univ = E['univ']
    ent = O.shift(-1); Cf = C.ffill(); dd0 = pd.Timestamp('2016-02-01')
    lu_next = (O.shift(-1) / Cf - 1 >= 0.095)
    rows = []
    for grp, pool in (('強勢股', strong), ('全市場', univ)):
        base_m = (pool & ~lu_next & ent.notna()).loc[dd0:]
        for k, M in list(REVEV.items()) + [('對照：收斂突破事件', anyev)]:
            if grp == '全市場' and k.startswith('對照'): continue
            o = {'群體': grp, '事件': k}
            for h in (5, 20, 60):
                Rr = (Cf.shift(-h) / ent - 1).loc[dd0:]; m = base_m & M.loc[dd0:] & Rr.notna()
                ex = Rr.sub(Rr.where(base_m).mean(1), axis=0).where(m).stack().dropna(); rr = Rr.where(m).stack().dropna()
                yr = ex.groupby(ex.index.get_level_values(0).year).mean()
                o[f'{h}日超額'] = f'{ex.mean():+.2%}'; o[f'{h}日正年'] = f'{(yr > 0).sum()}/{len(yr)}'
                if h == 20: o['20日勝率'] = f'{(rr > 0).mean():.0%}'
            o['每年次數'] = int(m.sum().sum() / 10.7); rows.append(o); print(o, flush=True)
    D = pd.DataFrame(rows); pd.set_option('display.width', 260); print(D.to_string(index=False))
    D.to_csv('/home/claude/bt/fac/rev_events.csv', index=False, encoding='utf-8-sig')
