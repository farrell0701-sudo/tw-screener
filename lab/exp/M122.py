# M-122：財報（綜合損益表）因子（依 M-119；奇數代碼）。定義見 registry_M.md M-122。
import sys
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'exp'))
import M120

TWD = M120.TWD; TY = ['Revenue', 'GrossProfit', 'OperatingIncome', 'EPS']


def fin_factors(dates):
    fs = []
    for y in range(2002, 2027):
        f = TWD / 'derived' / 'financials' / f'{y}.parquet'
        if not f.exists(): continue
        d = pd.read_parquet(f); d = d[(d.statement == 'income') & d.type.isin(TY) & d.security_key.notna() & d.security_key.ne('')]
        fs.append(d[['security_key', 'period_end', 'type', 'value', 'available_date']])
    D = pd.concat(fs)
    W = D.pivot_table(index=['security_key', 'period_end'], columns='type', values='value', aggfunc='last')
    W['avail'] = D.groupby(['security_key', 'period_end']).available_date.max()
    W = W.reset_index(); W['q'] = W.period_end.dt.to_period('Q')
    W.loc[~(W.Revenue > 0), 'Revenue'] = np.nan
    W['GM'] = W.GrossProfit / W.Revenue; W['OM'] = W.OperatingIncome / W.Revenue
    out = []
    for k, g in W.groupby('security_key'):
        g = g.set_index('q').sort_index(); g = g[~g.index.duplicated(keep='last')]
        g = g.reindex(pd.period_range(g.index.min(), g.index.max(), freq='Q'))
        oi4 = g.OperatingIncome.shift(4)
        with np.errstate(all='ignore'):
            f = pd.DataFrame(index=g.index)
            f['Q1'] = (g.OperatingIncome / oi4 - 1).where(oi4 > 0)
            f['Q2'] = g.GM - g.GM.shift(4)
            f['Q3'] = g.OM - g.OM.shift(4)
            de = g.EPS - g.EPS.shift(4); sd = de.shift(1).rolling(8, min_periods=6).std()
            f['Q4'] = de / sd.where(sd > 0)
            f['Q5'] = g.GM - g.GM.shift(1).rolling(8, min_periods=6).max()
        f['key'] = k; f['avail'] = g.avail; f['pe'] = g.period_end
        out.append(f.dropna(subset=['avail']))
    Fq = pd.concat(out).sort_values('avail')
    ds = pd.Series(dates, index=dates); s = ds[ds.dt.day > 10]; samp = s.groupby(s.dt.to_period('M')).min()
    S = pd.DataFrame({'t': samp.values}).sort_values('t')
    res = {}
    names = {'Q1': 'Q1 營益年增', 'Q2': 'Q2 毛利率年變化', 'Q3': 'Q3 營益率年變化', 'Q4': 'Q4 EPS驚喜', 'Q5': 'Q5 毛利率創高'}
    frames = {c: [] for c in names}
    for t in S.t:
        # 可取得日 < 取樣日 的最新一季
        sub = Fq[Fq.avail < t]
        last = sub.groupby('key').tail(1)
        last = last[(t - last.pe).dt.days <= 210]
        for c in names: frames[c].append(last.set_index('key')[c].rename(t.to_period('M') - 1))
    for c, nm in names.items():
        res[nm] = pd.DataFrame(frames[c]).sort_index()
    return res


def main():
    ref = {'R4': M120.rev_factors()['R4 營收創高']}
    L, summ = M120.evaluate(1, None, fac_fn=fin_factors, ref=ref)
    head = ['# M-122 結果：財報因子（M-119 標準；奇數代碼；F 計數 10）\n', '| 因子 | 層 | 期間 | IC5 | IC20 | IC60 | IC120 | 高−低 60 日（t） |', '|---|---|---|---|---|---|---|---|']
    for fn, la, era, ic, sp, t in summ:
        head.append(f"| {fn} | {la} | {era} | " + ' | '.join(f'{v:+.3f}' for v in ic) + f" | {sp:+.2%}（{t:.1f}） |")
    return dict(summary='\n'.join(head + [''] + L), headline='財報因子評估完成', n_variants=0)
