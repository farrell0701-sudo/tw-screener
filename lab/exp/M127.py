# M-127：集保大戶持股因子（依 M-119；奇數代碼；期間 2010–18／2019–26）。定義見 registry_M.md M-127。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'exp')); sys.path.insert(0, str(LAB / 'core'))
import M120
from chips import _staged

ERAS = (('2010–18', '2010-01-01', '2018-12-31'), ('2019–26', '2019-01-01', '2026-12-31'))
BIG1000 = ['more than 1,000,001']
BIG400 = ['400,001-600,000', '600,001-800,000', '800,001-1,000,000', 'more than 1,000,001']
RETAIL = ['1-999', '1,000-5,000', '5,001-10,000', '10,001-15,000', '15,001-20,000', '20,001-30,000', '30,001-40,000', '40,001-50,000']


def tdcc_factors(dates):
    d = _staged('TaiwanStockHoldingSharesPer', ['date', 'stock_id', 'HoldingSharesLevel', 'people', 'percent'])
    piv = lambda lv, col='percent': d[d.HoldingSharesLevel.isin(lv)].groupby(['date', 'stock_id'])[col].sum().unstack()
    W = {'b1000': piv(BIG1000), 'b400': piv(BIG400), 'ret': piv(RETAIL), 'ppl': piv(['total'], 'people').astype(float)}
    obs = W['b1000'].index.sort_values()
    ds = pd.Series(dates, index=dates); s = ds[ds.dt.day > 10]; samp = s.groupby(s.dt.to_period('M')).min()
    rows = {k: {} for k in ('B1', 'B2', 'B3', 'B4', 'B5')}
    for p, t in samp.items():
        ti = dates.get_loc(t)
        if ti < 2: continue
        lim = dates[ti - 2]; o = obs[obs <= lim]
        if len(o) == 0: continue
        d0 = o[-1]; o1 = obs[obs <= d0 - pd.Timedelta(days=28)]; o3 = obs[obs <= d0 - pd.Timedelta(days=84)]
        if len(o1) == 0 or len(o3) == 0 or (d0 - o1[-1]).days > 45 or (d0 - o3[-1]).days > 110: continue
        d1, d3 = o1[-1], o3[-1]
        rows['B1'][p - 1] = W['b1000'].loc[d0] - W['b1000'].loc[d1]
        rows['B2'][p - 1] = W['b1000'].loc[d0] - W['b1000'].loc[d3]
        rows['B3'][p - 1] = W['b400'].loc[d0] - W['b400'].loc[d1]
        rows['B4'][p - 1] = W['ret'].loc[d0] - W['ret'].loc[d1]
        with np.errstate(all='ignore'): rows['B5'][p - 1] = W['ppl'].loc[d0] / W['ppl'].loc[d1] - 1
    names = {'B1': 'B1 千張大戶 1 月變化', 'B2': 'B2 千張大戶 3 月變化', 'B3': 'B3 400 張以上 1 月變化', 'B4': 'B4 散戶 1 月變化', 'B5': 'B5 股東人數 1 月變化率'}
    return {names[k]: pd.DataFrame(v).T.sort_index() for k, v in rows.items()}


def main():
    L, summ = M120.evaluate(1, None, fac_fn=tdcc_factors, ref={'R4': M120.rev_factors()['R4 營收創高']}, eras=ERAS)
    head = ['# M-127 結果：集保大戶持股因子（奇數代碼；期間 2010–18／2019–26；F 計數 21）\n', '| 因子 | 層 | 期間 | IC5 | IC20 | IC60 | IC120 | 高−低 60 日（t） |', '|---|---|---|---|---|---|---|---|']
    for fn, la, era, ic, sp, t in summ:
        head.append(f"| {fn} | {la} | {era} | " + ' | '.join(f'{v:+.3f}' for v in ic) + f" | {sp:+.2%}（{t:.1f}） |")
    return dict(summary='\n'.join(head + [''] + L), headline='集保大戶因子評估完成', n_variants=0)
