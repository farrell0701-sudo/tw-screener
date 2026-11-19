# M-125：估值因子（依 M-119；奇數代碼）。定義見 registry_M.md M-125。
import sys
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'exp')); sys.path.insert(0, str(LAB / 'core'))
import M120
from chips import _staged


def monthly_from_daily(df, cols, dates, fn):
    """df：date, stock_id, cols；回傳 {名稱: 以（取樣月−1）為索引的寬表}，值取取樣日前一交易日。"""
    ds = pd.Series(dates, index=dates); s = ds[ds.dt.day > 10]; samp = s.groupby(s.dt.to_period('M')).min()
    prev = {p: dates[dates.get_loc(t) - 1] for p, t in samp.items() if dates.get_loc(t) > 0}
    out = {}
    for c in cols:
        W = df.pivot_table(index='date', columns='stock_id', values=c, aggfunc='last').reindex(dates).ffill(limit=5)
        out[c] = W
    res = fn(out)
    return {k: pd.DataFrame({p - 1: v.loc[d] for p, d in prev.items() if d in v.index}).T.sort_index() for k, v in res.items()}


def val_factors(dates):
    df = _staged('TaiwanStockPER', ['date', 'stock_id', 'dividend_yield', 'PER', 'PBR'])
    def fn(W):
        pb, pe = W['PBR'].where(W['PBR'] > 0), W['PER'].where(W['PER'] > 0)
        return {'V1 B/P': 1 / pb, 'V2 E/P': 1 / pe, 'V3 現金殖利率': W['dividend_yield']}
    return monthly_from_daily(df, ['PBR', 'PER', 'dividend_yield'], dates, fn)


def report(title, fac_fn):
    L, summ = M120.evaluate(1, None, fac_fn=fac_fn, ref={'R4': M120.rev_factors()['R4 營收創高']})
    head = [title, '| 因子 | 層 | 期間 | IC5 | IC20 | IC60 | IC120 | 高−低 60 日（t） |', '|---|---|---|---|---|---|---|---|']
    for fn, la, era, ic, sp, t in summ:
        head.append(f"| {fn} | {la} | {era} | " + ' | '.join(f'{v:+.3f}' for v in ic) + f" | {sp:+.2%}（{t:.1f}） |")
    return '\n'.join(head + [''] + L)


def main():
    return dict(summary=report('# M-125 結果：估值因子（奇數代碼；F 計數 13）\n', val_factors), headline='估值因子評估完成', n_variants=0)
