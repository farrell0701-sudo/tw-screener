# M-134：融資融券因子（依 M-119；奇數代碼）。定義見 registry_M.md M-134。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'exp')); sys.path.insert(0, str(LAB / 'core'))
import M120
sys.path.insert(0, str(LAB / 'data'))
from M125 import monthly_from_daily
from M127 import tdcc_factors
from chips import _staged


def mg_factors(dates):
    df = _staged('TaiwanStockMarginPurchaseShortSale', ['date', 'stock_id', 'MarginPurchaseTodayBalance', 'MarginPurchaseLimit', 'ShortSaleTodayBalance'])
    def fn(W):
        mb, lim, sb = W['MarginPurchaseTodayBalance'], W['MarginPurchaseLimit'], W['ShortSaleTodayBalance']
        mbp = mb.where(mb > 0)
        use = (mb / lim.where(lim > 0)) * 100; ssr = sb / mbp
        return {'M1 融資 20 日變化率': mb / mbp.shift(20) - 1, 'M2 融資 60 日變化率': mb / mbp.shift(60) - 1,
                'M3 融資使用率 20 日變化': use - use.shift(20), 'M4 券資比': ssr, 'M5 券資比 20 日變化': ssr - ssr.shift(20)}
    return monthly_from_daily(df, ['MarginPurchaseTodayBalance', 'MarginPurchaseLimit', 'ShortSaleTodayBalance'], dates, fn)


def main():
    import pandas as pd
    from panel import load as load_panel
    dates = load_panel()['dates']
    ref = {'R4': M120.rev_factors()['R4 營收創高'], 'B4': tdcc_factors(dates)['B4 散戶 1 月變化']}
    L, summ = M120.evaluate(1, None, fac_fn=mg_factors, ref=ref)
    head = ['# M-134 結果：融資融券因子（奇數代碼；F 計數 33）\n', '| 因子 | 層 | 期間 | IC5 | IC20 | IC60 | IC120 | 高−低 60 日（t） |', '|---|---|---|---|---|---|---|---|']
    for fn, la, era, ic, sp, t in summ:
        head.append(f"| {fn} | {la} | {era} | " + ' | '.join(f'{v:+.3f}' for v in ic) + f" | {sp:+.2%}（{t:.1f}） |")
    return dict(summary='\n'.join(head + [''] + L), headline='融資融券因子評估完成', n_variants=0)
