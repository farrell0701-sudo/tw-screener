# M-138：當沖比因子（依 M-119；奇數代碼；2016–20／2021–26）。定義見 registry_M.md M-138。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
import M120
from M127 import tdcc_factors
from chips import _staged, to_grid

ERAS = (('2016–20', '2016-01-01', '2020-12-31'), ('2021–26', '2021-01-01', '2026-12-31'))


def dt_factors(dates):
    P = load_panel(); keys = P['meta']['security_key'].astype(str).to_numpy()
    d = _staged('TaiwanStockDayTrading', ['date', 'stock_id', 'Volume'])
    DT = pd.DataFrame(to_grid(d, 'Volume', P), index=dates).astype('float64'); del d
    V = pd.DataFrame(np.asarray(P['V'], np.float64), index=dates); V[~(V > 0)] = np.nan
    DT = DT.where(V.notna())
    r20 = DT.rolling(20, min_periods=15).sum() / V.rolling(20, min_periods=15).sum()
    r5 = DT.rolling(5, min_periods=4).sum() / V.rolling(5, min_periods=4).sum()
    r60 = DT.rolling(60, min_periods=45).sum() / V.rolling(60, min_periods=45).sum()
    dl = DT / V; a250 = dl.rolling(250, min_periods=150).mean()
    res = {'D1 當沖比 20 日': r20, 'D2 當沖比突增': r5 - r60, 'D3 當沖比相對自身': r20 / a250.where(a250 > 0)}
    ds = pd.Series(dates, index=dates); s = ds[ds.dt.day > 10]; samp = s.groupby(s.dt.to_period('M')).min()
    prev = {p: dates[dates.get_loc(t) - 1] for p, t in samp.items() if dates.get_loc(t) > 0 and t >= pd.Timestamp('2016-01-01')}
    out = {}
    for k, W in res.items():
        W.columns = keys; W = W.loc[:, ~W.columns.duplicated()]
        out[k] = pd.DataFrame({p - 1: W.loc[dd] for p, dd in prev.items()}).T.sort_index()
    return out


def main():
    dates = load_panel()['dates']
    ref = {'R4': M120.rev_factors()['R4 營收創高'], 'B4': tdcc_factors(dates)['B4 散戶 1 月變化']}
    L, summ = M120.evaluate(1, None, fac_fn=dt_factors, ref=ref, eras=ERAS)
    head = ['# M-138 結果：當沖比因子（奇數代碼；期間 2016–20／2021–26；F 計數 43）\n', '| 因子 | 層 | 期間 | IC5 | IC20 | IC60 | IC120 | 高−低 60 日（t） |', '|---|---|---|---|---|---|---|---|']
    for fn, la, era, ic, sp, t in summ:
        head.append(f"| {fn} | {la} | {era} | " + ' | '.join(f'{v:+.3f}' for v in ic) + f" | {sp:+.2%}（{t:.1f}） |")
    return dict(summary='\n'.join(head + [''] + L), headline='當沖比因子評估完成', n_variants=0)
