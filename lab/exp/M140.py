# M-140：上發散＋（過熱或最陡）是否最容易快速回跌（強勢池、奇數代碼）。定義見 registry_M.md M-140。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals
from M107 import nw_t
from M113 import states

ERAS = (('2005–15', '2005-01-01', '2015-12-31'), ('2016–26', '2016-01-01', '2026-12-31'))
GN = {'A': '上發散＋（過熱或最陡）', 'B': '上發散、非過熱非最陡', 'C': '其他狀態＋（過熱或最陡）', 'D': '其他'}


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    S = signals.compute('fix', P=P, F=F); d = S['_dbg']; U, pool, hot, steep = d['U'], d['pool'], d['hot'], d['steep']
    ATRP = np.asarray(S['ATRP'], np.float64); del S
    st, _ = states(F)
    Cf = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill().to_numpy(); VALID = np.asarray(F['VALID'])
    HA = np.asarray(F['HA'], np.float64); LA = np.asarray(F['LA'], np.float64)
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CX = np.cumsum(R - ew[:, None], axis=0)
    odd = (P['meta']['stock_id'].astype(str).str[-1].astype(int) % 2 == 1).to_numpy()
    L = ['# M-140 結果：上發散＋（過熱或最陡）是否最容易快速回跌（強勢池、奇數代碼）\n',
         '自取樣日收盤起；ATR 跌幅＝10 日內最低價跌幅 ÷（ATR％ × 收盤）；先跌／先漲＝20 日內先觸及 −10% 或 +15%；差＝與同期強勢池平均之差。\n']
    for era, lo, hi in ERAS:
        t_lo = int(dates.searchsorted(pd.Timestamp(lo))); t_hi = min(int(dates.searchsorted(pd.Timestamp(hi), side='right')) - 1, T - 61)
        rows, per = [], []
        for t in range(t_lo, t_hi + 1, 21):
            sm = pool[t] & odd & (st[t] >= 0)
            if sm.sum() < 15: continue
            jj = np.nonzero(sm)[0]; c0 = Cf[t, jj]
            hs = hot[t, jj] | steep[t, jj]; div = st[t, jj] == 2
            g = np.where(div & hs, 'A', np.where(div, 'B', np.where(hs, 'C', 'D')))
            with np.errstate(all='ignore'):
                lo10 = np.nanmin(LA[t + 1:t + 11, jj], axis=0) / c0 - 1; lo20 = np.nanmin(LA[t + 1:t + 21, jj], axis=0) / c0 - 1
                hi20 = np.nanmax(HA[t + 1:t + 21, jj], axis=0) / c0 - 1
                atr = ATRP[t, jj]; atrdrop = -lo10 / atr
                dn_hit = (LA[t + 1:t + 21, jj] / c0 - 1) <= -0.10; up_hit = (HA[t + 1:t + 21, jj] / c0 - 1) >= 0.15
            fd = np.where(dn_hit.any(0), dn_hit.argmax(0), 99); fu = np.where(up_hit.any(0), up_hit.argmax(0), 99)
            first = np.where((fd < fu), 'down', np.where(fu < fd, 'up', np.where(fd == 99, 'none', 'same')))
            e20 = CX[t + 20, jj] - CX[t, jj]; e60 = CX[t + 60, jj] - CX[t, jj]
            df = pd.DataFrame(dict(g=g, d10=lo10 <= -0.10, d20=lo20 <= -0.10, d20b=lo20 <= -0.15, atr3=atrdrop >= 3, u15=hi20 >= 0.15,
                                   fdown=first == 'down', fup=first == 'up', e20=e20, e60=e60))
            for k, x in df.groupby('g'): per.append(dict(g=k, d20=x.e20.mean() - e20.mean(), d60=x.e60.mean() - e60.mean()))
            rows.append(df)
        X = pd.concat(rows, ignore_index=True); Pd = pd.DataFrame(per)
        L += [f'## {era}（期數 {len(rows)}，觀測 {len(X):,}）\n',
              '| 組 | 觀測 | 10日跌≥10% | 20日跌≥10% | 20日跌≥15% | 10日跌≥3ATR | 20日漲≥15% | 先跌10% | 先漲15% | 後20日差（t） | 後60日差（t） |', '|---|---|---|---|---|---|---|---|---|---|---|']
        for k in 'ABCD':
            x = X[X.g == k]; pp = Pd[Pd.g == k]
            if len(x) < 30: L.append(f'| {GN[k]} | {len(x)} | （樣本 < 30） |||||||||'); continue
            L.append(f"| {GN[k]} | {len(x)} | {x.d10.mean():.1%} | {x.d20.mean():.1%} | {x.d20b.mean():.1%} | {x.atr3.mean():.1%} | {x.u15.mean():.1%} | {x.fdown.mean():.1%} | {x.fup.mean():.1%} | "
                     f"{pp.d20.mean():+.2%}（{nw_t(pp.d20, 1):.1f}） | {pp.d60.mean():+.2%}（{nw_t(pp.d60, 3):.1f}） |")
        L.append('')
    return dict(summary='\n'.join(L), headline='快速回跌位階觀察完成', n_variants=0)
