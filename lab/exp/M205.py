# M-205：L-131 POS250 剔除的機制描述（奇數代碼、排除最近兩年）。定義見 registry_M.md M-205。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals
import M190
from M195 import HOLDOUT, ERAS, COST


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    signals.build('fix', mode='fix', P=P, F=F)
    S_ = signals.compute('fix', P=P, F=F); dbg = S_['_dbg']; cols = dbg['cols']
    def full(a, fill):
        a = np.asarray(a)
        if a.shape[1] == N: return a
        o = np.full((T, N), fill, a.dtype); o[:, cols] = a; return o
    U = full(dbg['U'], False); pool = full(dbg['pool'], False); hot = full(dbg['hot'], False); steep = full(dbg['steep'], False)
    Tr = full(np.asarray(dbg['Tr'], np.float64), np.nan); del S_
    FE = M190.features(P, F, Tr); POS = FE['POS250']
    VALID = np.asarray(F['VALID']); OA = np.asarray(F['OA'], np.float64); HA = pd.DataFrame(np.asarray(F['HA'], np.float64)).ffill()
    Cf = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill().to_numpy(); ATRP = np.asarray(signals.load_sig('fix')['ATRP'])
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CX = np.cumsum(R - ew[:, None], axis=0)
    H250 = HA.rolling(250, min_periods=200).max().to_numpy()
    odd = (P['meta']['stock_id'].astype(str).str[-1].astype(int) % 2 == 1).to_numpy()
    t0 = dates.searchsorted(pd.Timestamp('2005-01-01')); t1 = min(dates.searchsorted(HOLDOUT), T - 62)
    rows = []
    for t in range(t0, t1, 5):
        jp = np.nonzero(pool[t])[0]
        if len(jp) < 10: continue
        pr = pd.Series(POS[t, jp]).rank(pct=True).to_numpy(); low = pr < 0.2
        sel = odd[jp]; jj = jp[sel]; lw = low[sel]
        ent = np.where(np.isfinite(OA[t + 1, jj]), OA[t + 1, jj], Cf[t + 1, jj]); e60 = CX[t + 60, jj] - CX[t, jj]
        q = np.nanpercentile(CX[t + 60, U[t]] - CX[t, U[t]], [10, 90])
        rows.append(pd.DataFrame(dict(t=t, low=lw, hot=hot[t, jj], steep=steep[t, jj], tld=Tr[t, jj], dt10=Tr[t, jj] - Tr[t - 10, jj], r20=Cf[t, jj] / Cf[t - 20, jj] - 1,
                                      atrp=ATRP[t, jj], dh=Cf[t, jj] / H250[t, jj] - 1, e20=CX[t + 20, jj] - CX[t, jj], e60=e60, n20=Cf[t + 20, jj] / ent - 1 - COST,
                                      left=(e60 <= q[0]).astype(float), right=(e60 >= q[1]).astype(float))))
    D = pd.concat(rows, ignore_index=True); D['date'] = dates[D.t.to_numpy()]
    L = ['# M-205 結果：POS250 剔除的機制（強勢池、奇數代碼、排除最近兩年）\n']
    for era, lo, hi in ERAS:
        E = D[(D.date >= lo) & (D.date <= hi)]; cand = E[~E.hot & ~E.steep]
        lowc = cand[cand.low]; keep = cand[~cand.low]
        L += [f'## {era}（強勢池 {len(E):,} 筆；X1 候選＝剔 hot／steep 後 {len(cand):,} 筆）\n',
              f"POS 後 20% 中同時是 hot：{E[E.low].hot.mean():.0%}、steep：{E[E.low].steep.mean():.0%}；hot 中是 POS 後 20%：{E[E.hot].low.mean():.0%}；X1 候選中被 POS 剔除：{cand.low.mean():.0%}\n",
              '| 組 | 筆數 | T_ld | ΔT10 | 近 20 日漲幅 | ATR% | 距 250 日高 | 20日超額 | 60日超額 | 20日淨勝率 | 左尾 | 右尾 |', '|' + '---|' * 12]
        for nm, x in (('X1 候選：被 POS 剔除', lowc), ('X1 候選：保留', keep)):
            L.append(f"| {nm} | {len(x):,} | {x.tld.median():.2f} | {x.dt10.median():+.2f} | {x.r20.median():+.1%} | {x.atrp.median():.1%} | {x.dh.median():+.1%} | {x.e20.mean():+.2%} | {x.e60.mean():+.2%} | {(x.n20 > 0).mean():.0%} | {x.left.mean():.1%} | {x.right.mean():.1%} |")
        L.append('')
    return dict(summary='\n'.join(L), headline='POS 剔除機制', n_variants=0)
