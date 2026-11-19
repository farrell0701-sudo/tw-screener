# M-117：動能脈衝在 T_ld 前 50% 股池的分辨力（觀察式，奇數代碼）。定義見 registry_M.md M-117。
import sys
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals
from M107 import nw_t, NAMES
from M113 import states

ERAS = (('2005–15', '2005-01-01', '2015-12-31'), ('2016–26', '2016-01-01', '2026-12-31'))


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; U, pool, Tr = dbg['U'], dbg['pool'], dbg['Tr']
    if Tr.shape[1] != N:
        full = np.full((T, N), np.nan); full[:, dbg['cols']] = Tr; Tr = full
    pr = pd.DataFrame(np.where(U, Tr, np.nan)).rank(axis=1, pct=True).to_numpy()
    top50 = U & (np.nan_to_num(pr) >= 0.5)
    st, _ = states(F)
    Cf = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill().to_numpy(); VALID = np.asarray(F['VALID'])
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CX = np.cumsum(R - ew[:, None], axis=0)
    odd = (P['meta']['stock_id'].astype(str).str[-1].astype(int) % 2 == 1).to_numpy()
    L = ['# M-117 結果：動能脈衝在 T_ld 前 50% 股池的分辨力（觀察式，奇數代碼）\n',
         '超額＝扣可交易池等權；「差」＝該組平均 − 同期同層平均（期數平均，NW t）；左／右尾＝同期同層後 20 日超額的最差／最好 10%。\n']
    for era, lo, hi in ERAS:
        t_lo = int(dates.searchsorted(pd.Timestamp(lo))); t_hi = min(int(dates.searchsorted(pd.Timestamp(hi), side='right')) - 1, T - 61)
        for layer, mask in (('前 50% 全部', top50), ('前 10%（強勢池）', pool), ('10–50%', top50 & ~pool)):
            rows, per = [], []
            for t in range(t_lo, t_hi + 1, 21):
                Sm = mask[t] & odd & (st[t] >= 0)
                if Sm.sum() < 15: continue
                jj = np.nonzero(Sm)[0]
                e20 = CX[t + 20, jj] - CX[t, jj]; e60 = CX[t + 60, jj] - CX[t, jj]
                with np.errstate(all='ignore'): a20 = Cf[t + 20, jj] / Cf[t, jj] - 1
                q10, q90 = np.nanpercentile(e20, [10, 90]); s = st[t, jj]
                for k in range(6):
                    m = s == k
                    if m.sum(): per.append(dict(k=k, d20=e20[m].mean() - e20.mean(), d60=e60[m].mean() - e60.mean(), da=np.nanmean(a20[m]) - np.nanmean(a20)))
                rows.append(pd.DataFrame(dict(k=s, e20=e20, e60=e60, a20=a20, left=e20 <= q10, right=e20 >= q90)))
            X = pd.concat(rows, ignore_index=True); Pd = pd.DataFrame(per)
            L += [f'## {era}｜{layer}（期數 {len(rows)}，觀測 {len(X):,}）\n',
                  '| 狀態 | 占比 | 後20日超額 | 差（t） | 後60日超額 | 差（t） | 後20日絕對 | 差（t） | 勝率 | 左尾率 | 右尾率 |', '|---|---|---|---|---|---|---|---|---|---|---|']
            for k, nm in enumerate(NAMES):
                x = X[X.k == k]; p = Pd[Pd.k == k]
                if len(x) < 30: L.append(f'| {nm} | {len(x) / len(X):.0%} | （樣本 < 30） | | | | | | | | |'); continue
                L.append(f"| {nm} | {len(x) / len(X):.0%} | {x.e20.mean():+.2%} | {p.d20.mean():+.2%}（{nw_t(p.d20, 2):.1f}） | {x.e60.mean():+.2%} | {p.d60.mean():+.2%}（{nw_t(p.d60, 3):.1f}） | "
                         f"{np.nanmean(x.a20):+.2%} | {p.da.mean():+.2%}（{nw_t(p.da, 2):.1f}） | {(x.e20 > 0).mean():.0%} | {x.left.mean():.0%} | {x.right.mean():.0%} |")
            L.append('')
    return dict(summary='\n'.join(L), headline='前 50% 股池脈衝觀察完成', n_variants=0)
