# M-118：動能脈衝 × 過熱 × 最陡（強勢池內，觀察式，奇數代碼）。定義見 registry_M.md M-118。
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
SN = ['上收斂', '上平行', '上發散', '下']


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; U, pool, hot, steep = dbg['U'], dbg['pool'], dbg['hot'], dbg['steep']; del S
    st, _ = states(F); s4 = np.where(st >= 3, 3, st)
    Cf = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill().to_numpy(); VALID = np.asarray(F['VALID'])
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CX = np.cumsum(R - ew[:, None], axis=0)
    odd = (P['meta']['stock_id'].astype(str).str[-1].astype(int) % 2 == 1).to_numpy()
    L = ['# M-118 結果：動能脈衝 × 過熱 × 最陡（強勢池內，奇數代碼）\n',
         '「差」＝該格平均 − 同期整個強勢池平均（期數平均，NW t）；左／右尾＝同期強勢池後 20 日超額最差／最好 10%。\n']
    for era, lo, hi in ERAS:
        t_lo = int(dates.searchsorted(pd.Timestamp(lo))); t_hi = min(int(dates.searchsorted(pd.Timestamp(hi), side='right')) - 1, T - 61)
        rows, per = [], []
        for t in range(t_lo, t_hi + 1, 21):
            Sm = pool[t] & odd & (s4[t] >= 0)
            if Sm.sum() < 15: continue
            jj = np.nonzero(Sm)[0]
            e20 = CX[t + 20, jj] - CX[t, jj]; e60 = CX[t + 60, jj] - CX[t, jj]
            with np.errstate(all='ignore'): a20 = Cf[t + 20, jj] / Cf[t, jj] - 1
            q10, q90 = np.nanpercentile(e20, [10, 90])
            h, s_, k = hot[t, jj], steep[t, jj], s4[t, jj]
            grp = np.where(h & s_, '過熱且最陡', np.where(h, '過熱', np.where(s_, '最陡', '合格')))
            df = pd.DataFrame(dict(g=grp, k=k, e20=e20, e60=e60, a20=a20, left=e20 <= q10, right=e20 >= q90))
            m20, m60, ma = e20.mean(), e60.mean(), np.nanmean(a20)
            for (g, kk), x in df.groupby(['g', 'k']): per.append(dict(g=g, k=kk, d20=x.e20.mean() - m20, d60=x.e60.mean() - m60, da=np.nanmean(x.a20) - ma))
            for g, x in df.groupby('g'): per.append(dict(g=g, k=-1, d20=x.e20.mean() - m20, d60=x.e60.mean() - m60, da=np.nanmean(x.a20) - ma))
            rows.append(df)
        X = pd.concat(rows, ignore_index=True); Pd = pd.DataFrame(per)
        L += [f'## {era}（期數 {len(rows)}，觀測 {len(X):,}）\n', '### 重疊\n']
        L.append('| 組 | 占強勢池 | ' + ' | '.join(SN) + ' |'); L.append('|---|---|---|---|---|---|')
        for g in ('合格', '過熱', '最陡', '過熱且最陡'):
            x = X[X.g == g]
            L.append(f"| {g} | {len(x) / len(X):.0%} | " + ' | '.join(f"{(x.k == k).mean():.0%}" for k in range(4)) + ' |')
        L += ['\n### 各格後續報酬\n', '| 組 | 狀態 | 觀測 | 後20日超額 | 差（t） | 後60日差（t） | 後20日絕對差（t） | 左尾率 | 右尾率 |', '|---|---|---|---|---|---|---|---|---|']
        for g in ('合格', '過熱', '最陡', '過熱且最陡'):
            for k in [-1] + list(range(4)):
                x = X[X.g == g] if k < 0 else X[(X.g == g) & (X.k == k)]; p = Pd[(Pd.g == g) & (Pd.k == k)]
                nm = '（全部）' if k < 0 else SN[k]
                if len(x) < 30: L.append(f'| {g} | {nm} | {len(x)} | （樣本 < 30） | | | | | |'); continue
                L.append(f"| {g} | {nm} | {len(x)} | {x.e20.mean():+.2%} | {p.d20.mean():+.2%}（{nw_t(p.d20, 2):.1f}） | {p.d60.mean():+.2%}（{nw_t(p.d60, 3):.1f}） | "
                         f"{p.da.mean():+.2%}（{nw_t(p.da, 2):.1f}） | {x.left.mean():.0%} | {x.right.mean():.0%} |")
        L.append('')
    return dict(summary='\n'.join(L), headline='脈衝 × 過熱 × 最陡觀察完成', n_variants=0)
