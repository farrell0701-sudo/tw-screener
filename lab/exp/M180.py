# M-180：強勢池（X1 候選）內 D2 對 X1 分數的增量與頭部選股——偶數代碼確認集。定義見 registry_M.md M-180。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals
from M107 import nw_t
import M165

PARITY = 0


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    par = (P['meta']['stock_id'].astype(str).str[-1].astype(int) % 2 == PARITY).to_numpy()
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; cols = dbg['cols']
    def full(a, fill):
        if a.shape[1] == N: return a
        o = np.full((T, N), fill, a.dtype); o[:, cols] = a; return o
    U, pool, hot, steep = full(dbg['U'], False), full(dbg['pool'], False), full(dbg['hot'], False), full(dbg['steep'], False)
    Tr = full(dbg['Tr'], np.nan); del S
    cand = pool & ~hot & ~steep
    VALID = np.asarray(F['VALID']); CAv = np.where(VALID, np.asarray(F['CA'], np.float64), np.nan); LA = np.asarray(F['LA'], np.float64)
    Cf = pd.DataFrame(CAv).ffill().to_numpy()
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CX = np.cumsum(R - ew[:, None], axis=0)
    pr = lambda a: pd.Series(a).rank(pct=True).to_numpy()
    sp = lambda a, b: pd.Series(a).rank().corr(pd.Series(b).rank())
    inc, rows, ovl = [], [], []
    for m, (t, tp) in M165.samp_prev(dates).items():
        if dates[t] < pd.Timestamp('2005-01-01') or t + 121 >= T: continue
        era = '2005–15' if dates[t] <= pd.Timestamp('2015-12-31') else '2016–26'
        s0 = dates.searchsorted((dates[tp].to_period('M') - 1).start_time)
        jj = np.nonzero(cand[tp] & par)[0]
        if len(jj) < 12: continue
        w = CAv[s0:tp + 1][:, jj]; cnt = np.isfinite(w).sum(0)
        with np.errstate(all='ignore'): h, l = np.nanmax(w, 0), np.nanmin(w, 0)
        c = Cf[tp, jj]; tl = Tr[tp, jj]; dtl = Tr[tp, jj] - Tr[max(0, tp - 10), jj]
        e60 = CX[t + 60, jj] - CX[t, jj]; e120 = CX[t + 120, jj] - CX[t, jj]
        ok = (cnt >= 15) & (h > l) & np.isfinite(c) & np.isfinite(tl) & np.isfinite(dtl) & np.isfinite(e60) & np.isfinite(e120)
        jj, h, l, c, tl, dtl, e60, e120 = jj[ok], h[ok], l[ok], c[ok], tl[ok], dtl[ok], e60[ok], e120[ok]
        if len(jj) < 12: continue
        d2 = (c - (h + l) / 2) / (h - l)
        x1 = 0.5 * pr(tl) + 0.5 * pr(dtl); cmb = (pr(tl) + pr(dtl) + pr(d2)) / 3
        X = np.column_stack([np.ones(len(jj)), x1]); res = d2 - X @ np.linalg.lstsq(X, d2, rcond=None)[0]
        inc.append((era, sp(res, e60), sp(res, e120)))
        e60u = (CX[t + 60] - CX[t])[U[t]]; q10, q90 = np.nanpercentile(e60u, [10, 90])
        a6 = np.argsort(-x1, kind='stable')[:6]; b6 = np.argsort(-cmb, kind='stable')[:6]
        ovl.append((era, len(set(a6) & set(b6)) / 6))
        for g, idx in (('X1 前 6', a6), ('合併前 6', b6), ('候選全部', np.arange(len(jj)))):
            for k in idx: rows.append((str(m), era, g, e60[k], e120[k], q10, q90))
    I = pd.DataFrame(inc, columns=['era', 'ic60', 'ic120']); D = pd.DataFrame(rows, columns=['m', 'era', 'g', 'e60', 'e120', 'q10', 'q90'])
    D['L'] = D.e60 <= D.q10; D['Rt'] = D.e60 >= D.q90; O = pd.DataFrame(ovl, columns=['era', 'o'])
    L = ['# M-180 結果：強勢池 D2 對 X1 分數的增量與頭部選股｜偶數代碼確認集\n']
    r1 = {}
    tab1 = ['## 判準 1：殘差 IC\n', '| 期間 | 殘差 IC60（t） | 殘差 IC120（t） | 期數 |', '|---|---|---|---|']
    for era, x in I.groupby('era'):
        t60, t120 = nw_t(x.ic60.to_numpy(), 3), nw_t(x.ic120.to_numpy(), 3); r1[era] = (x.ic60.mean(), t60, x.ic120.mean())
        tab1.append(f"| {era} | {x.ic60.mean():+.3f}（{t60:.1f}） | {x.ic120.mean():+.3f}（{t120:.1f}） | {len(x)} |")
    a, b = r1['2005–15'], r1['2016–26']
    c1 = a[0] > 0 and b[0] > 0 and a[2] > 0 and b[2] > 0 and max(a[1], b[1]) >= 2 and min(a[1], b[1]) >= 1
    tab2 = ['', '## 判準 2：頭部選股\n', '| 組 | 期間 | 筆數 | 60 日 | 120 日 | 60 日中位數 | 左尾 | 右尾 |', '|---|---|---|---|---|---|---|---|']
    r2 = {}
    for era in ('2005–15', '2016–26'):
        for g in ('候選全部', 'X1 前 6', '合併前 6'):
            x = D[(D.era == era) & (D.g == g)]
            tab2.append(f"| {g} | {era} | {len(x)} | {x.e60.mean():+.2%} | {x.e120.mean():+.2%} | {x.e60.median():+.2%} | {x.L.mean():.0%} | {x.Rt.mean():.0%} |")
        ga = D[(D.era == era) & (D.g == '合併前 6')]; gb = D[(D.era == era) & (D.g == 'X1 前 6')]
        d = (ga.groupby('m').e60.mean() - gb.groupby('m').e60.mean()).dropna()
        r2[era] = (d.mean(), nw_t(d.to_numpy(), 3), ga.L.mean(), gb.L.mean())
    a, b = r2['2005–15'], r2['2016–26']
    c2 = a[0] > 0 and b[0] > 0 and max(a[1], b[1]) >= 1.5 and a[2] <= a[3] + 0.01 and b[2] <= b[3] + 0.01
    tab2 += ['', '| 期間 | 合併前 6 − X1 前 6（60 日，NW t） | 前 6 重疊比例 |', '|---|---|---|']
    for era in ('2005–15', '2016–26'):
        tab2.append(f"| {era} | {r2[era][0]:+.2%}（{r2[era][1]:.1f}） | {O[O.era == era].o.mean():.0%} |")
    mk = lambda f: '✓' if f else '✗'
    L.append(f"**{'確認通過' if (c1 and c2) else '確認不通過'}**（判準 1 增量 {mk(c1)}；判準 2 頭部 {mk(c2)}）\n")
    return dict(summary='\n'.join(L + tab1 + tab2), headline=f"強勢池 D2 確認集：增量{mk(c1)} 頭部{mk(c2)}", n_variants=0)
