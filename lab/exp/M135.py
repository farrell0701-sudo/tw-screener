# M-135：量價匹配因子（Alpha#15）短線表現（每日取樣、奇數代碼）。定義見 registry_M.md M-135。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals
from M107 import nw_t

HS = (1, 2, 5, 10, 20); ERAS = (('2005–15', '2005-01-01', '2015-12-31'), ('2016–26', '2016-01-01', '2026-12-31'))
COST = 0.0077


def xrank(A, M):
    return pd.DataFrame(np.where(M, A, np.nan)).rank(axis=1, pct=True)


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    S = signals.compute('fix', P=P, F=F); U, pool = S['_dbg']['U'], S['_dbg']['pool']; del S
    HA = np.asarray(F['HA'], np.float64); V = np.asarray(P['V'], np.float64); V[~(V > 0)] = np.nan
    hr, vr = xrank(HA, U), xrank(V, U)
    mx, my = hr.rolling(3, min_periods=3).mean(), vr.rolling(3, min_periods=3).mean()
    cxy = (hr * vr).rolling(3, min_periods=3).mean() - mx * my
    sx = (hr ** 2).rolling(3, min_periods=3).mean() - mx ** 2; sy = (vr ** 2).rolling(3, min_periods=3).mean() - my ** 2
    with np.errstate(all='ignore'): corr = cxy / np.sqrt(sx * sy)
    corr = corr.where((sx > 1e-12) & (sy > 1e-12))
    fac = -(xrank(corr.to_numpy(), U).rolling(3, min_periods=3).sum()).to_numpy()
    del hr, vr, mx, my, cxy, sx, sy, corr
    Cf = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill().to_numpy(); VALID = np.asarray(F['VALID'])
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CX = np.cumsum(R - ew[:, None], axis=0)
    with np.errstate(all='ignore'): r5 = Cf / np.vstack([np.full((5, N), np.nan), Cf[:-5]]) - 1
    R20 = np.asarray(F['R20'], np.float64)
    odd = (P['meta']['stock_id'].astype(str).str[-1].astype(int) % 2 == 1).to_numpy()
    rk = lambda a: pd.Series(a).rank().to_numpy()
    L = ['# M-135 結果：量價匹配因子（Alpha#15）短線表現（每日取樣、奇數代碼）\n',
         '超額＝扣可交易池等權；「延遲」＝從 t+1 收盤起算；換手成本＝(1 − 頭組逐日保留) × 0.77%（每日）。\n']
    for layer, mask, ng in (('全市場', U, 10), ('強勢池', pool, 5)):
        for era, lo, hi in ERAS:
            t_lo = int(dates.searchsorted(pd.Timestamp(lo))); t_hi = min(int(dates.searchsorted(pd.Timestamp(hi), side='right')) - 1, T - 22)
            ic = {(d, h): [] for d in (0, 1) for h in HS}; sp = {h: [] for h in (1, 5, 20)}; keep = []; prev = None; c5, c20 = [], []
            for t in range(max(t_lo, 10), t_hi + 1):
                sel = mask[t] & odd & np.isfinite(fac[t])
                if sel.sum() < ng * 5: continue
                jj = np.nonzero(sel)[0]; x = fac[t, jj]; xr = rk(x)
                for d in (0, 1):
                    for h in HS:
                        y = CX[t + d + h, jj] - CX[t + d, jj]
                        ic[(d, h)].append(np.corrcoef(xr, rk(y))[0, 1])
                g = pd.qcut(pd.Series(x).rank(method='first'), ng, labels=False).to_numpy()
                for h in sp:
                    y = CX[t + h, jj] - CX[t, jj]; sp[h].append(np.nanmean(y[g == ng - 1]) - np.nanmean(y[g == 0]))
                top = set(jj[g == ng - 1])
                if prev: keep.append(len(top & prev) / len(prev))
                prev = top
                if t % 5 == 0:
                    for arr, lst in ((r5, c5), (R20, c20)):
                        a = arr[t, jj]; ok = np.isfinite(a)
                        if ok.sum() > 10: lst.append(np.corrcoef(xr[ok], rk(a[ok]))[0, 1])
            kp = np.mean(keep)
            L += [f'## {layer}｜{era}（{len(sp[1])} 日；與 5 日報酬相關 {np.mean(c5):+.2f}、與 R20 相關 {np.mean(c20):+.2f}；頭組逐日保留 {kp:.0%}，每日換手成本約 {(1 - kp) * COST:.2%}）\n',
                  '| 起點 | ' + ' | '.join(f'IC{h}' for h in HS) + ' |', '|---|---|---|---|---|---|']
            for d, nm in ((0, 't 收盤'), (1, '延遲一日')):
                L.append(f'| {nm} | ' + ' | '.join(f"{np.mean(ic[(d, h)]):+.3f}（{nw_t(ic[(d, h)], h):.1f}）" for h in HS) + ' |')
            L += ['', '| 持有 | 高−低（t） | 每日平均 | 每日換手成本 |', '|---|---|---|---|']
            for h in sp:
                L.append(f"| {h} 日 | {np.mean(sp[h]):+.2%}（{nw_t(sp[h], h):.1f}） | {np.mean(sp[h]) / h:+.3%} | {(1 - kp) * COST:.3%} |")
            L.append('')
    return dict(summary='\n'.join(L), headline='量價匹配因子短線評估完成', n_variants=0)
