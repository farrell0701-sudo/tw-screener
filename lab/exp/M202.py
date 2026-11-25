# M-202：底層邏輯堆疊地圖（全市場、強勢池逐層增量；奇數代碼、排除最近兩年）。定義見 registry_M.md M-202。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals
import M120, M180, M190
from L123 import daily_from_monthly
from M195 import HOLDOUT, ERAS
from M107 import nw_t

ORDER = ['T_ld', 'R6', 'POS250', 'QUIET', 'R5', 'R4', 'ΔT60']


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    signals.build('fix', mode='fix', P=P, F=F)
    S_ = signals.compute('fix', P=P, F=F); dbg = S_['_dbg']; cols = dbg['cols']
    def full(a, fill):
        a = np.asarray(a)
        if a.shape[1] == N: return a
        o = np.full((T, N), fill, a.dtype); o[:, cols] = a; return o
    U = full(dbg['U'], False); pool = full(dbg['pool'], False); Tr = full(np.asarray(dbg['Tr'], np.float64), np.nan); del S_
    VALID = np.asarray(F['VALID'])
    Cf = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill().to_numpy()
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CX = np.cumsum(R - ew[:, None], axis=0)
    keys = P['meta']['security_key'].astype(str).to_numpy(); odd = (P['meta']['stock_id'].astype(str).str[-1].astype(int) % 2 == 1).to_numpy()
    FE = M190.features(P, F, Tr); rv = M120.rev_factors(dates)
    FD = {'T_ld': Tr, 'R6': daily_from_monthly(M180.r6_factors(dates)['R6 營收短長比'], dates, keys), 'POS250': FE['POS250'],
          'VC': FE['VC'], 'VX': FE['VX'], 'R5': daily_from_monthly(rv['R5 年增驚喜'], dates, keys), 'R4': daily_from_monthly(rv['R4 營收創高'], dates, keys),
          'ΔT60': Tr - np.vstack([np.full((60, N), np.nan), Tr[:-60]])}
    t0 = dates.searchsorted(pd.Timestamp('2005-01-01')); t1 = min(dates.searchsorted(HOLDOUT), T - 62)
    L = ['# M-202 結果：底層邏輯堆疊地圖（奇數代碼、排除最近兩年；每 5 日取樣）\n', '殘差 IC＝該層因子對前面所有層的百分位回歸後殘差的 IC；NW t。\n']
    for layer, M in (('全市場 U', U), ('強勢池（T_ld 前 10%）', pool)):
        rows = []
        for t in range(t0, t1, 5):
            jj = np.nonzero(M[t] & VALID[t] & odd)[0]
            if len(jj) < 30: continue
            d = dict(t=t, e20=CX[t + 20, jj] - CX[t, jj], e60=CX[t + 60, jj] - CX[t, jj])
            for k, A in FD.items(): d[k] = np.asarray(A[t, jj], np.float64)
            rows.append(pd.DataFrame(d))
        D = pd.concat(rows, ignore_index=True); D['date'] = dates[D.t.to_numpy()]
        D = D[np.isfinite(D.e60)]
        for era, lo, hi in ERAS:
            E = D[(D.date >= lo) & (D.date <= hi)].copy()
            g = E.groupby('t')
            pc = {k: g[k].rank(pct=True) for k in FD}
            E = E.assign(**{k + '_p': pc[k] for k in FD}); E['QUIET_p'] = (2 - E.VC_p - E.VX_p) / 2
            E = E.dropna(subset=[k + '_p' for k in ORDER])
            L += [f"## {layer}｜{era}（{len(E):,} 筆、{E.t.nunique()} 個取樣日）\n", '| 層 | 因子 | 原始 IC20（t） | 原始 IC60（t） | 殘差 IC20（t） | 殘差 IC60（t） | 對前面各層的 R² |', '|---|---|---|---|---|---|---|']
            for i, f in enumerate(ORDER):
                def stat(col, y):
                    s = E.groupby('t').apply(lambda g: g[col].rank().corr(g[y].rank())).dropna(); return s.mean(), nw_t(s, 4 if y == 'e20' else 12)
                prev = [o + '_p' for o in ORDER[:i]]
                if prev:
                    def resid(g):
                        X = np.column_stack([np.ones(len(g))] + [g[c] for c in prev]); y = g[f + '_p'].to_numpy()
                        b = np.linalg.lstsq(X, y, rcond=None)[0]; r = y - X @ b
                        return pd.Series(r, index=g.index)
                    E['_r'] = E.groupby('t', group_keys=False).apply(resid)
                    r2 = 1 - E._r.var() / E[f + '_p'].var()
                else: E['_r'] = E[f + '_p']; r2 = 0.0
                a20, a60 = stat(f + '_p', 'e20'), stat(f + '_p', 'e60'); b20, b60 = stat('_r', 'e20'), stat('_r', 'e60')
                L.append(f"| L{i + 1} | {f} | {a20[0]:+.3f}（{a20[1]:.1f}） | {a60[0]:+.3f}（{a60[1]:.1f}） | {b20[0]:+.3f}（{b20[1]:.1f}） | {b60[0]:+.3f}（{b60[1]:.1f}） | {r2:.2f} |")
            cm = E[[o + '_p' for o in ORDER]].corr(); cm.index = cm.columns = ORDER
            L += ['', '相關（|ρ| ≥ 0.3）：' + ('；'.join(f'{a}–{b} {cm.loc[a, b]:+.2f}' for i, a in enumerate(ORDER) for b in ORDER[i + 1:] if abs(cm.loc[a, b]) >= 0.3) or '無') + '\n']
    return dict(summary='\n'.join(L), headline='堆疊地圖', n_variants=0)
