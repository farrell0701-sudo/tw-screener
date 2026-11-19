# M-131：E4（穩定成長）控制 R4 後的增量資訊（強勢池、奇數代碼）。定義見 registry_M.md M-131。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, M120
from M107 import nw_t


def cond_spread(a, b, y):
    """先依 a 分三組，組內依 b 分三組，回傳各 a 組內 b 高−低 的 y 平均差（list）。"""
    ga = pd.qcut(pd.Series(a).rank(method='first'), 3, labels=False).to_numpy(); out = []
    for k in range(3):
        m = ga == k
        if m.sum() < 9: out.append(np.nan); continue
        gb = pd.qcut(pd.Series(b[m]).rank(method='first'), 3, labels=False).to_numpy()
        out.append(np.nanmean(y[m][gb == 2]) - np.nanmean(y[m][gb == 0]))
    return out


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    S = signals.compute('fix', P=P, F=F); U, pool = S['_dbg']['U'], S['_dbg']['pool']; del S
    Cf = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill().to_numpy(); VALID = np.asarray(F['VALID'])
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CX = np.cumsum(R - ew[:, None], axis=0)
    odd = (P['meta']['stock_id'].astype(str).str[-1].astype(int) % 2 == 1).to_numpy()
    keys = P['meta']['security_key'].astype(str).to_numpy()
    base = M120.rev_factors(); R1, R2, R4 = base['R1 單月年增'], base['R2 三月年增'], base['R4 營收創高']
    sd = R1.rolling(12, min_periods=10).std(); E4 = R2 / sd.where(sd > 0)
    jm = pd.Index(R4.columns.astype(str)).get_indexer(keys)
    ds = pd.Series(dates, index=dates); s = ds[ds.dt.day > 10]; samp = s.groupby(s.dt.to_period('M')).min()
    L = ['# M-131 結果：E4（穩定成長）控制 R4 後的增量資訊（強勢池、奇數代碼）\n',
         '後 60 日超額；條件排序 3×3；殘差 IC＝rank(E4) 對 rank(R4) 迴歸的殘差與後 60 日超額的等級相關。\n']
    res = {}
    for era, lo, hi in M120.ERAS:
        e4c, r4c, ric, per_k = [], [], [], []
        for p, t0 in samp.items():
            if not (pd.Timestamp(lo) <= t0 <= pd.Timestamp(hi)): continue
            t = dates.get_loc(t0); m = p - 1
            if t + 60 >= T or m not in R4.index: continue
            take = lambda W: np.where(jm >= 0, W.loc[m].to_numpy(float)[np.clip(jm, 0, None)], np.nan)
            a, b = take(R4), take(E4)
            sel = pool[t] & odd & np.isfinite(a) & np.isfinite(b)
            if sel.sum() < 30: continue
            jj = np.nonzero(sel)[0]; y = CX[t + 60, jj] - CX[t, jj]; a, b = a[jj], b[jj]
            cs = cond_spread(a, b, y); e4c.append(np.nanmean(cs)); per_k.append(cs)
            r4c.append(np.nanmean(cond_spread(b, a, y)))
            ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
            beta = np.polyfit(ra, rb, 1); resid = rb - np.polyval(beta, ra)
            ok = np.isfinite(y)
            ric.append(np.corrcoef(pd.Series(resid[ok]).rank(), pd.Series(y[ok]).rank())[0, 1])
        pk = np.array(per_k)
        res[era] = (np.mean(e4c), nw_t(e4c, 3), np.mean(ric), nw_t(ric, 3))
        L += [f'## {era}（期數 {len(e4c)}）\n', '| 項目 | 平均 | t |', '|---|---|---|',
              f"| 控制 R4 後的 E4 高−低 | {np.mean(e4c):+.2%} | {nw_t(e4c, 3):.1f} |",
              f"| 控制 E4 後的 R4 高−低 | {np.mean(r4c):+.2%} | {nw_t(r4c, 3):.1f} |",
              f"| E4 殘差 IC60 | {np.mean(ric):+.3f} | {nw_t(ric, 3):.1f} |",
              f"| E4 價差：R4 低組內 | {np.nanmean(pk[:, 0]):+.2%} | {nw_t(pk[:, 0][np.isfinite(pk[:, 0])], 3):.1f} |",
              f"| E4 價差：R4 中組內 | {np.nanmean(pk[:, 1]):+.2%} | {nw_t(pk[:, 1][np.isfinite(pk[:, 1])], 3):.1f} |",
              f"| E4 價差：R4 高組內 | {np.nanmean(pk[:, 2]):+.2%} | {nw_t(pk[:, 2][np.isfinite(pk[:, 2])], 3):.1f} |", '']
    ok = all(v[0] > 0.015 and v[1] >= 2 and v[2] > 0 for v in res.values())
    L.append(f"**提名：{'是，送偶數代碼確認集' if ok else '否，E4 視為 R4 的變形，結案'}**")
    return dict(summary='\n'.join(L), headline=f"E4 增量{'提名' if ok else '不提名'}", n_variants=0)
