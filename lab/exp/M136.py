# M-136：營收合成、R5、散戶（反向）的獨立性與合成（強勢池、奇數代碼、2010–18／2019–26）。定義見 registry_M.md M-136。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, M120
from M107 import nw_t
from M127 import tdcc_factors, ERAS
from M131 import cond_spread

NAMES = ['RC 營收合成', 'R5 年增驚喜', 'B4 散戶減（反向）']


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    S = signals.compute('fix', P=P, F=F); U, pool = S['_dbg']['U'], S['_dbg']['pool']; del S
    Cf = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill().to_numpy(); VALID = np.asarray(F['VALID'])
    HA = np.asarray(F['HA'], np.float64); LA = np.asarray(F['LA'], np.float64)
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CX = np.cumsum(R - ew[:, None], axis=0)
    odd = (P['meta']['stock_id'].astype(str).str[-1].astype(int) % 2 == 1).to_numpy()
    keys = P['meta']['security_key'].astype(str).to_numpy()
    base = M120.rev_factors(); R1, R2, R4, R5 = base['R1 單月年增'], base['R2 三月年增'], base['R4 營收創高'], base['R5 年增驚喜']
    sd = R1.rolling(12, min_periods=10).std(); E4 = R2 / sd.where(sd > 0)
    B4 = tdcc_factors(dates)['B4 散戶 1 月變化']
    def getter(W):
        jm = pd.Index(W.columns.astype(str)).get_indexer(keys)
        return lambda m: (np.where(jm >= 0, W.loc[m].to_numpy(float)[np.clip(jm, 0, None)], np.nan) if m in W.index else np.full(N, np.nan))
    g4, ge, g5, gb = getter(R4), getter(E4), getter(R5), getter(B4)
    ds = pd.Series(dates, index=dates); s = ds[ds.dt.day > 10]; samp = s.groupby(s.dt.to_period('M')).min()
    pr = lambda a: pd.Series(a).rank(pct=True).to_numpy()
    L = ['# M-136 結果：營收合成、R5、散戶（反向）的獨立性與合成（強勢池、奇數代碼）\n']
    verdict = {}
    for era, lo, hi in ERAS:
        cor = {(0, 1): [], (0, 2): [], (1, 2): []}; inc = {k: [] for k in range(3)}
        grp = ['RC', 'R5', 'B4', 'C3']; rec = {g: dict(d20=[], d60=[], n=[], left=[], right=[], up=[], dn=[], keep=[]) for g in grp}; prev = {g: None for g in grp}
        for p, t0 in samp.items():
            if not (pd.Timestamp(lo) <= t0 <= pd.Timestamp(hi)): continue
            t = dates.get_loc(t0); m = p - 1
            if t + 60 >= T: continue
            a4, ae, a5, ab = g4(m), ge(m), g5(m), gb(m)
            sel = pool[t] & odd & np.isfinite(a4) & np.isfinite(ae) & np.isfinite(a5) & np.isfinite(ab)
            if sel.sum() < 30: continue
            jj = np.nonzero(sel)[0]
            rc = pr((pr(a4[jj]) + pr(ae[jj])) / 2); f5 = pr(a5[jj]); fb = pr(-ab[jj]); X = [rc, f5, fb]
            for (i, k) in cor: cor[(i, k)].append(np.corrcoef(X[i], X[k])[0, 1])
            y60 = CX[t + 60, jj] - CX[t, jj]; y20 = CX[t + 20, jj] - CX[t, jj]
            for k in range(3):
                others = (sum(X) - X[k]) / 2
                inc[k].append(np.nanmean(cond_spread(others, X[k], y60)))
            c3 = pr((rc + f5 + fb) / 3)
            G = {'RC': rc > 0.8, 'R5': f5 > 0.8, 'B4': fb > 0.8, 'C3': c3 > 0.8}
            q10, q90 = np.nanpercentile(y60, [10, 90])
            with np.errstate(all='ignore'):
                up = np.nanmax(HA[t + 1:t + 61, jj], axis=0) / Cf[t, jj] - 1; dn = np.nanmin(LA[t + 1:t + 61, jj], axis=0) / Cf[t, jj] - 1
            mu, md = np.nanmedian(up), np.nanmedian(dn)
            for g in grp:
                msk = G[g]; r = rec[g]
                r['d20'].append(np.nanmean(y20[msk]) - np.nanmean(y20)); r['d60'].append(np.nanmean(y60[msk]) - np.nanmean(y60)); r['n'].append(msk.sum())
                r['left'].append(np.nanmean(y60[msk] <= q10)); r['right'].append(np.nanmean(y60[msk] >= q90))
                r['up'].append(np.nanmean(up[msk]) - mu); r['dn'].append(np.nanmean(dn[msk]) - md)
                cur = set(jj[msk])
                if prev[g]: r['keep'].append(len(cur & prev[g]) / len(prev[g]))
                prev[g] = cur
        L += [f'## {era}（期數 {len(inc[0])}）\n', '### 兩兩相關\n', f"- RC × R5 {np.mean(cor[(0, 1)]):+.2f}；RC × 散戶減 {np.mean(cor[(0, 2)]):+.2f}；R5 × 散戶減 {np.mean(cor[(1, 2)]):+.2f}\n",
              '### 控制另外兩者後的增量（後 60 日高−低）\n', '| 因子 | 增量（t） |', '|---|---|']
        for k in range(3): L.append(f"| {NAMES[k]} | {np.mean(inc[k]):+.2%}（{nw_t(inc[k], 3):.1f}） |")
        L += ['\n### 前 20% 組（與同期強勢池平均之差）\n', '| 組 | 平均檔數 | 後20日差（t） | 後60日差（t） | 左尾率 | 右尾率 | 60日最高漲幅 | 60日最大跌幅 | 逐月保留 |', '|---|---|---|---|---|---|---|---|---|']
        nm = {'RC': '營收合成', 'R5': 'R5', 'B4': '散戶減', 'C3': '三者合成'}
        for g in grp:
            r = rec[g]
            L.append(f"| {nm[g]} | {np.mean(r['n']):.1f} | {np.mean(r['d20']):+.2%}（{nw_t(r['d20'], 1):.1f}） | {np.mean(r['d60']):+.2%}（{nw_t(r['d60'], 3):.1f}） | {np.mean(r['left']):.0%} | {np.mean(r['right']):.0%} | {np.mean(r['up']):+.1%} | {np.mean(r['dn']):+.1%} | {np.mean(r['keep']):.0%} |")
        best = max(np.mean(rec[g]['d60']) for g in ('RC', 'R5', 'B4'))
        verdict[era] = (all(np.mean(inc[k]) > 0 for k in range(3)), np.mean(rec['C3']['d60']) - best >= 0.005, np.mean(rec['C3']['d60']) - best)
        L.append('')
    L += ['## 判讀（事前寫定）\n'] + [f"- {e}：可相加 {'✓' if v[0] else '✗'}；合成有益（C3 − 最佳單一 {v[2]:+.2%}，需 ≥ +0.5pt）{'✓' if v[1] else '✗'}" for e, v in verdict.items()]
    return dict(summary='\n'.join(L), headline='三因子獨立性與合成完成', n_variants=0)
