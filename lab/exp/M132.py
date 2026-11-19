# M-132：R4 與 E4 的三種融合（強勢池、奇數代碼）。定義見 registry_M.md M-132。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, M120
from M107 import nw_t

GN = {'0a': 'R4 單獨', '0b': 'E4 單獨', 'A': '都要成立', 'O': '成立其一', 'C': '等權合成'}
PARITY = 1


def revenue_E4():
    base = M120.rev_factors(); R1, R2, R4 = base['R1 單月年增'], base['R2 三月年增'], base['R4 營收創高']
    sd = R1.rolling(12, min_periods=10).std(); return R4, R2 / sd.where(sd > 0)


def run(parity, groups):
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    S = signals.compute('fix', P=P, F=F); U, pool = S['_dbg']['U'], S['_dbg']['pool']; del S
    Cf = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill().to_numpy(); VALID = np.asarray(F['VALID'])
    HA = np.asarray(F['HA'], np.float64); LA = np.asarray(F['LA'], np.float64)
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CX = np.cumsum(R - ew[:, None], axis=0)
    par = (P['meta']['stock_id'].astype(str).str[-1].astype(int) % 2 == parity).to_numpy()
    keys = P['meta']['security_key'].astype(str).to_numpy()
    R4, E4 = revenue_E4(); jm = pd.Index(R4.columns.astype(str)).get_indexer(keys)
    ds = pd.Series(dates, index=dates); s = ds[ds.dt.day > 10]; samp = s.groupby(s.dt.to_period('M')).min()
    out = {}
    for era, lo, hi in M120.ERAS:
        rec = {g: dict(d20=[], d60=[], n=[], left=[], right=[], up=[], dn=[], keep=[]) for g in groups}; prev = {g: None for g in groups}
        for p, t0 in samp.items():
            if not (pd.Timestamp(lo) <= t0 <= pd.Timestamp(hi)): continue
            t = dates.get_loc(t0); m = p - 1
            if t + 60 >= T or m not in R4.index: continue
            take = lambda W: np.where(jm >= 0, W.loc[m].to_numpy(float)[np.clip(jm, 0, None)], np.nan)
            a, b = take(R4), take(E4)
            sel = pool[t] & par & np.isfinite(a) & np.isfinite(b)
            if sel.sum() < 30: continue
            jj = np.nonzero(sel)[0]; a, b = a[jj], b[jj]
            pa, pb = pd.Series(a).rank(pct=True).to_numpy(), pd.Series(b).rank(pct=True).to_numpy()
            pc = pd.Series((pa + pb) / 2).rank(pct=True, method='first').to_numpy()
            ta, tb = pa > 0.8, pb > 0.8
            G = {'0a': ta, '0b': tb, 'A': ta & tb, 'O': ta | tb, 'C': pc > 0.8}
            e20 = CX[t + 20, jj] - CX[t, jj]; e60 = CX[t + 60, jj] - CX[t, jj]; q10, q90 = np.nanpercentile(e60, [10, 90])
            with np.errstate(all='ignore'):
                up = np.nanmax(HA[t + 1:t + 61, jj], axis=0) / Cf[t, jj] - 1; dn = np.nanmin(LA[t + 1:t + 61, jj], axis=0) / Cf[t, jj] - 1
            mu, md = np.nanmedian(up), np.nanmedian(dn)
            for g in groups:
                msk = G[g]
                if msk.sum() == 0: continue
                r = rec[g]
                r['d20'].append(np.nanmean(e20[msk]) - np.nanmean(e20)); r['d60'].append(np.nanmean(e60[msk]) - np.nanmean(e60)); r['n'].append(msk.sum())
                r['left'].append(np.nanmean(e60[msk] <= q10)); r['right'].append(np.nanmean(e60[msk] >= q90))
                r['up'].append(np.nanmean(up[msk]) - mu); r['dn'].append(np.nanmean(dn[msk]) - md)
                cur = set(jj[msk])
                if prev[g]: r['keep'].append(len(cur & prev[g]) / len(prev[g]))
                prev[g] = cur
        out[era] = rec
    return out


def table(out, groups):
    L = []
    for era, rec in out.items():
        L += [f'## {era}\n', '| 組 | 平均檔數 | 後20日差（t） | 後60日差（t） | 左尾率 | 右尾率 | 60日最高漲幅（扣中位） | 60日最大跌幅（扣中位） | 逐月保留 |', '|---|---|---|---|---|---|---|---|---|']
        for g in groups:
            r = rec[g]
            L.append(f"| {GN[g]} | {np.mean(r['n']):.1f} | {np.mean(r['d20']):+.2%}（{nw_t(r['d20'], 1):.1f}） | {np.mean(r['d60']):+.2%}（{nw_t(r['d60'], 3):.1f}） | "
                     f"{np.mean(r['left']):.0%} | {np.mean(r['right']):.0%} | {np.mean(r['up']):+.1%} | {np.mean(r['dn']):+.1%} | {np.mean(r['keep']):.0%} |")
        L.append('')
    return L


def main():
    groups = ['0a', '0b', 'A', 'O', 'C']
    out = run(PARITY, groups)
    L = ['# M-132 結果：R4 與 E4 的三種融合（強勢池、奇數代碼）\n', '「差」＝該組後續超額 − 同期強勢池平均（期數平均，NW t）。\n'] + table(out, groups)
    score = {}
    for g in ('A', 'O', 'C'):
        v = [(np.mean(out[e][g]['d60']), nw_t(out[e][g]['d60'], 3)) for e in out]
        score[g] = (min(x[0] for x in v), min(x[1] for x in v))
    best = max(score, key=lambda g: score[g])
    L += ['## 挑選（兩期後 60 日差的較小值）\n'] + [f"- {GN[g]}：{score[g][0]:+.2%}（較小 t {score[g][1]:.1f}）" for g in ('A', 'O', 'C')] + [f"\n**挑出：{GN[best]}**（送偶數代碼確認集，另登記）"]
    return dict(summary='\n'.join(L), headline=f'挑出 {GN[best]}', n_variants=0)
