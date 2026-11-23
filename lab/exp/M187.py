# M-187（M-186 確認集，偶數代碼）：營收池 × 產業自身相對動能（奇數代碼；到 2024-11-13）。定義見 registry_M.md M-186。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals
import M165
from L140 import fixed_industry

END = pd.Timestamp('2024-11-13')
LS = (60, 20, 120)


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    meta = P['meta']; keys = meta['security_key'].astype(str).to_numpy()
    ind = np.array([str(v) if isinstance(v, str) else '缺' for v in fixed_industry(meta)[0].astype(object).replace({'化學生技醫療': '生技醫療業', '金融業': '金融保險'})])
    odd = (meta['stock_id'].astype(str).str[-1].astype(int) % 2 == 0).to_numpy()
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; cols = dbg['cols']
    def full(a, fill):
        if a.shape[1] == N: return a
        o = np.full((T, N), fill, a.dtype); o[:, cols] = a; return o
    U, pool = full(dbg['U'], False), full(dbg['pool'], False); del S
    VALID = np.asarray(F['VALID']); CA = np.where(VALID, np.asarray(F['CA'], np.float64), np.nan)
    Cf = pd.DataFrame(CA).ffill().to_numpy()
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CX = np.cumsum(R - ew[:, None], axis=0)
    A = M165.layers(U, pool, dates, keys)[0][1]
    codes, inv = np.unique(ind, return_inverse=True); K = len(codes)
    rows = []
    for m, (t, tp) in sorted(M165.samp_prev(dates).items(), key=lambda x: x[1][0]):
        if dates[t] < pd.Timestamp('2005-01-01') or dates[t] > END or t + 61 >= T: continue
        era = '2005–15' if dates[t] <= pd.Timestamp('2015-12-31') else '2016–24/11'
        Ut = U[t]; e60 = CX[t + 60] - CX[t]; q10 = np.nanpercentile(e60[Ut], 10)
        grp = np.where(pool[t], '強勢池', np.where(A[t], '營收池', '其餘'))
        half = {}
        for L in LS:
            if tp - L < 0: break
            past = CX[tp] - CX[tp - L]; ok = Ut & ~np.isnan(past)
            s = np.bincount(inv[ok], past[ok], K); c = np.bincount(inv[ok], minlength=K)
            loo_c = c[inv] - ok; im = np.where(loo_c >= 5, (s[inv] - np.where(ok, past, 0)) / np.maximum(loo_c, 1), np.nan)
            q = pd.Series(np.where(Ut, im, np.nan)).rank(pct=True).to_numpy()
            half[L] = np.where(np.isnan(q), '', np.where(q > 0.5, '上半', '下半'))
        if len(half) < 3: continue
        for j in np.nonzero(Ut & odd)[0]:
            rows.append((era, grp[j], half[60][j], half[20][j], half[120][j], e60[j], e60[j] <= q10))
    D = pd.DataFrame(rows, columns=['era', 'pool', 'h60', 'h20', 'h120', 'e60', 'L'])
    out = ['# M-187 結果（偶數代碼確認）：營收池 × 產業自身相對動能（奇數代碼；60 日超額；左尾 = 當日 U 的後 10%）\n']
    verdict = {}
    for L in LS:
        h = f'h{L}'
        out += [f'## 產業動能 L = {L}\n', '| 期間 | IM | 營收池 平均（中位）左尾 n | 強勢池 平均 | 其餘 平均 | 營收池 − 同組其餘 |', '|---|---|---|---|---|---|']
        for era in ('2005–15', '2016–24/11'):
            for hv in ('上半', '下半'):
                x = D[(D.era == era) & (D[h] == hv)]
                r, s, o = x[x.pool == '營收池'], x[x.pool == '強勢池'], x[x.pool == '其餘']
                verdict[(L, era, hv)] = (r.e60.mean(), o.e60.mean())
                out.append(f"| {era} | {hv} | {r.e60.mean():+.1%}（{r.e60.median():+.1%}）{r.L.mean():.0%} n={len(r)} | {s.e60.mean():+.1%} | {o.e60.mean():+.1%} | {r.e60.mean() - o.e60.mean():+.1%} |")
        out.append('')
    out += ['## 確認判準\n']
    ok_all = True
    for L in LS:
        for era in ('2005–15', '2016–24/11'):
            (up, ou), (dn, od) = verdict[(L, era, '上半')], verdict[(L, era, '下半')]
            if L == 60:
                c = (up - dn >= 0.01) and (up - dn > ou - od)
            else:
                c = up > dn
            ok_all &= c
            out.append(f"- L={L} {era}：營收池 上半−下半 = {up - dn:+.1%}；其餘股票 = {ou - od:+.1%} → {'✓' if c else '✗'}")
    out.append(f"\n**整體：{'確認' if ok_all else '未確認'}**")
    return dict(summary='\n'.join(out), headline=('確認' if ok_all else '未確認') + '：營收池 × 產業動能（偶數代碼）', n_variants=1)
