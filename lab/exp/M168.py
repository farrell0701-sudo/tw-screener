# M-168：營收池訊號品質——進場後 ΔT_ld10 轉弱（奇數代碼）。定義見 registry_M.md M-168。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals
from M107 import nw_t
from M122 import fin_factors
import M165

CPS = (20, 40, 60); FH = 60; STOP = 0.85
VARS = {'V1 次數（ΔT_ld10<0 天數占比）': +1, 'V2 幅度（T_ld 第c日−進場日）': -1, 'V3 自高點回落': +1, 'V4 第一次轉負時間': -1}   # +1：值越高越弱


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    keys = P['meta']['security_key'].astype(str).to_numpy()
    odd = (P['meta']['stock_id'].astype(str).str[-1].astype(int) % 2 == 1).to_numpy()
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; U, pool, Tr = dbg['U'], dbg['pool'], dbg['Tr']; del S
    if Tr.shape[1] != N: full = np.full((T, N), np.nan); full[:, dbg['cols']] = Tr; Tr = full
    dT = Tr - np.vstack([np.full((10, N), np.nan), Tr[:-10]])
    VALID = np.asarray(F['VALID']); CA = np.asarray(F['CA'], np.float64); OA = np.asarray(F['OA'], np.float64); LA = np.asarray(F['LA'], np.float64)
    Cf = pd.DataFrame(CA).ffill().to_numpy()
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CX = np.cumsum(R - ew[:, None], axis=0)
    A, _ = M165.layers(U, pool, dates, keys)[0][1], None
    Q5 = fin_factors(dates)['Q5 毛利率創高']; qj = pd.Index(Q5.columns.astype(str)).get_indexer(keys)
    rows = []; nsig = 0; nq = 0
    for m, (t, _) in M165.samp_prev(dates).items():
        if dates[t] < pd.Timestamp('2005-01-01') or t + 1 + 120 >= T or not A[t].any(): continue
        mem = np.nonzero(A[t])[0]
        q = np.where(qj[mem] >= 0, Q5.loc[m].to_numpy(float)[np.clip(qj[mem], 0, None)], np.nan) if m in Q5.index else np.full(len(mem), np.nan)
        fin = np.isfinite(q)
        drop = np.zeros(len(mem), bool)
        if fin.sum() >= 20: drop[fin] = q[fin] <= np.nanpercentile(q[fin], 20)
        nq += drop.sum()
        sig = mem[~drop & odd[mem]]
        era = '2005–15' if dates[t] <= pd.Timestamp('2015-12-31') else '2016–26'
        for j in sig:
            ent = OA[t + 1, j]
            if not (VALID[t + 1, j] and np.isfinite(ent) and ent > 0) or not np.isfinite(Tr[t, j]): continue
            nsig += 1
            cl = CA[t + 1:t + 121, j]
            hit = np.nonzero(cl < ent * STOP)[0]; kst = hit[0] + 1 if len(hit) else 999
            d = dT[t + 1:t + 61, j]; tl = Tr[t + 1:t + 61, j]
            for c in CPS:
                if t + c + FH >= T: continue
                rec = dict(m=str(m), era=era, c=c, stopped=kst <= c)
                if kst > c:
                    dd = d[:c]; ok = np.isfinite(dd)
                    neg = np.nonzero(ok & (dd < 0))[0]
                    tc = tl[c - 1]
                    rec.update({'V1 次數（ΔT_ld10<0 天數占比）': (dd[ok] < 0).mean() if ok.any() else np.nan,
                                'V2 幅度（T_ld 第c日−進場日）': tc - Tr[t, j],
                                'V3 自高點回落': np.nanmax(tl[:c]) - tc,
                                'V4 第一次轉負時間': (neg[0] + 1) if len(neg) else c + 1,
                                'y': CX[t + c + FH, j] - CX[t + c, j],
                                'dd': np.nanmin(LA[t + c + 1:t + c + FH + 1, j]) / Cf[t + c, j] - 1,
                                'st60': float(kst <= c + FH), 'rem': CX[t + 120, j] - CX[t + c, j] if c < 120 else np.nan})
                rows.append(rec)
    D = pd.DataFrame(rows); D['stopped'] = D.stopped.astype(bool); D['st60'] = D.st60.astype(float)
    L = ['# M-168 結果：營收池訊號品質——進場後 ΔT_ld10 轉弱（奇數代碼）\n',
         f'訊號＝池外營收池剔除 Q5 最低 20%（剔除 {nq} 檔次，全部代碼）；奇數代碼訊號 {nsig} 筆。超額＝第 c 日收盤到 c＋60 日，扣可交易池等權。\n',
         '| 檢查點 | 期間 | 訊號數 | 已停損（剔除） | 後 60 日超額 | 左尾門檻／右尾門檻 | 剩餘到第 120 日超額 |', '|---|---|---|---|---|---|---|']
    for c in CPS:
        for era in ('2005–15', '2016–26'):
            x = D[(D.c == c) & (D.era == era)]; y = x[~x.stopped]
            if not len(y): continue
            L.append(f"| {c} | {era} | {len(x)} | {x.stopped.mean():.0%} | {y.y.mean():+.2%} | {y.y.quantile(.1):+.1%}／{y.y.quantile(.9):+.1%} | {y.rem.mean():+.2%} |")
    verdict = ['## 判準（最弱組 vs 其餘四組；1 差值兩期 < 0 且 t ≤ −2、≤ −1／2 左尾較高／3 右尾不較高；c＝20、40 為主）\n', '| 變數 | 檢查點 | 1／2／3 | 結論 |', '|---|---|---|---|']
    det = []
    for v, sg in VARS.items():
        det += [f'## {v}\n', '| 檢查點 | 期間 | 最弱組 60 日超額 | 其餘 | 差（t） | 最弱 左尾／右尾 | 其餘 左尾／右尾 | 最弱 最大跌幅 | 其餘 | 最弱 60 日內停損 | 其餘 | 最弱 剩餘到 120 日 | 其餘 |',
                '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for c in CPS:
            res = {}
            for era in ('2005–15', '2016–26'):
                x = D[(D.c == c) & (D.era == era) & ~D.stopped].dropna(subset=[v, 'y']).copy()
                if len(x) < 100: continue
                q10, q90 = x.y.quantile([.1, .9])
                x['w'] = x.groupby('m')[v].transform(lambda s: (s * sg).rank(pct=True, method='first') > 0.8 if len(s) >= 5 else pd.Series(False, index=s.index)).astype(bool)
                a, b = x[x.w], x[~x.w]
                g = x.groupby(['m', 'w']).y.mean().unstack()
                dif = (g[True] - g[False]).dropna() if True in g and False in g else pd.Series(dtype=float)
                tt = nw_t(dif.to_numpy(), 3)
                la, ra, lb, rb = (a.y <= q10).mean(), (a.y >= q90).mean(), (b.y <= q10).mean(), (b.y >= q90).mean()
                res[era] = (dif.mean(), tt, la, ra, lb, rb)
                det.append(f"| {c} | {era} | {a.y.mean():+.2%} | {b.y.mean():+.2%} | {dif.mean():+.2%}（{tt:.1f}） | {la:.0%}／{ra:.0%} | {lb:.0%}／{rb:.0%} | {a.dd.mean():+.1%} | {b.dd.mean():+.1%} | {a.st60.mean():.0%} | {b.st60.mean():.0%} | {a.rem.mean():+.2%} | {b.rem.mean():+.2%} |")
            if len(res) == 2:
                r1, r2 = res['2005–15'], res['2016–26']
                c1 = r1[0] < 0 and r2[0] < 0 and min(r1[1], r2[1]) <= -2.0 and max(r1[1], r2[1]) <= -1.0
                c2 = r1[2] > r1[4] and r2[2] > r2[4]; c3 = r1[3] <= r1[5] and r2[3] <= r2[5]
                verdict.append(f"| {v} | {c}{'（只報告）' if c == 60 else ''} | {'✓' if c1 else '✗'}／{'✓' if c2 else '✗'}／{'✓' if c3 else '✗'} | {'**有資訊**' if (c1 and c2 and c3) else '否'} |")
        det.append('')
    n = sum('有資訊' in x for x in verdict if '只報告' not in x)
    return dict(summary='\n'.join(L + [''] + verdict + [''] + det), headline=f'營收池訊號轉弱：{n} 組有資訊（c＝20、40）', n_variants=0)
