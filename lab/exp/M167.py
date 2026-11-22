# M-167：持有 60 日內 ΔT_ld10 轉弱與後續報酬（奇數代碼探索集）。定義見 registry_M.md M-167。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim
from M107 import nw_t

NPATH = 20; K = 60; H = 20
ERAS = (('2005–15', '2005-01-01', '2015-12-31'), ('2016–26', '2016-01-01', '2026-12-31'))
BUCK = ((1, 20), (21, 40), (41, 60))


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    keys = P['meta']['security_key'].astype(str).to_numpy(); kix = {k: i for i, k in enumerate(keys)}
    odd = (P['meta']['stock_id'].astype(str).str[-1].astype(int) % 2 == 1).to_numpy()
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; U, Tr = dbg['U'], dbg['Tr']; del S
    if Tr.shape[1] != N: full = np.full((T, N), np.nan); full[:, dbg['cols']] = Tr; Tr = full
    dT = Tr - np.vstack([np.full((10, N), np.nan), Tr[:-10]])
    pr = signals.prk(dT, U)
    VALID = np.asarray(F['VALID']); CA = np.asarray(F['CA'], np.float64); LA = np.asarray(F['LA'], np.float64)
    Cf = pd.DataFrame(CA).ffill().to_numpy()
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CX = np.cumsum(R - ew[:, None], axis=0)
    RP = sim.run_paths('fix', NPATH)
    trades = {}
    for kind in ('X', 'V'):
        seen = {}
        for i in range(NPATH):
            for (code, d0, d1, wt, ret, hold, why, st, lk) in RP[kind][i][1]:
                seen.setdefault((code, d0), (d1, why))
        trades[kind] = seen
    obs, first = [], []
    for kind, seen in trades.items():
        for (code, d0), (d1, why) in seen.items():
            j = kix.get(str(code))
            if j is None or not odd[j]: continue
            t0 = dates.get_loc(pd.Timestamp(d0)); t1 = dates.get_loc(pd.Timestamp(d1))
            if t0 + K + H >= T or dates[t0] < pd.Timestamp('2005-01-01'): continue
            era = '2005–15' if dates[t0] <= pd.Timestamp('2015-12-31') else '2016–26'
            f1 = f2 = None
            for k in range(1, K + 1):
                t = t0 + k - 1
                if not VALID[t, j] or not np.isfinite(dT[t, j]): continue
                w1 = dT[t, j] < 0; w2 = np.isfinite(pr[t, j]) and pr[t, j] <= 0.2
                y = CX[t + H, j] - CX[t, j]
                with np.errstate(all='ignore'): dd = np.nanmin(LA[t + 1:t + H + 1, j]) / Cf[t, j] - 1
                obs.append((kind, era, k, t, w1, w2, y, dd))
                if w1 and f1 is None: f1 = (k, t)
                if w2 and f2 is None: f2 = (k, t)
            rem = lambda f: (CX[t0 + K - 1, j] - CX[f[1], j]) if f else np.nan
            y20 = lambda f: (CX[f[1] + H, j] - CX[f[1], j]) if f else np.nan
            first.append(dict(kind=kind, era=era, k1=f1[0] if f1 else np.nan, k2=f2[0] if f2 else np.nan,
                              rem1=rem(f1), rem2=rem(f2), y1=y20(f1), y2=y20(f2),
                              pre1=(t1 <= f1[1]) if f1 else np.nan, pre2=(t1 <= f2[1]) if f2 else np.nan,
                              full=CX[t0 + K - 1, j] - CX[t0 - 1, j]))
    O = pd.DataFrame(obs, columns=['kind', 'era', 'k', 't', 'W1', 'W2', 'y', 'dd'])
    O['b'] = pd.cut(O.k, [0, 20, 40, 60], labels=['1–20', '21–40', '41–60'])
    Fr = pd.DataFrame(first)
    L = ['# M-167 結果：持有 60 日內 ΔT_ld10 轉弱與後續 20 日超額（奇數代碼）\n',
         f"交易數（去重）：X1 {(Fr.kind == 'X').sum()}、V4.1 {(Fr.kind == 'V').sum()}；觀測 {len(O)}。20 日超額＝第 k 日收盤到 k＋20 日收盤，扣可交易池等權。\n"]
    verdict = []
    for kind, kn in (('X', 'X1'), ('V', 'V4.1')):
        for st, sn in (('W1', 'W1 轉負'), ('W2', 'W2 深度轉弱（池內 ≤20%）')):
            L += [f'## {kn}｜{sn}\n', '| 期間 | 持有日 | 成立 n | 不成立 n | 成立 20 日超額 | 不成立 | 差（NW t） | 成立 左尾／右尾 | 不成立 左尾／右尾 | 成立 20 日最大跌幅 | 不成立 |',
                  '|---|---|---|---|---|---|---|---|---|---|---|']
            res = {}
            for era, _, _ in ERAS:
                for bl in ('1–20', '21–40', '41–60', '21–60'):
                    D = O[(O.kind == kind) & (O.era == era) & (O.b.isin(['21–40', '41–60']) if bl == '21–60' else O.b == bl)]
                    if len(D) < 100: continue
                    q10, q90 = D.y.quantile([.1, .9])
                    a, b = D[D[st]], D[~D[st]]
                    g = D.groupby(['t', st]).y.mean().unstack()
                    dser = (g[True] - g[False]).dropna() if True in g and False in g else pd.Series(dtype=float)
                    tt = nw_t(dser.to_numpy(), 20); md = dser.mean()
                    la, ra, lb, rb = (a.y <= q10).mean(), (a.y >= q90).mean(), (b.y <= q10).mean(), (b.y >= q90).mean()
                    res[(era, bl)] = (md, tt, la, ra, lb, rb)
                    L.append(f"| {era} | {bl} | {len(a)} | {len(b)} | {a.y.mean():+.2%} | {b.y.mean():+.2%} | {md:+.2%}（{tt:.1f}） | {la:.0%}／{ra:.0%} | {lb:.0%}／{rb:.0%} | {a.dd.mean():+.1%} | {b.dd.mean():+.1%} |")
            r1, r2 = res.get(('2005–15', '21–60')), res.get(('2016–26', '21–60'))
            if r1 and r2:
                c1 = r1[0] < 0 and r2[0] < 0 and min(r1[1], r2[1]) <= -2.0 and max(r1[1], r2[1]) <= -1.0
                c2 = r1[2] > r1[4] and r2[2] > r2[4]
                c3 = r1[3] <= r1[5] and r2[3] <= r2[5]
                ok = c1 and c2 and c3
                verdict.append(f"| {kn} | {sn} | {'✓' if c1 else '✗'}／{'✓' if c2 else '✗'}／{'✓' if c3 else '✗'} | {'**有轉弱資訊**' if ok else '否'} |")
            L.append('')
    L += ['## 表二：第一次觸發\n', '| 策略 | 狀態 | 期間 | 60 日內觸發 | 觸發持有日中位數 | 觸發後 20 日超額 | 觸發後到第 60 日剩餘超額 | 剩餘超額 > 0 比例 | 不觸發者 60 日超額 | 現行規則已在觸發前出場 |', '|---|---|---|---|---|---|---|---|---|---|']
    for kind, kn in (('X', 'X1'), ('V', 'V4.1')):
        for s, sn in (('1', 'W1'), ('2', 'W2')):
            for era, _, _ in ERAS:
                D = Fr[(Fr.kind == kind) & (Fr.era == era)]
                if not len(D): continue
                hit = D[f'k{s}'].notna(); Dh = D[hit]
                L.append(f"| {kn} | {sn} | {era} | {hit.mean():.0%} | {Dh[f'k{s}'].median():.0f} | {Dh[f'y{s}'].mean():+.2%} | {Dh[f'rem{s}'].mean():+.2%} | {(Dh[f'rem{s}'] > 0).mean():.0%} | {D[~hit].full.mean():+.2%} | {Dh[f'pre{s}'].astype(float).mean():.0%} |")
    head = ['## 判準（21–60 持有日；1 差值兩期 < 0 且 t ≤ −2、≤ −1／2 成立組左尾較高／3 成立組右尾不較高）\n', '| 策略 | 狀態 | 1／2／3 | 結論 |', '|---|---|---|---|'] + verdict + ['']
    nh = sum('有轉弱資訊' in v for v in verdict)
    return dict(summary='\n'.join(L[:2] + head + L[2:]), headline=f'ΔT_ld10 轉弱：{nh} 個組合有資訊', n_variants=0)
