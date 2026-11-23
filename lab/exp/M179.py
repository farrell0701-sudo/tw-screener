# M-179：個股多空點扣除 R20、T_ld、ΔT_ld10 後的增量（奇數代碼）。定義見 registry_M.md M-179。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals
import M120, M165
from M107 import nw_t

FN = ['D1 多空狀態（線上＝1）', 'D2 相對位置']
_C = {}


def build(dates):
    if 'D' in _C: return _C['D']
    P = load_panel(); F = load_feat('AF'); T, N = P['C'].shape
    keys = P['meta']['security_key'].astype(str).to_numpy()
    VALID = np.asarray(F['VALID']); CA = np.where(VALID, np.asarray(F['CA'], np.float64), np.nan)
    D1, D2 = {}, {}
    for m, (t, tp) in M165.samp_prev(dates).items():
        s0 = dates.searchsorted((dates[tp].to_period('M') - 1).start_time)
        w = CA[s0:tp + 1]; cnt = np.isfinite(w).sum(0)
        with np.errstate(all='ignore'): h, l = np.nanmax(w, 0), np.nanmin(w, 0)
        c = pd.DataFrame(CA[max(0, tp - 5):tp + 1]).ffill().to_numpy()[-1]
        ok = np.isfinite(h) & np.isfinite(l) & (cnt >= 15) & np.isfinite(c) & (h > l)
        mid = (h + l) / 2
        D1[m] = np.where(ok, (c > mid).astype(float), np.nan)
        D2[m] = np.where(ok, (c - mid) / (h - l), np.nan)
    out = {}
    for nm, d in ((FN[0], D1), (FN[1], D2)):
        W = pd.DataFrame(d, index=keys).T.sort_index(); out[nm] = W.loc[:, ~W.columns.duplicated()]
    _C['D'] = out; return out


def layers(U, pool, dates, keys):
    A = M165.layers(U, pool, dates, keys)[0][1]
    return (('全市場', U, 10), ('強勢池', pool, 5), ('營收池', A & U, 5))


def main():
    # D1 二分表
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    keys = P['meta']['security_key'].astype(str).to_numpy()
    odd = (P['meta']['stock_id'].astype(str).str[-1].astype(int) % 2 == 1).to_numpy()
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; U, pool = dbg['U'], dbg['pool']; dbg_Tr = dbg['Tr']; del S
    if dbg_Tr.shape[1] != N: _o = np.full((T, N), np.nan); _o[:, dbg['cols']] = dbg_Tr; dbg_Tr = _o
    VALID = np.asarray(F['VALID']); CA = np.asarray(F['CA'], np.float64)
    Cf = pd.DataFrame(np.where(VALID, CA, np.nan)).ffill().to_numpy()
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CX = np.cumsum(R - ew[:, None], axis=0)
    A = M165.layers(U, pool, dates, keys)[0][1]
    D1 = build(dates)[FN[0]]; jx = pd.Index(D1.columns.astype(str)).get_indexer(keys)
    rows = []
    for m, (t, _) in M165.samp_prev(dates).items():
        if dates[t] < pd.Timestamp('2005-01-01') or t + 61 >= T or m not in D1.index: continue
        v = np.where(jx >= 0, D1.loc[m].to_numpy(float)[np.clip(jx, 0, None)], np.nan)
        e60 = CX[t + 60] - CX[t]; q10, q90 = np.nanpercentile(e60[U[t]], [10, 90])
        era = '2005–15' if dates[t] <= pd.Timestamp('2015-12-31') else '2016–26'
        for ln, mask in (('全市場', U[t]), ('強勢池', pool[t]), ('營收池', A[t] & U[t])):
            s = mask & odd & np.isfinite(v)
            for j in np.nonzero(s)[0]: rows.append((str(m), era, ln, v[j], e60[j], q10, q90))
    Dd = pd.DataFrame(rows, columns=['m', 'era', 'layer', 'up', 'e60', 'q10', 'q90'])
    Bt = ['## D1 二分：線上 vs 線下（後 60 日超額）\n', '| 層 | 期間 | 線上占比 | 線上 | 線下 | 差（逐月 NW t） | 線上 左尾／右尾 | 線下 左尾／右尾 |', '|---|---|---|---|---|---|---|---|']
    for (ln, era), x in Dd.groupby(['layer', 'era'], sort=False):
        a, b = x[x.up == 1], x[x.up == 0]
        g = x.groupby(['m', 'up']).e60.mean().unstack()
        dif = (g[1.0] - g[0.0]).dropna() if (1.0 in g and 0.0 in g) else pd.Series(dtype=float)
        Bt.append(f"| {ln} | {era} | {len(a) / len(x):.0%} | {a.e60.mean():+.2%} | {b.e60.mean():+.2%} | {dif.mean():+.2%}（{nw_t(dif.to_numpy(), 3):.1f}） | "
                  f"{(a.e60 <= a.q10).mean():.0%}／{(a.e60 >= a.q90).mean():.0%} | {(b.e60 <= b.q10).mean():.0%}／{(b.e60 >= b.q90).mean():.0%} |")
    # 增量：殘差 IC
    Tr = dbg_Tr
    D2w = build(dates)[FN[1]]; j2 = pd.Index(D2w.columns.astype(str)).get_indexer(keys)
    rr = []
    for m, (t, tp) in M165.samp_prev(dates).items():
        if dates[t] < pd.Timestamp('2005-01-01') or t + 121 >= T or m not in D1.index: continue
        era = '2005–15' if dates[t] <= pd.Timestamp('2015-12-31') else '2016–26'
        d1 = np.where(jx >= 0, D1.loc[m].to_numpy(float)[np.clip(jx, 0, None)], np.nan)
        d2 = np.where(j2 >= 0, D2w.loc[m].to_numpy(float)[np.clip(j2, 0, None)], np.nan)
        r20 = Cf[tp] / Cf[max(0, tp - 20)] - 1; tl = Tr[tp]; dtl = Tr[tp] - Tr[max(0, tp - 10)]
        e60 = CX[t + 60] - CX[t]; e120 = CX[t + 120] - CX[t]
        for ln, mask in (('全市場', U[t]), ('強勢池', pool[t]), ('營收池', A[t] & U[t])):
            s_ = mask & odd & np.isfinite(r20) & np.isfinite(tl) & np.isfinite(dtl) & np.isfinite(e60) & np.isfinite(e120)
            for dn, dv in (('D1', d1), ('D2', d2)):
                ss = s_ & np.isfinite(dv)
                if ss.sum() < 30 or np.nanstd(dv[ss]) == 0: continue
                pr_ = lambda a: pd.Series(a).rank(pct=True).to_numpy()
                X = np.column_stack([np.ones(ss.sum()), pr_(r20[ss]), pr_(tl[ss]), pr_(dtl[ss])])
                beta = np.linalg.lstsq(X, dv[ss], rcond=None)[0]; res = dv[ss] - X @ beta
                X2 = np.column_stack([np.ones(ss.sum()), 0.5 * pr_(tl[ss]) + 0.5 * pr_(dtl[ss])])
                b2 = np.linalg.lstsq(X2, dv[ss], rcond=None)[0]; res2 = dv[ss] - X2 @ b2
                sp = lambda a, b: pd.Series(a).rank().corr(pd.Series(b).rank())
                rr.append((ln, era, dn, sp(res, e60[ss]), sp(res, e120[ss]), sp(dv[ss], r20[ss]), sp(dv[ss], tl[ss]), sp(dv[ss], e60[ss]), sp(dv[ss], dtl[ss]), sp(res2, e60[ss])))
    R_ = pd.DataFrame(rr, columns=['layer', 'era', 'd', 'ic60', 'ic120', 'cR20', 'cTld', 'raw60', 'cdT', 'ic60x1'])
    Inc = ['## 增量檢查：扣除 R20、T_ld、ΔT_ld10 後的殘差 IC（逐月平均，NW t）\n', '| 層 | 因子 | 期間 | 原始 IC60 | 殘差 IC60（t） | 殘差 IC120（t） | 與 R20 相關 | 與 T_ld 相關 | 與 ΔT_ld10 相關 | 只扣 X1 分數的殘差 IC60（t） |', '|---|---|---|---|---|---|---|---|---|---|']
    verdict = {}
    for (ln, dn), g in R_.groupby(['layer', 'd'], sort=False):
        ok = []
        for era, x in g.groupby('era'):
            t60, t120 = nw_t(x.ic60.to_numpy(), 3), nw_t(x.ic120.to_numpy(), 3)
            ok.append((x.ic60.mean(), t60, x.ic120.mean(), t120))
            Inc.append(f"| {ln} | {dn} | {era} | {x.raw60.mean():+.3f} | {x.ic60.mean():+.3f}（{t60:.1f}） | {x.ic120.mean():+.3f}（{t120:.1f}） | {x.cR20.mean():+.2f} | {x.cTld.mean():+.2f} | {x.cdT.mean():+.2f} | {x.ic60x1.mean():+.3f}（{nw_t(x.ic60x1.to_numpy(), 3):.1f}） |")
        if len(ok) == 2:
            pos = all(o[0] > 0 and o[2] > 0 for o in ok); ts = sorted([ok[0][1], ok[1][1]])
            verdict[(ln, dn)] = pos and ts[1] >= 2.0 and ts[0] >= 1.0
    Inc += ['', '增量判定（殘差 IC60、IC120 兩期 > 0，殘差 IC60 一期 t ≥ 2、另一期 ≥ 1）：' + '；'.join(f"{k[0]} {k[1]} {'**有增量**' if v else '無'}" for k, v in verdict.items()), '']
    return dict(summary='\n'.join(['# M-179 結果：個股多空點扣除 R20、T_ld、ΔT_ld10 後的增量（奇數代碼）\n'] + Inc), headline='多空點增量（扣 ΔT_ld）：' + ('、'.join(f'{k[0]}{k[1]}' for k, v in verdict.items() if v) or '無'), n_variants=0)
    S1 = ['## 摘要（IC 5／20／60／120；高−低 60 日）\n', '| 因子 | 層 | 期間 | IC5 | IC20 | IC60 | IC120 | 高−低 60 日（t） |', '|---|---|---|---|---|---|---|---|']
    for fn, la, era, ic, sp, t in summ:
        S1.append(f"| {fn} | {la} | {era} | " + ' | '.join(f'{v:+.3f}' for v in ic) + f" | {sp:+.2%}（{t:.1f}） |")
    return dict(summary='\n'.join(['# M-176 結果：個股多空點（正確定義）＋增量檢查（奇數代碼；F 計數 54）\n'] + Inc + S1 + [''] + Bt + [''] + L), headline='個股多空點（正確定義）：' + '、'.join(f'{k[0]}{k[1]}' for k, v in verdict.items() if v) + ' 有增量' if any(verdict.values()) else '個股多空點（正確定義）：無增量', n_variants=0)
