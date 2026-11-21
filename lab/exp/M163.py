# M-163：營收高的 X1 候選中，剔除進場前已漲多者（探索集提名 → 確認集）。定義與判準見 registry_M.md M-163。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals
from M120 import rev_factors
from L123 import daily_from_monthly

ERAS = (('2005–15', '2005-01-01', '2015-12-31'), ('2016–26', '2016-01-01', '2026-12-31'))
COST = 0.0077; SLIP = 0.0015


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    keys = P['meta']['security_key'].astype(str).to_numpy()
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; ATRP = np.asarray(S['ATRP'], np.float64); del S
    U, pool, hot, steep = dbg['U'], dbg['pool'], dbg['hot'], dbg['steep']; assert U.shape[1] == N
    cand = pool & ~hot & ~steep
    r4p = signals.prk(daily_from_monthly(rev_factors(dates)['R4 營收創高'], dates, keys), pool)
    VALID = np.asarray(F['VALID']); CA = np.asarray(F['CA'], np.float64); OA = np.asarray(F['OA'], np.float64)
    Cf = pd.DataFrame(CA).ffill().to_numpy()
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CEW = np.cumprod(1 + ew)
    par = (P['meta']['stock_id'].astype(str).str[-1].astype(int) % 2).to_numpy()
    per = dates.to_period('M'); first = np.r_[True, per[1:] != per[:-1]]; ms = np.maximum.accumulate(np.where(first, np.arange(T), 0))
    ds = pd.Series(dates, index=dates); d10 = ds[ds.dt.day > 10]; samp = d10.groupby(d10.dt.to_period('M')).min()
    samp = [dates.get_loc(d) for d in samp if d >= pd.Timestamp('2005-01-01') and dates.get_loc(d) + 121 < T]

    def stop_ret(t, j):
        if not (VALID[t + 1, j] and np.isfinite(OA[t + 1, j]) and OA[t + 1, j] > 0): return np.nan, False
        px = OA[t + 1, j] * (1 + SLIP); d = ATRP[t, j] * 4; d = 0.04 if not np.isfinite(d) else min(max(d, 0.04), 0.15); hi = px
        for k in range(t + 1, t + 121):
            if not VALID[k, j]: continue
            cc = CA[k, j]
            if not np.isfinite(cc): continue
            hi = max(hi, cc)
            if cc < hi * (1 - d) or (hi >= px * 1.06 and cc < px * 1.005):
                return cc / px - 1 - COST + SLIP, (k - t) <= 20
        return Cf[t + 120, j] / px - 1 - COST + SLIP, False

    rows = []
    for t in samp:
        jj = np.nonzero(cand[t])[0]
        if len(jj) < 10: continue
        with np.errstate(all='ignore'):
            r20 = Cf[t, jj] / Cf[t - 20, jj] - 1; mtd = Cf[t, jj] / Cf[ms[t] - 1, jj] - 1
            ent = OA[t + 1, jj]; y60 = Cf[t + 60, jj] / ent - 1 - (CEW[t + 60] / CEW[t] - 1)
        for k, j in enumerate(jj):
            if not np.isfinite(ent[k]) or ent[k] <= 0 or not np.isfinite(r20[k]): continue
            sr, fast = stop_ret(t, j)
            rows.append(dict(t=t, j=j, par=int(par[j]), hi4=bool(r4p[t, j] >= 0.8), r20=r20[k], mtd=mtd[k], Y=y60[k], STOP=sr, fast=fast))
    D = pd.DataFrame(rows); D['date'] = dates[D.t]
    D['era'] = np.where(D.date <= '2015-12-31', '2005–15', '2016–26')

    def mark(X, col, frac):
        """每期在樣本內依 col 排名，剔除最高 frac。"""
        rk = X.groupby('t')[col].rank(pct=True, method='first'); return rk > 1 - frac

    def evaluate(X, col, frac):
        out = {}
        for en, _, _ in ERAS:
            E = X[X.era == en].copy(); rm = mark(E, col, frac)
            q10 = E.groupby('t').Y.transform(lambda v: v.quantile(0.1)); q90 = E.groupby('t').Y.transform(lambda v: v.quantile(0.9))
            g = lambda m: dict(n=int(m.sum()), Y=E.Y[m].mean(), STOP=E.STOP[m].mean(), fast=E.fast[m].mean(), left=(E.Y[m] <= q10[m]).mean(), right=(E.Y[m] >= q90[m]).mean())
            out[en] = (g(rm), g(~rm))
        return out

    def table(res, title):
        L = [f'### {title}\n', '| 期間 | 組 | n | Y60 | 含停損報酬 | 20 日內停損／保本 | 左尾 | 右尾 |', '|---|---|---|---|---|---|---|---|']
        for en, (a, b) in res.items():
            for nm, z in (('被剔', a), ('保留', b)):
                L.append(f"| {en} | {nm} | {z['n']} | {z['Y']:+.2%} | {z['STOP']:+.2%} | {z['fast']:.0%} | {z['left']:.0%} | {z['right']:.0%} |")
        return L

    def judge(X):
        main = evaluate(X, 'r20', 1 / 3); p20 = evaluate(X, 'r20', 0.2); p50 = evaluate(X, 'r20', 0.5)
        c = {}
        for en in main:
            a, b = main[en]
            c[en] = [a['STOP'] <= b['STOP'] - 0.01, a['fast'] >= b['fast'] + 0.05, a['Y'] <= b['Y'] + 0.01,
                     p20[en][0]['STOP'] < p20[en][1]['STOP'] and p50[en][0]['STOP'] < p50[en][1]['STOP']]
        ok = all(all(v) for v in c.values())
        L = table(main, '主版本：剔除近 20 日漲幅最高 1/3') + [''] + table(p20, '高原：剔除最高 20%') + [''] + table(p50, '高原：剔除最高 50%') + ['']
        L += table(evaluate(X, 'mtd', 1 / 3), '並報：剔除本月以來漲幅最高 1/3') + ['']
        L += ['| 判準 | ' + ' | '.join(c) + ' |', '|---|---|---|']
        for i, nm in enumerate(['1 被剔含停損低 ≥ 1pt', '2 被剔快速出場高 ≥ 5pt', '3 被剔 Y60 不高於保留 +1pt', '4 高原方向相同']):
            L.append(f"| {nm} | " + ' | '.join('✓' if c[e][i] else '✗' for e in c) + ' |')
        return ok, L

    L = ['# M-163 結果：營收高的 X1 候選中，剔除進場前已漲多者\n',
         '樣本＝X1 候選（池內、剔過熱、剔最陡）且 R4 池內百分位 ≥ 0.8；每月 10 日後第一個交易日。含停損報酬＝t+1 開盤進、X1 移動停損與保本、最長 120 日、扣 0.77%。\n']
    ok1, t1 = judge(D[(D.par == 1) & D.hi4])
    L += ['## 探索集（奇數代碼）\n'] + t1 + [f"\n**{'提名，進確認集' if ok1 else '不提名，結案（不看偶數代碼）'}**\n"]
    ok2 = None
    if ok1:
        ok2, t2 = judge(D[(D.par == 0) & D.hi4])
        L += ['## 確認集（偶數代碼，只測一次）\n'] + t2 + [f"\n**{'通過，可登記 L 系列' if ok2 else '不通過，結案'}**\n"]
    _, ta = judge(D[D.par == 1])
    L += ['## 只報告：全部 X1 候選（奇數代碼，不限營收）\n'] + ta
    head = f"探索集{'提名' if ok1 else '不提名'}" + ('' if ok2 is None else f"；確認集{'通過' if ok2 else '不通過'}")
    return dict(summary='\n'.join(L), headline=head, n_variants=0)
