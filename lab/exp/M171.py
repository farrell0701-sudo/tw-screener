# M-171：營收策略 vs 合併策略差異拆解（母體 × 邏輯）。定義見 registry_M.md M-171。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals
from signals import prk
from M122 import fin_factors
import M165

FEE, TAX, SLIP = 0.00085, 0.003, 0.0015
SEGS = (('2005–15', '2005-01-01', '2015-12-31'), ('2016–20', '2016-01-01', '2020-12-31'), ('2021–26', '2021-01-01', '2026-12-31'), ('2016–26', '2016-01-01', '2026-12-31'))


def seg(r, lo, hi):
    r = r.loc[lo:hi]; n = len(r); eq = (1 + r).cumprod(); ann = eq.iloc[-1] ** (245 / n) - 1; dd = (eq / eq.cummax() - 1).min()
    return ann, dd, (ann / -dd if dd < 0 else np.nan)


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    keys = P['meta']['security_key'].astype(str).to_numpy()
    odd = (P['meta']['stock_id'].astype(str).str[-1].astype(int) % 2 == 1).to_numpy()
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; U, pool, Tr, hot, steep = dbg['U'], dbg['pool'], dbg['Tr'], dbg['hot'], dbg['steep']
    cols = dbg['cols']; R_ = S['rules']; del S
    def full(a, fill):
        if a.shape[1] == N: return a
        o = np.full((T, N), fill, a.dtype); o[:, cols] = a; return o
    U, pool, hot, steep = full(U, False), full(pool, False), full(hot, False), full(steep, False); Tr = full(Tr, np.nan)
    dT = Tr - np.vstack([np.full((10, N), np.nan), Tr[:-10]])
    candT = pool & ~hot & ~steep
    VALID = np.asarray(F['VALID']); CA = np.asarray(F['CA'], np.float64); OA = np.asarray(F['OA'], np.float64); LA = np.asarray(F['LA'], np.float64)
    ATRP = np.asarray(F['ATR'], np.float64) / CA
    Cf = pd.DataFrame(np.where(VALID, CA, np.nan)).ffill().to_numpy()
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CX = np.cumsum(R - ew[:, None], axis=0)
    ix = P['index']['TAIEX_P'].reindex(dates).ffill(); below = (ix < ix.rolling(R_['ma']).mean()).to_numpy()
    # 營收池（每月）→ 每日母體 V（當月沿用到下個取樣日）
    A = M165.layers(U, pool, dates, keys)[0][1]
    Q5 = fin_factors(dates)['Q5 毛利率創高']; qj = pd.Index(Q5.columns.astype(str)).get_indexer(keys)
    candV = np.zeros((T, N), bool); MONTH = {}
    sp = sorted(M165.samp_prev(dates).items(), key=lambda x: x[1][0])
    for i, (m, (t, _)) in enumerate(sp):
        if not A[t].any(): continue
        mem = np.nonzero(A[t])[0]
        q = np.where(qj[mem] >= 0, Q5.loc[m].to_numpy(float)[np.clip(qj[mem], 0, None)], np.nan) if m in Q5.index else np.full(len(mem), np.nan)
        fin = np.isfinite(q); drop = np.zeros(len(mem), bool)
        if fin.sum() >= 20: drop[fin] = q[fin] <= np.nanpercentile(q[fin], 20)
        s = mem[~drop]; MONTH[t] = s
        t2 = sp[i + 1][1][0] if i + 1 < len(sp) else T
        candV[t:t2, s] = True
    candV &= U
    scT = 0.5 * np.nan_to_num(prk(Tr, candT)) + 0.5 * np.nan_to_num(prk(dT, candT))
    scV = 0.5 * np.nan_to_num(prk(Tr, candV)) + 0.5 * np.nan_to_num(prk(dT, candV))

    # ---- 拆解一：訊號層（每月取樣日、奇數代碼）
    rows = []
    for t, s in MONTH.items():
        if dates[t] < pd.Timestamp('2005-01-01') or t + 121 >= T: continue
        era = '2005–15' if dates[t] <= pd.Timestamp('2015-12-31') else '2016–26'
        e60u = (CX[t + 60] - CX[t])[U[t]]; q10, q90 = np.nanpercentile(e60u, [10, 90])
        jT = np.nonzero(candT[t] & odd)[0]; jV = np.nonzero(candV[t] & odd)[0]
        top = lambda jj, sc: jj[np.argsort(-sc[t, jj])[:6]] if len(jj) else jj
        for g, jj in (('S1 X1 候選全部', jT), ('S2 X1 候選前 6', top(jT, scT)), ('S3 營收池全部', jV), ('S4 營收池前 6（X1 分數）', top(jV, scV))):
            for j in jj:
                rows.append((g, era, CX[t + 20, j] - CX[t, j], CX[t + 60, j] - CX[t, j], CX[t + 120, j] - CX[t, j], q10, q90))
    D = pd.DataFrame(rows, columns=['g', 'era', 'e20', 'e60', 'e120', 'q10', 'q90'])
    L = ['# M-171 結果：營收策略 vs 合併策略差異拆解\n', '## 拆解一：訊號層（每月取樣日、奇數代碼；超額扣可交易池等權）\n',
         '| 組 | 期間 | 筆數 | 20 日 | 60 日 | 120 日 | 60 日中位數 | 60 日勝率 | 左尾率 | 右尾率 |', '|---|---|---|---|---|---|---|---|---|---|']
    for (g, era), x in D.groupby(['g', 'era']):
        L.append(f"| {g} | {era} | {len(x)} | {x.e20.mean():+.2%} | {x.e60.mean():+.2%} | {x.e120.mean():+.2%} | {x.e60.median():+.2%} | {(x.e60 > 0).mean():.0%} | {(x.e60 <= x.q10).mean():.0%} | {(x.e60 >= x.q90).mean():.0%} |")

    # ---- 拆解二：組合層
    def runX(cand, sc, n):
        cash = 1.0; pos = {}; eq = np.full(T, np.nan); t0 = int(dates.searchsorted(pd.Timestamp('2004-07-01')))
        buy = []
        for t in range(t0, T):
            if buy:
                free = n - len(pos); bb = [j for j in buy if j not in pos][:free]
                if bb and cash > 1e-9:
                    val = cash + sum(p['sh'] * Cf[t - 1, j] for j, p in pos.items() if np.isfinite(Cf[t - 1, j]))
                    for j in bb:
                        if not (VALID[t, j] and np.isfinite(OA[t, j]) and OA[t, j] > 0): continue
                        a = min(cash, val / n); px = OA[t, j] * (1 + SLIP); d = 4 * ATRP[t - 1, j]
                        d = 0.04 if not np.isfinite(d) else min(max(d, 0.04), 0.15)
                        pos[j] = dict(sh=a * (1 - FEE) / px, px=px, hi=px, d=d, t0=t); cash -= a
            sell = []
            for j, p in pos.items():
                c = Cf[t, j]
                if not np.isfinite(c): continue
                p['hi'] = max(p['hi'], c)
                if c < p['hi'] * (1 - p['d']) or (p['hi'] >= p['px'] * 1.06 and c < p['px'] * 1.005) or t - p['t0'] + 1 >= 250: sell.append(j)
            for j in sell:
                p = pos.pop(j); cash += p['sh'] * Cf[t, j] * (1 - FEE - TAX - SLIP)
            eq[t] = cash + sum(p['sh'] * Cf[t, j] for j, p in pos.items() if np.isfinite(Cf[t, j]))
            buy = []
            if len(pos) < n and not below[t]:
                jj = np.nonzero(cand[t])[0]
                buy = list(jj[np.argsort(-sc[t, jj])]) if len(jj) else []
        return pd.Series(eq, index=dates).dropna().pct_change().fillna(0)

    def runR(sigs):
        SIG = {t + 1: s for t, s in sigs.items()}
        cash = 1.0; pos = {}; eq = np.full(T, np.nan)
        for t in range(min(SIG), T):
            if t in SIG:
                s = [j for j in SIG[t] if j not in pos and VALID[t, j] and np.isfinite(OA[t, j]) and OA[t, j] > 0]
                if s and cash > 1e-9:
                    a = cash / len(s)
                    for j in s:
                        px = OA[t, j] * (1 + SLIP); d = 4 * ATRP[t - 1, j]; d = 0.04 if not np.isfinite(d) else min(max(d, 0.04), 0.15)
                        pos[j] = dict(sh=a * (1 - FEE) / px, px=px, hi=px, d=d, t0=t)
                    cash = 0.0
            for j in [j for j, p in pos.items() if np.isfinite(Cf[t, j]) and (Cf[t, j] < max(p['hi'], Cf[t, j]) * (1 - p['d']) or t - p['t0'] + 1 >= 120)]:
                p = pos.pop(j); cash += p['sh'] * Cf[t, j] * (1 - FEE - TAX - SLIP)
            for j, p in pos.items():
                if np.isfinite(Cf[t, j]): p['hi'] = max(p['hi'], Cf[t, j])
            eq[t] = cash + sum(p['sh'] * Cf[t, j] for j, p in pos.items() if np.isfinite(Cf[t, j]))
        return pd.Series(eq, index=dates).dropna().pct_change().fillna(0)

    sigT = {t: np.nonzero(candT[t])[0] for t in MONTH}
    OUT = {'P1 強勢候選 × X1 式（6 檔）': runX(candT, scT, 6), 'P2 營收池 × X1 式（6 檔）': runX(candV, scV, 6),
           'P3 強勢候選 × 營收式（全買、每月）': runR(sigT), 'P4 營收池 × 營收式（＝L-127 B1）': runR(MONTH),
           'P5 營收池 × X1 式（20 檔）': runX(candV, scV, 20), 'P6 強勢候選 × X1 式（20 檔）': runX(candT, scT, 20)}
    L += ['', '## 拆解二：組合層（單一路徑、全部代碼；年化／回撤／Calmar）\n', '| 格 | ' + ' | '.join(s[0] for s in SEGS) + ' |', '|---|---|---|---|---|']
    for nm, r in OUT.items():
        L.append(f'| {nm} | ' + ' | '.join('{:+.1%}／{:.1%}／{:.2f}'.format(*seg(r, lo, hi)) for _, lo, hi in SEGS) + ' |')
    k = list(OUT)
    a = lambda i, lo, hi, w=2: seg(OUT[k[i]], lo, hi)[w]
    L += ['', '## 效果（Calmar 差）\n', '| 效果 | ' + ' | '.join(s[0] for s in SEGS) + ' |', '|---|---|---|---|---|']
    for nm, (x, y) in {'母體（X1 式下：P1−P2）': (0, 1), '母體（營收式下：P3−P4）': (2, 3), '邏輯（強勢母體：P1−P3）': (0, 2), '邏輯（營收母體：P2−P4）': (1, 3),
                       '集中度（營收母體：P2−P5，6 vs 20 檔）': (1, 4), '集中度（強勢母體：P1−P6）': (0, 5)}.items():
        L.append(f'| {nm} | ' + ' | '.join(f'{a(x, lo, hi) - a(y, lo, hi):+.2f}' for _, lo, hi in SEGS) + ' |')
    L += ['', '## 效果（年化差，pt）\n', '| 效果 | ' + ' | '.join(s[0] for s in SEGS) + ' |', '|---|---|---|---|---|']
    for nm, (x, y) in {'母體（X1 式下：P1−P2）': (0, 1), '母體（營收式下：P3−P4）': (2, 3), '邏輯（強勢母體：P1−P3）': (0, 2), '邏輯（營收母體：P2−P4）': (1, 3)}.items():
        L.append(f'| {nm} | ' + ' | '.join(f'{(a(x, lo, hi, 0) - a(y, lo, hi, 0)) * 100:+.1f}' for _, lo, hi in SEGS) + ' |')
    return dict(summary='\n'.join(L), headline='營收 vs 合併差異拆解完成', n_variants=0)
