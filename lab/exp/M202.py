# M-202：X1 與 V4.1 的互補來自進場還是出場——進出場交叉互換。定義見 registry_M.md M-202。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim, evalu as E

END = pd.Timestamp('2024-11-13')
SEGS = (('2005–15', '2005-01-01', '2015-12-31'), ('2016–20', '2016-01-01', '2020-12-31'), ('2021–24/11', '2021-01-01', '2024-11-13'), ('16–24/11', '2016-01-01', '2024-11-13'))


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    meta = P['meta']; odd = (meta['stock_id'].astype(str).str[-1].astype(int) % 2 == 1).to_numpy()
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; cols = dbg['cols']
    def full(a, fill):
        a = np.asarray(a)
        if a.shape[1] == N: return a
        o = np.full((T, N), fill, a.dtype); o[:, cols] = a; return o
    U = full(dbg['U'], False); RKX = full(S['RKX'], np.nan); RKV = full(S['RKV'], np.nan); below = S['daily']['below'].to_numpy(bool); del S
    VALID = np.asarray(F['VALID']); CA = np.where(VALID, np.asarray(F['CA'], np.float64), np.nan); O = np.asarray(F['OA'], np.float64)
    ATRP = np.asarray(F['ATR'], np.float64) / pd.DataFrame(CA).ffill().to_numpy()
    tend = int(dates.searchsorted(END, side='right')) - 1; s0 = int(dates.searchsorted(pd.Timestamp('2005-01-01')))
    acc = {k: np.zeros(T) for k in ('XT', 'XF', 'VT', 'VF')}; cnt = {k: np.zeros(T) for k in acc}
    ntr = {k: 0 for k in acc}
    def book(key, j, t, k_last, exit_px, px):
        # t 日：開盤 → 收盤；t+1..k_last：收盤 → 收盤；k_last+1：收盤 → 開盤（出場）
        prev = px
        for d in range(t, k_last + 1):
            cc = CA[d, j]
            if not np.isfinite(cc): continue
            acc[key][d] += cc / prev - 1; cnt[key][d] += 1; prev = cc
        if k_last + 1 <= tend:
            acc[key][k_last + 1] += exit_px / prev - 1; cnt[key][k_last + 1] += 1
        ntr[key] += 1
    for s in range(s0, tend - 21):
        for kind, rk, lim in (('X', RKX, 6), ('V', RKV, 4)):
            if kind == 'X' and below[s]: continue
            for j in np.nonzero((rk[s] <= lim) & odd)[0]:
                t = s + 1; px = O[t, j]
                if not np.isfinite(px) or px <= 0: continue
                # T：移動停損
                d = ATRP[s, j] * 4; d = 0.04 if not np.isfinite(d) else min(max(d, 0.04), 0.15)
                hi = px; ex = None
                for k in range(t, min(t + 250, tend)):
                    cc = CA[k, j]
                    if not np.isfinite(cc): continue
                    hi = max(hi, cc)
                    if cc < hi * (1 - d) or (hi >= px * 1.06 and cc < px * 1.005) or k - t + 1 >= 250:
                        o = O[k + 1, j]; ex = (k, o if np.isfinite(o) else cc); break
                if ex is not None: book(kind + 'T', j, t, ex[0], ex[1], px)
                # F：固定 20 日
                k = t + 19
                if k + 1 <= tend:
                    o = O[k + 1, j]; book(kind + 'F', j, t, k, o if np.isfinite(o) else CA[k, j], px)
    idx = dates
    Ser = pd.DataFrame({k: np.where(cnt[k] > 0, acc[k] / np.maximum(cnt[k], 1), 0.0) for k in acc}, index=idx).loc['2005-01-01':END]
    # U 等權（奇數代碼）
    with np.errstate(all='ignore'):
        C = pd.DataFrame(CA).ffill(limit=5).to_numpy(); Rr = C / np.vstack([np.full((1, N), np.nan), C[:-1]]) - 1
        rU = np.nanmean(np.where(np.vstack([np.zeros((1, N), bool), U[:-1]]) & odd & np.isfinite(Rr), np.clip(Rr, -0.5, 1), np.nan), axis=1)
    rU = pd.Series(np.nan_to_num(rU), index=idx).loc['2005-01-01':END]
    out = ['# M-202 結果：進出場交叉互換（奇數代碼；不限名額、每筆獨立、等權；無成本；到 2024-11-13）\n',
           'X = X1 進場（名次 ≤ 6、閘門開）；V = V4.1 進場（名次 ≤ 4）；T = 移動停損；F = 固定 20 日。\n',
           '交易筆數：' + '、'.join(f'{k} {v}' for k, v in ntr.items()) + '\n', '## 各序列（年化／最大回撤／Calmar）\n',
           '| 序列 | ' + ' | '.join(s[0] for s in SEGS) + ' |', '|---|---|---|---|---|']
    for k in Ser:
        out.append(f'| {k} | ' + ' | '.join('{:+.1%}／{:.1%}／{:.2f}'.format(*E.met(Ser[k].loc[lo:hi])[:3]) for _, lo, hi in SEGS) + ' |')
    pairs = [('XT', 'XF', '同進場 X、不同出場'), ('VT', 'VF', '同進場 V、不同出場'), ('XT', 'VT', '不同進場、同出場 T'), ('XF', 'VF', '不同進場、同出場 F'), ('XT', 'VF', '實際組合'), ('XF', 'VT', '互換組合')]
    out += ['', '## 相關（日報酬｜扣掉 U 的超額）\n', '| 配對 | 意義 | 2005–15 | 2016–24/11 |', '|---|---|---|---|']
    for a, b, nm in pairs:
        c = []
        for lo, hi in (('2005-01-01', '2015-12-31'), ('2016-01-01', '2024-11-13')):
            x, y, u = Ser[a].loc[lo:hi], Ser[b].loc[lo:hi], rU.loc[lo:hi]
            c.append(f"{x.corr(y):.2f}｜{(x - u).corr(y - u):+.2f}")
        out.append(f"| {a} × {b} | {nm} | {c[0]} | {c[1]} |")
    out += ['', '## 60/40 組合（每月再平衡；年化／最大回撤／Calmar）\n', '| 組合 | ' + ' | '.join(s[0] for s in SEGS) + ' |', '|---|---|---|---|---|']
    for a, b in (('XT', 'VF'), ('XT', 'VT'), ('XF', 'VF'), ('XF', 'VT')):
        m = sim.mix(Ser[a], Ser[b])
        out.append(f'| {a} 60／{b} 40 | ' + ' | '.join('{:+.1%}／{:.1%}／{:.2f}'.format(*E.met(m.loc[lo:hi])[:3]) for _, lo, hi in SEGS) + ' |')
    return dict(summary='\n'.join(out), headline='進出場交叉互換描述', n_variants=0)
