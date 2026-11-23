# M-175：個股「多空點」（前 2 個日曆月還原收盤高低點中點）有沒有資訊（M-119 評估；奇數代碼）。定義見 registry_M.md M-175。
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
    mon = dates.to_period('M')
    df = pd.DataFrame(CA, index=dates)
    g = df.groupby(mon); mx, mn, cnt = g.max(), g.min(), g.count()
    hi2 = mx.rolling(2, min_periods=2).max().shift(1); lo2 = mn.rolling(2, min_periods=2).min().shift(1); n2 = cnt.rolling(2).sum().shift(1)
    D1, D2 = {}, {}
    for m, (t, tp) in M165.samp_prev(dates).items():
        M = dates[t].to_period('M')
        if M not in hi2.index: continue
        h, l, c2 = hi2.loc[M].to_numpy(), lo2.loc[M].to_numpy(), n2.loc[M].to_numpy()
        c = pd.DataFrame(CA[max(0, tp - 5):tp + 1]).ffill().to_numpy()[-1]
        ok = np.isfinite(h) & np.isfinite(l) & (c2 >= 20) & np.isfinite(c) & (h > l)
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
    L, summ = M120.evaluate(1, None, fac_fn=build, layers=layers)
    # D1 二分表
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    keys = P['meta']['security_key'].astype(str).to_numpy()
    odd = (P['meta']['stock_id'].astype(str).str[-1].astype(int) % 2 == 1).to_numpy()
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; U, pool = dbg['U'], dbg['pool']; del S
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
    S1 = ['## 摘要（IC 5／20／60／120；高−低 60 日）\n', '| 因子 | 層 | 期間 | IC5 | IC20 | IC60 | IC120 | 高−低 60 日（t） |', '|---|---|---|---|---|---|---|---|']
    for fn, la, era, ic, sp, t in summ:
        S1.append(f"| {fn} | {la} | {era} | " + ' | '.join(f'{v:+.3f}' for v in ic) + f" | {sp:+.2%}（{t:.1f}） |")
    return dict(summary='\n'.join(['# M-175 結果：個股多空點的資訊（奇數代碼；F 計數 52）\n'] + S1 + [''] + Bt + [''] + L), headline='個股多空點評估完成', n_variants=0)
