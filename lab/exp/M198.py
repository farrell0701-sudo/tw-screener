# M-198：空頭時兩池怎麼走。定義見 registry_M.md M-198。
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from poolret import pool_returns, END


def main():
    df, ix = pool_returns()
    p = ix['TAIEX_P'].loc['2004-01-01':END]
    eq = (1 + df).cumprod()
    # 空頭事件：回落 ≥ 15%
    ev, pk, pkd, i, vals, idx = [], p.iloc[0], p.index[0], 0, p.values, p.index
    in_bear = False; tr, trd = None, None
    for d, v in zip(idx, vals):
        if not in_bear:
            if v > pk: pk, pkd = v, d
            elif v <= pk * 0.85: in_bear, tr, trd = True, v, d
        else:
            if v < tr: tr, trd = v, d
            if v >= pk: ev.append((pkd, trd)); in_bear = False; pk, pkd = v, d
    if in_bear: ev.append((pkd, trd))
    ev = [(a, b) for a, b in ev if a >= pd.Timestamp('2005-01-01')]
    out = ['# M-198 結果：空頭時兩池怎麼走（TAIEX 回落 ≥ 15% 事件；奇數代碼；到 2024-11-13）\n',
           '| 事件（前高 → 谷底） | 指數跌幅 | 下跌段 S｜A｜U | 前 1/3 段 S｜A｜U | 谷底後 60 日 S｜A｜U | 谷底後 120 日 S｜A｜U | 各自谷底相對指數（交易日）S｜A｜U |', '|---|---|---|---|---|---|---|']
    D = df.index
    agg = []
    for a, b in ev:
        ai, bi = D.searchsorted(a), min(D.searchsorted(b), len(D) - 1)
        third = ai + (bi - ai) // 3
        rows = {}
        for k in ('S', 'A', 'U'):
            seg = eq[k].iloc[ai:bi + 1]
            dn = eq[k].iloc[bi] / eq[k].iloc[ai] - 1
            t1 = eq[k].iloc[third] / eq[k].iloc[ai] - 1
            u60 = eq[k].iloc[min(bi + 60, len(D) - 1)] / eq[k].iloc[bi] - 1
            u120 = eq[k].iloc[min(bi + 120, len(D) - 1)] / eq[k].iloc[bi] - 1
            w = eq[k].iloc[ai:min(bi + 60, len(D) - 1) + 1]
            lag = D.get_loc(w.idxmin()) - bi
            rows[k] = (dn, t1, u60, u120, lag)
        agg.append(rows)
        f = lambda j, pct=True: '｜'.join((f"{rows[k][j]:+.0%}" if pct else f"{rows[k][j]:+d}") for k in ('S', 'A', 'U'))
        out.append(f"| {a:%Y-%m-%d} → {b:%Y-%m-%d} | {p.loc[b] / p.loc[a] - 1:+.0%} | {f(0)} | {f(1)} | {f(2)} | {f(3)} | {f(4, False)} |")
    m = lambda j, k: np.mean([x[k][j] for x in agg])
    out += ['', f"平均（{len(agg)} 次）：下跌段 S {m(0, 'S'):+.1%}｜A {m(0, 'A'):+.1%}｜U {m(0, 'U'):+.1%}；前 1/3 段 S {m(1, 'S'):+.1%}｜A {m(1, 'A'):+.1%}｜U {m(1, 'U'):+.1%}；谷底後 60 日 S {m(2, 'S'):+.1%}｜A {m(2, 'A'):+.1%}｜U {m(2, 'U'):+.1%}；120 日 S {m(3, 'S'):+.1%}｜A {m(3, 'A'):+.1%}｜U {m(3, 'U'):+.1%}",
            f"見底時間平均（相對指數谷底）：S {m(4, 'S'):+.1f}｜A {m(4, 'A'):+.1f}｜U {m(4, 'U'):+.1f} 交易日", '']
    d = df.S - df.A
    out += ['## 領先落後：S − A 日報酬差與前 k 日 U 報酬的相關\n', '| k | 2005–15 | 2016–24/11 |', '|---|---|---|']
    for k in range(0, 6):
        c = [d.loc[lo:hi].corr(df.U.shift(k).loc[lo:hi]) for lo, hi in (('2005', '2015'), ('2016', END))]
        out.append(f"| {k} | {c[0]:+.3f} | {c[1]:+.3f} |")
    return dict(summary='\n'.join(out), headline=f'{len(agg)} 次空頭事件', n_variants=0)
