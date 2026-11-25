# M-203：L-131 P 系列在最近兩年保留集（2024-11-25 之後）的驗收。定義見 registry_L.md M-203。
import numpy as np, pandas as pd
from pathlib import Path
LAB = Path(__file__).resolve().parents[1]


def st(r):
    n = len(r); eq = (1 + r).cumprod(); ann = eq.iloc[-1] ** (245 / n) - 1; dd = (eq / eq.cummax() - 1).min()
    return ann, dd, eq.iloc[-1] - 1


def main():
    z = np.load(LAB / 'results' / 'L131' / 'series.npz', allow_pickle=True)
    idx = pd.to_datetime(z['dates'], unit='D'); S = {str(n): pd.Series(z['values'][:, i], index=idx) for i, n in enumerate(z['names'])}
    lo = pd.Timestamp('2024-11-25'); base = S['base'].loc[lo:]
    L = [f'# M-203 結果：L-131 最近兩年保留集驗收（{base.index[0].date()}～{base.index[-1].date()}，{len(base)} 個交易日）\n',
         '| 版本 | 年化 | 累積 | 最大回撤 | 年化差 | 回撤差 |', '|---|---|---|---|---|---|']
    ab, db, cb = st(base); L.append(f'| 新基準 | {ab:+.1%} | {cb:+.1%} | {db:.1%} | | |')
    res = {}
    for k in ('P20（主）', 'P10', 'P30', 'R20（主）'):
        if k not in S: continue
        a, d, c = st(S[k].loc[lo:]); res[k] = (a - ab, d - db)
        L.append(f'| {k} | {a:+.1%} | {c:+.1%} | {d:.1%} | {a - ab:+.1%} | {(d - db) * 100:+.1f}pt |')
    da, dd = res['P20（主）']; ok = da >= -0.02 and dd >= -0.03
    v = f"判準（P20 年化 ≥ 新基準 − 2pt、回撤不深於 3pt）：{'✓ 通過' if ok else '✗ 不通過'}（年化差 {da:+.1%}、回撤差 {dd * 100:+.1f}pt）"
    return dict(summary='\n'.join(L[:1] + [v + '\n'] + L[1:]), headline=v, n_variants=0)
