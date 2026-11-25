# M-199：beta 中性組合。定義見 registry_M.md M-199。
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'core'))
from poolret import pool_returns, END
import evalu as E

SEGS = (('2005–15', '2005-01-01', '2015-12-31'), ('2016–20', '2016-01-01', '2020-12-31'), ('2021–24/11', '2021-01-01', '2024-11-13'), ('16–24/11', '2016-01-01', '2024-11-13'))


def rbeta(y, x, w=250):
    cov = y.rolling(w, min_periods=200).cov(x); var = x.rolling(w, min_periods=200).var()
    return (cov / var).shift(1)


def main():
    df, ix = pool_returns()
    rix = ix['TAIEX_TR'].pct_change().reindex(df.index)
    bS, bA = rbeta(df.S, rix), rbeta(df.A, rix)
    mth = df.index.to_period('M'); first = ~pd.Series(mth, index=df.index).duplicated()
    wS = (1 / bS) / (1 / bS + 1 / bA); wS = wS.where(first).ffill()
    V = pd.DataFrame({'S': df.S, 'A': df.A, '50/50': 0.5 * df.S + 0.5 * df.A, '反beta權重': wS * df.S + (1 - wS) * df.A,
                      'S_h': df.S - bS * rix, 'A_h': df.A - bA * rix})
    V['50/50_h'] = 0.5 * V.S_h + 0.5 * V.A_h
    V = V.loc['2006-01-01':].dropna()
    out = ['# M-199 結果：beta 中性組合（奇數代碼；避險 = 放空 滾動β × TAIEX_TR；忽略期貨成本；2006 起，因滾動 beta 需 250 日）\n',
           '格式：年化／最大回撤／Calmar／Sharpe\n', '| 版本 | ' + ' | '.join(s[0] for s in SEGS) + ' |', '|---|---|---|---|---|']
    for k in V:
        out.append(f'| {k} | ' + ' | '.join('{:+.1%}／{:.1%}／{:.2f}／{:.2f}'.format(*E.met(V[k].loc[lo:hi])) for _, lo, hi in SEGS) + ' |')
    out += ['', f"滾動 beta 中位數：S {bS.median():.2f}、A {bA.median():.2f}；反 beta 權重中 S 的平均 {wS.mean():.0%}",
            '避險後相關（S_h 與 A_h）：' + '；'.join(f"{nm} {V.S_h.loc[lo:hi].corr(V.A_h.loc[lo:hi]):+.2f}" for nm, lo, hi in SEGS), '',
            '## 重點年份報酬\n', '| 年 | S | A | 50/50 | S_h | A_h | 50/50_h | TAIEX_TR |', '|---|---|---|---|---|---|---|---|']
    for y in ('2008', '2011', '2015', '2018', '2020', '2022'):
        g = lambda s: (1 + s.loc[y]).prod() - 1
        out.append(f"| {y} | " + ' | '.join(f"{g(V[k]):+.0%}" for k in ('S', 'A', '50/50', 'S_h', 'A_h', '50/50_h')) + f" | {g(rix.loc['2006':]):+.0%} |")
    return dict(summary='\n'.join(out), headline='beta 中性組合描述', n_variants=0)
