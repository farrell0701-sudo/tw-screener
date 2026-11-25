# M-180：營收短長比 R6 與扣除 R1、R4 後的增量 R6r（M-119；奇數代碼）。定義見 registry_M.md M-180。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'exp'))
import M120

# 相容：新版 pyarrow 把 date 欄讀成 Python date 物件（pqshim 讀成 datetime64）；只轉換型別，不改數值。
_rp = pd.read_parquet
def _read_parquet(*a, **k):
    df = _rp(*a, **k)
    for c in ('period_end', 'available_date', 'date'):
        if c in df.columns and df[c].dtype == object: df[c] = pd.to_datetime(df[c])
    return df
pd.read_parquet = _read_parquet


def r6_factors(dates=None):
    base = M120.rev_factors(dates); R1, R4 = base['R1 單月年增'], base['R4 營收創高']
    d = pd.read_parquet(M120.TWD / 'derived' / 'month_revenue.parquet')
    d = d[d.security_key.notna() & d.security_key.ne('')]
    d['m'] = d.period_end.dt.to_period('M')
    W = d.pivot_table(index='m', columns='security_key', values='revenue', aggfunc='last').astype(float)
    W = W.reindex(pd.period_range(W.index.min(), W.index.max(), freq='M')); W[W <= 0] = np.nan
    with np.errstate(all='ignore'):
        R6 = W.rolling(3, min_periods=3).mean() / W.rolling(12, min_periods=12).mean()
    R1 = R1.reindex_like(R6); R4 = R4.reindex_like(R6)
    res = pd.DataFrame(np.nan, index=R6.index, columns=R6.columns)
    for m in R6.index:
        y, a, b = R6.loc[m], R1.loc[m], R4.loc[m]
        ok = y.notna() & a.notna() & b.notna()
        if ok.sum() < 50: continue
        Y = y[ok].rank(pct=True).to_numpy(); X = np.column_stack([np.ones(ok.sum()), a[ok].rank(pct=True), b[ok].rank(pct=True)])
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        res.loc[m, ok[ok].index] = Y - X @ beta
    return {'R6 營收短長比': R6, 'R6r 扣R1R4增量': res}


def main():
    base = M120.rev_factors()
    ref = {'R4': base['R4 營收創高'], 'R1': base['R1 單月年增']}
    L, summ = M120.evaluate(1, fac_fn=r6_factors, ref=ref)
    L[0] = '# M-180 結果：營收短長比 R6 與增量 R6r（M-119 標準；奇數代碼；F 計數 56）\n'
    head = ['# 摘要（IC 5／20／60／120 日；高−低 60 日超額）\n', '| 因子 | 層 | 期間 | IC5 | IC20 | IC60 | IC120 | 高−低 60 日（t） |', '|---|---|---|---|---|---|---|---|']
    for fn, la, era, ic, sp, t in summ:
        head.append(f"| {fn} | {la} | {era} | " + ' | '.join(f'{v:+.3f}' for v in ic) + f" | {sp:+.2%}（{t:.1f}） |")
    g = {(fn, la, era): (ic, sp, t) for fn, la, era, ic, sp, t in summ}
    k = [g.get(('R6r 扣R1R4增量', '全市場', e)) for e in ('2005–15', '2016–26')]
    ok = all(x is not None for x in k)
    c1 = ok and all(x[0][2] > 0 and x[0][3] > 0 for x in k)
    ts = sorted([x[2] for x in k]) if ok else [0, 0]
    c2 = ok and ts[1] >= 2.0 and ts[0] >= 1.0 and all(x[1] > 0 for x in k)
    verdict = f"判準（R6r 全市場）：IC60、IC120 兩期 > 0 {'✓' if c1 else '✗'}；高−低 60 日 t 一期 ≥ 2、另一期 ≥ 1 {'✓' if c2 else '✗'} → {'提名送確認集' if c1 and c2 else '不通過：R6 不獨立於 R1／R4'}\n"
    return dict(summary='\n'.join(head + ['', verdict, ''] + L), headline=verdict.split('→')[-1].strip(), n_variants=0)
