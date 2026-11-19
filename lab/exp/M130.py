# M-130：月營收延伸因子（依 M-119；奇數代碼）。定義見 registry_M.md M-130。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'exp')); sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data'))
import M120
from panel import load as load_panel
from feat import load_feat


def ext_factors(dates):
    base = M120.rev_factors()
    R1, R2, R4 = base['R1 單月年增'], base['R2 三月年增'], base['R4 營收創高']
    E1 = (R4 > 0).astype(float).where(R4.notna()).rolling(6, min_periods=6).sum()
    E3 = (R1 > 0).astype(float).where(R1.notna()).rolling(12, min_periods=12).sum()
    sd = R1.rolling(12, min_periods=10).std()
    E4 = R2 / sd.where(sd > 0)
    # E2：取樣日前一交易日的 60 日報酬
    P = load_panel(); F = load_feat('AF', fields=['CA'])
    Cf = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill()
    keys = P['meta']['security_key'].astype(str).to_numpy()
    ds = pd.Series(dates, index=dates); s = ds[ds.dt.day > 10]; samp = s.groupby(s.dt.to_period('M')).min()
    rows = {}
    for p, t0 in samp.items():
        t = dates.get_loc(t0)
        if t < 62: continue
        r = Cf.iloc[t - 1].to_numpy() / Cf.iloc[t - 61].to_numpy() - 1
        rows[p - 1] = pd.Series(r, index=keys)
    PR = pd.DataFrame(rows).T
    PR = PR.loc[:, ~PR.columns.duplicated()]
    cols = R4.columns.astype(str).intersection(PR.columns)
    r4 = R4.copy(); r4.columns = r4.columns.astype(str); r4 = r4.reindex(index=PR.index, columns=cols)
    pr = PR.reindex(columns=cols)
    both = r4.notna() & pr.notna()
    E2 = r4.where(both).rank(axis=1, pct=True) - pr.where(both).rank(axis=1, pct=True)
    return {'E1 創高持續': E1, 'E2 營收股價背離': E2, 'E3 年增連續性': E3, 'E4 穩定成長': E4}


def main():
    base = M120.rev_factors()
    L, summ = M120.evaluate(1, None, fac_fn=ext_factors, ref={'R4': base['R4 營收創高'], 'R5': base['R5 年增驚喜']})
    head = ['# M-130 結果：月營收延伸因子（奇數代碼；F 計數 25）\n', '| 因子 | 層 | 期間 | IC5 | IC20 | IC60 | IC120 | 高−低 60 日（t） |', '|---|---|---|---|---|---|---|---|']
    for fn, la, era, ic, sp, t in summ:
        head.append(f"| {fn} | {la} | {era} | " + ' | '.join(f'{v:+.3f}' for v in ic) + f" | {sp:+.2%}（{t:.1f}） |")
    return dict(summary='\n'.join(head + [''] + L), headline='營收延伸因子評估完成', n_variants=0)
