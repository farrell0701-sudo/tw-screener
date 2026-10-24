"""進場時點比較：t 收盤 / t+1 開盤 / t+1 收盤，出場固定在 t+1+h 收盤，只看進場價的差異"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
import numpy as np, pandas as pd
from panel import load_panel
from evaluate import prepare, _row_spearman, _nw_t
from factors import build
P = load_panel(); E = prepare(P); C, O = P['C'], P['O']; Cf = C.ffill(); d = E['dates']; U = E['univ']
D = pd.read_pickle('/home/claude/bt/struct/daily.pkl')
D = {k: v.reindex(index=C.index, columns=C.columns) for k, v in D.items()}
F = build(P, D)
ent = {'t收盤': C, 't+1開盤': O.shift(-1), 't+1收盤': C.shift(-1)}
legs = {'隔夜跳空 C[t]→O[t+1]': (C, O.shift(-1)), '隔日盤中 O[t+1]→C[t+1]': (O.shift(-1), C.shift(-1))}
rows = []
for fname in ['T_ld', 'S_ovh']:
    X, s, _ = F[fname]; G = (X * s).where(U).loc[d]
    q = G.rank(axis=1, pct=True)
    for h in (20, 60):
        ex = Cf.shift(-(1 + h))
        for k, en in ent.items():
            r = np.log(ex / en).where(C.shift(-1).notna()).loc[d]; r.iloc[-(h + 2):] = np.nan
            m = G.notna() & r.notna(); xr = r.where(m).sub(r.where(m).mean(1), axis=0)
            ic = _row_spearman(G, xr)
            q5 = xr.where(q > 0.8).mean(1).mean() * 100; q1 = xr.where(q <= 0.2).mean(1).mean() * 100
            rows.append(dict(因子=fname, h=h, 進場=k, IC=round(ic.mean(), 4), t=round(_nw_t(ic, h), 1),
                             Q5超額=round(q5, 2), Q1超額=round(q1, 2), 價差=round(q5 - q1, 2)))
    for k, (a, b) in legs.items():
        r = np.log(b / a).loc[d]; m = G.notna() & r.notna(); xr = r.where(m).sub(r.where(m).mean(1), axis=0)
        rows.append(dict(因子=fname, h='單段', 進場=k, Q5超額=round(xr.where(q > 0.8).mean(1).mean() * 100, 3),
                         Q1超額=round(xr.where(q <= 0.2).mean(1).mean() * 100, 3)))
R = pd.DataFrame(rows); pd.set_option('display.width', 200); print(R.to_string(index=False))
R.to_csv('entry_timing.csv', index=False, encoding='utf-8-sig')
# t+1 開盤即漲停（買不到）的比例：Q5 vs 全體
up = (P['O'].shift(-1) / C - 1 >= 0.095).loc[d]
for fname in ['T_ld', 'S_ovh']:
    X, s, _ = F[fname]; G = (X * s).where(U).loc[d]; q = G.rank(axis=1, pct=True)
    print(fname, '隔日開盤漲停比例 Q5 %.2f%%  全體 %.2f%%' % (up.where(q > 0.8).stack().mean() * 100, up.where(G.notna()).stack().mean() * 100))
