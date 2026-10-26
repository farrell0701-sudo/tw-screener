"""回撤比例 × 上漲速度 雙因子（9% 之字，近 120 日低點之後），強勢股內各取三等分，看 9 格的 60 日表現"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
import numpy as np, pandas as pd
from panel import load_panel; from evaluate import prepare; from factors3 import tld_floor
P = load_panel(); E = prepare(P); U = E['univ']; C, O = P['C'], P['O']; cols = C.columns
ldf = tld_floor(P).where(U)
indv = P['industry'].reindex(cols); known = indv.notna().values
g = ldf.loc[:, known].T.groupby(indv[known].values).transform('median').T.reindex(columns=cols)
gp = g.where(U).rank(axis=1, pct=True); cur = ldf.where((gp >= 0.5) | gp.isna())
strong = cur.rank(axis=1, pct=True) >= 0.9
F = pd.read_pickle('/home/claude/bt/fac/zz9.pkl')
d_ = E['dates']; st = strong.loc[d_]
sp = F['上漲波速度'].loc[d_].where(st); rt = F['平均回撤比例'].loc[d_].where(st)
ok = sp.notna() & rt.notna()
qs = sp.where(ok).rank(axis=1, pct=True); qr = rt.where(ok).rank(axis=1, pct=True)
cut = lambda q: (q > 1/3).astype(int) + (q > 2/3).astype(int)
bs, br = cut(qs).where(ok), cut(qr).where(ok)
ent = O.shift(-1)
for h in (20, 60):
    R = C.ffill().shift(-h).div(ent).sub(1).loc[d_].where(st); ex = R.sub(R.mean(1), axis=0)
    rows = []
    for i, sl in enumerate(('速度慢', '速度中', '速度快')):
        for k, rl in enumerate(('回撤淺', '回撤中', '回撤深')):
            m = (bs == i) & (br == k) & R.notna(); r = R.where(m).stack().dropna(); e = ex.where(m).mean(1)
            y = e.groupby(e.index.year).mean()
            rows.append(dict(速度=sl, 回撤比例=rl, 筆數占比=f'{m.sum().sum() / (ok & R.notna()).sum().sum():.0%}',
                             平均超額=f'{e.mean() * 100:+.2f}%', 崩跌超20=f'{(r < -.2).mean():.1%}', 噴漲超20=f'{(r > .2).mean():.1%}',
                             正年=f'{(y > 0).sum()}/{y.notna().sum()}'))
    print(f'== {h} 日（相對同日強勢股）'); print(pd.DataFrame(rows).to_string(index=False))
# 參考：兩者相關
print('速度與回撤比例的截面等級相關（強勢股內，日平均）', round(sp.rank(axis=1).corrwith(rt.rank(axis=1), axis=1).mean(), 2))
