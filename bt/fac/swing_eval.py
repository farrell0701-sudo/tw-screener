import sys; sys.path.insert(0, '/home/claude/bt/fac')
import numpy as np, pandas as pd
from panel import load_panel; from evaluate import prepare; from factors3 import tld_floor
P = load_panel(); E = prepare(P); U = E['univ']; C, O = P['C'], P['O']; dates = C.index; cols = C.columns
import os; F = pd.read_pickle(f"/home/claude/bt/fac/swing_shape_{os.environ.get('ZK','1.5')}_lo.pkl")
ldf = tld_floor(P).where(U)
indv = P['industry'].reindex(cols); known = indv.notna().values
g = ldf.loc[:, known].T.groupby(indv[known].values).transform('median').T.reindex(columns=cols)
gp = g.where(U).rank(axis=1, pct=True); cur = ldf.where((gp >= 0.5) | gp.isna())
strong = cur.rank(axis=1, pct=True) >= 0.9
# 類型：上漲速度、回檔速度各自在當日強勢股中高於／低於中位數
uh = F['上漲速度'].where(strong).rank(axis=1, pct=True) > 0.5
DK = os.environ.get('DK', '回檔深度÷前段漲幅')
dh = F[DK].where(strong).rank(axis=1, pct=True) > 0.5
ok = F['上漲速度'].notna() & F[DK].notna() & strong
TYPE = pd.DataFrame('', index=dates, columns=cols)
TYPE = TYPE.mask(ok & ~uh & dh, '緩漲急跌').mask(ok & uh & dh, '急漲急跌').mask(ok & uh & ~dh, '急漲整理').mask(ok & ~uh & ~dh, '緩漲整理')
cases = [('6223', '2025-09-26', '旺矽 +297%'), ('4743', '2020-04-29', '合一 +396%'), ('3231', '2023-02-21', '緯創 +291%'), ('6204', '2020-08-06', '艾華 +232%'),
         ('6215', '2025-02-18', '和椿 −30%'), ('6538', '2026-09-15', '倉和 −27%'), ('2485', '2026-03-31', '兆赫 −27%'), ('8171', '2017-12-20', '天宇 −36%')]
rows = []
for c, d, lab in cases:
    d = dates[dates.get_indexer([pd.Timestamp(d)], method='pad')[0]]
    o = {'個股': lab, '類型': TYPE.at[d, c]}
    for k, X in F.items():
        v = X.at[d, c]; pr = X.loc[d].where(strong.loc[d]).rank(pct=True).get(c)
        o[k] = f'{v:.3f}' + (f' ({pr:.0%})' if pr == pr else '') if v == v else '—'
    rows.append(o)
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 20)
print('個案（括號 = 當天在強勢股中的百分位）'); print(pd.DataFrame(rows).to_string(index=False))
d_ = E['dates']; ent = O.shift(-1)
st = strong.loc[d_]
for h in (20, 60):
    R = C.ffill().shift(-h).div(ent).sub(1).loc[d_].where(st)
    ex = R.sub(R.mean(1), axis=0)
    print(f'\n== {h} 日：四種類型（強勢股內，相對同日強勢股）')
    rr = []
    for ty in ('緩漲急跌', '急漲急跌', '急漲整理', '緩漲整理'):
        m = (TYPE.loc[d_] == ty) & R.notna()
        r = R.where(m).stack().dropna(); e = ex.where(m).mean(1)
        y = e.groupby(e.index.year).mean() * 100
        rr.append(dict(類型=ty, 占比=f'{m.sum().sum() / (st & R.notna()).sum().sum():.0%}', 平均超額=f'{e.mean() * 100:+.2f}%',
                       崩跌超20=f'{(r < -.2).mean():.1%}', 噴漲超20=f'{(r > .2).mean():.1%}',
                       **{str(k): round(v, 1) for k, v in y.items()}))
    print(pd.DataFrame(rr).to_string(index=False))
    if h == 60:
        print('\n== 60 日：各特徵五分組（強勢股內）')
        q_rows = []
        for k, X in F.items():
            q = X.loc[d_].where(st & R.notna()).rank(axis=1, pct=True); o = dict(特徵=k)
            for i in range(5):
                m = (q > i / 5) & (q <= (i + 1) / 5); r = R.where(m).stack().dropna()
                o[f'Q{i+1}'] = f'{ex.where(m).mean(1).mean() * 100:+.2f} ({(r < -.2).mean():.0%}/{(r > .2).mean():.0%})'
            dd = (ex.where(q > .8).mean(1) - ex.where(q <= .2).mean(1)).dropna()
            o['Q5>Q1 年數'] = f'{(dd.groupby(dd.index.year).mean() > 0).sum()}/{dd.index.year.nunique()}'
            q_rows.append(o)
        print('格式：平均超額% (崩跌超20%比例/噴漲超20%比例)'); print(pd.DataFrame(q_rows).to_string(index=False))
