"""3 根 K 分形之字：高點 = 前後各 3 根都沒超過它（3 天後才確認，不偷看）
近 120 日最低點以後的波段：每次回檔的「跌幅 ÷ 前段漲幅」（回撤比例）、上漲波平均漲幅、上漲速度"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('/home/claude/bt/fac/lock_ext.py').read().split("\ndef engine")[0])
from evaluate import prepare
E = prepare(P)
TH = -np.log(1 - 0.09)   # 9% 之字
lh, ll = np.log(Hd.values), np.log(Ld.values)
strong = (cur.rank(axis=1, pct=True) >= 0.9).values
names = ['平均回撤比例', '最大回撤比例', '最近一次回撤比例', '上漲波平均漲幅', '上漲波速度', '上漲波數']
OUT = {k: np.full(lh.shape, np.nan) for k in names}
T_, J_ = lh.shape
for j in range(J_):
    h, l = lh[:, j], ll[:, j]
    if np.isnan(h).all(): continue
    piv = []; tr = 1; ei = None
    for t in range(T_):
        if np.isnan(h[t]) or np.isnan(l[t]): continue
        if ei is None: ei = t; continue
        if tr == 1:
            if h[t] >= h[ei]: ei = t
            elif h[ei] - l[t] >= TH: piv.append([ei, h[ei], 1]); tr = -1; ei = t
        else:
            if l[t] <= l[ei]: ei = t
            elif h[t] - l[ei] >= TH: piv.append([ei, l[ei], -1]); tr = 1; ei = t
        if not strong[t, j]: continue
        lo0 = max(0, t - 120); w = l[lo0:t + 1]
        if np.isnan(w).all(): continue
        tmin = lo0 + int(np.nanargmin(w))
        seq = [(tmin, l[tmin], -1)] + [tuple(p) for p in piv if p[0] > tmin]
        # 最後補上今天的位置（進行中的波）
        if seq[-1][2] == -1: seq.append((t, h[t], 1))
        else: seq.append((t, l[t], -1))
        ups, rets = [], []
        for a, b in zip(seq[:-1], seq[1:]):
            if a[2] == -1 and b[2] == 1 and b[1] > a[1]: ups.append((b[1] - a[1], b[0] - a[0]))
            elif a[2] == 1 and b[2] == -1 and ups and ups[-1][0] > 0: rets.append((1 - np.exp(-(a[1] - b[1]))) / (np.exp(ups[-1][0]) - 1))   # 用 % 算，同看盤
        if not ups: continue
        OUT['上漲波平均漲幅'][t, j] = np.mean([u for u, _ in ups])
        OUT['上漲波速度'][t, j] = sum(u for u, _ in ups) / max(1, sum(n for _, n in ups))
        OUT['上漲波數'][t, j] = len(ups)
        if rets:
            OUT['平均回撤比例'][t, j] = np.mean(rets); OUT['最大回撤比例'][t, j] = max(rets); OUT['最近一次回撤比例'][t, j] = rets[-1]
F = {k: pd.DataFrame(v, index=C.index, columns=codes) for k, v in OUT.items()}
pd.to_pickle(F, '/home/claude/bt/fac/zz9.pkl')
S = cur.rank(axis=1, pct=True) >= 0.9
cases = [('6223', '2025-09-26', '旺矽 +297%'), ('4743', '2020-04-28', '合一 +396%'), ('3231', '2023-02-21', '緯創 +291%'), ('6204', '2020-08-06', '艾華 +232%'),
         ('6215', '2025-02-18', '和椿 −30%'), ('6538', '2026-09-15', '倉和 −27%'), ('2485', '2026-03-31', '兆赫 −27%'), ('8171', '2017-12-20', '天宇 −36%')]
rows = []
for c, d, lab in cases:
    d = C.index[C.index.get_indexer([pd.Timestamp(d)], method='pad')[0]]
    o = {'個股': lab}
    for k, X in F.items():
        v = X.at[d, c]; pr = X.loc[d].where(S.loc[d]).rank(pct=True).get(c)
        o[k] = f'{v:.2f} ({pr:.0%})' if v == v else '—'
    rows.append(o)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 20)
print('個案（括號 = 當天在強勢股中的百分位）'); print(pd.DataFrame(rows).to_string(index=False))
d_ = E['dates']; ent = O.shift(-1); st = S.loc[d_]
R = C.ffill().shift(-60).div(ent).sub(1).loc[d_].where(st); ex = R.sub(R.mean(1), axis=0)
rows = []
for k, X in F.items():
    q = X.loc[d_].where(st & R.notna()).rank(axis=1, pct=True); o = dict(特徵=k)
    for i in range(5):
        m = (q > i / 5) & (q <= (i + 1) / 5); r = R.where(m).stack().dropna()
        o[f'Q{i+1}'] = f'{ex.where(m).mean(1).mean() * 100:+.2f} ({(r < -.2).mean():.0%}/{(r > .2).mean():.0%})'
    dd = (ex.where(q > .8).mean(1) - ex.where(q <= .2).mean(1)).dropna()
    o['Q5>Q1年數'] = f'{(dd.groupby(dd.index.year).mean() > 0).sum()}/{dd.index.year.nunique()}'; rows.append(o)
print('\n60 日，強勢股內五分組：平均超額% (崩跌超20%/噴漲超20%)'); print(pd.DataFrame(rows).to_string(index=False))
