"""上漲路徑形狀（進場前 60 日）：穩定斜坡 vs 大漲大跌 vs 一路噴
全部無尺度：用日報酬"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('/home/claude/bt/fac/lock_ext.py').read().split("\nif __name__")[0])
from evaluate import prepare, _nw_t
r = np.log(C / C.ffill().shift(1))
W = 60
rw = r.rolling(W, min_periods=50)
F = {}
F['ER 效率'] = (np.log(C / C.shift(W))) / r.abs().rolling(W, min_periods=50).sum()
F['大振幅日比例 |r|≥5%'] = (r.abs() >= 0.05).astype(float).where(r.notna()).rolling(W, min_periods=50).mean()
F['報酬一階自相關'] = r.rolling(W, min_periods=50).corr(r.shift(1))
pos = r.clip(lower=0); neg = (-r).clip(lower=0)
F['下跌日波動 ÷ 上漲日波動'] = r.where(r < 0).rolling(W, min_periods=20).std() / r.where(r > 0).rolling(W, min_periods=20).std()
# 前 5 大漲日占全部上漲的比例（漲幅集中在少數幾天 = 短線資金）
def topk_share(x, k=5):
    a = np.sort(np.nan_to_num(np.clip(x, 0, None)), axis=0)[::-1]
    return a[:k].sum(0) / np.maximum(a.sum(0), 1e-9)
rv = r.values; ts = np.full(rv.shape, np.nan)
for t in range(W, len(rv)): ts[t] = topk_share(rv[t - W + 1:t + 1])
F['前5大漲日占上漲比例'] = pd.DataFrame(ts, index=C.index, columns=codes)
F['最大單日跌幅'] = r.rolling(W, min_periods=50).min()
F['回檔次數：跌破 5 日均線天數比例'] = (C < C.rolling(5).mean()).astype(float).where(C.notna()).rolling(W, min_periods=50).mean()
pd.to_pickle(F, '/home/claude/bt/fac/path_shape.pkl')
# ── 個案 ──
cases = [('6223', '2025-10-10'), ('4743', '2020-04-29'), ('3231', '2023-02-21'), ('6204', '2020-08-06'),
         ('6215', '2025-02-18'), ('6538', '2026-09-15'), ('2485', '2026-03-31'), ('8171', '2017-12-20'), ('6144', None), ('4946', None)]
T = pd.read_csv('/home/claude/bt/fac/trades_real_off0.csv', dtype={'代號': str})
rows = []
for c, d in cases:
    if d is None:
        x = T[T.代號 == c].iloc[0]; d = dates[int(x.進場) - 1]
    d = pd.Timestamp(d); d = dates[dates.get_indexer([d], method='pad')[0]]
    o = dict(代號=c, 名稱=NM.get(c, ''), 訊號日=d.date())
    for k, X in F.items(): o[k] = round(float(X.at[d, c]), 2)
    rows.append(o)
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 20)
print(pd.DataFrame(rows).to_string(index=False), flush=True)
# ── 強勢股內：各形狀分五組，60 日超額、跌超 20%、漲超 20% ──
E = prepare(P); d_ = E['dates']
strong = (cur.rank(axis=1, pct=True) >= 0.9).loc[d_]
ent = O.shift(-1)
HH = Hd.rolling(120, min_periods=100).max(); LL = Ld.rolling(120, min_periods=100).min()
iH = Hd.rolling(120, min_periods=100).apply(np.argmax, raw=True); iL = Ld.rolling(120, min_periods=100).apply(np.argmin, raw=True)
nd = iH - iL; sp = (np.log(HH / LL) / nd).where(nd > 0)
fast = (sp.where(cur.rank(axis=1, pct=True) >= 0.9).rank(axis=1, pct=True) > 0.75).loc[d_]
R = (Cfv_df := C.ffill()).shift(-60).div(ent).sub(1).loc[d_]
out2 = []
for grp_lab, G in (('全部強勢股', strong), ('強勢股中速度最快 25%', strong & fast)):
    Rg = R.where(G); ex = Rg.sub(R.where(strong).mean(1), axis=0)
    for k, X in F.items():
        q = X.loc[d_].where(G & Rg.notna()).rank(axis=1, pct=True)
        o = dict(群組=grp_lab, 特徵=k)
        for i in range(5):
            m = (q > i / 5) & (q <= (i + 1) / 5); rr = Rg.where(m).stack().dropna()
            o[f'Q{i+1} 超額'] = round(ex.where(m).mean(1).mean() * 100, 2)
            o[f'Q{i+1} 崩/噴'] = f'{(rr < -.2).mean():.0%}/{(rr > .2).mean():.0%}'
        e1 = ex.where(q > .8).mean(1); e0 = ex.where(q <= .2).mean(1); dd = (e1 - e0).dropna()
        o['Q5−Q1 正年'] = f'{(dd.groupby(dd.index.year).mean() > 0).sum()}/{dd.index.year.nunique()}'
        out2.append(o); print(o, flush=True)
D2 = pd.DataFrame(out2); D2.to_csv('/home/claude/bt/fac/path_shape.csv', index=False, encoding='utf-8-sig')
print(D2.to_string(index=False))
