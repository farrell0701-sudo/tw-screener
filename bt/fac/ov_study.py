"""T_ld 強勢股（T_ldf 前 10%、族群過濾通過）內：過去 20／60／120 根 K 棒的重疊比例 vs 未來報酬
重疊比例 = 今天 [低,高] 和前 k 根 K 棒有交集的比例，取 5 日平均（同 TradingView 指標）
看：強勢股內的條件 IC、五分位超額報酬（相對同日強勢股平均）、逐年、和 T_ldf 的相關"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
import numpy as np, pandas as pd
from panel import load_panel; from evaluate import prepare, _nw_t; from factors3 import tld_floor
P = load_panel(); E = prepare(P); U = E['univ']
C, H, L = P['C'], P['H'], P['L']; idx, cols = C.index, C.columns
ldf = tld_floor(P).where(U)
indv = P['industry'].reindex(cols); known = indv.notna().values
g = ldf.loc[:, known].T.groupby(indv[known].values).transform('median').T.reindex(columns=cols)
gp = g.where(U).rank(axis=1, pct=True)
cur = ldf.where((gp >= 0.5) | gp.isna())
strong = cur.rank(axis=1, pct=True) >= 0.9

Hv, Lv = H.values, L.values
band = {(1, 20): 0, (21, 60): 0, (61, 120): 0}
acc = {b: np.zeros_like(Hv) for b in band}
for k in range(1, 121):
    o = np.zeros_like(Hv); o[k:] = (np.minimum(Hv[k:], Hv[:-k]) - np.maximum(Lv[k:], Lv[:-k])) > 0
    o[k:][np.isnan(Hv[:-k]) | np.isnan(Hv[k:])] = np.nan
    for (a, b) in band:
        if a <= k <= b: acc[(a, b)] += o
sm = lambda x: pd.DataFrame(x, idx, cols).rolling(5).mean()
o20 = sm(acc[(1, 20)] / 20); o21_60 = sm(acc[(21, 60)] / 40); o61_120 = sm(acc[(61, 120)] / 60)
o60 = (o20 * 20 + o21_60 * 40) / 60; o120 = (o60 * 60 + o61_120 * 60) / 120
F = {
    'ov20 近 20 根重疊（收斂度）': o20,
    'ov60 近 60 根重疊': o60,
    'ov120 近 120 根重疊': o120,
    'ov21-60（中期舊區）': o21_60,
    'ov61-120（遠期舊區）': o61_120,
    'ov20 − ov21-60（OF 差值）': o20 - o21_60,
    'ov20 − ov61-120': o20 - o61_120,
    'ov21-60 − ov61-120': o21_60 - o61_120,
}
d = E['dates']; S = strong.loc[d]
tr = cur.loc[d].where(S)
rows, yrs = [], {}
for name, X in F.items():
    X = X.loc[d].where(S)
    r = dict(因子=name, 強勢內中位數=round(float(np.nanmedian(X.values)), 3))
    # 與 T_ldf 的截面相關（強勢股內）
    r['與T_ldf相關'] = round(X.rank(axis=1).corrwith(tr.rank(axis=1), axis=1).mean(), 2)
    for h in (20, 60):
        va = E['fwd'][h]['va'].loc[d].where(S); raw = E['fwd'][h]['raw'].loc[d].where(S)
        m = X.notna() & va.notna()
        ic = X.where(m).rank(axis=1).corrwith(va.where(m).rank(axis=1), axis=1).dropna()
        ic = ic[m.sum(1).reindex(ic.index) >= 15]
        y = ic.groupby(ic.index.year).mean(); yrs[(name, h)] = y
        r[f'IC{h}'] = round(ic.mean(), 3); r[f't{h}'] = round(_nw_t(ic, h), 1); r[f'正年{h}'] = f'{(y > 0).sum()}/{len(y)}'
        r[f'前半{h}'] = round(ic[:'2020'].mean(), 3); r[f'後半{h}'] = round(ic['2021':].mean(), 3)
        mm = X.notna() & raw.notna()
        ex = raw.where(mm).sub(raw.where(mm).mean(1), axis=0)
        q = X.where(mm).rank(axis=1, pct=True)
        for i in range(5):
            r[f'Q{i + 1}_{h}%'] = round(ex.where((q > i / 5) & (q <= (i + 1) / 5)).mean(1).mean() * 100, 2)
    rows.append(r); print(name, 'done', flush=True)
R = pd.DataFrame(rows)
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 40)
print(R[['因子', '強勢內中位數', '與T_ldf相關', 'IC20', 't20', '正年20', 'IC60', 't60', '正年60', '前半60', '後半60']].to_string(index=False))
print(R[['因子'] + [f'Q{i}_20%' for i in range(1, 6)] + [f'Q{i}_60%' for i in range(1, 6)]].to_string(index=False))
Y = pd.DataFrame({f'{k[0][:12]}|{k[1]}': v for k, v in yrs.items() if k[1] == 60}).round(3)
print(Y.T.to_string())
R.to_csv('/home/claude/bt/fac/ov_study.csv', index=False, encoding='utf-8-sig')
pd.to_pickle(dict(o20=o20, o21_60=o21_60, o61_120=o61_120), '/home/claude/bt/fac/ov_bands.pkl')
