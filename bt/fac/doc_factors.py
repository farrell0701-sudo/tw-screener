"""外部建議因子：強勢股內（T_ldf 前 10%＋族群過濾）的條件 IC，另看全市場扣 T_ldf 後的增量"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
import numpy as np, pandas as pd
from panel import load_panel; from evaluate import prepare, _nw_t, neutralize; from factors3 import tld_floor; from factors import base
from factors2 import rolling_r2_slope
P = load_panel(); E = prepare(P); U = E['univ']
C, H, L, V = P['C'], P['H'], P['L'], P['V']; idx, cols = C.index, C.columns
ldf = tld_floor(P).where(U)
indv = P['industry'].reindex(cols); known = indv.notna().values
g = ldf.loc[:, known].T.groupby(indv[known].values).transform('median').T.reindex(columns=cols)
gp = g.where(U).rank(axis=1, pct=True)
cur = ldf.where((gp >= 0.5) | gp.isna())
strong = cur.rank(axis=1, pct=True) >= 0.9
lr = np.log(C / C.shift(1)); sig60 = E['sig60']
# ADX
up = H.diff(); dn = -L.diff()
pdm = up.where((up > dn) & (up > 0), 0.0); ndm = dn.where((dn > up) & (dn > 0), 0.0)
tr = pd.concat([H - L, (H - C.shift(1)).abs(), (L - C.shift(1)).abs()]).groupby(level=0).max().reindex(idx)
w = lambda x: x.ewm(alpha=1 / 14, adjust=False).mean()
atr = w(tr); pdi = 100 * w(pdm) / atr; ndi = 100 * w(ndm) / atr
dx = 100 * (pdi - ndi).abs() / (pdi + ndi); adx = w(dx)
r = {n: np.log(C / C.shift(n)) for n in (20, 60, 120, 250)}
npos = sum((r[n] > 0).astype(float) for n in r).where(r[250].notna())
lc = np.log(C); mdd120 = (lc - lc.rolling(120, min_periods=100).max()).rolling(120, min_periods=100).min()
er60 = (lc - lc.shift(60)) / lr.abs().rolling(60).sum()
tx = np.log(P['taiex']); rs60 = r[60].sub(tx - tx.shift(60), axis=0)
upv = V.where(lr > 0).rolling(60, min_periods=30).sum(); dnv = V.where(lr < 0).rolling(60, min_periods=30).sum()
F = {
 'ER 趨勢效率 60 日': er60,
 'R² 直線度 120 日': rolling_r2_slope(lc, 120),
 'ADX14': adx,
 '+DI − −DI': pdi - ndi,
 'ADX>25 且 +DI>−DI (0/1)': ((adx > 25) & (pdi > ndi)).astype(float).where(adx.notna()),
 '多週期為正個數 (0–4)': npos,
 '120 日報酬 ÷ 波動': r[120] / (sig60 * np.sqrt(120)),
 '120 日報酬 ÷ 最大回撤': r[120] / (-mdd120).clip(lower=0.02),
 '最大回撤 120 日（淺＝好）': mdd120,
 '相對大盤 60 日': rs60,
 '上漲日量 ÷ 下跌日量 60 日': np.log(upv / dnv),
}
d = E['dates']; S = strong.loc[d]
rows = []
CT = [base(P)['atr'] / C, P['AMT'].rolling(20).mean(), tld_floor(P)]
for name, X in F.items():
    Xs = X.loc[d].where(S); o = dict(因子=name)
    o['與T_ldf相關'] = round(Xs.rank(axis=1).corrwith(cur.loc[d].where(S).rank(axis=1), axis=1).mean(), 2)
    for h in (20, 60):
        va = E['fwd'][h]['va'].loc[d].where(S); raw = E['fwd'][h]['raw'].loc[d].where(S)
        m = Xs.notna() & va.notna()
        ic = Xs.where(m).rank(axis=1).corrwith(va.where(m).rank(axis=1), axis=1)
        ic = ic[m.sum(1) >= 15].dropna(); y = ic.groupby(ic.index.year).mean()
        o[f'強內IC{h}'] = round(ic.mean(), 3); o[f't{h}'] = round(_nw_t(ic, h), 1); o[f'正年{h}'] = f'{(y > 0).sum()}/{len(y)}'
        if h == 60:
            mm = Xs.notna() & raw.notna(); ex = raw.where(mm).sub(raw.where(mm).mean(1), axis=0)
            q = Xs.where(mm).rank(axis=1, pct=True)
            o['Q1%'] = round(ex.where(q <= .2).mean(1).mean() * 100, 2); o['Q5%'] = round(ex.where(q > .8).mean(1).mean() * 100, 2)
    # 全市場、扣波動／成交額／T_ldf 後的增量 IC60
    Xa = X.loc[d].where(U.loc[d]); va = E['fwd'][60]['va'].loc[d]
    Fn = neutralize(Xa, [c.loc[d] for c in CT]); icn = Fn.rank(axis=1).corrwith(va.where(Fn.notna()).rank(axis=1), axis=1).dropna()
    o['全市場增量IC60'] = round(icn.mean(), 3); o['增量t'] = round(_nw_t(icn, 60), 1)
    rows.append(o); print(o, flush=True)
R = pd.DataFrame(rows); pd.set_option('display.width', 300)
print(R.to_string(index=False)); R.to_csv('/home/claude/bt/fac/doc_factors.csv', index=False, encoding='utf-8-sig')
