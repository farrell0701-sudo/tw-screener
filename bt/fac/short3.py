"""使用者觀察：站上 60 日密集區上緣（綠箭頭）＋ Overlap 類 MACD 柱為負且負值擴大 → 訊號 K 收盤進、持有 3 天收盤出
設定：Overlap near=20、far=60、平滑 5、訊號線 EMA9；密集區 W=60
乾淨資料、股票池（上市滿一年、成交額前 80%）；排除訊號當天收漲停（買不到）"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
import numpy as np, pandas as pd
from panel import load_panel; from evaluate import prepare
P = load_panel(); E = prepare(P); U = E['univ']
C, H, L, O = P['C'], P['H'], P['L'], P['O']
Hv, Lv = H.values, L.values
near = np.zeros_like(Hv); far = np.zeros_like(Hv)
for k in range(1, 61):
    o = np.zeros_like(Hv, dtype=bool)
    o[k:] = (np.minimum(Hv[k:], Hv[:-k]) - np.maximum(Lv[k:], Lv[:-k])) > 0
    if k <= 20: near += o
    else: far += o
near /= 20; far /= 40; near[:60] = np.nan; far[:60] = np.nan
idx, cols = C.index, C.columns
n5 = pd.DataFrame(near, idx, cols).rolling(5).mean(); f5 = pd.DataFrame(far, idx, cols).rolling(5).mean()
d = n5 - f5; h = d - d.ewm(span=9, adjust=False).mean()
red_exp = (h < 0) & (h < h.shift(1))
medH = H.rolling(60, min_periods=60).median().shift(1)
cross = (C > medH) & (C.shift(1) <= medH.shift(1))
lu = C / C.shift(1) - 1 >= 0.095                       # 收漲停買不到
base = U & ~lu & C.notna()
COST = 0.0047
mkt = {}
def fwd(n, entry='close'):
    en = C if entry == 'close' else O.shift(-1)
    return C.shift(-n) / en - 1
rows = []
d0 = pd.Timestamp('2016-02-01')
for n in (1, 3, 5, 10):
    R = fwd(n)
    Rm = R.where(base).mean(1)                           # 同期股票池平均（同樣持有 n 天）
    for lab, cond in (('全部股票（基準）', base), ('只有站上密集區', base & cross), ('只有紅柱擴大', base & red_exp),
                      ('站上密集區＋紅柱擴大', base & cross & red_exp), ('站上密集區＋綠柱（對照）', base & cross & (h > 0))):
        m = cond.loc[d0:] & R.loc[d0:].notna()
        r = R.loc[d0:].where(m).stack().dropna(); ex = R.loc[d0:].sub(Rm.loc[d0:], axis=0).where(m).stack().dropna()
        rows.append(dict(持有天數=n, 條件=lab, 筆數=len(r), 每年筆數=round(len(r) / 10.7), 勝率_未扣成本=f'{(r > 0).mean():.1%}',
                         勝率_扣成本=f'{(r > COST).mean():.1%}', 平均=f'{r.mean():.2%}', 中位=f'{r.median():.2%}',
                         扣成本平均=f'{(r - COST).mean():.2%}', 相對大盤超額=f'{ex.mean():.2%}'))
R = pd.DataFrame(rows); pd.set_option('display.width', 250); pd.set_option('display.max_columns', 20)
print(R.to_string(index=False))
# 3 天版逐年
R3 = fwd(3); Rm3 = R3.where(base).mean(1)
m = (base & cross & red_exp).loc[d0:] & R3.loc[d0:].notna()
s = R3.loc[d0:].where(m).stack().dropna(); ex = R3.loc[d0:].sub(Rm3.loc[d0:], axis=0).where(m).stack().dropna()
y = pd.DataFrame({'r': s, 'ex': ex}).reset_index(level=0)
print('\n站上密集區＋紅柱擴大，持有 3 天，逐年：')
print(y.groupby(y.iloc[:, 0].dt.year).agg(筆數=('r', 'size'), 勝率=('r', lambda x: f'{(x > 0).mean():.0%}'), 平均=('r', lambda x: f'{x.mean():.2%}'),
                                         扣成本=('r', lambda x: f'{(x - COST).mean():.2%}'), 超額=('ex', lambda x: f'{x.mean():.2%}')).to_string())
# 隔日開盤進場（不用盤中判斷）
R3o = fwd(3, 'open'); s2 = R3o.loc[d0:].where(m & R3o.loc[d0:].notna()).stack().dropna()
print(f'\n改成隔日開盤進、持有到第 3 天收盤：勝率 {(s2 > 0).mean():.1%}、平均 {s2.mean():.2%}、扣成本 {(s2 - COST).mean():.2%}')
# 只看強勢股（T_ldf 前 10%）、以及大盤多頭時
from factors3 import tld_floor
tp = tld_floor(P).where(U).rank(axis=1, pct=True)
tx = P['taiex']; e60 = tx.ewm(span=60, adjust=False).mean(); bull = (e60 / e60.shift(20) - 1) > 0
for lab, extra in (('＋T_ldf 前 10%', tp >= 0.9), ('＋大盤多頭', pd.DataFrame(np.repeat(bull.values[:, None], C.shape[1], 1), index=idx, columns=cols))):
    for nm, cond in (('訊號', base & cross & red_exp & extra), ('同條件全部股票', base & extra)):
        mm = cond.loc[d0:] & R3.loc[d0:].notna()
        s = R3.loc[d0:].where(mm).stack().dropna(); ex = R3.loc[d0:].sub(Rm3.loc[d0:], axis=0).where(mm).stack().dropna()
        print(f'{lab}｜{nm}：{len(s)} 筆，3 日勝率 {(s > 0).mean():.1%}，平均 {s.mean():.2%}，扣成本 {(s - COST).mean():.2%}，超額 {ex.mean():.2%}')
