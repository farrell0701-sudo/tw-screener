"""微策略事件研究：方向 = 強勢股（T_ldf 前 10%、族群過濾通過），時機 = 60 日密集區 × Overlap
所有訊號用 t 日收盤以前的資料；t+1 開盤進場（不需盤中判斷），持有 h 日到收盤出場
比較基準 = 同一天所有強勢股（同樣 t+1 開盤進），看「時機」本身有沒有加分；2016–2020／2021–2026 分開看"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
import numpy as np, pandas as pd
from panel import load_panel; from evaluate import prepare; from factors import base; from factors3 import tld_floor
P = load_panel(); E = prepare(P); U = E['univ']
C, H, L, O = P['C'], P['H'], P['L'], P['O']
idx, cols = C.index, C.columns
atr = base(P)['atr']
# 強勢股
ldf = tld_floor(P).where(U)
indv = P['industry'].reindex(cols); known = indv.notna().values
g = ldf.loc[:, known].T.groupby(indv[known].values).transform('median').T.reindex(columns=cols)
gp = g.where(U).rank(axis=1, pct=True)
cur = ldf.where((gp >= 0.5) | gp.isna())
strong = cur.rank(axis=1, pct=True) >= 0.9
# 60 日密集區（到昨天為止）
medH = H.rolling(60, min_periods=60).median().shift(1); medL = L.rolling(60, min_periods=60).median().shift(1)
# Overlap 20/60
Hv, Lv = H.values, L.values
near = np.zeros_like(Hv); far = np.zeros_like(Hv)
for k in range(1, 61):
    o = np.zeros_like(Hv, dtype=bool); o[k:] = (np.minimum(Hv[k:], Hv[:-k]) - np.maximum(Lv[k:], Lv[:-k])) > 0
    if k <= 20: near += o
    else: far += o
near /= 20; far /= 40; near[:60] = np.nan; far[:60] = np.nan
n5 = pd.DataFrame(near, idx, cols).rolling(5).mean(); f5 = pd.DataFrame(far, idx, cols).rolling(5).mean()
ofd = n5 - f5; hist = ofd - ofd.ewm(span=9, adjust=False).mean()
pos_in_zone = (C - medH) / atr                    # 收盤在密集區上緣上方幾個 ATR
r3 = C / C.shift(3) - 1
below5 = (C < medH).astype(float).rolling(5).max().shift(1) > 0
SIG = {
    '突破：收盤首次越過密集區上緣': (C > medH) & (C.shift(1) <= medH.shift(1)),
    '回測後站回：近 5 日曾跌破上緣、今天站回': (C > medH) & below5 & (C.shift(1) <= medH.shift(1)),
    '回到密集區內（上下緣之間）': (C <= medH) & (C >= medL),
    '貼近支撐：在上緣上方 0～1 ATR': (pos_in_zone >= 0) & (pos_in_zone <= 1),
    '遠離支撐：在上緣上方 >4 ATR': pos_in_zone > 4,
    '強勢中 3 日回檔、仍在上緣之上': (r3 < 0) & (C > medH),
    '強勢中 3 日回檔、回到上緣附近 0～1.5 ATR': (r3 < 0) & (pos_in_zone >= 0) & (pos_in_zone <= 1.5),
    'OF 差值 > 0（近期收斂、已脫離舊區）': ofd > 0,
    'OF 類 MACD 紅柱擴大': (hist < 0) & (hist < hist.shift(1)),
    'OF 類 MACD 綠柱翻正': (hist > 0) & (hist.shift(1) <= 0),
    '跌破密集區下緣（對照：失敗）': C < medL,
}
lu_next = O.shift(-1) / C - 1 >= 0.095             # 隔日開盤漲停買不到
ent = O.shift(-1)
base_m = strong & ~lu_next & ent.notna()
d0 = pd.Timestamp('2016-02-01'); mid = pd.Timestamp('2021-01-01')
rows = []
for h in (5, 10, 20):
    R = C.shift(-h) / ent - 1
    Rb = R.where(base_m).mean(1)                     # 同日強勢股平均
    for lab, sig in SIG.items():
        m = (base_m & sig.fillna(False)).loc[d0:] & R.loc[d0:].notna()
        r = R.loc[d0:].where(m).stack().dropna()
        ex = R.loc[d0:].sub(Rb.loc[d0:], axis=0).where(m).stack().dropna()
        dts = ex.index.get_level_values(0)
        e1, e2 = ex[dts < mid], ex[dts >= mid]
        frac = m.sum().sum() / base_m.loc[d0:].sum().sum()
        rows.append(dict(持有=h, 訊號=lab, 筆數=len(r), 占強勢股日比例=f'{frac:.1%}', 勝率=f'{(r > 0).mean():.1%}', 平均=f'{r.mean():.2%}',
                         相對強勢股=f'{ex.mean():+.2%}', 前半=f'{e1.mean():+.2%}', 後半=f'{e2.mean():+.2%}',
                         t值=round(ex.mean() / ex.std() * np.sqrt(len(ex) / max(1, h)), 1)))
    print(f'持有 {h} 日完成', flush=True)
R = pd.DataFrame(rows); pd.set_option('display.width', 250); pd.set_option('display.max_columns', 20)
print(R.to_string(index=False))
R.to_csv('/home/claude/bt/fac/micro_events.csv', index=False, encoding='utf-8-sig')
