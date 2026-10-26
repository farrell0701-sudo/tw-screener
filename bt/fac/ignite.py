"""主升段點火事件（T_ldf 前 10%＋族群過濾的強勢股內）
隔日開盤進場，持有 h 日，和「同一天所有強勢股」比；另看 60 日內漲超 30% 的比例（抓到主升段）"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
import numpy as np, pandas as pd
from panel import load_panel; from evaluate import prepare; from factors import base; from factors3 import tld_floor
P = load_panel(); E = prepare(P); U = E['univ']
C, H, L, O, V = P['C'], P['H'], P['L'], P['O'], P['V']; cols = C.columns
atr = base(P)['atr']
ldf = tld_floor(P).where(U)
indv = P['industry'].reindex(cols); known = indv.notna().values
g = ldf.loc[:, known].T.groupby(indv[known].values).transform('median').T.reindex(columns=cols)
gp = g.where(U).rank(axis=1, pct=True); cur = ldf.where((gp >= 0.5) | gp.isna())
strong = cur.rank(axis=1, pct=True) >= 0.9
ema = lambda x, s: x.ewm(span=s, adjust=False).mean()
# 布林
m20 = C.rolling(20).mean(); sd20 = C.rolling(20).std(); bw = 4 * sd20 / m20
squeeze = bw.rolling(10).min().shift(1) <= bw.rolling(120, min_periods=100).quantile(0.2).shift(1)   # 近 10 日內曾處於自身 20% 低位的帶寬
bb_break = (C > m20 + 2 * sd20) & (C.shift(1) <= (m20 + 2 * sd20).shift(1))
bw_open = bw > bw.shift(3) * 1.15
# 整理：近 20 日區間 ÷ ATR 小、近 20 日沒創 60 日新高
rng20 = (H.rolling(20).max() - L.rolling(20).min()) / atr
tight = (rng20.shift(1) <= rng20.rolling(250, min_periods=150).quantile(0.3).shift(1))
hi20 = C > H.rolling(20).max().shift(1)
based = (C.rolling(20).max().shift(1) < C.rolling(60).max().shift(21).where(lambda x: x > 0, np.nan) * 1.0 + 1e-12) | tight   # 前 20 日未超過更早的高點 或 收斂
# 動能放大
roc10 = C / C.shift(10) - 1
mom_up = (roc10 > 0) & (roc10.shift(5) <= 0)                           # 10 日動能由負翻正
e12, e26 = ema(C, 12), ema(C, 26); macd = e12 - e26; hist = macd - ema(macd, 9)
macd_turn = (hist > 0) & (hist.rolling(10).max().shift(1) <= 0)        # MACD 柱翻紅，之前至少 10 天都是綠
# 短線 EMA 斜率交叉長線 EMA 斜率（ATR 單位），之前 10 日短線斜率都比較低
s10 = (ema(C, 10) - ema(C, 10).shift(5)) / atr; s50 = (ema(C, 50) - ema(C, 50).shift(5)) / atr
slope_x = (s10 > s50) & ((s10 - s50).rolling(10).max().shift(1) <= 0)
vol_x = V > V.rolling(20).mean().shift(1) * 2
EV = {
    '布林：收斂後首次突破上軌': squeeze & bb_break,
    '布林：收斂後帶寬張開 + 收上軌': squeeze & bw_open & (C > m20 + 2 * sd20),
    '整理後突破 20 日高': tight & hi20,
    '整理後突破 20 日高 + 量增 2 倍': tight & hi20 & vol_x,
    '10 日動能由負翻正': mom_up,
    'MACD 柱翻紅（之前綠 ≥10 天）': macd_turn,
    '短 EMA 斜率上穿長 EMA 斜率（之前 ≥10 天在下）': slope_x,
    '斜率交叉 + 突破 20 日高': slope_x & hi20,
    '對照：單純突破 20 日高': hi20,
    '對照：單純量增 2 倍': vol_x,
}
lu_next = O.shift(-1) / C.ffill() - 1 >= 0.095
ent = O.shift(-1)
d0 = pd.Timestamp('2016-02-01'); mid = pd.Timestamp('2021-01-01')
base_m = (strong & ~lu_next & ent.notna()).loc[d0:]
rows = []
for h in (5, 20, 60):
    R = (C.ffill().shift(-h) / ent - 1).loc[d0:]
    Rb = R.where(base_m).mean(1)
    bb = R.where(base_m).stack().dropna()
    if h == 60: rows.append(dict(持有=h, 事件='（全部強勢股）', 筆數=len(bb), 占強勢股日=f'100%', 平均=f'{bb.mean():.2%}', 相對=f'+0.00%', 前半='', 後半='', t值='', 漲超30=f'{(bb > .3).mean():.1%}', 跌超20=f'{(bb < -.2).mean():.1%}'))
    for lab, ev in EV.items():
        m = base_m & ev.loc[d0:].fillna(False) & R.notna()
        r = R.where(m).stack().dropna(); ex = R.sub(Rb, axis=0).where(m).stack().dropna()
        dts = ex.index.get_level_values(0); e1, e2 = ex[dts < mid], ex[dts >= mid]
        yr = ex.groupby(dts.year).mean()
        rows.append(dict(持有=h, 事件=lab, 筆數=len(r), 占強勢股日=f'{m.sum().sum() / base_m.sum().sum():.1%}', 平均=f'{r.mean():.2%}',
                         相對=f'{ex.mean():+.2%}', 前半=f'{e1.mean():+.2%}', 後半=f'{e2.mean():+.2%}', 正年=f'{(yr > 0).sum()}/{len(yr)}',
                         t值=round(ex.mean() / ex.std() * np.sqrt(len(ex) / h), 1), 漲超30=f'{(r > .3).mean():.1%}', 跌超20=f'{(r < -.2).mean():.1%}'))
R_ = pd.DataFrame(rows); pd.set_option('display.width', 260); pd.set_option('display.max_columns', 20)
for h in (5, 20, 60): print(f'== 持有 {h} 日'); print(R_[R_.持有 == h].drop(columns='持有').to_string(index=False))
R_.to_csv('/home/claude/bt/fac/ignite.csv', index=False, encoding='utf-8-sig')
pd.to_pickle({k: v for k, v in EV.items()}, '/home/claude/bt/fac/ignite_ev.pkl')
