"""用「不創高＋重疊率」切波段（不用 ATR 門檻）
每天三種狀態：
  上攻：收盤創 10 日新高
  整理：沒創新高，且近期重疊率（今天和前 1～20 根 K 重疊的比例，5 日均）比 3 天前高 → 在區間內來回
  急跌：沒創新高，且重疊率下降 → 價格往下離開區間
特徵（近 120 日最低點以後）：上攻日比例、回檔日中急跌的比例、上攻速度（低點到區間最高 log 漲幅 ÷ 天數）"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('/home/claude/bt/fac/lock_ext.py').read().split("\ndef engine")[0])
from evaluate import prepare
E = prepare(P)
Hv, Lv = Hd.values, Ld.values
near = np.zeros_like(Hv)
for k in range(1, 21):
    o = np.zeros_like(Hv); o[k:] = (np.minimum(Hv[k:], Hv[:-k]) - np.maximum(Lv[k:], Lv[:-k])) > 0
    near += o
near = pd.DataFrame(near / 20, index=C.index, columns=codes).rolling(5).mean()
newhi = C >= C.rolling(10, min_periods=10).max()
rise = near > near.shift(3)
S_up = newhi; S_con = ~newhi & rise; S_dn = ~newhi & ~rise
# 近 120 日最低點以後
lc = np.log(C); LLc = lc.rolling(120, min_periods=100).min()
iL = lc.rolling(120, min_periods=100).apply(np.argmin, raw=True)          # 0..119
dsl = 119 - iL                                                             # 低點距今天數
def since_low_mean(B):
    """低點之後（含今天）B 為真的比例：用累積和相減"""
    b = B.astype(float).fillna(0).values; cs = np.vstack([np.zeros((1, b.shape[1])), np.cumsum(b, 0)])
    n = dsl.values; out = np.full(b.shape, np.nan)
    T = np.arange(b.shape[0])[:, None]
    ok = ~np.isnan(n)
    s = np.where(ok, T - np.nan_to_num(n).astype(int), 0)
    s = np.clip(s, 0, None)
    tot = cs[T + 1, np.arange(b.shape[1])[None, :]] - cs[s + 1, np.arange(b.shape[1])[None, :]]   # 低點隔天～今天
    days = np.where(ok, np.nan_to_num(n), np.nan)
    with np.errstate(invalid='ignore', divide='ignore'): out = tot / days
    return pd.DataFrame(np.where(days >= 10, out, np.nan), index=C.index, columns=codes)
f_up = since_low_mean(S_up); f_con = since_low_mean(S_con); f_dn = since_low_mean(S_dn)
pull_dn = f_dn / (f_con + f_dn)                                            # 回檔日裡急跌的比例
HHc = lc.rolling(120, min_periods=100).max(); iH = lc.rolling(120, min_periods=100).apply(np.argmax, raw=True)
ndh = iH - iL; speed = ((HHc - LLc) / ndh).where(ndh > 0)
FS = {'上攻日比例': f_up, '回檔中急跌比例': pull_dn, '上漲速度': speed}
pd.to_pickle(FS, '/home/claude/bt/fac/ovstate.pkl')
strong = cur.rank(axis=1, pct=True) >= 0.9
for c, d, lab in [('6223', '2025-09-26', '旺矽 +297%'), ('6215', '2025-02-18', '和椿 −30%'), ('4743', '2020-04-29', '合一 +396%'), ('3231', '2023-02-21', '緯創 +291%'),
                  ('6204', '2020-08-06', '艾華 +232%'), ('6538', '2026-09-15', '倉和 −27%'), ('2485', '2026-03-31', '兆赫 −27%'), ('8171', '2017-12-20', '天宇 −36%')]:
    d = pd.Timestamp(d)
    print(lab, {k: f'{v.at[d, c]:.3f} ({v.loc[d].where(strong.loc[d]).rank(pct=True).get(c):.0%})' for k, v in FS.items()})
d_ = E['dates']; ent = O.shift(-1); st = strong.loc[d_]
R = C.ffill().shift(-60).div(ent).sub(1).loc[d_].where(st); ex = R.sub(R.mean(1), axis=0)
rows = []
for k, X in FS.items():
    q = X.loc[d_].where(st & R.notna()).rank(axis=1, pct=True); o = dict(特徵=k)
    for i in range(5):
        m = (q > i / 5) & (q <= (i + 1) / 5); r = R.where(m).stack().dropna()
        o[f'Q{i+1}'] = f'{ex.where(m).mean(1).mean() * 100:+.2f} ({(r < -.2).mean():.0%}/{(r > .2).mean():.0%})'
    dd = (ex.where(q > .8).mean(1) - ex.where(q <= .2).mean(1)).dropna()
    o['Q5>Q1年數'] = f'{(dd.groupby(dd.index.year).mean() > 0).sum()}/{dd.index.year.nunique()}'; rows.append(o)
pd.set_option('display.width', 250)
print('\n60 日，強勢股內五分組：平均超額% (崩跌超20%/噴漲超20%)'); print(pd.DataFrame(rows).to_string(index=False))
# 四類：上漲速度 ≥2%/天 = 急漲；回檔中急跌比例 > 中位數 = 急跌，否則整理
med = pull_dn.where(strong).median(1)
fu = speed >= np.log(1.02); fd = pull_dn.gt(med, axis=0)
ok = strong & speed.notna() & pull_dn.notna()
TY = pd.DataFrame('', index=C.index, columns=codes)
TY = TY.mask(ok & ~fu & fd, '緩漲急跌').mask(ok & fu & fd, '急漲急跌').mask(ok & fu & ~fd, '急漲整理').mask(ok & ~fu & ~fd, '緩漲整理')
pd.to_pickle(TY, '/home/claude/bt/fac/ovstate_type.pkl')
for c, d, lab in [('6223', '2025-09-26', '旺矽'), ('6215', '2025-02-18', '和椿'), ('4743', '2020-04-29', '合一')]:
    print(lab, TY.at[pd.Timestamp(d), c], end='；')
rr = []
for ty in ('緩漲整理', '緩漲急跌', '急漲整理', '急漲急跌'):
    m = (TY.loc[d_] == ty) & R.notna(); r = R.where(m).stack().dropna(); e = ex.where(m).mean(1); y = e.groupby(e.index.year).mean()
    rr.append(dict(類型=ty, 占比=f'{m.sum().sum() / (st & R.notna()).sum().sum():.0%}', 平均超額=f'{e.mean() * 100:+.2f}%',
                   崩跌超20=f'{(r < -.2).mean():.1%}', 噴漲超20=f'{(r > .2).mean():.1%}', 正年=f'{(y > 0).sum()}/{y.notna().sum()}'))
print('\n60 日四類'); print(pd.DataFrame(rr).to_string(index=False))
