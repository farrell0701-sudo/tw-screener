"""在三種收斂突破事件上加：①盤整多久 ②收斂期量縮 ③突破後回測量縮不破再轉強（新事件）"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('/home/claude/bt/fac/ignite.py').read().split("lu_next = ")[0])
anyev = (EV['布林：收斂後首次突破上軌'] | EV['布林：收斂後帶寬張開 + 收上軌'] | EV['整理後突破 20 日高']).fillna(False)
lu_next = O.shift(-1) / C.ffill() - 1 >= 0.095; ent = O.shift(-1)
d0 = pd.Timestamp('2016-02-01')
base_m = (strong & ~lu_next & ent.notna()).loc[d0:]
Cf = C.ffill()
def bars_since(b):
    b = b.astype(float).fillna(0); cs = b.cumsum()
    idx = pd.DataFrame(np.arange(len(b))[:, None].repeat(b.shape[1], 1), index=b.index, columns=b.columns).astype(float)
    last = idx.where(b > 0).ffill()
    return idx - last
# ① 盤整多久
newhi60 = C >= C.rolling(60, min_periods=60).max()
base_len = bars_since(newhi60).shift(1)                                       # 到昨天為止，距上次創 60 日收盤新高幾天
sq = bw <= bw.rolling(120, min_periods=100).quantile(0.3)
sq_len = sq.astype(float).groupby((~sq).cumsum(), axis=0).cumsum() if False else None
s = sq.astype(int).values; run = np.zeros_like(s, dtype=float)
for t in range(1, len(s)): run[t] = (run[t - 1] + 1) * s[t]
sq_len = pd.DataFrame(run, index=C.index, columns=cols).shift(1).rolling(10, min_periods=1).max()   # 近 10 日內最長的連續低帶寬天數
# ② 收斂期量縮
vdry = V.rolling(10).mean().shift(1) / V.rolling(60).mean().shift(11)
# ③ 回測量縮不破
seg = anyev.astype(int).cumsum()
Cb = C.where(anyev).ffill(); Lb = L.where(anyev).ffill(); Vb = V.where(anyev).ffill()
since = bars_since(anyev)
minC = pd.DataFrame(index=C.index, columns=cols, dtype=float); avgV = minC.copy()
for c in cols:
    g = seg[c]
    cc = C[c].where(~anyev[c])                     # 突破日之後的收盤（不含突破日）
    minC[c] = cc.groupby(g).cummin()
    avgV[c] = V[c].where(~anyev[c]).groupby(g).transform(lambda x: x.expanding().mean())
minC_y, avgV_y = minC.shift(1), avgV.shift(1)       # 到昨天為止的回檔
resume = C > H.shift(1)
win = (since >= 3) & (since <= 15)
retest = win & (minC_y <= Cb) & (minC_y >= Lb) & resume
retest_dry = retest & (avgV_y <= 0.7 * Vb)
retest_wet = retest & (avgV_y > 0.7 * Vb)
def study(mask, lab):
    m0 = base_m & mask.loc[d0:].fillna(False)
    o = {'條件': lab, '占強勢股日': f'{m0.sum().sum() / base_m.sum().sum():.2%}'}
    for h in (5, 20, 60):
        R = (Cf.shift(-h) / ent - 1).loc[d0:]
        m = m0 & R.notna()
        ex = R.sub(R.where(base_m).mean(1), axis=0).where(m).stack().dropna(); r = R.where(m).stack().dropna()
        yr = ex.groupby(ex.index.get_level_values(0).year).mean()
        o[f'{h}日超額'] = f'{ex.mean():+.2%}'
        if h == 20: o['20日正年'] = f'{(yr > 0).sum()}/{len(yr)}'; o['20日跌超20'] = f'{(r < -.2).mean():.1%}'
        if h == 60: o['60日漲超30'] = f'{(r > .3).mean():.1%}'
    o['筆數'] = int(m0.sum().sum())
    return o
rows = [study(anyev, '三種事件任一（基準）')]
for a, b, lab in ((0, 10, '0～10 天'), (10, 20, '10～20 天'), (20, 40, '20～40 天'), (40, 80, '40～80 天'), (80, 999, '80 天以上')):
    rows.append(study(anyev & (base_len >= a) & (base_len < b), f'①底部長度（距上次 60 日新高）{lab}'))
for a, b, lab in ((0, 3, '<3 天'), (3, 8, '3～8 天'), (8, 15, '8～15 天'), (15, 999, '15 天以上')):
    rows.append(study(anyev & (sq_len >= a) & (sq_len < b), f'①連續低帶寬 {lab}'))
qv = vdry.where(anyev & strong).rank(axis=1, pct=True)
vq = vdry.where(anyev & strong)
for a, b, lab in ((0, 0.7, '<0.7（明顯量縮）'), (0.7, 0.9, '0.7～0.9'), (0.9, 1.1, '0.9～1.1'), (1.1, 1.5, '1.1～1.5'), (1.5, 99, '>1.5（收斂期量增）')):
    rows.append(study(anyev & (vdry >= a) & (vdry < b), f'②收斂期均量比 {lab}'))
rows.append(study(retest, '③回測不破再轉強（不看量）'))
rows.append(study(retest_dry, '③回測量縮（<70% 突破日量）不破再轉強'))
rows.append(study(retest_wet, '③回測未量縮 不破再轉強（對照）'))
D = pd.DataFrame(rows); pd.set_option('display.width', 260); pd.set_option('display.max_columns', 20)
print(D.to_string(index=False))
pd.to_pickle(dict(base_len=base_len, sq_len=sq_len, vdry=vdry, retest=retest, retest_dry=retest_dry), '/home/claude/bt/fac/ignite_mod.pkl')
