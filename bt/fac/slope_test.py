"""使用者提議：近 120 日 (最高−最低)/天數 → 上漲速度。修成無尺度版本後測：強勢股內 IC、交易分組、組合（真實執行＋近5日一字跌停不買為基準）"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('/home/claude/bt/fac/lock_ext.py').read().split("\nif __name__")[0])
from evaluate import prepare, _nw_t
E = prepare(P); out_ = out
N = 120
Hh = Hd.rolling(N, min_periods=100); Ll = Ld.rolling(N, min_periods=100)
HH, LL = Hh.max(), Ll.min()
iH = Hd.rolling(N, min_periods=100).apply(np.argmax, raw=True)     # 窗內位置 0..N-1
iL = Ld.rolling(N, min_periods=100).apply(np.argmin, raw=True)
ndays = (iH - iL)                                                  # 低點到高點經過幾天（>0 才是上漲）
up = ndays > 0
F = {
 '原式 (高−低)/120（價格單位）': (HH - LL) / N,
 '振幅 log(高/低)/120': np.log(HH / LL) / N,
 '速度 log(高/低)/低到高天數': (np.log(HH / LL) / ndays).where(up),
 '速度（ATR 單位/天）': ((HH - LL) / Av_df / ndays).where(up) if False else None,
}
atr = pd.DataFrame(Av, index=C.index, columns=codes)
F['速度（ATR 單位/天）'] = ((HH - LL) / atr / ndays).where(up)
F['低到高天數'] = ndays.where(up)
strong = cur.rank(axis=1, pct=True) >= 0.9
d = E['dates']; S = strong.loc[d]; rows = []
for name, X in F.items():
    Xs = X.loc[d].where(S); o = dict(因子=name)
    o['與T_ldf相關'] = round(Xs.rank(axis=1).corrwith(cur.loc[d].where(S).rank(axis=1), axis=1).mean(), 2)
    for h in (20, 60):
        va = E['fwd'][h]['va'].loc[d].where(S); raw = E['fwd'][h]['raw'].loc[d].where(S)
        m = Xs.notna() & va.notna()
        ic = Xs.where(m).rank(axis=1).corrwith(va.where(m).rank(axis=1), axis=1); ic = ic[m.sum(1) >= 15].dropna()
        y = ic.groupby(ic.index.year).mean()
        o[f'IC{h}'] = round(ic.mean(), 3); o[f't{h}'] = round(_nw_t(ic, h), 1); o[f'正年{h}'] = f'{(y > 0).sum()}/{len(y)}'
        if h == 60:
            mm = Xs.notna() & raw.notna(); ex = raw.where(mm).sub(raw.where(mm).mean(1), axis=0); q = Xs.where(mm).rank(axis=1, pct=True)
            for i in range(5): o[f'Q{i+1}'] = round(ex.where((q > i / 5) & (q <= (i + 1) / 5)).mean(1).mean() * 100, 2)
    rows.append(o)
pd.set_option('display.width', 250)
print(pd.DataFrame(rows).to_string(index=False), flush=True)
ldn5 = pd.DataFrame(lockdn, index=C.index, columns=codes).rolling(5).sum() > 0
base_sc = cur.where(~ldn5)
T = engine(base_sc, '基準：真實執行＋近5日一字跌停不買', log=True)
sd = T.進場.values - 1; jj = T.j.values
for col, X in (('速度%/天', F['速度 log(高/低)/低到高天數']), ('速度ATR/天', F['速度（ATR 單位/天）'])):
    T[col] = X.values[sd, jj]
    q = pd.qcut(T[col], 5, duplicates='drop')
    print(f'\n== 交易依進場前 {col} 分五組')
    print(T.groupby(q, observed=True).報酬.agg(筆數='size', 平均=lambda x: f'{x.mean():.1%}', 中位=lambda x: f'{x.median():.1%}',
          勝率=lambda x: f'{(x > 0).mean():.0%}', 賺1倍=lambda x: int((x > 1).sum()), 虧20=lambda x: int((x < -0.2).sum())).to_string(), flush=True)
sp = F['速度 log(高/低)/低到高天數']; spa = F['速度（ATR 單位/天）']
for X, lab in ((sp, '%/天'), (spa, 'ATR/天')):
    q = X.where(strong).rank(axis=1, pct=True)
    for x in (0.8, 0.9):
        engine(base_sc.where(~(q > x)), f'排除強勢股內速度({lab})最快 {int(round((1-x)*100))}%')
print(pd.DataFrame(out).drop(columns='資料').to_string())
