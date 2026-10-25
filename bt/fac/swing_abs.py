import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('/home/claude/bt/fac/lock_ext.py').read().split("\nif __name__")[0])
from evaluate import prepare
E = prepare(P); out_ = out
F = pd.read_pickle('/home/claude/bt/fac/swing_shape_1.5_lo.pkl')
strong = cur.rank(axis=1, pct=True) >= 0.9
us, ds, dp = F['上漲速度'], F['回檔速度'], F['回檔深度÷前段漲幅']
TH = np.log(1.02)
fast_up = us >= TH; fast_dn = ds >= TH
ok = strong & us.notna()
TY = pd.DataFrame('', index=C.index, columns=codes)
TY = TY.mask(ok & ~fast_up & ds.notna() & fast_dn, '緩漲急跌').mask(ok & fast_up & ds.notna() & fast_dn, '急漲急跌') \
       .mask(ok & fast_up & (ds.isna() | ~fast_dn), '急漲整理').mask(ok & ~fast_up & (ds.isna() | ~fast_dn), '緩漲緩回')
for c, d, lab in [('6223', '2025-09-26', '旺矽'), ('6215', '2025-02-18', '和椿'), ('4743', '2020-04-29', '合一'), ('3231', '2023-02-21', '緯創'),
                  ('6204', '2020-08-06', '艾華'), ('6538', '2026-09-15', '倉和'), ('2485', '2026-03-31', '兆赫'), ('8171', '2017-12-20', '天宇')]:
    print(lab, TY.at[pd.Timestamp(d), c], end='；')
print()
d_ = E['dates']; ent = O.shift(-1); st = strong.loc[d_]
for h in (20, 60):
    R = C.ffill().shift(-h).div(ent).sub(1).loc[d_].where(st); ex = R.sub(R.mean(1), axis=0); rr = []
    for ty in ('緩漲緩回', '緩漲急跌', '急漲整理', '急漲急跌'):
        m = (TY.loc[d_] == ty) & R.notna(); r = R.where(m).stack().dropna(); e = ex.where(m).mean(1)
        y = e.groupby(e.index.year).mean()
        rr.append(dict(類型=ty, 占比=f'{m.sum().sum() / (st & R.notna()).sum().sum():.0%}', 平均超額=f'{e.mean() * 100:+.2f}%',
                       崩跌超20=f'{(r < -.2).mean():.1%}', 噴漲超20=f'{(r > .2).mean():.1%}', 正年=f'{(y > 0).sum()}/{y.notna().sum()}'))
    print(f'== {h} 日'); print(pd.DataFrame(rr).to_string(index=False), flush=True)
ldn5 = pd.DataFrame(lockdn, index=C.index, columns=codes).rolling(5).sum() > 0
base_sc = cur.where(~ldn5)
engine(base_sc, '基準（真實執行＋近5日一字跌停不買）')
engine(base_sc.where(TY != '急漲急跌'), '排除 急漲急跌')
engine(base_sc.where(~TY.isin(['急漲急跌', '急漲整理'])), '排除 所有急漲（對照）')
engine(base_sc.where(TY != '急漲整理'), '排除 急漲整理（對照）')
print(pd.DataFrame(out).drop(columns='資料').to_string())
