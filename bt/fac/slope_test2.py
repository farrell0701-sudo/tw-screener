import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('/home/claude/bt/fac/lock_ext.py').read().split("\nif __name__")[0])
out_ = out
N = 120
HH = Hd.rolling(N, min_periods=100).max(); LL = Ld.rolling(N, min_periods=100).min()
iH = Hd.rolling(N, min_periods=100).apply(np.argmax, raw=True); iL = Ld.rolling(N, min_periods=100).apply(np.argmin, raw=True)
nd = iH - iL
sp = (np.log(HH / LL) / nd).where(nd > 0); amp = np.log(HH / LL)
atrp = pd.DataFrame(Av, index=C.index, columns=codes) / C
strong = cur.rank(axis=1, pct=True) >= 0.9
ldn5 = pd.DataFrame(lockdn, index=C.index, columns=codes).rolling(5).sum() > 0
base_sc = cur.where(~ldn5)
for X, lab in ((sp, '速度%/天'), (amp, '120日振幅%'), (atrp, 'ATR%（純波動，對照）')):
    q = X.where(strong).rank(axis=1, pct=True)
    for x in ((0.7, 0.75, 0.85) if lab == '速度%/天' else (0.8,)):
        engine(base_sc.where(~(q > x)), f'排除強勢股內{lab}最高 {int(round((1-x)*100))}%')
print(pd.DataFrame(out).drop(columns='資料').to_string())
