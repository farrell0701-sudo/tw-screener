import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('/home/claude/bt/fac/micro_port.py').read().split("\nout_ = out")[0])
out_ = out
lc = np.log(C); mdd = (lc - lc.rolling(120, min_periods=100).max()).rolling(120, min_periods=100).min()
strong = cur.rank(axis=1, pct=True) >= 0.9
qs = mdd.where(strong).rank(axis=1, pct=True)
eng(cur, '標準版', 'atr或medL')
for th in (0.2, 0.4):
    eng(cur.where(~(qs <= th)), f'排除強勢股內回撤最深 {int(th*100)}%', 'atr或medL')
eng(cur.where(mdd > -np.log(1 + 0.30)), '排除 120 日最大回撤 >~26%', 'atr或medL')
print(pd.DataFrame(out).to_string())
