import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('/home/claude/bt/fac/lock_ext.py').read().split("\nif __name__")[0])
out_ = out
strong = cur.rank(axis=1, pct=True) >= 0.9
engine(cur, '真實執行（基準）')
ldn5 = pd.DataFrame(lockdn, index=C.index, columns=codes).rolling(5).sum() > 0
engine(cur.where(~ldn5), '近 5 日有一字跌停不買')
for x in (1.0, 1.5, 2.0):
    engine(cur.where(~(run120 > np.log(1 + x))), f'排除 120 日低點起漲幅 >{int(x*100)}%')
for x in (0.8, 0.9):
    q = run250.where(strong).rank(axis=1, pct=True)
    engine(cur.where(~(q > x)), f'排除強勢股內 250 日漲幅前 {int(round((1-x)*100))}%')
print(pd.DataFrame(out).drop(columns='資料').to_string())
