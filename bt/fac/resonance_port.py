import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('/home/claude/bt/fac/sizing.py').read().split("\nif __name__")[0])
ldn5 = pd.DataFrame(lockdn, index=C.index, columns=codes).rolling(5).sum() > 0
sc = cur.where(~ldn5); strong = cur.rank(axis=1, pct=True) >= 0.9
F = pd.read_pickle('/home/claude/bt/fac/zz9.pkl')
sp, rt = F['上漲波速度'].where(strong), F['平均回撤比例'].where(strong)
qs = sp.rank(axis=1, pct=True); qr = rt.where(sp.notna()).rank(axis=1, pct=True)
fast = qs > 2/3; shallow_mid = qr <= 2/3
half = lambda m: m.astype(float).replace({1.0: 0.5, 0.0: 1.0})
engine_w(sc, '標準版（等權）')
engine_w(sc, '速度快（前 1/3）半個部位', half(fast))
engine_w(sc, '速度快 且 回撤淺／中 半個部位（快＋深 全額）', half(fast & shallow_mid))
engine_w(sc, '速度快 半個部位，其中快＋淺 不買', half(fast).where(~(fast & (qr <= 1/3)), 0.0))
print(pd.DataFrame(out).drop(columns='資料').to_string())
