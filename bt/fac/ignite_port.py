import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('/home/claude/bt/fac/order_test.py').read().split("\nif __name__")[0])
EV = pd.read_pickle('/home/claude/bt/fac/ignite_ev.pkl')
ldn5 = pd.DataFrame(lockdn, index=C.index, columns=codes).rolling(5).sum() > 0
sc = cur.where(~ldn5)
tp = sc.rank(axis=1, pct=True)
engine_o(sc, sc, '標準版')
for lab in ('布林：收斂後帶寬張開 + 收上軌', '整理後突破 20 日高', '布林：收斂後首次突破上軌'):
    rec = EV[lab].reindex_like(sc).fillna(False).astype(float).rolling(5, min_periods=1).max() > 0   # 近 5 日出現過
    engine_o(sc, tp + rec.astype(float), f'優先買：近 5 日有「{lab}」')
    engine_o(sc, tp.where(rec), f'只在近 5 日有「{lab}」才買', need=True)
print(pd.DataFrame(out).drop(columns='資料').to_string())
