import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('dense_ctrl.py').read().split("rows = []")[0])
rows = []
exec(open('dense_ctrl.py').read().split("rows = []")[1].split("for K in (30,):")[0])
F4 = pd.read_pickle('/home/claude/bt/fac/f4.pkl'); vp = F4['B_vcp'][0].where(U).rank(axis=1, pct=True)
base_cur = cur
for lab, flt in (('', None), ('＋排除波動擴張20%', vp <= 0.8)):
    c2 = base_cur if flt is None else base_cur.where(flt)
    globals()['RK'] = c2.where(pd.DataFrame(BUY, index=dates, columns=C.columns)).rank(axis=1, ascending=False).values
    globals()['RKall'] = c2.rank(axis=1, ascending=False).values
    for K in (20, 30, 50):
        go(allT, f'每日補位{lab}', K)
    go(dense_signals(P, atr_, U, 20, 0.3, 'dense'), f'密集上緣＋整理{lab}', 30)
pd.DataFrame(rows).to_csv('/home/claude/bt/fac/dense_vcp.csv', index=False, encoding='utf-8-sig')
