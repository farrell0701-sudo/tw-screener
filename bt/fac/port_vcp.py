import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('/home/claude/bt/fac/port_struct.py').read().split("\nif __name__ == '__main__'")[0])
F4 = pd.read_pickle('/home/claude/bt/fac/f4.pkl')
v = F4['B_vcp'][0].where(U)
vp = v.rank(axis=1, pct=True)          # 越高 = 波動擴張越大
rows = []
for n in (10, 20):
    rows.append(test(cur, '現行', n))
    for q in (0.95, 0.9, 0.8, 0.7, 0.6):
        rows.append(test(cur.where(vp <= q), f'排除波動擴張最大的 {round((1-q)*100)}%', n))
    for x in (1.5, 2.0):
        rows.append(test(cur.where(v <= x), f'排除 ATR20/ATR120 > {x}', n))
pd.DataFrame(rows).to_csv('/home/claude/bt/fac/port_vcp.csv', index=False, encoding='utf-8-sig')
