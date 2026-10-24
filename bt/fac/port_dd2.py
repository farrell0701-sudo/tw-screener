import sys; sys.path.insert(0, '/home/claude/bt/fac')
src = open('/home/claude/bt/fac/port_dd.py').read().split("for K in (20, 30, 40):")[0]
exec(src)
for K in (20, 30):
    pool = rk <= K
    xp = F3['R_downday'][0].where(U & pool).rank(axis=1, pct=True)
    lp = base_sc.where(pool).rank(axis=1, pct=True)
    go(lambda sd: (xp + lp).where(pool), '候選內 T_ldf 與抗跌各半', K)
