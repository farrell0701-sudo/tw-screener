import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('/home/claude/bt/fac/port_dd_bear.py').read().split("rows = []")[0])
for name in ('T_ldf（現行）', '抗跌最高'):
    globals()['Sv'] = V[name].values
    out = []
    for off in range(0, 20, 2):
        s = run(off, False, 4, 3)[0]; dd = s / s.cummax() - 1; b = dd.idxmin(); a = s[:b].idxmax()
        out.append(f'{a.date()}→{b.date()} {dd.min():.1%}')
    print(name); print('  ' + '\n  '.join(out))
