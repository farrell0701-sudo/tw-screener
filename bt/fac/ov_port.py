import sys; sys.path.insert(0, '/home/claude/bt/fac')
src = open('/home/claude/bt/fac/micro_port.py').read().split("\nout_ = out")[0]
exec(src)
out_ = out
B = pd.read_pickle('/home/claude/bt/fac/ov_bands.pkl')
o120 = (B['o20']*20 + B['o21_60']*40 + B['o61_120']*60)/120
strong = cur.rank(axis=1, pct=True) >= 0.9
qs = o120.where(strong).rank(axis=1, pct=True)
eng(cur, '標準版', 'atr或medL')
for th in (0.2, 0.4):
    eng(cur.where(~(qs < th)), f'排除強勢股內 ov120 最低 {int(th*100)}%', 'atr或medL')
eng(cur.where(~(o120 <= 0.02)), '排除 ov120≤0.02（幾乎完全脫離舊區）', 'atr或medL')
print(pd.DataFrame(out).to_string())
