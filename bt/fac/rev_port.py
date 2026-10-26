import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('/home/claude/bt/fac/sleeve.py').read().split("\nif __name__")[0])
RV = pd.read_pickle('/home/claude/bt/fac/rev_events.pkl')
def recent(m, w): return m.astype(float).rolling(w, min_periods=1).max() > 0
R1, R5 = RV['R1 單月營收創 12 月新高'], RV['R5 營收加速前 20%（排除前 2%）']
sleeve(anyev, '基準：收斂突破，分批 15/20/25')
for lab, M in (('R1', R1), ('R5', R5)):
    sleeve(M, f'只用營收事件 {lab}，分批 15/20/25')
    sleeve(M, f'只用營收事件 {lab}，分批 40/50/60', ladder=(40, 50, 60))
    sleeve(anyev | M, f'收斂突破 ∪ {lab}，分批 15/20/25')
    sleeve(anyev & recent(M, 30), f'收斂突破，且 30 日內有 {lab}（營收佐證）')
    tp = cur.rank(axis=1, pct=True)
    sleeve(anyev, f'收斂突破，30 日內有 {lab} 的優先買', prio=tp + recent(M, 30).astype(float))
pd.set_option('display.width', 260); print(pd.DataFrame(RES).to_string(index=False))
