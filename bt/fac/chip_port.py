import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('/home/claude/bt/fac/sleeve.py').read().split("\nif __name__")[0])
X = pd.read_pickle('/home/claude/bt/fac/chip_events.pkl'); MOD, CEV = X['MOD'], X['CEV']
inst20 = MOD['法人合計 20 日買超／成交量'].reindex_like(C); fo5 = MOD['外資 5 日買超／成交量'].reindex_like(C)
tp = cur.rank(axis=1, pct=True)
sleeve(anyev, '基準：收斂突破')
for th in (0.05, 0.07, 0.10):
    sleeve(anyev & ~(inst20 > th), f'排除：法人 20 日買超 > {int(th*100)}% 成交量')
sleeve(anyev, '降低順序：法人 20 日買超 > 7% 的最後買', prio=tp - (inst20 > 0.07).astype(float))
sleeve(anyev & ~(fo5 > 0.12), '排除：外資 5 日買超 > 12% 成交量')
sleeve(anyev | CEV['C1 投信首買（60 日未買、≥1% 成交量）'], '加入投信首買事件')
pd.set_option('display.width', 260); print(pd.DataFrame(RES).to_string(index=False))
