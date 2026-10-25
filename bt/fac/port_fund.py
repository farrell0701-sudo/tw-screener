"""基本面因子當過濾（標準版：T_ldf＋族群＋每日補位＋出場 ATR 或跌破 60 日密集區下緣）
沒有資料的股票一律視為通過（避免用到「之後下市／資料缺漏」的資訊）"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('/home/claude/bt/fac/micro_port.py').read().split("out_ = out")[0])
out_ = out
from factors6 import build6
W = pd.read_pickle('/home/claude/bt/fac/fund.pkl'); F6 = build6(P, W)
names = sys.argv[1:] or ['F_REVMOM', 'F_REVYOY', 'F_BP', 'F_DY', 'F_EP']
eng(cur, '標準版', 'atr或medL')
for k in names:
    X, s, _ = F6[k]; xp = (X * s).where(U).rank(axis=1, pct=True)
    for th in (0.2, 0.4, 0.6):
        f = cur.where((xp >= th) | xp.isna())
        eng(f, f'{k} 百分位≥{th}', 'atr或medL')
pd.DataFrame(out).to_csv('/home/claude/bt/fac/port_fund_' + '_'.join(names)[:50] + '.csv', index=False, encoding='utf-8-sig')
