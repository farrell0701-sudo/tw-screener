"""量價因子當過濾（乾淨資料、每日補位、族群過濾）"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('/home/claude/bt/fac/fin_const.py').read().split("\nif __name__ == '__main__':")[0])
from factors5 import build5
F5 = build5(P)
daily2(cur, cur, '現行')
for k in ('V_updn60', 'V_obv60', 'V_amtgrow', 'V_spike20'):
    X, s, _ = F5[k]; xp = (X * s).where(U).rank(axis=1, pct=True)
    for th in (0.3, 0.5):
        daily2(cur.where(xp >= th), cur.where(xp >= th), f'{k} 百分位≥{th}')
pd.DataFrame(out).to_csv('/home/claude/bt/fac/port_vol.csv', index=False, encoding='utf-8-sig')
