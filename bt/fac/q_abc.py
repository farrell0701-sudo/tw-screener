"""T_ld 選出的股票，有多少還在結構轉變期？"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('port_timing.py').read().split("\nif __name__ == '__main__'")[0])
from factors import base
atrp = (base(P)['atr'] / P['C']).where(U).rank(axis=1, pct=True)
ev = EVP['A出現']
reb = range(d0, len(dates) - 1, 20)
for w in (20, 60, 120):
    rec = ev.rolling(w, min_periods=1).max().astype(bool).values
    for n in (10, 50):
        hit = [rec[t, np.where(RK[t] <= n)[0]].mean() for t in reb]
        allu = [rec[t, np.where(U.values[t])[0]].mean() for t in reb]
        print(f'近 {w} 日出現過 A：T_ld 前 {n} 名 {np.mean(hit):.1%}　全市場 {np.mean(allu):.1%}')
# 事件發生當天，T_ld 排名在哪
e = EV[EV.kind == 'A出現即進']
print('A 出現當天的 T_ldf 百分位：中位', round(e.ldp.median(), 2), ' 前 5%：', f'{(e.ldp >= 0.95).mean():.1%}', ' 前 1%：', f'{(e.ldp >= 0.99).mean():.1%}')
# 前 10 名中「近 60 日有 A」的股票 ATR% 分位
rec = ev.rolling(60, min_periods=1).max().astype(bool).values
a1, a0 = [], []
for t in reb:
    js = np.where(RK[t] <= 10)[0]
    for j in js: (a1 if rec[t, j] else a0).append(atrp.values[t, j])
print(f'前 10 名中：近 60 日有 A 的 ATR% 分位中位 {np.nanmedian(a1):.2f}（n={len(a1)}），沒有 A 的 {np.nanmedian(a0):.2f}（n={len(a0)}）')
