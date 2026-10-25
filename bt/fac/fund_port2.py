"""基本面 × 標準版（真實執行＋近5日一字跌停不買）：過濾、截頂營收加速、合成分數、獨立基本面組合與相關性"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('/home/claude/bt/fac/sizing.py').read().split("\nif __name__")[0])
from factors6 import build6
W = pd.read_pickle('/home/claude/bt/fac/fund.pkl'); F6 = build6(P, W)
ldn5 = pd.DataFrame(lockdn, index=C.index, columns=codes).rolling(5).sum() > 0
sc = cur.where(~ldn5)
def pct(X, s=1): return (X * s).where(U).rank(axis=1, pct=True)
REV = F6['F_REVMOM'][0]; REVc = REV.clip(upper=1.4)
fac = {'營收加速 F_REVMOM': pct(REV), '營收加速（截頂 1.4）': pct(REVc), '淨值市價比 F_BP': pct(F6['F_BP'][0]),
       '現金殖利率 F_DY': pct(F6['F_DY'][0]), '盈餘殖利率 F_EP': pct(F6['F_EP'][0]), '近四季淨利/市值 F_EPfin': pct(F6['F_EPfin'][0]),
       '小市值 F_SIZE': pct(F6['F_SIZE'][0], -1)}
res0 = engine_w(sc, '標準版')
for k, xp in fac.items():
    for th in (0.3,):
        engine_w(sc.where((xp >= th) | xp.isna()), f'過濾：{k} 排除最低 30%')
# 截頂營收加速：只排除最高 1%（一次性暴衝）
q = pct(REV); engine_w(sc.where(~(q > 0.99)), '過濾：排除營收加速最高 1%')
# 合成分數：池內排序 = T_ldf 百分位 + w × 基本面百分位（族群過濾、一字跌停規則照舊）
tp = cur.rank(axis=1, pct=True)
for k in ('營收加速（截頂 1.4）', '淨值市價比 F_BP', '現金殖利率 F_DY'):
    for w in (0.3, 0.5):
        blend = (tp + w * fac[k].fillna(0.5)).where(sc.notna())
        engine_w(blend, f'合成：T_ldf + {w}×{k}')
# 獨立基本面組合：同樣引擎，排序改用基本面（只限流動性股票池），看與主策略的相關
alone = {}
for k in ('營收加速（截頂 1.4）', '淨值市價比 F_BP'):
    s_ = fac[k].where(U & ~ldn5)
    alone[k] = engine_w(s_, f'獨立組合：只用 {k} 排序')
r0 = res0[0].pct_change().dropna()
for k, rs in alone.items():
    r1 = rs[0].pct_change().dropna(); j = r0.index.intersection(r1.index)
    m0 = res0[0].resample('ME').last().pct_change(); m1 = rs[0].resample('ME').last().pct_change()
    print(f'{k} 獨立組合 vs 標準版：日報酬相關 {r0[j].corr(r1[j]):.2f}，月報酬相關 {m0.corr(m1):.2f}')
    for wt in (0.2, 0.3, 0.5):
        mix = (1 - wt) * r0[j] + wt * r1[j]; eq = (1 + mix).cumprod()
        yrs = (eq.index[-1] - eq.index[0]).days / 365.25
        print(f'   資金 {int(wt*100)}% 配基本面組合：年化 {eq.iloc[-1] ** (1 / yrs) - 1:.1%}，回撤 {(eq / eq.cummax() - 1).min():.1%}，Sharpe {mix.mean() / mix.std() * np.sqrt(252):.2f}')
D = pd.DataFrame(out).drop(columns='資料'); D.to_csv('/home/claude/bt/fac/fund_port2.csv', index=False, encoding='utf-8-sig')
pd.set_option('display.width', 250); print(D.to_string())
