"""標準版（每日補位、K=30/2K=60、ATR 多4空3 或跌破 60 日密集區下緣出場、10 檔）逐筆交易紀錄"""
import sys; sys.path.insert(0, '/home/claude/bt/fac'); sys.path.insert(0, '/home/claude/bt')
exec(open('/home/claude/bt/fac/micro_port.py').read().split("\ndef eng")[0])
from load import load_all
NM = load_all()[0].set_index('code')['name']
codes = C.columns
def run_log(sc, off, K=30, n=10, kb=4, kr=3):
    RK = sc.where(BUYdf).rank(axis=1, ascending=False).values; RKall = sc.rank(axis=1, ascending=False).values
    reb = set(range(d0 + off, len(dates) - 2, REB))
    cash = 1.0; pos = {}; sell_next, buy_next = {}, []; T = []
    for t in range(d0, len(dates)):
        for j, why in sell_next.items():
            if j in pos:
                px = Ov[t, j] if not np.isnan(Ov[t, j]) else Cfv[t, j]
                p = pos.pop(j); cash += p[0] * px * (1 - FEE - TAX)
                T.append(dict(代號=codes[j], 名稱=NM.get(codes[j], ''), 進場日=dates[p[3]].date(), 出場日=dates[t].date(),
                              持有交易日=t - p[3], 報酬=px * (1 - FEE - TAX) / (p[1] / (1 - FEE)) - 1, 出場原因=why))
        if buy_next:
            val = cash + sum(p[0] * Cfv[t - 1, j] for j, p in pos.items())
            for j in buy_next:
                if j in pos or len(pos) >= n: continue
                px = Ov[t, j]
                if np.isnan(px): continue
                amt = min(val / n, cash)
                if amt <= 0: break
                pos[j] = [amt * (1 - FEE) / px, px, px, t]; cash -= amt
        sell_next, buy_next = {}, []
        k = kb if BULL[t] else kr
        for j, p in pos.items():
            c = Cv[t, j]
            if np.isnan(c): continue
            p[2] = max(p[2], c)
            if c < p[2] - k * Av[t, j]: sell_next[j] = 'ATR 移動停損'
            elif c < ML[t, j]: sell_next[j] = '跌破密集區下緣'
        if t in reb:
            for j in pos:
                if j not in sell_next and not (RKall[t, j] <= 2 * K): sell_next[j] = '換股日跌出前 60'
        free = n - len(pos) + len(set(sell_next) & set(pos))
        if free > 0:
            cand = np.where(RK[t] <= K)[0]
            buy_next = [j for j in cand[np.argsort(RK[t, cand])] if j not in pos][:free]
    for j, p in pos.items():
        T.append(dict(代號=codes[j], 名稱=NM.get(codes[j], ''), 進場日=dates[p[3]].date(), 出場日='持有中',
                      持有交易日=len(dates) - 1 - p[3], 報酬=Cfv[-1, j] / (p[1] / (1 - FEE)) - 1, 出場原因='持有中'))
    return pd.DataFrame(T)
allT = []
for off in range(20):
    x = run_log(cur, off); x['off'] = off; allT.append(x)
A = pd.concat(allT)
A['出場日'] = A['出場日'].astype(str)
G = A.groupby(['代號', '名稱', '進場日', '出場日', '持有交易日', '出場原因']).agg(報酬=('報酬', 'first'), 出現於幾個起始日=('off', 'nunique')).reset_index()
G.to_csv('/home/claude/bt/fac/trades_std.csv', index=False, encoding='utf-8-sig')
T0 = allT[0]; T0.to_csv('/home/claude/bt/fac/trades_std_off0.csv', index=False, encoding='utf-8-sig')
pd.set_option('display.width', 250)
for lab, X in (('起始日 0', T0), ('20 個起始日去重', G)):
    X = X.copy(); X['報酬%'] = (X['報酬'] * 100).round(1)
    c = ['代號', '名稱', '進場日', '出場日', '持有交易日', '報酬%', '出場原因'] + (['出現於幾個起始日'] if '出現於幾個起始日' in X else [])
    print(f'== {lab}：{len(X)} 筆，勝率 {(X.報酬 > 0).mean():.1%}，平均 {X.報酬.mean():.2%}，持有中位 {X.持有交易日.median():.0f} 日')
    print(X.nlargest(5, '報酬')[c].to_string(index=False)); print(X.nsmallest(5, '報酬')[c].to_string(index=False))
