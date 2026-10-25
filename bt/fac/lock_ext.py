"""1) 執行真實化：一字跌停（H==L 且跌 ≥9.5%）當天賣不掉，順延；一字漲停買不到（原本只擋開盤漲停）
2) 進場前漲幅：訊號日收盤 ÷ 近 120／250 日最低價，和交易報酬的關係
3) 過濾：前一日跌停不買、近 N 日有一字板不買、排除漲幅過大"""
import sys; sys.path.insert(0, '/home/claude/bt/fac'); sys.path.insert(0, '/home/claude/bt')
exec(open('/home/claude/bt/fac/micro_port.py').read().split("\ndef eng")[0])
from load import load_all
NM = load_all()[0].set_index('code')['name']; codes = C.columns
Hd, Ld = P['H'], P['L']
prevC = C.ffill().shift(1)
ret1 = C / prevC - 1
one = (Hd == Ld) & C.notna()
lockdn = (one & (ret1 <= -0.095)).values
lockup = (one & (ret1 >= 0.095)).values
limdn = (ret1 <= -0.095)
def engine(sc, label, real=True, n=10, K=30, kb=4, kr=3, offs=range(20), log=False):
    RK = sc.where(BUYdf).rank(axis=1, ascending=False).values; RKall = sc.rank(axis=1, ascending=False).values
    res, T = [], []
    for off in offs:
        reb = set(range(d0 + off, len(dates) - 2, REB))
        cash = 1.0; pos = {}; sell_next, buy_next = {}, []; eq = np.full(len(dates), np.nan)
        for t in range(d0, len(dates)):
            keep = {}
            for j, why in sell_next.items():
                if j not in pos: continue
                if real and lockdn[t, j]: keep[j] = why; continue          # 一字跌停賣不掉，明天再賣
                px = Ov[t, j] if not np.isnan(Ov[t, j]) else Cfv[t, j]
                p = pos.pop(j); cash += p[0] * px * (1 - FEE - TAX)
                if log and off == 0:
                    T.append(dict(j=j, 進場=p[3], 出場=t, 報酬=px * (1 - FEE - TAX) / (p[1] / (1 - FEE)) - 1, 原因=why))
            if buy_next:
                val = cash + sum(p[0] * Cfv[t - 1, j] for j, p in pos.items())
                for j in buy_next:
                    if j in pos or len(pos) >= n: continue
                    px = Ov[t, j]
                    if np.isnan(px) or (real and lockup[t, j]): continue
                    amt = min(val / n, cash)
                    if amt <= 0: break
                    pos[j] = [amt * (1 - FEE) / px, px, px, t]; cash -= amt
            sell_next, buy_next = dict(keep), []
            eq[t] = cash + sum(p[0] * Cfv[t, j] for j, p in pos.items())
            k = kb if BULL[t] else kr
            for j, p in pos.items():
                c = Cv[t, j]
                if np.isnan(c) or j in sell_next: continue
                p[2] = max(p[2], c)
                if c < p[2] - k * Av[t, j]: sell_next[j] = 'ATR'
                elif c < ML[t, j]: sell_next[j] = '密集區下緣'
            if t in reb:
                for j in pos:
                    if j not in sell_next and not (RKall[t, j] <= 2 * K): sell_next[j] = '跌出前60'
            free = n - len(pos) + len(set(sell_next) & set(pos))
            if free > 0:
                cand = np.where(RK[t] <= K)[0]
                buy_next = [j for j in cand[np.argsort(RK[t, cand])] if j not in pos][:free]
        res.append(pd.Series(eq, index=dates).dropna())
    summ(res, label)
    return pd.DataFrame(T)
lc = np.log(C)
run120 = lc - np.log(Ld.rolling(120, min_periods=100).min())
run250 = lc - np.log(Ld.rolling(250, min_periods=200).min())
sig60 = np.log(C / C.shift(1)).rolling(60, min_periods=40).std()
run250v = run250 / (sig60 * np.sqrt(250))
if __name__ == '__main__':
    engine(cur, '原版執行（只擋開盤漲停）', real=False)
    T = engine(cur, '真實執行（一字跌停賣不掉、一字漲停買不到）', log=True)
    # 交易層：進場前漲幅分組（訊號日 = 進場前一天）
    sd = T.進場.values - 1; jj = T.j.values
    T['漲幅120'] = np.exp(run120.values[sd, jj]) - 1; T['漲幅250'] = np.exp(run250.values[sd, jj]) - 1
    T['漲幅250÷波動'] = run250v.values[sd, jj]
    T['近10日一字板'] = (one.rolling(10).sum() > 0).values[sd, jj]
    T['前日跌停'] = limdn.values[sd, jj]
    T['持有'] = T.出場 - T.進場; T['代號'] = codes[jj]; T['名稱'] = [NM.get(c, '') for c in T.代號]
    T.to_csv('/home/claude/bt/fac/trades_real_off0.csv', index=False, encoding='utf-8-sig')
    for col in ('漲幅120', '漲幅250', '漲幅250÷波動'):
        q = pd.qcut(T[col], 5, duplicates='drop')
        print(f'\n== 依進場前 {col} 五分組（起始日 0，{len(T)} 筆）')
        print(T.groupby(q, observed=True).報酬.agg(筆數='size', 平均=lambda x: f'{x.mean():.1%}', 中位=lambda x: f'{x.median():.1%}',
              勝率=lambda x: f'{(x > 0).mean():.0%}', 大贏家數=lambda x: int((x > 1).sum()), 大輸家數=lambda x: int((x < -0.2).sum())).to_string())
    for col in ('近10日一字板', '前日跌停'):
        print(f'\n== {col}'); print(T.groupby(col).報酬.agg(['size', 'mean', 'median']).round(3).to_string())
    pd.set_option('display.width', 250)
    T['報酬%'] = (T.報酬 * 100).round(1)
    show = ['代號', '名稱', '持有', '報酬%', '漲幅120', '漲幅250', '漲幅250÷波動']
    print(T.nlargest(10, '報酬')[show].round(2).to_string(index=False)); print(T.nsmallest(10, '報酬')[show].round(2).to_string(index=False))
