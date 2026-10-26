"""候選池固定 = T_ldf 前 K 名（族群過濾＋近5日一字跌停不買），只改「池內買進順序」，看年化有多敏感"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('/home/claude/bt/fac/sizing.py').read().split("\nif __name__")[0])
def engine_o(sc, prio, label, n=10, K=30, kb=4, kr=3, offs=range(20), need=False):
    """sc 決定候選池與持股檢查；prio 決定池內先買誰（越大越先買）"""
    RK = sc.where(BUYdf).rank(axis=1, ascending=False).values; RKall = sc.rank(axis=1, ascending=False).values
    PR = prio.reindex_like(sc).values
    res = []
    for off in offs:
        reb = set(range(d0 + off, len(dates) - 2, REB))
        cash = 1.0; pos = {}; sell_next, buy_next = {}, []; eq = np.full(len(dates), np.nan)
        for t in range(d0, len(dates)):
            keep = {}
            for j, why in sell_next.items():
                if j not in pos: continue
                if lockdn[t, j]: keep[j] = why; continue
                px = Ov[t, j] if not np.isnan(Ov[t, j]) else Cfv[t, j]
                p = pos.pop(j); cash += p[0] * px * (1 - FEE - TAX)
            if buy_next:
                val = cash + sum(p[0] * Cfv[t - 1, j] for j, p in pos.items())
                for j in buy_next:
                    if j in pos or len(pos) >= n: continue
                    px = Ov[t, j]
                    if np.isnan(px) or lockup[t, j]: continue
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
                if c < p[2] - k * Av[t, j]: sell_next[j] = 'A'
                elif c < ML[t, j]: sell_next[j] = 'L'
            if t in reb:
                for j in pos:
                    if j not in sell_next and not (RKall[t, j] <= 2 * K): sell_next[j] = 'R'
            free = n - len(pos) + len(set(sell_next) & set(pos))
            if free > 0:
                cand = np.where(RK[t] <= K)[0]
                pr = np.nan_to_num(PR[t, cand], nan=-np.inf)
                if need: cand = cand[~np.isnan(PR[t, cand])]; pr = PR[t, cand]
                buy_next = [j for j in cand[np.argsort(-pr, kind='stable')] if j not in pos][:free]
        res.append(pd.Series(eq, index=dates).dropna())
    summ(res, label); return res
if __name__ == '__main__':
    ldn5 = pd.DataFrame(lockdn, index=C.index, columns=codes).rolling(5).sum() > 0
    sc = cur.where(~ldn5)
    atrp = pd.DataFrame(Av, index=C.index, columns=codes) / C
    HH = Hd.rolling(120, min_periods=100).max(); LL = Ld.rolling(120, min_periods=100).min()
    iH = Hd.rolling(120, min_periods=100).apply(np.argmax, raw=True); iL = Ld.rolling(120, min_periods=100).apply(np.argmin, raw=True)
    nd = iH - iL; sp = (np.log(HH / LL) / nd).where(nd > 0)
    engine_o(sc, sc, 'T_ldf 高→低（現行）')
    engine_o(sc, -sc, 'T_ldf 低→高（反過來）')
    rng = np.random.default_rng(0)
    for s in range(5):
        R = pd.DataFrame(rng.random(sc.shape), index=sc.index, columns=sc.columns)
        engine_o(sc, R, f'隨機 #{s + 1}')
    engine_o(sc, -P['RC'], '低價優先（V3C 式）')
    engine_o(sc, P['RC'], '高價優先')
    engine_o(sc, -atrp, '低波動優先')
    engine_o(sc, atrp, '高波動優先')
    engine_o(sc, -sp, '漲得慢的優先')
    for K in (10, 60):
        engine_o(sc, sc, f'T_ldf 高→低，候選池改 {K} 名', K=K)
    D = pd.DataFrame(out).drop(columns='資料'); pd.set_option('display.width', 250); print(D.to_string())
    D.to_csv('/home/claude/bt/fac/order_test.csv', index=False, encoding='utf-8-sig')
