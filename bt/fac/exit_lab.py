"""出場研究：基準＝標準版（真實執行＋近5日一字跌停不買）
參數：rank_k（換股日跌出前 rank_k×K 才賣；None＝不做換股檢查）、kb/kr（多空 ATR 倍數）、
      bonus（[(獲利門檻, 加多少倍 ATR)]：賺越多停損越寬）、use_medL（跌破密集區下緣出場）
記錄每筆出場，並看賣掉後該股 20／60 日走勢（相對同日強勢股平均）"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('/home/claude/bt/fac/sizing.py').read().split("\nif __name__")[0])
ldn5 = pd.DataFrame(lockdn, index=C.index, columns=codes).rolling(5).sum() > 0
sc0 = cur.where(~ldn5)
strong = cur.rank(axis=1, pct=True) >= 0.9
Cfv_ = Cfv
def eng_x(sc, label, rank_k=2, kb=4, kr=3, bonus=(), use_medL=True, medL_max_gain=None, n=10, K=30, offs=range(20), log=False, W=None):
    RK = sc.where(BUYdf).rank(axis=1, ascending=False).values; RKall = sc.rank(axis=1, ascending=False).values
    Wv = None if W is None else W.reindex_like(sc).values
    res, T = [], []
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
                if log and off == 0: T.append(dict(j=j, 進=p[3], 出=t, 報酬=px * (1 - FEE - TAX) / (p[1] / (1 - FEE)) - 1, 原因=why))
            if buy_next:
                val = cash + sum(p[0] * Cfv[t - 1, j] for j, p in pos.items())
                for j in buy_next:
                    if j in pos or len(pos) >= n: continue
                    px = Ov[t, j]
                    if np.isnan(px) or lockup[t, j]: continue
                    w = 1.0 if Wv is None or np.isnan(Wv[t - 1, j]) else Wv[t - 1, j]
                    amt = min(val / n * w, cash)
                    if amt <= 0: break
                    pos[j] = [amt * (1 - FEE) / px, px, px, t]; cash -= amt
            sell_next, buy_next = dict(keep), []
            eq[t] = cash + sum(p[0] * Cfv[t, j] for j, p in pos.items())
            k0 = kb if BULL[t] else kr
            for j, p in pos.items():
                c = Cv[t, j]
                if np.isnan(c) or j in sell_next: continue
                p[2] = max(p[2], c)
                k = k0
                gain = p[2] / p[1] - 1
                for th, add in bonus:
                    if gain >= th: k = k0 + add
                if c < p[2] - k * Av[t, j]: sell_next[j] = 'ATR 停損'
                elif use_medL and c < ML[t, j] and (medL_max_gain is None or gain < medL_max_gain): sell_next[j] = '跌破密集區下緣'
            if rank_k and t in reb:
                for j in pos:
                    if j not in sell_next and not (RKall[t, j] <= rank_k * K): sell_next[j] = f'換股日跌出前 {rank_k * K}'
            free = n - len(pos) + len(set(sell_next) & set(pos))
            if free > 0:
                cand = np.where(RK[t] <= K)[0]
                buy_next = [j for j in cand[np.argsort(RK[t, cand])] if j not in pos][:free]
        res.append(pd.Series(eq, index=dates).dropna())
    summ(res, label)
    return pd.DataFrame(T)
if __name__ == '__main__':
    T = eng_x(sc0, '標準版', log=True)
    # 賣掉之後：該股 20／60 日報酬（從出場價起）− 同期強勢股平均
    Cf = C.ffill()
    for h in (20, 60):
        fwd = (Cf.shift(-h) / Cf - 1)
        bench = fwd.where(strong).mean(1)
        T[f'賣後{h}日'] = [fwd.values[t, j] for t, j in zip(T.出, T.j)]
        T[f'賣後{h}日相對強勢股'] = T[f'賣後{h}日'] - bench.values[T.出.values]
    T['持有'] = T.出 - T.進
    T['賣時已賺>50%'] = T.報酬 > 0.5
    g = T.groupby('原因').agg(筆數=('報酬', 'size'), 平均報酬=('報酬', 'mean'), 持有中位=('持有', 'median'),
                            賣後20日=('賣後20日', 'mean'), 賣後60日=('賣後60日', 'mean'), 賣後60日相對=('賣後60日相對強勢股', 'mean'),
                            賣後60日漲超30比例=('賣後60日', lambda x: (x > 0.3).mean()))
    pd.set_option('display.width', 250)
    print(g.round(3).to_string())
    print('\n賣出時已獲利 >50% 的交易（讓獲利奔跑的關鍵）')
    print(T[T['賣時已賺>50%']].groupby('原因').agg(筆數=('報酬', 'size'), 賣後60日=('賣後60日', 'mean'), 賣後60日相對=('賣後60日相對強勢股', 'mean')).round(3).to_string())
    T.to_csv('/home/claude/bt/fac/exit_trades.csv', index=False, encoding='utf-8-sig')
