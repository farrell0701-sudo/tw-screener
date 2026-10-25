"""事件研究唯一顯著的結果：強勢股「跌回 60 日密集區內」之後 5～10 日落後同儕 → 拿來當出場／排除條件
乾淨資料、每日補位（現行主策略），只改出場或進場排除"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('/home/claude/bt/fac/rerun_clean.py').read().split("if __name__ == '__main__':")[0])
medH = P['H'].rolling(60, min_periods=60).median().shift(1); medL = P['L'].rolling(60, min_periods=60).median().shift(1)
MH, ML = medH.values, medL.values


def eng(sc, label, stop='atr', K=30, n=10, kb=4, kr=3, offs=range(20), maxhold=None):
    RK = sc.where(BUYdf).rank(axis=1, ascending=False).values; RKall = sc.rank(axis=1, ascending=False).values
    res, nst = [], []
    for off in offs:
        reb = set(range(d0 + off, len(dates) - 2, REB))
        cash = 1.0; pos = {}; sell_next, buy_next = set(), []; eq = np.full(len(dates), np.nan); ns = 0
        for t in range(d0, len(dates)):
            for j in sell_next:
                if j in pos:
                    px = Ov[t, j] if not np.isnan(Ov[t, j]) else Cfv[t, j]
                    cash += pos.pop(j)[0] * px * (1 - FEE - TAX)
            if buy_next:
                val = cash + sum(p[0] * Cfv[t - 1, j] for j, p in pos.items())
                for j in buy_next:
                    if j in pos or len(pos) >= n: continue
                    px = Ov[t, j]
                    if np.isnan(px): continue
                    amt = min(val / n, cash)
                    if amt <= 0: break
                    pos[j] = [amt * (1 - FEE) / px, px, px, t]; cash -= amt
            sell_next, buy_next = set(), []
            eq[t] = cash + sum(p[0] * Cfv[t, j] for j, p in pos.items())
            k = kb if BULL[t] else kr
            for j, p in pos.items():
                c = Cv[t, j]
                if np.isnan(c): continue
                p[2] = max(p[2], c)
                hit_atr = c < p[2] - k * Av[t, j]
                hit_h = c < MH[t, j]; hit_l = c < ML[t, j]
                out = {'atr': hit_atr, 'medH': hit_h, 'medL': hit_l, 'atr或medH': hit_atr or hit_h, 'atr或medL': hit_atr or hit_l}[stop]
                if maxhold and t - p[3] >= maxhold: out = True
                if out: sell_next.add(j); ns += 1
            if t in reb: sell_next |= {j for j in pos if not (RKall[t, j] <= 2 * K)}
            free = n - len(pos) + len(sell_next & set(pos))
            if free > 0:
                cand = np.where(RK[t] <= K)[0]
                buy_next = [j for j in cand[np.argsort(RK[t, cand])] if j not in pos][:free]
        res.append(pd.Series(eq, index=dates).dropna()); nst.append(ns)
    summ(res, label); out_[-1]['停損次數每年'] = round(np.median(nst) / 10.7)


out_ = out
eng(cur, '現行：ATR 出場')
eng(cur, '出場：跌回密集區上緣以下', 'medH')
eng(cur, '出場：跌破密集區下緣', 'medL')
eng(cur, '出場：ATR 或 跌回上緣（先到者）', 'atr或medH')
eng(cur, '出場：ATR 或 跌破下緣（先到者）', 'atr或medL')
inz = (C <= medH)            # 收盤在密集區上緣以下
eng(cur.where(~inz), '進場排除：已跌回密集區上緣以下的不買')
eng(cur.where(~inz), '排除＋出場 ATR 或 跌破下緣', 'atr或medL')
pd.DataFrame(out).to_csv('/home/claude/bt/fac/micro_port.csv', index=False, encoding='utf-8-sig')
