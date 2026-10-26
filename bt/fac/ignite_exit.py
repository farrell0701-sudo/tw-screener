"""事件進場（強勢股出現三種收斂突破事件之一 → 隔日開盤買），前 MIN 日不賣，之後依狀態出場"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('/home/claude/bt/fac/sizing.py').read().split("\nif __name__")[0])
EV = pd.read_pickle('/home/claude/bt/fac/ignite_ev.pkl')
anyev = EV['布林：收斂後首次突破上軌'] | EV['布林：收斂後帶寬張開 + 收上軌'] | EV['整理後突破 20 日高']
ldn5 = pd.DataFrame(lockdn, index=C.index, columns=codes).rolling(5).sum() > 0
sc = cur.where(~ldn5)
strong = (cur.rank(axis=1, pct=True) >= 0.9) & ~ldn5
RKall = sc.rank(axis=1, ascending=False).values
med = {w: (Ld.rolling(w, min_periods=w).median().shift(1).values, Hd.rolling(w, min_periods=w).median().shift(1).values) for w in (20, 60)}
def ev_eng(label, rules, MIN=10, n=10, kb=4, kr=3, maxhold=None):
    S = strong & anyev.reindex_like(strong).fillna(False)
    RKs = cur.where(S & BUYdf).rank(axis=1, ascending=False).values
    cash = 1.0; pos = {}; sell_next, buy_next = set(), []; eq = np.full(len(dates), np.nan); T = []
    for t in range(d0, len(dates)):
        keep = set()
        for j in sell_next:
            if j not in pos: continue
            if lockdn[t, j]: keep.add(j); continue
            px = Ov[t, j] if not np.isnan(Ov[t, j]) else Cfv[t, j]
            p = pos.pop(j); cash += p[0] * px * (1 - FEE - TAX); T.append((px * (1 - FEE - TAX) / (p[1] / (1 - FEE)) - 1, t - p[3]))
        if buy_next:
            val = cash + sum(p[0] * Cfv[t - 1, j] for j, p in pos.items())
            for j in buy_next:
                if j in pos or len(pos) >= n: continue
                px = Ov[t, j]
                if np.isnan(px) or lockup[t, j]: continue
                amt = min(val / n, cash)
                if amt <= 0: break
                pos[j] = [amt * (1 - FEE) / px, px, px, t]; cash -= amt
        sell_next, buy_next = keep, []
        eq[t] = cash + sum(p[0] * Cfv[t, j] for j, p in pos.items())
        k = kb if BULL[t] else kr
        for j, p in pos.items():
            c = Cv[t, j]
            if j in sell_next: continue
            held = t - p[3] + 1
            if np.isnan(c):
                if held > 20 and np.isnan(RKall[t, j]): sell_next.add(j)      # 停牌／離開股票池
                continue
            p[2] = max(p[2], c)
            if held < MIN: continue
            ex_ = False
            for r in rules:
                if r == 'ATR' and c < p[2] - k * Av[t, j]: ex_ = True
                elif r == '排名' and not (RKall[t, j] <= 60): ex_ = True
                elif r == '下緣20' and c < med[20][0][t, j]: ex_ = True
                elif r == '下緣60' and c < med[60][0][t, j]: ex_ = True
                elif r == '上緣20' and c < med[20][1][t, j]: ex_ = True
            if maxhold and held >= maxhold: ex_ = True
            if ex_: sell_next.add(j)
        free = n - len(pos) + len(sell_next & set(pos))
        if free > 0:
            cand = np.where(~np.isnan(RKs[t]))[0]
            buy_next = [j for j in cand[np.argsort(RKs[t, cand])] if j not in pos][:free]
    s = pd.Series(eq, index=dates).dropna(); summ([s], label)
    r = np.array([x for x, _ in T]); h = np.array([y for _, y in T])
    out[-1].update({'交易數/年': round(len(r) / 10.7), '勝率': f'{(r > 0).mean():.0%}', '每筆平均': f'{r.mean():.2%}', '持有中位': int(np.median(h))})
    return s
S = {}
S['固定20'] = ev_eng('對照：固定持有 20 日', [], maxhold=20)
S['std'] = ev_eng('10 日後：標準版規則（ATR＋60 下緣＋跌出前 60）', ['ATR', '下緣60', '排名'])
ev_eng('10 日後：ATR＋跌出前 60', ['ATR', '排名'])
ev_eng('10 日後：ATR＋60 日密集區下緣', ['ATR', '下緣60'])
ev_eng('10 日後：ATR＋20 日密集區下緣', ['ATR', '下緣20'])
ev_eng('10 日後：只看 60 日密集區下緣', ['下緣60'])
S['l20'] = ev_eng('10 日後：只看 20 日密集區下緣', ['下緣20'])
ev_eng('10 日後：只看 20 日密集區上緣（較嚴）', ['上緣20'])
ev_eng('10 日後：20 日下緣＋跌出前 60', ['下緣20', '排名'])
ev_eng('不設最短持有：標準版規則', ['ATR', '下緣60', '排名'], MIN=1)
ev_eng('最短持有 5 日：標準版規則', ['ATR', '下緣60', '排名'], MIN=5)
ev_eng('最短持有 15 日：標準版規則', ['ATR', '下緣60', '排名'], MIN=15)
D = pd.DataFrame(out).drop(columns=['資料', '最低', '最差回撤'], errors='ignore'); pd.set_option('display.width', 260); print(D.to_string())
pd.to_pickle(S, '/home/claude/bt/fac/ignite_exit_eq.pkl')
