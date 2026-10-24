"""混合：現行每 20 日換股（前 10 名）＋ 停損後空出的位置，改成「每天用前 K 名內剛出現事件的股票補上」，不再空等到下個換股日"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('port_timing.py').read().split("\nif __name__ == '__main__'")[0])


def run_hyb(evm, K, off=0, kb=4, kr=3, n=10):
    ev = evm.values if evm is not None else None
    start = d0; end = len(dates) - 1
    reb = set(range(start + off, end - 1, REB))
    cash = 1.0; pos = {}; sell_next, buy_next = set(), []
    eq = np.full(len(dates), np.nan); expo = []
    for t in range(start, end + 1):
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
                pos[j] = [amt * (1 - FEE) / px, px, px]; cash -= amt
        sell_next, buy_next = set(), []
        eq[t] = cash + sum(p[0] * Cfv[t, j] for j, p in pos.items()); expo.append(1 - cash / eq[t])
        k = kb if BULL[t] else kr
        for j, p in pos.items():
            c = Cv[t, j]
            if np.isnan(c): continue
            p[2] = max(p[2], c)
            if c < p[2] - k * Av[t, j]: sell_next.add(j)
        if t in reb:
            ok = np.where(RK[t] <= n)[0]; top = list(ok[np.argsort(RK[t, ok])])
            sell_next |= {j for j in pos if j not in top}
            buy_next = [j for j in top if j not in pos]
        elif ev is not None:
            free = n - len(pos) + len(sell_next)
            if free > 0:
                cand = np.where(ev[t] & (RK[t] <= K))[0]
                buy_next = [j for j in cand[np.argsort(RK[t, cand])] if j not in pos][:free]
    return pd.Series(eq, index=dates).dropna(), np.mean(expo)


rows = []
for label, evm, K in (('現行（同引擎）', None, 0), ('停損空位用 A 出現補，前100', EVP['A出現'], 100), ('停損空位用 A 出現補，前300', EVP['A出現'], 300),
                      ('停損空位用上緣突破補，前100', EVP['上緣突破'], 100), ('停損空位用上緣突破補，前300', EVP['上緣突破'], 300)):
    out = [run_hyb(evm, K, off) for off in range(20)]
    r = summarize([s for s, _ in out], label); r['平均持股比例'] = round(np.median([x for _, x in out]) * 100, 1); rows.append(r)
    print('   平均持股比例', r['平均持股比例'], flush=True)
pd.DataFrame(rows).to_csv('/home/claude/bt/fac/port_hybrid.csv', index=False, encoding='utf-8-sig')
