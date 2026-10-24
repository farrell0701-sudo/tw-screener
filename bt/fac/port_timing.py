"""事件當進場時機：方向由現行分數（T_ldf＋族群過濾）決定，進場時點由結構事件決定
版本
  現行：每 20 日換股，買進前 N 名
  定期＋近期事件：每 20 日換股，但只買「排名前 K 且近 w 日出現事件」的股票
  每日事件進場：每天收盤檢查，排名前 K 的股票當天出現事件就隔日開盤買（有空位才買）；
               每 20 日把跌出前 2K 名的持股賣掉；停損同現行
事件：A 出現（原版）／收盤突破 A 區上緣（改進版）"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('/home/claude/bt/fac/port_struct.py').read().split("\nif __name__ == '__main__'")[0])
import numpy as np, pandas as pd
EV = pd.read_pickle('/home/claude/bt/fac/struct_events.pkl')


def ev_panel(kind):
    m = pd.DataFrame(False, index=dates, columns=C.columns)
    e = EV[EV.kind == kind]
    for (d, c) in zip(e.date, e.code): m.at[d, c] = True
    return m


EVP = {'A出現': ev_panel('A出現即進'), '上緣突破': ev_panel('上緣突破進')}
RK = cur.where(pd.DataFrame(BUY, index=dates, columns=C.columns)).rank(axis=1, ascending=False).values
RKall = cur.rank(axis=1, ascending=False).values


def run_evt(evm, K, off=0, kb=4, kr=3, n=10):
    ev = evm.values
    start = d0; end = len(dates) - 1
    reb = set(range(start + off, end - 1, REB))
    cash = 1.0; pos = {}; sell_next, buy_next = set(), []
    eq = np.full(len(dates), np.nan); nb = 0
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
                pos[j] = [amt * (1 - FEE) / px, px, px]; cash -= amt; nb += 1
        sell_next, buy_next = set(), []
        eq[t] = cash + sum(p[0] * Cfv[t, j] for j, p in pos.items())
        k = kb if BULL[t] else kr
        for j, p in pos.items():
            c = Cv[t, j]
            if np.isnan(c): continue
            p[2] = max(p[2], c)
            if c < p[2] - k * Av[t, j]: sell_next.add(j)
        if t in reb:
            sell_next |= {j for j in pos if not (RKall[t, j] <= 2 * K)}
        free = n - len(pos) + len(sell_next & set(pos))
        if free > 0:
            cand = np.where(ev[t] & (RK[t] <= K))[0]
            cand = [j for j in cand[np.argsort(RK[t, cand])] if j not in pos]
            buy_next = cand[:free]
    return pd.Series(eq, index=dates).dropna(), nb


def summarize(res, label):
    st = pd.DataFrame([dict(**stats(s), 前半=stats(s[:mid])['CAGR'], 後半=stats(s[mid:])['CAGR']) for s in res]); m = st.median()
    r = dict(版本=label, 年化=round(m.CAGR * 100, 1), 最低=round(st.CAGR.min() * 100, 1), 回撤=round(m.MDD * 100, 1),
             Sharpe=round(m.Sharpe, 2), 前半=round(m.前半 * 100, 1), 後半=round(m.後半 * 100, 1))
    print(r, flush=True); return r


if __name__ == '__main__':
    rows = [test(cur, '現行（每 20 日、前 10）', 10)]
    for name, evm in EVP.items():
        for w in (5, 20):
            rec = evm.rolling(w, min_periods=1).max().astype(bool)
            for K in (50, 150):
                sc = cur.where(rec & (pd.DataFrame(RK, index=dates, columns=C.columns) <= K))
                rows.append(test(sc, f'定期＋近{w}日{name}，前{K}名內', 10))
        for K in (30, 100):
            res = [run_evt(evm, K, off)[0] for off in range(0, 20, 4)]
            rows.append(summarize(res, f'每日{name}進場，前{K}名內'))
    pd.DataFrame(rows).to_csv('/home/claude/bt/fac/port_timing.csv', index=False, encoding='utf-8-sig')
