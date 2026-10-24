"""結構當停損：現行選股（T_ldf＋族群過濾，10 檔），只換出場規則
ATR：收盤 < 持有期最高收盤 − k×ATR（多4空3，現行）
結構低點：收盤 < 進場前 n 日最低價（固定不動）
ATR＋結構：兩者取較高（較緊）的那條
通道：收盤 < 最近 n 日最低價（隨時間上移的結構支撐）
停損後的空位留現金，到下次換股日再補"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
exec(open('/home/claude/bt/fac/port_struct.py').read().split("\nif __name__ == '__main__'")[0])
import numpy as np, pandas as pd
Lv = P['L'].values
LOW = {n: P['L'].rolling(n, min_periods=1).min().values for n in (5, 10, 20, 40)}


def run_stop(mode, nlow, off=0, n=10, kb=4, kr=3):
    start = d0; end = len(dates) - 1
    reb = set(range(start + off, end - 1, REB))
    cash = 1.0; pos = {}; sell_next, buy_next = set(), []
    eq = np.full(len(dates), np.nan); nstop = 0
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
                fixed = LOW[nlow][t - 1, j] if nlow else np.nan      # 進場前 n 日低點（訊號日以前）
                pos[j] = [amt * (1 - FEE) / px, px, px, fixed]; cash -= amt
        sell_next, buy_next = set(), []
        eq[t] = cash + sum(p[0] * Cfv[t, j] for j, p in pos.items())
        k = kb if BULL[t] else kr
        for j, p in pos.items():
            c = Cv[t, j]
            if np.isnan(c): continue
            p[2] = max(p[2], c)
            atr_line = p[2] - k * Av[t, j]
            if mode == 'ATR': line = atr_line
            elif mode == '結構低點': line = p[3]
            elif mode == 'ATR＋結構': line = np.nanmax([atr_line, p[3]])
            elif mode == '通道': line = LOW[nlow][t - 1, j]
            elif mode == '通道＋ATR': line = np.nanmax([atr_line, LOW[nlow][t - 1, j]])
            if c < line: sell_next.add(j); nstop += 1
        if t in reb:
            ok = np.where(RK_[t] <= n)[0]; top = list(ok[np.argsort(RK_[t, ok])])
            sell_next |= {j for j in pos if j not in top}
            buy_next = [j for j in top if j not in pos]
    return pd.Series(eq, index=dates).dropna(), nstop


RK_ = cur.where(pd.DataFrame(BUY, index=dates, columns=C.columns)).rank(axis=1, ascending=False).values
rows = []
for mode, nl in (('ATR', 0), ('結構低點', 10), ('結構低點', 20), ('結構低點', 40), ('ATR＋結構', 10), ('ATR＋結構', 20),
                 ('通道', 10), ('通道', 20), ('通道', 40), ('通道＋ATR', 20)):
    out = [run_stop(mode, nl, off) for off in range(20)]
    st = pd.DataFrame([dict(**stats(s), 前半=stats(s[:mid])['CAGR'], 後半=stats(s[mid:])['CAGR']) for s, _ in out]); m = st.median()
    r = dict(出場=f'{mode}{nl or ""}', 年化=round(m.CAGR * 100, 1), 最低=round(st.CAGR.min() * 100, 1), 回撤=round(m.MDD * 100, 1),
             回撤最差=round(st.MDD.min() * 100, 1), Sharpe=round(m.Sharpe, 2), 前半=round(m.前半 * 100, 1), 後半=round(m.後半 * 100, 1),
             停損次數每年=round(np.median([x for _, x in out]) / 10.7, 0))
    rows.append(r); print(r, flush=True)
pd.DataFrame(rows).to_csv('/home/claude/bt/fac/port_stop.csv', index=False, encoding='utf-8-sig')
