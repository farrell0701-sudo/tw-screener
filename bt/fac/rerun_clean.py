"""在乾淨資料（CLEAN=1）上重跑關鍵結論，並和舊資料並列
用法：CLEAN=1 python3 rerun_clean.py   （不設 CLEAN 則跑舊資料，方便對照）"""
import os, sys, time
sys.path.insert(0, '/home/claude/bt/fac')
exec(open('/home/claude/bt/fac/port2.py').read().split("if __name__")[0])
from factors import base, build as build1
from factors3 import tld_floor
from evaluate import evaluate
import numpy as np, pandas as pd

TAG = '乾淨' if os.environ.get('CLEAN') == '1' else '舊'
t0 = time.time()
B = base(P); atr = B['atr']
e12 = C.ewm(span=12, adjust=False).mean(); e200 = C.ewm(span=200, adjust=False).mean()
T_ld = ((e12 - e200) / atr).where(U)
T_ldf = tld_floor(P).where(U)
indv = P['industry'].reindex(C.columns); known = indv.notna().values
def ind_filter(sc):
    g = sc.loc[:, known].T.groupby(indv[known].values).transform('median').T.reindex(columns=C.columns)
    gp = g.where(U).rank(axis=1, pct=True)
    return sc.where((gp >= 0.5) | gp.isna())
vcp = (atr.rolling(20).mean() / atr.rolling(120).mean()).where(U)
vp = vcp.rank(axis=1, pct=True)
cur = ind_filter(T_ldf)
mid = dates[dates.get_indexer([pd.Timestamp('2021-01-04')], method='bfill')[0]]
BUYdf = pd.DataFrame(BUY, index=dates, columns=C.columns)
LOW20 = P['L'].rolling(20, min_periods=1).min().values
out = []


def summ(res, label):
    st = pd.DataFrame([dict(**stats(s), 前半=stats(s[:mid])['CAGR'], 後半=stats(s[mid:])['CAGR']) for s in res]); m = st.median()
    r = dict(資料=TAG, 版本=label, 年化=round(m.CAGR * 100, 1), 最低=round(st.CAGR.min() * 100, 1), 回撤=round(m.MDD * 100, 1),
             最差回撤=round(st.MDD.min() * 100, 1), Sharpe=round(m.Sharpe, 2), 前半=round(m.前半 * 100, 1), 後半=round(m.後半 * 100, 1))
    out.append(r); print(r, flush=True)


def old_engine(sc, label, kb=4, kr=3, n=10):
    globals()['N'] = n; globals()['Sv'] = sc.values
    summ([run(off, False, kb, kr)[0] for off in range(20)], label)


def daily_engine(sc, label, K=30, n=10, kb=4, kr=3, stop='atr', offs=range(20)):
    RK = sc.where(BUYdf).rank(axis=1, ascending=False).values; RKall = sc.rank(axis=1, ascending=False).values
    res = []
    for off in offs:
        reb = set(range(d0 + off, len(dates) - 2, REB))
        cash = 1.0; pos = {}; sell_next, buy_next = set(), []; eq = np.full(len(dates), np.nan)
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
                    pos[j] = [amt * (1 - FEE) / px, px, px]; cash -= amt
            sell_next, buy_next = set(), []
            eq[t] = cash + sum(p[0] * Cfv[t, j] for j, p in pos.items())
            k = kb if BULL[t] else kr
            for j, p in pos.items():
                c = Cv[t, j]
                if np.isnan(c): continue
                p[2] = max(p[2], c)
                line = p[2] - k * Av[t, j] if stop == 'atr' else LOW20[t - 1, j]
                if c < line: sell_next.add(j)
            if t in reb: sell_next |= {j for j in pos if not (RKall[t, j] <= 2 * K)}
            free = n - len(pos) + len(sell_next & set(pos))
            if free > 0:
                cand = np.where(RK[t] <= K)[0]
                buy_next = [j for j in cand[np.argsort(RK[t, cand])] if j not in pos][:free]
        res.append(pd.Series(eq, index=dates).dropna())
    summ(res, label)


if __name__ == '__main__':
    print(f'資料：{TAG}，股票池中位 {int(U.loc[dates[d0]:].sum(1).median())} 檔', flush=True)
    # 1) 單因子 IC
    F1 = build1(P); CTRL = [F1['X_atrpct'][0], F1['X_amt'][0]]
    for name, X, s in (('T_ld', T_ld, 1), ('T_ldf', T_ldf, 1), ('E_pct（百分比距離）', ((e12 - e200) / C).where(U), 1),
                       ('B_vcp（收縮好）', vcp, -1), ('X_atrpct（低波動）', F1['X_atrpct'][0], -1)):
        o, _ = evaluate(name, X, s, E, P, CTRL if not name.startswith('X_') else None)
        r = dict(資料=TAG, 因子=name, IC20=round(o['ICva20'], 3), IC60=round(o['ICva60'], 3), t60=round(o['t60'], 1), 正年=o['正年60'],
                 增量60=round(o.get('ICn60', np.nan), 3), 前20=round(o['Top20_60'], 2), 前50=round(o['Top50_60'], 2))
        print(r, flush=True); pd.DataFrame([r]).to_csv(f'/home/claude/bt/fac/rerun_ic_{TAG}.csv', mode='a', header=False, index=False)
    # 2) 組合
    old_engine(T_ld, '舊引擎｜T_ld（無下限）')
    old_engine(T_ldf, '舊引擎｜T_ldf')
    old_engine(cur, '舊引擎｜T_ldf＋族群（舊現行）')
    old_engine(cur.where(vp <= 0.8), '舊引擎｜＋排除波動擴張20%')
    old_engine(cur, '舊引擎｜全程 4ATR', 4, 4)
    old_engine(cur, '舊引擎｜不停損', None, None)
    daily_engine(cur, '每日補位｜T_ldf＋族群')
    daily_engine(cur.where(vp <= 0.8), '每日補位｜＋排除波動擴張20%')
    daily_engine(T_ldf, '每日補位｜無族群過濾')
    daily_engine(cur, '每日補位｜停損改 20 日低點', stop='low20')
    daily_engine(cur, '每日補位｜K=50', K=50)
    pd.DataFrame(out).to_csv(f'/home/claude/bt/fac/rerun_port_{TAG}.csv', index=False, encoding='utf-8-sig')
    print(f'完成 {time.time() - t0:.0f}s')
