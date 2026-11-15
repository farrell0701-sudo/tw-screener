"""用 2004–2016/06 舊股價（FinMind 未還原、含下市股）檢驗凍結版 vs 波動收斂版
邏輯與 finlab_short_v2.py 相同；模擬引擎與自建引擎相同（20 路徑：起點錯開 0–4 日、隨機剔除 10% 訊號，取中位數）"""
import glob, sys, numpy as np, pandas as pd
src = open('/home/claude/bt/finlab_short_v2.py').read().split('\nt0 = time.time()')[0]
exec(src)
D0 = '/home/claude/old/d'
lst = pd.read_csv('/home/claude/hdata/stock_list.csv', dtype={'code': str}).set_index('code')
fr = {}
for f in glob.glob(f'{D0}/[0-9]*.csv'):
    c = f.split('/')[-1][:-4]
    if len(c) != 4 or c.startswith('0'): continue
    x = pd.read_csv(f, parse_dates=['date'])
    if len(x) < 30: continue
    fr[c] = x.set_index('date')
W = lambda k: pd.DataFrame({c: x[k] for c, x in fr.items()}).sort_index().astype(float)
O, H, L, C, V = W('o'), W('h'), W('l'), W('c'), W('v')
for X_ in (O, H, L, C): X_[X_ <= 0] = np.nan
O = O.where(C.notna()); H = np.fmax(np.fmax(H, O), C); L = np.fmin(np.fmin(L, O), C)
ix = pd.read_csv(f'{D0}/TAIEX.csv', parse_dates=['date']).set_index('date').c.reindex(C.index).ffill()
ind = lst.industry.reindex(C.columns)
print(f'股票 {C.shape[1]} 檔（有產業 {ind.notna().sum()}）、{C.index[0].date()} ~ {C.index[-1].date()}', flush=True)
# 還原：未還原價遇除權/減資/分割會跳動；超過當時漲跌幅（2015/06 前 7%）的單日變動視為非交易造成，向前調整
r = C / C.ffill().shift(1) - 1
lim = pd.Series(np.where(C.index < pd.Timestamp('2015-06-01'), 0.075, 0.105), index=C.index)
bad = r.abs().gt(lim, axis=0) & C.notna() & (C.notna().cumsum() > 5)
print(f'超過漲跌幅的單日跳動：{int(bad.sum().sum())} 筆（已修正）；現金股利未還原', flush=True)
k = (1 + r.where(bad, 0)).iloc[::-1].cumprod().iloc[::-1].shift(-1).fillna(1.0)
D = dict(O=O * k, H=H * k, L=L * k, C=C * k, RC=C, RH=H, RL=L, ROP=O, V=V, ix=ix, ind=ind)
X = prepare(D)
Cf = D['C'].ffill().values; Ov = X['O']; LD, LU, OFF = X['LD'], X['LU'], X['OFF']; dates = X['dates']


def run(RK, start, end, off=0, drop=0.1, seed=0, n=N_SLOTS, Hd=HOLD):
    rng = np.random.default_rng(seed)
    s0 = dates.get_indexer([pd.Timestamp(start)], method='bfill')[0] + off
    s1 = dates.get_indexer([pd.Timestamp(end)], method='ffill')[0]
    cash = 1.0; pos = {}; sell, buy = set(), []; eq = np.full(len(dates), np.nan); ntr = 0
    for t in range(s0, s1 + 1):
        keep = set()
        for j in sell:
            if j not in pos: continue
            if LD[t, j]: keep.add(j); continue
            px = Ov[t, j] if not np.isnan(Ov[t, j]) else Cf[t, j]
            p = pos.pop(j); cash += p[0] * px * (1 - FEE - TAX)
        if buy:
            val = cash + sum(p[0] * Cf[t - 1, j] for j, p in pos.items())
            for j in buy:
                if j in pos or len(pos) >= n: continue
                px = Ov[t, j]
                if np.isnan(px) or LU[t, j]: continue
                amt = min(val / n, cash)
                if amt <= 0: break
                pos[j] = [amt * (1 - FEE) / px, t]; cash -= amt; ntr += 1
        sell, buy = keep, []
        eq[t] = cash + sum(p[0] * Cf[t, j] for j, p in pos.items())
        for j, p in pos.items():
            if j not in sell and (t - p[1] + 1 >= Hd or (OFF[t] and t - p[1] >= 1)): sell.add(j)
        free = n - len(pos) + len(sell & set(pos))
        if free > 0 and not OFF[t]:
            c = np.where(~np.isnan(RK[t]))[0]
            if drop: c = c[rng.random(len(c)) >= drop]
            buy = [j for j in c[np.argsort(RK[t, c])] if j not in pos][:free]
    e = pd.Series(eq, index=dates).dropna()
    return e.pct_change().fillna(0), ntr


def stat(r):
    e = (1 + r).cumprod(); y = (e.index[-1] - e.index[0]).days / 365.25
    yr = e.resample('YE').last(); yr = yr.pct_change().fillna(yr.iloc[0] - 1)
    return dict(年化=e.iloc[-1] ** (1 / y) - 1, 回撤=(e / e.cummax() - 1).min(), Sharpe=r.mean() / r.std() * np.sqrt(252), 最差年=yr.min()), yr


ix_r = ix.pct_change()
for a, b in (('2005-01-03', '2016-06-30'),):
    rows, YR = [], {}
    for vk in ('凍結版', '波動收斂版'):
        paths = []
        for o in range(20):
            r_, nt = run(X['RK'][vk], a, b, off=o % 5, seed=o); paths.append(r_)
        S = pd.DataFrame([stat(p)[0] for p in paths]).median()
        rows.append(dict(版本=vk, 年化=f'{S.年化:.1%}', 回撤=f'{S.回撤:.1%}', Sharpe=f'{S.Sharpe:.2f}', 最差年=f'{S.最差年:.1%}', 每年交易=int(nt / ((pd.Timestamp(b) - pd.Timestamp(a)).days / 365.25))))
        for p1, p2 in (('2005', '2009'), ('2010', '2016')):
            s_ = pd.DataFrame([stat(p.loc[p1:p2])[0] for p in paths]).median(); rows[-1][f'{p1}-{p2[2:]}'] = f'{s_.年化:.0%}/{s_.回撤:.0%}/{s_.Sharpe:.2f}'
        m = pd.concat(paths, axis=1).mean(axis=1); YR[vk] = stat(m)[1]
    YR['加權指數'] = stat(ix_r.loc[a:b].fillna(0))[1]
    pd.set_option('display.width', 250)
    print(pd.DataFrame(rows).to_string(index=False))
    Y = pd.DataFrame(YR); Y.index = Y.index.year; print('\n逐年報酬（20 路徑平均）'); print((Y * 100).round(1).to_string())
