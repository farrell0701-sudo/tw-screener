"""v4 指標「跳過缺日」精確版：每檔股票只用自己有成交的日子計算時間序列指標
（ATR、EMA、布林、20 日區間／高點、自身分位、20 日前比較、成交額均值）；
跨股票的排名、門檻、大盤條件與交易規則不變。另加規則修正 B（停損依進場時多空）。
用法：python skip.py old|new → /home/claude/old/sk_{which}.pkl"""
import sys, numpy as np, pandas as pd
exec(open('/home/claude/old/hs.py').read().split('SAVE = {')[0])   # 原始資料與 v4 全部變數（未補值）
C0, H0, L0 = D['C'], D['H'], D['L']
RCV = (D['RC'] * D['V'])
cols = C0.columns; idx = C0.index; T_, N_ = C0.shape
names = ['atr', 'e12', 'e200', 'amt20', 'up', 'upprev', 'cprev', 'bwmin', 'bwq', 'rng', 'rngq', 'hh20', 'atr20ago_raw']
P = {k: np.full((T_, N_), np.nan) for k in names}
for j, c in enumerate(cols):
    ok = C0[c].notna().values
    if ok.sum() < 30: continue
    ii = np.where(ok)[0]
    Cs = C0[c].values[ii]; Hs = H0[c].values[ii]; Ls = L0[c].values[ii]
    s = pd.DataFrame({'C': Cs, 'H': Hs, 'L': Ls, 'A': RCV[c].values[ii]})
    pc = s.C.shift(1)
    tr = np.maximum(s.H - s.L, np.maximum((s.H - pc).abs(), (s.L - pc).abs()))
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    m20 = s.C.rolling(20).mean(); sd20 = s.C.rolling(20).std(); up = m20 + 2 * sd20; bw = 4 * sd20 / m20
    rng = (s.H.rolling(20).max() - s.L.rolling(20).min()) / atr
    vals = {'atr': atr, 'e12': s.C.ewm(span=12, adjust=False).mean(), 'e200': s.C.ewm(span=200, adjust=False).mean(),
            'amt20': s.A.rolling(20, min_periods=15).mean(), 'up': up, 'upprev': up.shift(1), 'cprev': pc,
            'bwmin': bw.rolling(10).min().shift(1), 'bwq': bw.rolling(120, min_periods=100).quantile(0.2).shift(1),
            'rng': rng, 'rngq': rng.rolling(250, min_periods=150).quantile(0.3).shift(1),
            'hh20': s.H.rolling(20).max().shift(1), 'atr20ago_raw': atr.shift(20)}
    for k, v in vals.items(): P[k][ii, j] = v.values
    if j % 400 == 0: print('per-stock', j, flush=True)
F = {k: pd.DataFrame(v, index=idx, columns=cols) for k, v in P.items()}
# ---- 與 prepare / bt14 相同的橫斷面運算 ----
atr = F['atr']; floor = (atr / C0).quantile(0.05, axis=1)
den = np.maximum(atr, C0.mul(floor, axis=0))
# 20 個「自身交易日」前的 den：用 20 日前 ATR 與當時 floor
den_prev = pd.DataFrame(np.nan, index=idx, columns=cols)
for j, c in enumerate(cols):
    ok = C0[c].notna().values; ii = np.where(ok)[0]
    if len(ii) < 21: continue
    dv = den[c].values[ii]; den_prev.values[ii[20:], j] = dv[:-20]
univ = C0.notna() & (C0.notna().cumsum() >= 250) & F['amt20'].ge(F['amt20'].quantile(0.20, axis=1), axis=0)
score = ((F['e12'] - F['e200']) / den).where(univ)
ind = D['ind'].reindex(cols); known = ind.notna().values
g = score.loc[:, known].T.groupby(ind[known].values).transform('median').T.reindex(columns=cols)
score = score.where((g.where(univ).rank(axis=1, pct=True) >= 0.5) | g.isna())
strong = score.rank(axis=1, pct=True) >= 0.9
lim = pd.Series(np.where(idx < pd.Timestamp('2015-06-01'), 0.07, 0.10), index=idx) - 0.005
RC_, RH_, RL_, ROP_ = D['RC'], D['RH'], D['RL'], D['ROP']
r1 = RC_ / RC_.ffill().shift(1) - 1; one = (RH_ == RL_) & RC_.notna()
lockdn = one & r1.le(-lim, axis=0); ldn5k = lockdn.rolling(5, min_periods=1).sum() > 0
buy_ok = ~(ROP_.shift(-1) / RC_ - 1).ge(lim.shift(-1), axis=0) & D['O'].shift(-1).notna()
ev1 = (F['bwmin'] <= F['bwq']) & (C0 > F['up']) & (F['cprev'] <= F['upprev'])
ev2 = (F['rng'].groupby(level=0, axis=0).first() if False else None)
tight = pd.DataFrame(np.nan, index=idx, columns=cols)
# tight 使用「前一個自身交易日」的區間比值 ≤ 前一日門檻（與原式 rng.shift(1) <= rngq 一致，rngq 已含 shift）
rprev = pd.DataFrame(np.nan, index=idx, columns=cols)
for j, c in enumerate(cols):
    ok = C0[c].notna().values; ii = np.where(ok)[0]
    if len(ii) < 2: continue
    rv = F['rng'][c].values[ii]; rprev.values[ii[1:], j] = rv[:-1]
tight = rprev <= F['rngq']
ev2 = tight & (C0 > F['hh20'])
OFFdf = pd.DataFrame(np.repeat(X['OFF'][:, None], N_, 1), index=idx, columns=cols)
base_ev = strong & ~ldn5k & (ev1 | ev2) & ~OFFdf & buy_ok
RK0s = score.where(base_ev).rank(axis=1, ascending=False, method='first').values
stk_up_s = (den >= den_prev).values
bull_ = ~br_[:, None]; bear_ = br_[:, None]
RKs = np.where((bull_ & (~stk_up_s | rl[:, None])) | (bear_ & ~mvup[:, None]), RK0s, np.nan)
e_old, e_new = ~np.isnan(RKv4), ~np.isnan(RKs)
print(f'事件數：原始 {e_old.sum()}、跳過缺日 {e_new.sum()}；兩者共同 {(e_old & e_new).sum()}；只在原始 {(e_old & ~e_new).sum()}；只在跳過版 {(~e_old & e_new).sum()}', flush=True)
def runB(RK, start, end, off=0, drop=0.1, seed=0, n=N_SLOTS, Hd=HOLD, stop=0.05, entry_regime=True):
    rng_ = np.random.default_rng(seed)
    s0 = dates.get_indexer([pd.Timestamp(start)], method='bfill')[0] + off
    s1 = dates.get_indexer([pd.Timestamp(end)], method='ffill')[0]
    cash = 1.0; pos = {}; sell, buy = set(), []; eq = np.full(len(dates), np.nan)
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
                pos[j] = [amt * (1 - FEE) / px, t, bool(MD[t - 1]), px]; cash -= amt
        sell, buy = keep, []
        eq[t] = cash + sum(p[0] * Cf[t, j] for j, p in pos.items())
        for j, p in pos.items():
            if j in sell: continue
            reg = (MD[t] or p[2]) if entry_regime else MD[t]
            stp = reg and Cf[t, j] < p[3] * (1 - stop)
            if t - p[1] + 1 >= Hd or (OFF[t] and t - p[1] >= 1) or stp: sell.add(j)
        free = n - len(pos) + len(sell & set(pos))
        if free > 0 and not OFF[t]:
            c = np.where(~np.isnan(RK[t]))[0]
            if drop: c = c[rng_.random(len(c)) >= drop]
            buy = [j for j in c[np.argsort(RK[t, c])] if j not in pos][:free]
    e = pd.Series(eq, index=dates).dropna()
    return e.pct_change().fillna(0)
SAVE = {'v4': pd.read_pickle(f'/home/claude/old/r14_{which}.pkl')['v4']}
SAVE['跳過缺日'] = [runB(RKs, a, b, off=o % 5, seed=o, entry_regime=False) for o in range(20)]; print('skip', flush=True)
SAVE['跳過缺日＋B（v4.1）'] = [runB(RKs, a, b, off=o % 5, seed=o, entry_regime=True) for o in range(20)]; print('skip+B', flush=True)
SAVE['原始＋B'] = [runB(RKv4, a, b, off=o % 5, seed=o, entry_regime=True) for o in range(20)]; print('orig+B', flush=True)
pd.to_pickle(SAVE, f'/home/claude/old/sk_{which}.pkl')
np.save(f'/home/claude/old/RKs_{which}.npy', RKs)
