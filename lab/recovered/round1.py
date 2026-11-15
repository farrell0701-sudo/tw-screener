"""第一輪：H-001 ~ H-008（登記內容見 registry.md）
用法：python round1.py old|new   → /home/claude/old/r1_{which}.pkl"""
import sys, numpy as np, pandas as pd
exec(open('/home/claude/old/hs.py').read().split('SAVE = {')[0])   # 載入資料、v4（RKv4, MD=br_, rl）、run2
V_ = D['V'].reindex_like(Cc).astype(float)
cand = ~np.isnan(RKv4)
def cmed(X):                       # 每日：v4 候選中的中位數
    Y = np.where(cand, X, np.nan)
    with np.errstate(all='ignore'): return np.nanmedian(Y, axis=1, keepdims=True)
pri = lambda M: np.where(cand, RKv4 - np.nan_to_num(M).astype(bool) * 1e6, np.nan)
flt = lambda M: np.where(cand & np.nan_to_num(M).astype(bool), RKv4, np.nan)
VAR = {'v4': RKv4}
# H-001 量能結構
v20 = V_.rolling(20, min_periods=15).mean().shift(1); v250 = V_.rolling(250, min_periods=200).mean().shift(1)
dry = (v20 / v250 < 1).values; burst = (V_ / v20 > 1).values
VAR['H1a 量縮且量增'] = flt(dry & burst); VAR['H1b 只看整理量縮'] = flt(dry); VAR['H1c 量縮量增優先'] = pri(dry & burst)
# H-002 族群共振
m20 = Cc.rolling(20).mean(); sd20 = Cc.rolling(20).std(); upb = m20 + 2 * sd20; bw = 4 * sd20 / m20
sq = bw.rolling(10).min().shift(1) <= bw.rolling(120, min_periods=100).quantile(0.2).shift(1)
ev1 = sq & (Cc > upb) & (Cc.shift(1) <= upb.shift(1))
rng20 = (Hc.rolling(20).max() - Lc.rolling(20).min()) / atr_
tight = rng20.shift(1) <= rng20.rolling(250, min_periods=150).quantile(0.3).shift(1)
ev2 = tight & (Cc > Hc.rolling(20).max().shift(1))
Eall = ((ev1 | ev2) & univ).astype(float)
had = (Eall.rolling(20, min_periods=1).max() > 0) & univ
ind = D['ind'].reindex(Cc.columns)
known = ind.notna().values
sh = had.loc[:, known].T.groupby(ind[known].values).mean().T            # 各產業：近 20 日有事件的比例
cnt = univ.loc[:, known].T.groupby(ind[known].values).sum().T
sh = sh.where(cnt >= 2)                                                   # 單一檔的產業無從共振
ind_med = sh.median(axis=1)
res_ind = sh.gt(ind_med, axis=0)
res = pd.DataFrame(False, index=Cc.index, columns=Cc.columns)
for g in res_ind.columns:
    cols = Cc.columns[known][ind[known].values == g]
    res.loc[:, cols] = np.repeat(res_ind[g].values[:, None], len(cols), 1)
res = res.values
VAR['H2a 只買族群共振'] = flt(res); VAR['H2b 族群共振優先'] = pri(res)
# H-003 250 日新高
nh = (Cc >= Hc.shift(1).rolling(250, min_periods=200).max()).values
VAR['H3a 只買250日新高'] = flt(nh); VAR['H3b 250日新高優先'] = pri(nh)
# ---- 籌碼（只在新資料）----
if which == 'new':
    def ld(pref, col):
        out = {}
        for c in Cc.columns:
            try:
                x = pd.read_csv(f'/home/claude/chip/{pref}_{c}.csv', parse_dates=['date']).set_index('date')
                out[c] = x[col] if isinstance(col, str) else x[col].sum(axis=1)
            except Exception: pass
        return pd.DataFrame(out).reindex(index=Cc.index, columns=Cc.columns).astype(float)
    inst = ld('inst', ['foreign', 'trust']); mgb = ld('mg', 'MarginPurchaseTodayBalance'); dtv = ld('dt', 'Volume')
    net = (inst.rolling(20, min_periods=15).sum() / V_.rolling(20, min_periods=15).sum()).values
    VAR['H6a 法人淨買超高於中位優先'] = pri(net > cmed(net)); VAR['H6b 只買法人淨買超'] = flt(net > 0)
    mgchg = (mgb / mgb.shift(20) - 1).values
    dtr = (dtv.rolling(5, min_periods=4).sum() / V_.rolling(5, min_periods=4).sum()).values
    VAR['H7a 融資增幅低優先'] = pri(mgchg < cmed(mgchg)); VAR['H7b 當沖比低優先'] = pri(dtr < cmed(dtr))
    VAR['H7c 剔除融資與當沖皆高'] = np.where(cand & ~((mgchg > cmed(mgchg)) & (dtr > cmed(dtr))), RKv4, np.nan)
    md = (mgb < mgb.shift(20)).values
    VAR['H8a 融資減少優先'] = pri(md); VAR['H8b 只買融資減少'] = flt(md)
    print('籌碼覆蓋率（候選中有融資資料）', f'{np.isfinite(np.where(cand, mgchg, np.nan)).sum() / cand.sum():.0%}', flush=True)
# ---- H-004 / H-005 需要改引擎 ----
atrp = (atr_ / Cc).values
strong_rank = sc_.rank(axis=1, pct=True).values >= 0.9
def run6(RK, start, end, off=0, drop=0.1, seed=0, n=N_SLOTS, Hd=HOLD, stop=0.05, wmode=None, xmode=None):
    rng = np.random.default_rng(seed)
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
            ref = np.nanmedian(np.where(cand[t - 1], atrp[t - 1], np.nan)) if wmode else None
            for j in buy:
                if j in pos or len(pos) >= n: continue
                px = Ov[t, j]
                if np.isnan(px) or LU[t, j]: continue
                k = 1.0
                if wmode and np.isfinite(atrp[t - 1, j]) and atrp[t - 1, j] > 0 and np.isfinite(ref): k = min(ref / atrp[t - 1, j], 2.0)
                amt = min(val / n * k, cash)
                if amt <= 0: break
                pos[j] = [amt * (1 - FEE) / px, t, bool(MD[t - 1]), px]; cash -= amt
        sell, buy = keep, []
        eq[t] = cash + sum(p[0] * Cf[t, j] for j, p in pos.items())
        for j, p in pos.items():
            if j in sell: continue
            hd = t - p[1] + 1
            stp = MD[t] and Cf[t, j] < p[3] * (1 - stop)
            if xmode == 'a': tx = not strong_rank[t, j]
            elif xmode == 'b': tx = (hd >= Hd and not strong_rank[t, j]) or hd >= 2 * Hd
            else: tx = hd >= Hd
            if tx or (OFF[t] and t - p[1] >= 1) or stp: sell.add(j)
        free = n - len(pos) + len(sell & set(pos))
        if free > 0 and not OFF[t]:
            c = np.where(~np.isnan(RK[t]))[0]
            if drop: c = c[rng.random(len(c)) >= drop]
            buy = [j for j in c[np.argsort(RK[t, c])] if j not in pos][:free]
    e = pd.Series(eq, index=dates).dropna()
    return e.pct_change().fillna(0)
SAVE = {}
for k, RK in VAR.items():
    SAVE[k] = [run2(RK, a, b, off=o % 5, seed=o)[0] for o in range(20)]; print(k, flush=True)
for k, kw in (('H4 風險均衡權重', dict(wmode=1)), ('H5a 跌出強勢池才出場', dict(xmode='a')), ('H5b 20日到期仍強勢則續抱', dict(xmode='b'))):
    SAVE[k] = [run6(RKv4, a, b, off=o % 5, seed=o, **kw) for o in range(20)]; print(k, flush=True)
pd.to_pickle(SAVE, f'/home/claude/old/r1_{which}.pkl')
print('done', flush=True)
