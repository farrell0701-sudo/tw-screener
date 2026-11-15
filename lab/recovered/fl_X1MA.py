# L-069：精簡版 X1 vs 高波動持股 MA20 出場 —— FinLab 外部資料驗證（只考一次）
# 用法：在已登入 FinLab 的 notebook 貼上整格執行，把最後印出的表格整段貼回。
import gc
for _n in [n for n in list(globals()) if not n.startswith('_') and n not in ('gc', 'In', 'Out', 'get_ipython', 'exit', 'quit')]:
    del globals()[_n]
gc.collect()
import numpy as np, pandas as pd
from finlab import data
FEE, TAX, SLIP, RATE, LEVX = 0.00085, 0.003, 0.0015, 0.0645, 2.5
MI = RATE / 252
f32 = np.float32
sc = data.get('security_categories'); sc = sc.set_index('stock_id') if 'stock_id' in sc.columns else sc
bad = set(sc[sc['category'].astype(str).str.contains('存託憑證|ETF|ETN|受益', regex=True)].index.astype(str)); del sc
def load(key, cols=None):
    x = data.get(key)
    if cols is None:
        cols = [c for c in x.columns if len(str(c)) == 4 and str(c).isdigit() and not str(c).startswith('0') and str(c) not in bad]
    y = x.reindex(columns=cols).astype(f32); del x; gc.collect(); return y, cols
RC, cols = load('price:收盤價')
C, _ = load('etl:adj_close', cols); RH, _ = load('price:最高價', cols); RL, _ = load('price:最低價', cols)
ROP, _ = load('price:開盤價', cols); V, _ = load('price:成交股數', cols)
RC, RH, RL, ROP, V = [x.reindex(C.index) for x in (RC, RH, RL, ROP, V)]
_cnt = pd.Series((pd.DataFrame(C).notna().values & (pd.DataFrame(V).fillna(0).values > 0)).sum(1), index=C.index)
_last = _cnt[_cnt >= 0.5 * _cnt[_cnt > 0].median()].index[-1]
C, RC, RH, RL, ROP, V = [x.loc[:_last] for x in (C, RC, RH, RL, ROP, V)]
RH = np.fmax(np.fmax(RH, ROP), RC); RL = np.fmin(np.fmin(RL, ROP), RC)
k_ = C / RC; O, H, L = ROP * k_, RH * k_, RL * k_; del k_
VALID = C.notna() & (V.fillna(0) > 0)
dates = C.index; T_, N_ = C.shape
print('資料', dates[0].date(), '～', dates[-1].date(), '｜股票', N_, flush=True)
def idx_series(key, col):
    s = pd.DataFrame(data.get(key)); s = s[col] if col in s.columns else s.iloc[:, 0]; s.index = pd.to_datetime(s.index); return s.astype(float).dropna()
best = None
for key, col in (('stock_index_price:收盤指數', '上市發行量加權股價指數'), ('benchmark_return:發行量加權股價報酬指數', None), ('taiex_total_index:收盤指數', None)):
    try:
        s_ = idx_series(key, col)
        if best is None or s_.index[0] < best[1].index[0]: best = (key, s_)
    except Exception as e:
        print('大盤資料讀取失敗', key, str(e)[:60])
ixc = best[1].reindex(dates).ffill()
if best[1].index[0] > dates[0] + pd.Timedelta(days=30):
    w_ = pd.DataFrame((RC * V).values, index=dates).shift(1).rolling(20, min_periods=5).mean()
    rr_ = pd.DataFrame((RC / RC.shift(1) - 1).values, index=dates).clip(-0.11, 0.11)
    pr_ = (rr_ * w_).sum(axis=1) / w_.where(rr_.notna()).sum(axis=1); s0_ = best[1].index[0]
    px_ = (1 + pr_.fillna(0)).cumprod(); px_ = px_ / px_.loc[px_.index[px_.index.get_indexer([s0_], method='bfill')[0]]] * best[1].iloc[0]
    ixc = ixc.where(ixc.index >= s0_, px_); del w_, rr_, pr_, px_
print('大盤序列', best[0], flush=True)
# ---- 個股指標（自身交易日）
ATR = np.full((T_, N_), np.nan, f32); E12 = ATR.copy(); E200 = ATR.copy(); AMT = ATR.copy(); R20 = ATR.copy(); MA20 = ATR.copy()
OLD = np.zeros((T_, N_), bool)
Cn, Hn, Ln, VV = C.values, H.values, L.values, VALID.values; An = (RC * V).values
for j in range(N_):
    ii = np.where(VV[:, j])[0]
    if len(ii) < 30: continue
    s = pd.DataFrame({'C': Cn[ii, j].astype(float), 'H': Hn[ii, j].astype(float), 'L': Ln[ii, j].astype(float), 'A': An[ii, j].astype(float)})
    pc = s.C.shift(1); tr = np.maximum(s.H - s.L, np.maximum((s.H - pc).abs(), (s.L - pc).abs()))
    ATR[ii, j] = tr.ewm(alpha=1 / 14, adjust=False).mean().values
    E12[ii, j] = s.C.ewm(span=12, adjust=False).mean().values; E200[ii, j] = s.C.ewm(span=200, adjust=False).mean().values
    AMT[ii, j] = s.A.rolling(20, min_periods=15).mean().values; R20[ii, j] = (s.C / s.C.shift(20) - 1).values
    MA20[ii, j] = s.C.rolling(20, min_periods=15).mean().values
    OLD[ii[249:], j] = True
del An; gc.collect(); print('個股指標完成', flush=True)
floor = np.nanquantile(np.where(VV, ATR / Cn, np.nan), 0.05, axis=1).astype(f32)
DEN = np.maximum(ATR, Cn * floor[:, None]); Tr = (E12 - E200) / DEN
amt_q = np.nanquantile(np.where(VV & OLD, AMT, np.nan), 0.20, axis=1)
U = VV & OLD & (AMT >= amt_q[:, None])
def prk(x, m): return pd.DataFrame(np.where(m, x, np.nan)).rank(axis=1, pct=True).values
pool = U & (np.nan_to_num(prk(Tr, U)) >= 0.9)
hot = np.nan_to_num(prk(R20, pool)) > 0.8
ATRP = ATR / Cn
Wv = np.nan_to_num(np.clip(np.nanmedian(np.where(pool, ATRP, np.nan), 1)[:, None] / ATRP, 0.5, 2.0), nan=1.0)
TL = pd.DataFrame(Tr, index=dates)
SC = 0.5 * np.nan_to_num(prk(Tr, pool), nan=0) + 0.5 * np.nan_to_num(prk((TL - TL.shift(10)).values, pool), nan=0)
Sx = pool & ~hot
HV = np.nan_to_num(prk(ATRP, Sx), nan=0) >= 2 / 3
below = (ixc < ixc.rolling(200).mean()).values
bullv = ~below
lim = np.where(dates < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None] - 0.005
r1 = (RC / RC.ffill().shift(1) - 1).values; one = ((RH == RL) & RC.notna()).values
LD = one & (r1 <= -lim); LU = one & (r1 >= lim); del r1, one
Ov, Cf = O.values * (1 + 0), C.ffill().values
del H, L, RH, RL, ROP, V, ATR, E12, E200, AMT, R20; gc.collect()
def rk_by(sc_, m): return pd.DataFrame(np.where(m, sc_, np.nan)).rank(axis=1, ascending=False, method='first').values
RK = rk_by(SC, Sx)
START = max(pd.Timestamp('2008-06-02'), dates[300])
def runM(off=0, seed=0, lev=True, rule=None, n=15, drop=0.2, cap=250):
    rng = np.random.default_rng(seed)
    s0 = dates.get_indexer([START], method='bfill')[0] + off
    cash = 1.0; pos = {}; sell, buy = set(), []; eq = np.full(T_, np.nan); force = False
    for t in range(s0, T_):
        keep = set()
        if force: sell |= {j for j, p in pos.items() if p[5] > 0}; force = False
        for j in sell:
            if j not in pos: continue
            if LD[t, j]: keep.add(j); continue
            px = (Ov[t, j] if not np.isnan(Ov[t, j]) else Cf[t, j]) * (1 - SLIP)
            p = pos.pop(j); cash += p[0] * px * (1 - FEE - TAX) - p[5]
        if buy:
            E = cash + sum(p[0] * Cf[t - 1, j] - p[5] for j, p in pos.items())
            for j in buy:
                if j in pos or len(pos) >= n: continue
                px = Ov[t, j]
                if np.isnan(px) or LU[t, j]: continue
                px = px * (1 + SLIP)
                own = min(Wv[t - 1, j] * E / n, cash)
                if own <= 0: break
                Lx = LEVX if (lev and bullv[t - 1]) else 1.0
                val = own * Lx; loan = val - own
                d = ATRP[t - 1, j] * 4; d = 0.04 if not np.isfinite(d) else min(max(d, 0.04), 0.15)
                pos[j] = [val * (1 - FEE) / px, t, px, d, px, loan, bool(HV[t - 1, j])]; cash -= own
        sell, buy = keep, []
        loans = sum(p[5] for p in pos.values()); cash -= loans * MI
        Vt = sum(p[0] * Cf[t, j] for j, p in pos.items()); eq[t] = cash + Vt - loans
        fv = sum(p[0] * Cf[t, j] for j, p in pos.items() if p[5] > 0)
        if loans > 0 and fv / loans < 1.3: force = True
        for j, p in pos.items():
            c = Cf[t, j]
            if np.isnan(c): continue
            p[4] = max(p[4], c)
            ext = rule == 'MA' and p[6] and (t - p[1] >= 5) and np.isfinite(MA20[t, j]) and c < MA20[t, j]
            if c < p[4] * (1 - p[3]) or (p[4] >= p[2] * 1.06 and c < p[2] * 1.005) or (t - p[1] + 1 >= cap) or ext: sell.add(j)
        free = n - len(pos) + len(sell & set(pos))
        if free > 0 and not below[t]:
            cnd = np.where(~np.isnan(RK[t]))[0]
            if drop: cnd = cnd[rng.random(len(cnd)) >= drop]
            buy = [j for j in cnd[np.argsort(RK[t, cnd])] if j not in pos and j not in sell][:free]
    return pd.Series(eq, index=dates).dropna().pct_change().fillna(0)
def met(r):
    if len(r) < 60: return (np.nan, np.nan, np.nan)
    e = (1 + r).cumprod(); y = len(r) / 252; cg = e.iloc[-1] ** (1 / y) - 1; md = (e / e.cummax() - 1).min()
    return cg, md, cg / -md if md < 0 else np.nan
END = dates[-1]
SEG = (('2008/6–2015', START, pd.Timestamp('2015-12-31')), (f'2016–{END:%Y/%m}', pd.Timestamp('2016-01-01'), END))
RES = {}
for nm, rule in (('精簡版 X1', None), ('高波動 MA20 出場', 'MA')):
    for lev in (True, False):
        RES[(nm, lev)] = [runM(off=i, seed=1000 + i, lev=lev, rule=rule) for i in range(20)]
        print(nm, '槓桿' if lev else '1倍', '完成', flush=True); gc.collect()
rows = []
for (nm, lev), P in RES.items():
    for tag, lo, hi in SEG:
        M = np.array([met(p.loc[lo:hi]) for p in P])
        rows.append(dict(版本=nm, 模式='槓桿' if lev else '1倍', 期間=tag, 年化=f'{np.median(M[:, 0]):.1%}', 回撤=f'{np.median(M[:, 1]):.1%}', Calmar=round(float(np.median(M[:, 2])), 2)))
T = pd.DataFrame(rows); print(T.to_string(index=False))
print('--- 判準（槓桿 Calmar 兩段都高於精簡版 X1 才通過）')
ok = True
for tag, lo, hi in SEG:
    b = [met(p.loc[lo:hi])[2] for p in RES[('精簡版 X1', True)]]; v = [met(p.loc[lo:hi])[2] for p in RES[('高波動 MA20 出場', True)]]
    win = np.mean(np.array(v) > np.array(b)); ok &= np.median(v) > np.median(b)
    print(f'{tag}: 精簡版 {np.median(b):.2f} → MA20 {np.median(v):.2f} ｜ 同路徑勝出比例 {win:.0%}')
print('結論：', '通過' if ok else '不通過')
