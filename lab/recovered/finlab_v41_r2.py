"""台股短線事件策略 v4.1 與 R2（finlab 單檔：同一份資料產生兩個持股表 → sim → 比較）

v4.1 ＝ v4 ＋ 兩項規則修正：
  1. 跳過缺日：所有時間序列指標（ATR、EMA、布林、區間、20 日高、自身分位、20 日前比較、成交額均值）
     只用個股「有成交」的日子計算（成交量 0 或無收盤價視為缺日）
  2. 空頭進場停損：進場當時為空頭模式的部位，整段持有期都套用 5% 停損（即使後來廣度爆發轉多頭）
R2 ＝ v4.1 ＋ 資金中性權重（研究中已判定為波動縮放、不列候選；此處僅作 finlab 資料的獨立驗證）
  左尾風險：MAX20（前 20 日最大單日報酬）與 IVOL120（前 120 日市場模型殘差標準差）在強勢池內百分位平均，前 1/3 → ×0.5
  右尾結構：三角收斂 W60a 突破 × 震盪箱 20（(20 日高−低)/(ATR×√20) 在流動性池百分位 ≤ 0.5）→ ×1.5
  單檔權重 ＝ 0.1 × 原始權重 ÷ 過去 250 日訊號平均原始權重（當日合計 > 100% 時等比例縮小）
其餘規則同 v4：T_ld 前 10%＋產業條件、事件 1／2、多空條件、廣度爆發、IAKF(250/10/2.2) 停止進場並清倉、
  10 格、持有 20 日、隔日開盤進場、手續費 0.085%、證交稅 0.3%
輸出：兩份 finlab 報告＋分期（2008–15、2016–）1 倍與 2.5 倍（多頭 2.5 倍／空頭 1 倍）績效比較
"""
import numpy as np
import pandas as pd
from finlab import data
from finlab.backtest import sim

# ───────────── 參數 ─────────────
DATA_START = '2007-01-01'
START = '2008-06-02'
N_SLOTS, HOLD = 10, 20
FEE, TAX = 0.00085, 0.003
BEAR_STOP = 0.05
ATR_LOOKBACK = 20
THRUST_UP, THRUST_LEAD, THRUST_DAYS = 0.05, 0.02, 20
IAKF_SMOOTH, IAKF_NLOOK, IAKF_THR, IAKF_REJ, IAKF_WIN = 250, 10, 2.2, 0.05, 5
R2_LO, R2_HI = 0.5, 1.5
LEV, LOAN_RATE = 2.5, 0.0645
UPLOAD = False
NAMES = ('短線事件v4.1', '短線事件v4.1_R2')

# ───────────── 資料 ─────────────
raw = data.get('price:收盤價'); adj = data.get('etl:adj_close')
hi = data.get('price:最高價'); lo = data.get('price:最低價'); op = data.get('price:開盤價'); vol = data.get('price:成交股數')
sc = data.get('security_categories')
sc = sc.set_index('stock_id') if 'stock_id' in sc.columns else sc
bad = set(sc[sc['category'].astype(str).str.contains('存託憑證|ETF|ETN|受益', regex=True)].index.astype(str))
cols = [c for c in raw.columns if len(str(c)) == 4 and str(c).isdigit() and not str(c).startswith('0')
        and str(c) not in bad and all(c in x.columns for x in (adj, hi, lo, vol, op))]
f = lambda x: x.reindex(columns=cols).loc[DATA_START:].astype(float)
RC, C, RH, RL, ROP, V = map(f, (raw, adj, hi, lo, op, vol))
RH = np.fmax(np.fmax(RH, ROP), RC); RL = np.fmin(np.fmin(RL, ROP), RC)
k = C / RC
O, H, L = ROP * k, RH * k, RL * k
r_ = C / C.ffill().shift(1) - 1
jump = (r_.abs() > 0.11) & (C.notna().cumsum() > 5) & C.notna()
fix = (1 + r_.where(jump, 0)).iloc[::-1].cumprod().iloc[::-1].shift(-1).fillna(1.0)
O, H, L, C = O * fix, H * fix, L * fix, C * fix
VALID = C.notna() & (V.fillna(0) > 0)                      # 有成交的日子
Cv_ = C.where(VALID)                                        # 缺日為 NaN（交易用的價格另用 ffill）

ix = None
for key, col in (('stock_index_price:收盤指數', '上市發行量加權股價指數'), ('benchmark_return:發行量加權股價報酬指數', None)):
    try:
        s = data.get(key)
        s = (s[col] if col else (s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s)).astype(float).dropna()
        if s.index[0] <= pd.Timestamp(DATA_START) + pd.Timedelta(days=30):
            ix = s.loc[DATA_START:]; print(f'大盤序列：{key}'); break
    except Exception as e:  # noqa: BLE001
        print(f'大盤資料 {key} 讀取失敗：{e}')
if ix is None:
    raise SystemExit('找不到夠早的大盤序列')
ix = ix.reindex(C.index).ffill()
try:
    ind = data.get('director_compensation:產業類別').ffill().iloc[-1].reindex(cols)
except Exception as e:  # noqa: BLE001
    ind = pd.Series(np.nan, index=cols, dtype=object)
    import os
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd(), 'tw_industry.csv')
    if os.path.exists(p):
        ind = pd.read_csv(p, dtype={'code': str}).set_index('code')['industry'].reindex(cols)
        print(f'finlab 產業資料讀不到（{e}），改用 tw_industry.csv：{ind.notna().sum()} 檔有產業')
    else:
        print('產業資料讀取失敗，且找不到 tw_industry.csv，產業過濾停用：', e)


# ───────────── 指標（跳過缺日）─────────────
def iakf_down(x, smooth, nlook, thr, rej, decay=0.9):
    p_ = x.values.astype(float); ratio = (2.0 / (smooth + 1)) ** 2; a = 2.0 / (nlook + 1); thr2 = thr ** 2
    dn = np.zeros(len(p_), bool); started = False
    for t, p in enumerate(p_):
        if np.isnan(p): continue
        if not started:
            est, S, mS, R, calm, ev = p, 1e-4, 0.0, 1e-4, 1e-4, 1e-4; Q = 1e-4 * ratio; started = True
        pv = ev + Q; sur = p - est; cs = sur - mS; an = cs ** 2 / S > thr2
        mS = (1 - a) * mS + a * sur; S = (1 - a) * S + a * cs ** 2
        R = rej * cs ** 2 + R if an else (1 - decay) * S + decay * R
        calm = (1 - decay) * S + decay * calm
        K = pv / (pv + R); est = est + K * sur; ev = (1 - K) * pv; Q = S * ratio
        dn[t] = an and cs < 0
    return pd.Series(dn, index=x.index)


T_, N_ = C.shape
NAMES_F = ['atr', 'e12', 'e200', 'amt20', 'up', 'upprev', 'cprev', 'bwmin', 'bwq', 'rng_prev', 'rngq', 'hh20', 'den_prev_atr',
           'max20', 'ivol', 'tri', 'bx20', 'n_valid']
FX = {k_: np.full((T_, N_), np.nan) for k_ in NAMES_F}
Cn, Hn, Ln, An, VV = C.values, H.values, L.values, (RC * V).values, VALID.values
ixv = ix.values
for j in range(N_):
    ii = np.where(VV[:, j])[0]
    if len(ii) < 30: continue
    s = pd.DataFrame({'C': Cn[ii, j], 'H': Hn[ii, j], 'L': Ln[ii, j], 'A': An[ii, j], 'M': ixv[ii]})
    pc = s.C.shift(1)
    tr = np.maximum(s.H - s.L, np.maximum((s.H - pc).abs(), (s.L - pc).abs()))
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    m20 = s.C.rolling(20).mean(); sd20 = s.C.rolling(20).std(); up = m20 + 2 * sd20; bw = 4 * sd20 / m20
    rng = (s.H.rolling(20).max() - s.L.rolling(20).min()) / atr
    r = s.C.pct_change(); m = s.M.pct_change()
    b_ = r.rolling(120, min_periods=100).cov(m) / m.rolling(120, min_periods=100).var()
    Hp, Lp, Tp = s.H.shift(1), s.L.shift(1), tr.shift(1); h = 30
    h2, l2, t2 = Hp.rolling(h).max(), Lp.rolling(h).min(), Tp.rolling(h).mean()
    vals = {'atr': atr, 'e12': s.C.ewm(span=12, adjust=False).mean(), 'e200': s.C.ewm(span=200, adjust=False).mean(),
            'amt20': s.A.rolling(20, min_periods=15).mean(), 'up': up, 'upprev': up.shift(1), 'cprev': pc,
            'bwmin': bw.rolling(10).min().shift(1), 'bwq': bw.rolling(120, min_periods=100).quantile(0.2).shift(1),
            'rng_prev': rng.shift(1), 'rngq': rng.rolling(250, min_periods=150).quantile(0.3).shift(1),
            'hh20': s.H.rolling(20).max().shift(1), 'den_prev_atr': atr.shift(ATR_LOOKBACK),
            'max20': r.rolling(20).max().shift(1), 'ivol': (r - b_ * m).rolling(120, min_periods=100).std().shift(1),
            'tri': ((h2 < h2.shift(h)) & (l2 > l2.shift(h)) & (t2 < t2.shift(h)) & (s.C > h2)).astype(float),
            'bx20': ((s.H.rolling(20).max() - s.L.rolling(20).min()) / (atr * np.sqrt(20))).shift(1),
            'n_valid': pd.Series(np.arange(1, len(ii) + 1), dtype=float)}
    for k_, v in vals.items(): FX[k_][ii, j] = v.values
F = {k_: pd.DataFrame(v, index=C.index, columns=C.columns) for k_, v in FX.items()}
print('個股指標（跳過缺日）完成', flush=True)

atr = F['atr']
floor = (atr / C).quantile(0.05, axis=1)
den = np.maximum(atr, C.mul(floor, axis=0))
dp = np.full((T_, N_), np.nan); dv = den.values                             # 20 個自身交易日前的 den
for j in range(N_):
    ii = np.where(VV[:, j])[0]
    if len(ii) > ATR_LOOKBACK: dp[ii[ATR_LOOKBACK:], j] = dv[ii[:-ATR_LOOKBACK], j]
den_prev = pd.DataFrame(dp, index=C.index, columns=C.columns)
univ = VALID & (F['n_valid'] >= 250) & F['amt20'].ge(F['amt20'].quantile(0.20, axis=1), axis=0)
score = ((F['e12'] - F['e200']) / den).where(univ)
if ind.notna().any():
    kn = ind.notna().values
    g = score.loc[:, kn].T.groupby(ind[kn].values).transform('median').T.reindex(columns=C.columns)
    score = score.where((g.where(univ).rank(axis=1, pct=True) >= 0.5) | g.isna())
lim = pd.Series(np.where(C.index < pd.Timestamp('2015-06-01'), 0.07, 0.10), index=C.index) - 0.005
r1 = RC / RC.ffill().shift(1) - 1
one = (RH == RL) & RC.notna()
lockdn = one & r1.le(-lim, axis=0); lockup = one & r1.ge(lim, axis=0)
strong = (score.rank(axis=1, pct=True) >= 0.9) & ~(lockdn.rolling(5, min_periods=1).sum() > 0)
buy_ok = ~(ROP.shift(-1) / RC - 1).ge(lim.shift(-1), axis=0) & O.shift(-1).notna()
ev1 = (F['bwmin'] <= F['bwq']) & (Cv_ > F['up']) & (F['cprev'] <= F['upprev'])
ev2 = (F['rng_prev'] <= F['rngq']) & (Cv_ > F['hh20'])
atr_ok = den < den_prev

bear200 = ix < ix.rolling(200).mean()
mv = ix.pct_change().rolling(20).std(); mkt_vol_down = mv < mv.shift(20)
ew = C.pct_change().where(univ.shift(1, fill_value=False)).mean(axis=1).fillna(0)
ratio_ew = (1 + ew).cumprod() / ix
thrust = (ix.pct_change(10) >= THRUST_UP) & (ratio_ew.pct_change(10) >= THRUST_LEAD)
thrust_win = thrust.rolling(THRUST_DAYS, min_periods=1).max().astype(bool)
bear_mode = bear200 & ~thrust_win
off = iakf_down(ix.dropna(), IAKF_SMOOTH, IAKF_NLOOK, IAKF_THR, IAKF_REJ).reindex(C.index).fillna(False)
off = off.rolling(IAKF_WIN, min_periods=1).max().astype(bool)

B = lambda s: pd.DataFrame(np.repeat(s.values[:, None], C.shape[1], 1), index=C.index, columns=C.columns)
allow = (B(~bear_mode) & (atr_ok | B(thrust_win))) | (B(bear_mode) & B(mkt_vol_down))
signal = strong & (ev1 | ev2) & allow & ~B(off) & buy_ok
rank = score.where(signal).rank(axis=1, ascending=False, method='first')

# R2 權重
SPm = strong & univ
pr = lambda X: X.where(SPm).rank(axis=1, pct=True)
risk = ((pr(F['max20']) + pr(F['ivol'])) / 2).rank(axis=1, pct=True).fillna(0) >= 2 / 3
right = (F['tri'] > 0.5) & (F['bx20'].where(univ).rank(axis=1, pct=True).fillna(9) <= 0.5)
w_raw = pd.DataFrame(np.where(risk, R2_LO, 1.0) * np.where(right, R2_HI, 1.0), index=C.index, columns=C.columns)
sw = w_raw.where(signal).sum(axis=1); cnt = signal.sum(axis=1)
norm = (sw.rolling(250, min_periods=60).sum() / cnt.rolling(250, min_periods=60).sum()).shift(1).fillna(1.0)
W_R2 = (w_raw.div(norm, axis=0) / N_SLOTS)
print(f'空頭模式 {bear_mode.loc[START:].mean():.0%}｜廣度爆發 {int(thrust.loc[START:].sum())} 天｜IAKF 風險期 {off.loc[START:].mean():.0%}｜'
      f'訊號中高風險 {(risk & signal).loc[START:].sum().sum() / signal.loc[START:].sum().sum():.0%}、右尾結構 {(right & signal).loc[START:].sum().sum() / signal.loc[START:].sum().sum():.0%}')


# ───────────── 逐日持股 ─────────────
def build_position(Wt=None):
    dates, T, M = C.index, *C.shape
    RK, LD, LU, Ov, Cv = rank.values, lockdn.values, lockup.values, O.values, C.ffill().values
    BM, OFF = bear_mode.values, off.values
    WV = None if Wt is None else Wt.values
    s0 = dates.get_indexer([pd.Timestamp(START)], method='bfill')[0]
    out = np.zeros((T, M), np.float32); hold = {}            # 股票 → (訊號日, 進場時空頭, 權重)
    for t in range(s0, T):
        nxt = t + 1 < T
        for j in list(hold):
            s, bm0, w = hold[j]; held = t - s
            entry = Ov[s + 1, j] if s + 1 < T else np.nan
            stop = (BM[t] or bm0) and held >= 1 and Cv[t, j] < entry * (1 - BEAR_STOP)     # 規則 B
            if held >= HOLD or (OFF[t] and held >= 2) or stop:
                if nxt and LD[t + 1, j]: continue
                del hold[j]
        free = N_SLOTS - len(hold)
        if free > 0 and not OFF[t]:
            cand = np.where(~np.isnan(RK[t]))[0]
            for j in cand[np.argsort(RK[t, cand])]:
                if free <= 0: break
                if j in hold or (nxt and (LU[t + 1, j] or np.isnan(Ov[t + 1, j]))): continue
                hold[j] = (t, bool(BM[t]), 1.0 / N_SLOTS if WV is None else float(WV[t, j])); free -= 1
        tot = sum(v[2] for v in hold.values()); sc_ = 1.0 / tot if tot > 1 else 1.0
        for j, v in hold.items(): out[t, j] = v[2] * sc_
    pos = pd.DataFrame(out, index=dates, columns=C.columns).iloc[s0:]
    return pos.loc[:, pos.any()]


def stats(cr, lo_, hi_):
    c = cr.loc[lo_:hi_]; r = c.pct_change().dropna(); y = (c.index[-1] - c.index[0]).days / 365.25
    return f'{(c.iloc[-1] / c.iloc[0]) ** (1 / y) - 1:.1%} / {(c / c.cummax() - 1).min():.1%} / {r.mean() / r.std() * np.sqrt(252):.2f}'


def lever(r, Lv=LEV, reset=20):
    br = bear200.reindex(r.index).fillna(False).shift(1).fillna(False).values
    eq = 1.0; loan = 0.0; out = []; prev = False; k_ = 0
    for i, x in enumerate(r.values):
        want = Lv > 1 and not br[i]
        if want and (not prev or k_ >= reset): loan = (Lv - 1) * eq; k_ = 0
        prev = want
        if not want: loan = 0.0
        asset = (eq + loan) * (1 + x); eq_new = asset - loan - loan * LOAN_RATE / 252
        if loan > 0 and asset / loan < 1.3: loan = 0.0; k_ = 0
        eq = eq_new; out.append(eq); k_ += 1
    return pd.Series(out, index=r.index)


reports = {}
for nm, Wt in zip(NAMES, (None, W_R2)):
    position = build_position(Wt)
    print(f'{nm}：{position.shape[0]} 天 × {position.shape[1]} 檔，平均持股 {(position > 0).sum(axis=1).mean():.1f} 檔、平均曝險 {position.sum(axis=1).mean():.1%}')
    reports[nm] = sim(position, resample=None, trade_at_price='open', fee_ratio=FEE, tax_ratio=TAX, name=nm, upload=UPLOAD)

print('\n年報酬 / 最大回撤 / Sharpe')
rows = []
for nm, rp in reports.items():
    cr = rp.creturn.dropna(); cr = cr[cr.index >= pd.Timestamp(START)]
    lv = lever(cr.pct_change().fillna(0))
    rows.append({'版本': nm, '2008–2015（1倍）': stats(cr, START, '2015-12-31'), '2016–（1倍）': stats(cr, '2016-01-01', None),
                 '全期 1倍': stats(cr, START, None), f'全期 {LEV}倍': stats(lv, START, None)})
print(pd.DataFrame(rows).to_string(index=False))
r0 = reports[NAMES[0]].creturn.pct_change().dropna(); r1_ = reports[NAMES[1]].creturn.pct_change().dropna()
d = (r1_ - r0).dropna(); d = d[d.index >= pd.Timestamp(START)]
print(f'R2 − v4.1 日報酬差：平均 {d.mean() * 252:+.2%}/年，日相關 {pd.concat([r0, r1_], axis=1).dropna().corr().iloc[0, 1]:.3f}')
