"""台股短線事件策略 v2（finlab 單檔可執行）：凍結版 vs. 波動收斂版，保留期＋開發期一次跑完

選股池：4 位數普通股、上市滿 250 根、20 日均成交額 ≥ 全市場 20 分位
強勢股：T_ld = (EMA12 − EMA200) / max(ATR14, 全市場 ATR% 5 分位 × 股價)；
        所屬產業 T_ld 中位排名在前 50%；通過後 T_ld 前 10%；近 5 日無一字跌停
事件（當天收盤後判斷，任一成立）：
  事件1 布林收斂後首次突破上軌：近 10 日內帶寬（4σ÷中軌）曾 ≤ 自身前 120 日 20 分位，今天收盤首次站上上軌（20 日 2σ）
  事件2 整理後突破 20 日高：昨天為止 20 日高低區間 ÷ ATR ≤ 自身前 250 日 30 分位，今天收盤 > 前 20 日最高價
大盤風控 IAKF(250/2.2)：加權指數出現向下異常後 5 個交易日內，不買新股、持股全部於隔日開盤賣出
波動收斂版（候選）：另外要求事件當天 max(ATR14, 下限) < 20 個交易日前的同一數值，否則不買
進場：隔日開盤（開盤漲停／一字漲停買不到），同一天多檔依 T_ld 由高到低
出場：固定 10 檔、每檔 10%，持有 20 個交易日後隔日開盤賣出（一字跌停當天賣不掉、順延）；不設停損
成本：手續費 0.085%／邊、證交稅 0.3%
保留期 2010–2015：不得依結果回頭調參數（此樣本先前已使用三次，本次為第四次）
"""
import time
import numpy as np, pandas as pd

DATA_START = '2008-06-01'
PERIODS = {'保留期2010-2015': ('2010-01-04', '2015-12-31'), '開發期2016起': ('2016-02-01', None)}
N_SLOTS, HOLD, FEE, TAX = 10, 20, 0.00085, 0.003
IAKF_SMOOTH, IAKF_NLOOK, IAKF_THR, IAKF_REJ, IAKF_WIN = 250, 10, 2.2, 0.05, 5
ATR_LOOKBACK = 20
UPLOAD = True
# 自建引擎（2016 起，20 路徑中位數）：年化 / 最大回撤 / Sharpe
LOCAL = {'凍結版': (36.9, -25.1, 1.99), '波動收斂版': (31.8, -10.2, 2.38)}


def load():
    from finlab import data
    raw = data.get('price:收盤價'); adj = data.get('etl:adj_close')
    hi = data.get('price:最高價'); lo = data.get('price:最低價'); op = data.get('price:開盤價')
    vol = data.get('price:成交股數')
    sc = data.get('security_categories')
    sc = sc.set_index('stock_id') if 'stock_id' in sc.columns else sc
    bad = set(sc[sc['category'].astype(str).str.contains('存託憑證|ETF|ETN|受益', regex=True)].index.astype(str))
    cols = [c for c in raw.columns if len(str(c)) == 4 and str(c).isdigit() and not str(c).startswith('0')
            and str(c) not in bad and all(c in x.columns for x in (adj, hi, lo, vol, op))]
    f = lambda x: x.reindex(columns=cols).loc[DATA_START:].astype(float)
    raw, adj, hi, lo, op, vol = map(f, (raw, adj, hi, lo, op, vol))
    hi = np.fmax(np.fmax(hi, op), raw); lo = np.fmin(np.fmin(lo, op), raw)
    k = adj / raw
    ix = None
    for key, col in (('stock_index_price:收盤指數', '上市發行量加權股價指數'),
                     ('benchmark_return:發行量加權股價報酬指數', None)):
        try:
            s = data.get(key)
            s = (s[col] if col else (s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s)).astype(float).dropna()
            print(f'大盤資料 {key} 起始 {s.index[0].date()}', flush=True)
            if s.index[0] <= pd.Timestamp(DATA_START) + pd.Timedelta(days=30):
                ix = s.loc[DATA_START:]; break
        except Exception as e:  # noqa: BLE001
            print(f'大盤資料 {key} 讀取失敗：{e}', flush=True)
    if ix is None:
        raise SystemExit('找不到夠早的大盤序列，請回報。')
    try:
        ind = data.get('director_compensation:產業類別').ffill().iloc[-1]
    except Exception as e:  # noqa: BLE001
        print('產業資料讀取失敗，族群過濾停用：', e); ind = pd.Series(dtype=object)
    return dict(O=op * k, H=hi * k, L=lo * k, C=adj, RC=raw, RH=hi, RL=lo, ROP=op, V=vol,
                ix=ix.reindex(adj.index).ffill(), ind=ind.reindex(cols))


def fix_jumps(D, th=0.11):
    """漲跌幅上限 10%：上市 5 天後還有 >11% 的單日還原價變動視為資料問題，報酬歸零並向前調整"""
    C = D['C']; Cf = C.ffill(); age = C.notna().cumsum()
    r = C / Cf.shift(1) - 1
    bad = (r.abs() > th) & (age > 5) & C.notna()
    print(f'超過漲跌幅的單日跳動：{int(bad.sum().sum())} 筆（已修正）', flush=True)
    adj = (1 + r.where(bad, 0)).iloc[::-1].cumprod().iloc[::-1].shift(-1).fillna(1.0)
    for k in ('O', 'H', 'L', 'C'): D[k] = D[k] * adj
    return D


def iakf_down(x, smooth=IAKF_SMOOTH, nlook=IAKF_NLOOK, thr=IAKF_THR, rej=IAKF_REJ, decay=0.9):
    """IAKF（Innovation Adaptive Kalman Filter）單一序列版：回傳每天是否出現向下異常"""
    p_ = x.values.astype(float); ratio = (2.0 / (smooth + 1)) ** 2; a = 2.0 / (nlook + 1); thr2 = thr ** 2
    dn = np.zeros(len(p_), bool); started = False
    for t, p in enumerate(p_):
        if np.isnan(p): continue
        if not started:
            est, S, mS, R, calm, ev = p, 1e-4, 0.0, 1e-4, 1e-4, 1e-4; Q = 1e-4 * ratio; started = True
        pv = ev + Q; sur = p - est; cs = sur - mS
        an = cs ** 2 / S > thr2
        mS = (1 - a) * mS + a * sur; S = (1 - a) * S + a * cs ** 2
        R = rej * cs ** 2 + R if an else (1 - decay) * S + decay * R
        calm = (1 - decay) * S + decay * calm
        K = pv / (pv + R); est = est + K * sur; ev = (1 - K) * pv; Q = S * ratio
        dn[t] = an and cs < 0
    return pd.Series(dn, index=x.index)


def prepare(D):
    C, H, L, O = D['C'], D['H'], D['L'], D['O']
    RC, RH, RL, ROP = D['RC'], D['RH'], D['RL'], D['ROP']
    pc = C.shift(1)
    tr = np.maximum(H - L, np.maximum((H - pc).abs(), (L - pc).abs()))
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    e12 = C.ewm(span=12, adjust=False).mean(); e200 = C.ewm(span=200, adjust=False).mean()
    amt20 = (RC * D['V']).rolling(20, min_periods=15).mean()
    univ = C.notna() & (C.notna().cumsum() >= 250) & amt20.ge(amt20.quantile(0.20, axis=1), axis=0)
    floor = (atr / C).quantile(0.05, axis=1)
    den = np.maximum(atr, C.mul(floor, axis=0))
    score = ((e12 - e200) / den).where(univ)
    if D['ind'].notna().any():
        known = D['ind'].notna().values
        g = score.loc[:, known].T.groupby(D['ind'][known].values).transform('median').T.reindex(columns=C.columns)
        gp = g.where(univ).rank(axis=1, pct=True)
        score = score.where((gp >= 0.5) | gp.isna())
    strong = score.rank(axis=1, pct=True) >= 0.9
    lim = pd.Series(np.where(C.index < pd.Timestamp('2015-06-01'), 0.07, 0.10), index=C.index) - 0.005
    r1 = RC / RC.ffill().shift(1) - 1
    one = (RH == RL) & RC.notna()
    lockdn = one & r1.le(-lim, axis=0)
    lockup = one & r1.ge(lim, axis=0)
    ldn5 = lockdn.rolling(5, min_periods=1).sum() > 0
    nxt_open_up = (ROP.shift(-1) / RC - 1).ge(lim.shift(-1), axis=0)
    buy_ok = ~nxt_open_up & O.shift(-1).notna()
    # 事件
    m20 = C.rolling(20).mean(); sd20 = C.rolling(20).std(); up = m20 + 2 * sd20; bw = 4 * sd20 / m20
    squeeze = bw.rolling(10).min().shift(1) <= bw.rolling(120, min_periods=100).quantile(0.2).shift(1)
    ev1 = squeeze & (C > up) & (C.shift(1) <= up.shift(1))
    rng20 = (H.rolling(20).max() - L.rolling(20).min()) / atr
    tight = rng20.shift(1) <= rng20.rolling(250, min_periods=150).quantile(0.3).shift(1)
    ev2 = tight & (C > H.rolling(20).max().shift(1))
    # 大盤風控
    off = iakf_down(D['ix'].dropna()).reindex(C.index).fillna(False)
    off = off.rolling(IAKF_WIN, min_periods=1).max().astype(bool)
    print(f'IAKF 風險期占比：{off.loc["2010":].mean():.1%}', flush=True)
    OFF = pd.DataFrame(np.repeat(off.values[:, None], C.shape[1], 1), index=C.index, columns=C.columns)
    base_ev = strong & ~ldn5 & (ev1 | ev2) & ~OFF & buy_ok
    atr_ok = den < den.shift(ATR_LOOKBACK)
    rk = lambda ev: score.where(ev).rank(axis=1, ascending=False, method='first').values
    return dict(dates=C.index, cols=C.columns, O=O.values, LD=lockdn.values, LU=lockup.values, OFF=off.values,
                RK={'凍結版': rk(base_ev), '波動收斂版': rk(base_ev & atr_ok)})


def build_position(X, RKs, start, end=None, n=N_SLOTS, hold_days=HOLD):
    """逐日決定持股（訊號日收盤決定、隔日開盤成交），回傳 finlab 用的 position（權重）"""
    dates, cols = X['dates'], X['cols']
    T, M = len(dates), len(cols)
    LD, LU, Ov, OFF = X['LD'], X['LU'], X['O'], X['OFF']
    s0 = dates.get_indexer([pd.Timestamp(start)], method='bfill')[0]
    s1 = T if end is None else dates.get_indexer([pd.Timestamp(end)], method='ffill')[0] + 1
    out = np.zeros((T, M), np.float32); w = 1.0 / n
    hold = {}                                       # 股票 → 訊號日
    for t in range(s0, s1):
        nxt = t + 1 < T
        for j in [j for j, s in hold.items() if t - s >= hold_days or (OFF[t] and t - s >= 2)]:
            if nxt and LD[t + 1, j]: continue       # 隔天一字跌停賣不掉，順延
            del hold[j]
        free = n - len(hold)
        if free > 0 and not OFF[t]:
            cand = np.where(~np.isnan(RKs[t]))[0]
            for j in cand[np.argsort(RKs[t, cand])]:
                if free <= 0: break
                if j in hold: continue
                if nxt and (LU[t + 1, j] or np.isnan(Ov[t + 1, j])): continue
                hold[j] = t; free -= 1
        for j in hold: out[t, j] = w
    pos = pd.DataFrame(out, index=dates, columns=cols).iloc[s0:s1]
    return pos.loc[:, pos.any()]


def yearly(report):
    try:
        cr = report.creturn
        return (cr.resample('YE').last().pct_change().fillna(cr.resample('YE').last().iloc[0] / cr.iloc[0] - 1) * 100).round(1)
    except Exception:  # noqa: BLE001
        return None


t0 = time.time()
D = fix_jumps(load())
X = prepare(D)
print(f'準備完成 {time.time() - t0:.0f} 秒', flush=True)
from finlab.backtest import sim
rows, yrs = [], {}
for pk, (a, b) in PERIODS.items():
    for vk in ('凍結版', '波動收斂版'):
        pos = build_position(X, X['RK'][vk], a, b)
        name = f'短線事件v2_{vk}_{pk}'
        try:
            rep = sim(pos, trade_at_price='open', fee_ratio=FEE, tax_ratio=TAX, name=name, upload=UPLOAD)
        except TypeError:
            rep = sim(pos, fee_ratio=FEE, tax_ratio=TAX, name=name, upload=UPLOAD)
        m = rep.get_metrics()
        rows.append({'期間': pk, '版本': vk, '年化%': round(100 * float(m['profitability']['annualReturn']), 1),
                     '最大回撤%': round(100 * float(m['risk']['maxDrawdown']), 1), 'Sharpe': round(float(m['ratio']['sharpeRatio']), 2),
                     '平均持股': round(float((pos > 0).sum(1).mean()), 1), '總交易日': len(pos)})
        yrs[f'{vk}|{pk}'] = yearly(rep)
        print(rows[-1], flush=True)
print('\n====== 結果 ======')
print(pd.DataFrame(rows).to_string(index=False))
Y = pd.concat({k: v for k, v in yrs.items() if v is not None}, axis=1)
if len(Y): Y.index = Y.index.year; print('\n逐年報酬 %'); print(Y.to_string())
print('\n自建引擎（2016 起、20 路徑中位數）參考：', LOCAL)
