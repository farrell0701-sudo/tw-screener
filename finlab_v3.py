"""T_ld 標準版 v3（finlab 交叉驗證，單檔可直接執行）

選股（每天收盤後計算，只用當天以前的資料）
- 股票池：4 位數普通股、上市滿 250 根、20 日均成交額 ≥ 當日全市場 20 分位
- 分數 T_ldf = (EMA12 − EMA200) / max(ATR14, 當日全市場 ATR% 的 5 分位 × 股價)
- 族群過濾：所屬產業 T_ldf 中位數在全市場前 50%（查不到產業視為通過）
- 【新】上漲速度過濾：速度 = log(120 日最高價 ÷ 120 日最低價) ÷ 低點到高點經過的天數
       （高點在低點之前 = 下跌，不算）；在強勢股（分數前 10%）中排除速度最快的 25%
- 【新】近 5 日出現過一字跌停（最高 = 最低、跌幅 ≥ 跌停 −0.5%）不買
- 候選池 = 通過過濾後分數前 K 名（K=30）

進出場（持股 10 檔、等權）
- 每天收盤有空位就從候選池挑排名最高、未持有的，隔日開盤買
  隔日開盤漲停或一字漲停買不到 → 跳過換下一檔
- 出場（先到者）：收盤 < 持有期最高收盤 − k×ATR14（多頭 k=4、空頭 k=3；多空 = 加權指數 EMA60 的 20 日斜率 > 0）
                  或 收盤 < 60 日密集區下緣（前 60 根最低價的中位數，到昨天為止）
  → 隔日開盤賣；【新】當天一字跌停賣不掉，順延到打開那天
- 每 20 個交易日：排名跌出前 2K（60 名）的持股賣出
- 漲跌停：2015-06-01 前 7%，之後 10%
- 成本：手續費 0.085%/邊、證交稅 0.3%；20 個起始日取中位數

HOLDOUT=True 跑 2010–2015 保留樣本；False 跑 2016 起開發期，可和自建引擎（20 個起始日）對照。
"""
import time
import numpy as np, pandas as pd

HOLDOUT = True
DATA_START, EVAL_START, EVAL_END = ('2008-06-01', '2010-01-04', '2015-12-31') if HOLDOUT else ('2014-06-01', '2016-02-01', None)
N, K, REB, FEE, TAX = 10, 30, 20, 0.00085, 0.003
OFFSETS = tuple(range(20))   # 20 個起始日；只取 5 個的雜訊太大（模擬測試時速度過濾 25% 和 15% 差了 5 個百分點）
# 自建引擎結果（年化 / 回撤 / Sharpe），對照用
LOCAL = {'基準（無速度過濾）': (42.9, -25.9, 1.86), '速度過濾 15%': (41.8, -25.6, 1.95),
         '速度過濾 25%（標準）': (41.5, -24.0, 2.02), '速度過濾 30%': (41.6, -23.3, 2.07)}


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
    f = lambda x: x.reindex(columns=cols).loc[DATA_START:EVAL_END].astype(float)
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
                ix = s.loc[DATA_START:EVAL_END]; break
        except Exception as e:  # noqa: BLE001
            print(f'大盤資料 {key} 讀取失敗：{e}', flush=True)
    if ix is None:
        raise SystemExit('找不到夠早的大盤序列，請回報。')
    try:
        ind = data.get('director_compensation:產業類別').ffill().iloc[-1]
    except Exception as e:  # noqa: BLE001
        print('產業資料讀取失敗，族群過濾停用：', e); ind = pd.Series(dtype=object)
    return dict(O=op * k, H=hi * k, L=lo * k, C=adj, RC=raw, RH=hi, RL=lo, ROP=op, V=vol, ix=ix.reindex(adj.index).ffill(),
                ind=ind.reindex(cols))


def fix_jumps(D, th=0.11):
    """漲跌幅上限 10%：上市 5 天後還有 >11% 的單日還原價變動視為資料問題，報酬歸零並向前調整"""
    C = D['C']; Cf = C.ffill(); age = C.notna().cumsum()
    r = C / Cf.shift(1) - 1
    bad = (r.abs() > th) & (age > 5) & C.notna()
    print(f'超過漲跌幅的單日跳動：{int(bad.sum().sum())} 筆（已修正）', flush=True)
    adj = (1 + r.where(bad, 0)).iloc[::-1].cumprod().iloc[::-1].shift(-1).fillna(1.0)
    for k in ('O', 'H', 'L', 'C'): D[k] = D[k] * adj
    return D



def days_since(X, ext, n=120):
    """窗內極值距今幾天；同值取最舊的那一次（等同 rolling argmax/argmin 的第一個位置）"""
    a = X.values; e = ext.values; out = np.full(a.shape, np.nan)
    for k in range(n):
        sh = np.full(a.shape, np.nan); sh[k:] = a[:len(a) - k] if k else a
        out = np.where(sh == e, k, out)
    return out


def prepare(D, speed_cut=0.25):
    C, H, L, O = D['C'], D['H'], D['L'], D['O']
    RC, RH, RL, ROP = D['RC'], D['RH'], D['RL'], D['ROP']
    pc = C.shift(1)
    tr = np.maximum(H - L, np.maximum((H - pc).abs(), (L - pc).abs()))
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    e12 = C.ewm(span=12, adjust=False).mean(); e200 = C.ewm(span=200, adjust=False).mean()
    amt20 = (RC * D['V']).rolling(20, min_periods=15).mean()
    univ = C.notna() & (C.notna().cumsum() >= 250) & amt20.ge(amt20.quantile(0.20, axis=1), axis=0)
    floor = (atr / C).quantile(0.05, axis=1)
    score = ((e12 - e200) / np.maximum(atr, C.mul(floor, axis=0))).where(univ)
    if D['ind'].notna().any():
        known = D['ind'].notna().values
        g = score.loc[:, known].T.groupby(D['ind'][known].values).transform('median').T.reindex(columns=C.columns)
        gp = g.where(univ).rank(axis=1, pct=True)
        score = score.where((gp >= 0.5) | gp.isna())
    # 漲跌停門檻（未還原價）
    lim = pd.Series(np.where(C.index < pd.Timestamp('2015-06-01'), 0.07, 0.10), index=C.index) - 0.005
    rprev = RC.ffill().shift(1)
    r1 = RC / rprev - 1
    one = (RH == RL) & RC.notna()
    lockdn = one & r1.le(-lim, axis=0)
    lockup = one & r1.ge(lim, axis=0)
    # 上漲速度
    strong = score.rank(axis=1, pct=True) >= 0.9
    HH = H.rolling(120, min_periods=100).max(); LL = L.rolling(120, min_periods=100).min()
    nd = pd.DataFrame(days_since(L, LL) - days_since(H, HH), index=C.index, columns=C.columns)   # 低點到高點的天數
    speed = (np.log(HH / LL) / nd).where(nd > 0)
    score = score.where(~(lockdn.rolling(5, min_periods=1).sum() > 0))
    if speed_cut:
        q = speed.where(strong).rank(axis=1, pct=True)
        score = score.where(~(q > 1 - speed_cut))
    nxt_open_up = (ROP.shift(-1) / RC - 1).ge(lim.shift(-1), axis=0)
    buy_ok = ~nxt_open_up & O.shift(-1).notna()
    medL = L.rolling(60, min_periods=60).median().shift(1)
    e60 = D['ix'].ewm(span=60, adjust=False).mean(); bull = ((e60 / e60.shift(20) - 1) > 0)
    return dict(dates=C.index, C=C.values, O=O.values, Cf=C.ffill().values, A=atr.values, ML=medL.values,
                LD=lockdn.values, LU=lockup.values,
                RK=score.where(buy_ok).rank(axis=1, ascending=False).values,
                RKall=score.rank(axis=1, ascending=False).values,
                BULL=bull.values, cols=C.columns)


def run(X, off, kb=4, kr=3):
    dates = X['dates']; start = dates.get_indexer([pd.Timestamp(EVAL_START)], method='bfill')[0]; end = len(dates) - 1
    Cv, Ov, Cfv, Av, ML, LD, LU, RK, RKall, BULL = (X[k] for k in ('C', 'O', 'Cf', 'A', 'ML', 'LD', 'LU', 'RK', 'RKall', 'BULL'))
    reb = set(range(start + off, end - 1, REB))
    cash = 1.0; pos = {}; sell_next, buy_next = {}, []
    eq = np.full(len(dates), np.nan); log = []
    for t in range(start, end + 1):
        keep = {}
        for j, why in sell_next.items():
            if j not in pos: continue
            if LD[t, j]: keep[j] = why; continue                 # 一字跌停賣不掉
            px = Ov[t, j] if not np.isnan(Ov[t, j]) else Cfv[t, j]
            p = pos.pop(j); cash += p[0] * px * (1 - FEE - TAX)
            log.append(dict(code=X['cols'][j], entry=dates[p[3]], exit=dates[t], why=why,
                            ret=px * (1 - FEE - TAX) / (p[1] / (1 - FEE)) - 1))
        if buy_next:
            val = cash + sum(p[0] * Cfv[t - 1, j] for j, p in pos.items())
            for j in buy_next:
                if j in pos or len(pos) >= N: continue
                px = Ov[t, j]
                if np.isnan(px) or LU[t, j]: continue
                amt = min(val / N, cash)
                if amt <= 0: break
                pos[j] = [amt * (1 - FEE) / px, px, px, t]; cash -= amt
        sell_next, buy_next = dict(keep), []
        eq[t] = cash + sum(p[0] * Cfv[t, j] for j, p in pos.items())
        k = kb if BULL[t] else kr
        for j, p in pos.items():
            c = Cv[t, j]
            if np.isnan(c) or j in sell_next: continue
            p[2] = max(p[2], c)
            if c < p[2] - k * Av[t, j]: sell_next[j] = 'ATR'
            elif c < ML[t, j]: sell_next[j] = '密集區下緣'
        if t in reb:
            for j in pos:
                if j not in sell_next and not (RKall[t, j] <= 2 * K): sell_next[j] = '跌出前60'
        free = N - len(pos) + len(set(sell_next) & set(pos))
        if free > 0:
            cand = np.where(RK[t] <= K)[0]
            buy_next = [j for j in cand[np.argsort(RK[t, cand])] if j not in pos][:free]
    return pd.Series(eq, index=dates).dropna(), pd.DataFrame(log)


def stats(s):
    yrs = (s.index[-1] - s.index[0]).days / 365.25; r = s.pct_change().dropna()
    return dict(年化=(s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1, 最大回撤=(s / s.cummax() - 1).min(),
                Sharpe=r.mean() / r.std() * np.sqrt(252))



if True:  # finlab 線上環境直接執行
    t0 = time.time()
    D = fix_jumps(load())
    print(f'期間 {"保留樣本 2010–2015" if HOLDOUT else "開發期 2016 起"}；資料 {D["C"].index[0].date()} ~ {D["C"].index[-1].date()}，'
          f'{D["C"].shape[1]} 檔，產業資料 {int(D["ind"].notna().sum())} 檔，{time.time() - t0:.0f} 秒', flush=True)
    rows = []; keep = None
    for name, cut in (('基準（無速度過濾）', 0), ('速度過濾 15%', 0.15), ('速度過濾 25%（標準）', 0.25), ('速度過濾 30%', 0.30)):
        X = prepare(D, cut)
        res = [run(X, off) for off in OFFSETS]
        st = pd.DataFrame([stats(s) for s, _ in res]); m = st.median()
        loc = LOCAL[name] if not HOLDOUT else ('—',) * 3
        rows.append(dict(版本=name, 年化中位=f'{m.年化:.1%}', 年化範圍=f'{st.年化.min():.1%}～{st.年化.max():.1%}',
                         最大回撤=f'{m.最大回撤:.1%}', 最差回撤=f'{st.最大回撤.min():.1%}', Sharpe=round(m.Sharpe, 2),
                         自建引擎=f'{loc[0]}% / {loc[1]}% / {loc[2]}'))
        if cut == 0.25: keep = res[0]
        if cut == 0: base = res[0]
        print(rows[-1], f'{time.time() - t0:.0f} 秒', flush=True)
    pd.set_option('display.width', 220)
    print(pd.DataFrame(rows).to_string(index=False))
    for lab, (s, T) in (('基準', base), ('標準（速度 25%）', keep)):
        y = s.resample('YE').last(); yr = y.pct_change(); yr.iloc[0] = y.iloc[0] / s.iloc[0] - 1
        print(f'\n{lab}（起始日 0）逐年報酬 %：', dict(zip(yr.index.year, (yr * 100).round(1))))
        hold = (T.exit - T.entry).dt.days
        print(f'交易 {len(T)} 筆，勝率 {(T.ret > 0).mean():.1%}，平均 {T.ret.mean():.2%}，持有天數中位 {hold.median():.0f} 天，'
              f'虧損 >20% 的筆數 {(T.ret < -0.2).sum()}，獲利 >100% 的筆數 {(T.ret > 1).sum()}')
    keep[1].to_csv('tld_v3_trades.csv', index=False, encoding='utf-8-sig')
