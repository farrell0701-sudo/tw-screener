"""T_ld 新版：每日補位 ＋ 族群趨勢過濾 ＋ 排除波動擴張（finlab 交叉驗證，單檔可直接執行）

選股分數與過濾（每天收盤後計算，只用當天以前的資料）
- 股票池：4 位數普通股、上市滿 250 根、20 日均成交額 ≥ 當日全市場 20 分位
- 分數 T_ldf = (EMA12 − EMA200) / max(ATR14, 當日全市場 ATR% 的 5 分位 × 股價)
- 族群過濾：所屬產業 T_ldf 中位數，在全市場排前 50%（查不到產業的股票視為通過）
- 波動擴張過濾：ATR14 的 20 日均 ÷ 120 日均，排除全市場最高的 20%
- 候選池 = 通過過濾後分數前 K 名（K=30）

進出場（持股 10 檔、等權）
- 每天收盤：有空位（含當天要停損賣出的）就從候選池挑排名最高、尚未持有的，隔日開盤買進
  （隔日開盤漲停 ≥ 前收 +9.5% 買不到 → 跳過，換下一檔）
- 停損：收盤 < 持有期最高收盤 − k×ATR14 → 隔日開盤賣出；多頭 k=4、空頭 k=3
  多空 = 加權指數 EMA60 的 20 日斜率 > 0
- 每 20 個交易日檢查一次：排名跌出前 2K（60 名）的持股，隔日開盤賣出
- 成本：手續費 0.085%/邊、證交稅 0.3%
對照：舊版（每 20 日換成前 10 名）、新版不含波動擴張過濾。各 5 個起始日取中位數。

注意：預設只跑開發期 2016-02 起。HOLDOUT=True 會跑 2010–2015 保留樣本，請等全部定案後才開。
"""
import time
import numpy as np, pandas as pd

HOLDOUT = False
DATA_START, EVAL_START, EVAL_END = ('2008-06-01', '2010-01-04', '2015-12-31') if HOLDOUT else ('2014-06-01', '2016-02-01', None)
N, K, REB, FEE, TAX = 10, 30, 20, 0.00085, 0.003
OFFSETS = (0, 4, 8, 12, 16)
# 自建引擎結果（年化 / 回撤 / Sharpe），對照用
LOCAL = {'舊版：每20日換前10': (33.6, -22.5, 1.81), '新版（無波動過濾）': (42.4, -26.0, 1.94), '新版': (39.8, -20.5, 2.31)}


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
    return dict(O=op * k, H=hi * k, L=lo * k, C=adj, RC=raw, V=vol, ix=ix.reindex(adj.index).ffill(),
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


def prepare(D, vcp_filter=True, ind_filter=True):
    C, H, L, O = D['C'], D['H'], D['L'], D['O']
    pc = C.shift(1)
    tr = np.maximum(H - L, np.maximum((H - pc).abs(), (L - pc).abs()))
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    e12 = C.ewm(span=12, adjust=False).mean(); e200 = C.ewm(span=200, adjust=False).mean()
    amt20 = (D['RC'] * D['V']).rolling(20, min_periods=15).mean()
    univ = C.notna() & (C.notna().cumsum() >= 250) & amt20.ge(amt20.quantile(0.20, axis=1), axis=0)
    floor = (atr / C).quantile(0.05, axis=1)
    score = ((e12 - e200) / np.maximum(atr, C.mul(floor, axis=0))).where(univ)
    if ind_filter and D['ind'].notna().any():
        known = D['ind'].notna().values
        g = score.loc[:, known].T.groupby(D['ind'][known].values).transform('median').T.reindex(columns=C.columns)
        gp = g.where(univ).rank(axis=1, pct=True)
        score = score.where((gp >= 0.5) | gp.isna())
    if vcp_filter:
        vcp = (atr.rolling(20).mean() / atr.rolling(120).mean()).where(univ)
        score = score.where(vcp.rank(axis=1, pct=True) <= 0.8)
    buy_ok = (O.shift(-1) / C - 1 < 0.095) & O.shift(-1).notna()
    e60 = D['ix'].ewm(span=60, adjust=False).mean(); bull = ((e60 / e60.shift(20) - 1) > 0)
    return dict(dates=C.index, C=C.values, O=O.values, Cf=C.ffill().values, A=atr.values,
                RK=score.where(buy_ok).rank(axis=1, ascending=False).values,     # 買進用（排除隔日漲停）
                RKall=score.rank(axis=1, ascending=False).values,               # 持股檢查用
                BULL=bull.values, cols=C.columns)


def run(X, off, mode='new', kb=4, kr=3):
    dates = X['dates']; start = dates.get_indexer([pd.Timestamp(EVAL_START)], method='bfill')[0]; end = len(dates) - 1
    Cv, Ov, Cfv, Av, RK, RKall, BULL = (X[k] for k in ('C', 'O', 'Cf', 'A', 'RK', 'RKall', 'BULL'))
    reb = set(range(start + off, end - 1, REB))
    cash = 1.0; pos = {}; sell_next, buy_next = set(), []
    eq = np.full(len(dates), np.nan); log = []
    for t in range(start, end + 1):
        for j in sell_next:
            if j in pos:
                px = Ov[t, j] if not np.isnan(Ov[t, j]) else Cfv[t, j]
                p = pos.pop(j); cash += p[0] * px * (1 - FEE - TAX)
                log.append(dict(code=X['cols'][j], entry=dates[p[3]], exit=dates[t], ret=px * (1 - FEE - TAX) / (p[1] * (1 + FEE)) - 1))
        if buy_next:
            val = cash + sum(p[0] * Cfv[t - 1, j] for j, p in pos.items())
            for j in buy_next:
                if j in pos or len(pos) >= N: continue
                px = Ov[t, j]
                if np.isnan(px): continue
                amt = min(val / N, cash)
                if amt <= 0: break
                pos[j] = [amt * (1 - FEE) / px, px, px, t]; cash -= amt
        sell_next, buy_next = set(), []
        eq[t] = cash + sum(p[0] * Cfv[t, j] for j, p in pos.items())
        k = kb if BULL[t] else kr
        for j, p in pos.items():
            c = Cv[t, j]
            if np.isnan(c): continue
            p[2] = max(p[2], c)
            if c < p[2] - k * Av[t, j]: sell_next.add(j)
        if mode == 'old':                       # 舊版：每 20 日換成前 N 名，停損空位等到下次
            if t in reb:
                ok = np.where(RK[t] <= N)[0]; top = list(ok[np.argsort(RK[t, ok])])
                sell_next |= {j for j in pos if j not in top}
                buy_next = [j for j in top if j not in pos]
            continue
        if t in reb:                            # 新版：跌出前 2K 才換
            sell_next |= {j for j in pos if not (RKall[t, j] <= 2 * K)}
        free = N - len(pos) + len(sell_next & set(pos))
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
    print(f'資料 {D["C"].index[0].date()} ~ {D["C"].index[-1].date()}，{D["C"].shape[1]} 檔，產業資料 {int(D["ind"].notna().sum())} 檔，{time.time() - t0:.0f} 秒', flush=True)
    X_old = prepare(D, vcp_filter=False)
    X_nov = X_old
    X_new = prepare(D, vcp_filter=True)
    print(f'空頭天數比例（評估期）：{1 - X_new["BULL"][X_new["dates"] >= EVAL_START].mean():.0%}')
    rows = []; keep = None
    for name, X, mode in (('舊版：每20日換前10', X_old, 'old'), ('新版（無波動過濾）', X_nov, 'new'), ('新版', X_new, 'new')):
        res = [run(X, off, mode) for off in OFFSETS]
        st = pd.DataFrame([stats(s) for s, _ in res]); m = st.median()
        loc = LOCAL[name] if not HOLDOUT else (np.nan,) * 3
        rows.append(dict(版本=name, 年化中位=f'{m.年化:.1%}', 年化範圍=f'{st.年化.min():.1%}～{st.年化.max():.1%}',
                         最大回撤=f'{m.最大回撤:.1%}', 最差回撤=f'{st.最大回撤.min():.1%}', Sharpe=round(m.Sharpe, 2),
                         自建引擎=f'{loc[0]}% / {loc[1]}% / {loc[2]}'))
        if name == '新版': keep = res[0]
        print(rows[-1], flush=True)
    pd.set_option('display.width', 200)
    print(pd.DataFrame(rows).to_string(index=False))
    s, T = keep
    y = s.resample('YE').last(); yr = y.pct_change(); yr.iloc[0] = y.iloc[0] / s.iloc[0] - 1
    print('\n新版（起始日 0）逐年報酬：'); print((yr * 100).round(1).set_axis(yr.index.year).to_string())
    hold = (T.exit - T.entry).dt.days
    print(f'\n交易 {len(T)} 筆（每年約 {len(T) / ((s.index[-1] - s.index[0]).days / 365.25):.0f} 筆），勝率 {(T.ret > 0).mean():.1%}，'
          f'平均 {T.ret.mean():.2%}，中位 {T.ret.median():.2%}，持有天數中位 {hold.median():.0f} 天')
    T.to_csv('tld_v2_trades.csv', index=False, encoding='utf-8-sig')
