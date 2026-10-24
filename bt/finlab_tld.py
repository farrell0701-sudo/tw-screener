"""T_ld 排序策略：finlab 交叉驗證版（單檔，可直接在 finlab 線上環境執行）

策略
- 股票池：4 位數普通股、上市滿 250 根、20 日均成交額 ≥ 當日全市場 20 分位
- 分數 T_ld = (EMA12 − EMA200) / ATR14（還原價）
- 每 20 個交易日換股：t 收盤排序 → t+1 開盤；不在前 20 的賣出，空位以「當下總資產/20」買進（舊部位不重新平衡）
  開盤漲停（≥ 前收 +9.5%）買不到就往下遞補
- 停損：收盤 < 持有期最高收盤 − k×ATR14 → 隔日開盤賣出
  k = 4（多頭）／3（空頭）；多空 = 加權指數 EMA60 的 20 日斜率 > 0
- 被停掉的空位留現金，下次換股日再補；不設停利、不加大盤濾網
- 成本：手續費 0.085%/邊、證交稅 0.3%
同時跑「不停損」與「全程 4 ATR」做對照，各 5 個起始日取中位數。

注意：預設只跑開發期 2016-02 起。HOLDOUT=True 會跑 2010–2015 保留樣本，請等全部定案後才開。
"""
import time
import numpy as np, pandas as pd

HOLDOUT = False
DATA_START, EVAL_START, EVAL_END = ('2008-06-01', '2010-01-04', '2015-12-31') if HOLDOUT else ('2014-06-01', '2016-02-01', None)
N, REB, FEE, TAX = 20, 20, 0.00085, 0.003
OFFSETS = (0, 4, 8, 12, 16)
LOCAL = {'不停損': (35.1, -27.4, 1.79), '全程4ATR': (33.8, -20.7, 1.93), '多4空3': (33.3, -20.1, 1.98)}  # 自建引擎結果（對照用）


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
    return dict(O=op * k, H=hi * k, L=lo * k, C=adj, RC=raw, V=vol, ix=ix.reindex(adj.index).ffill())


def fix_jumps(D, th=0.11):
    """漲跌幅上限 10%：上市 5 天後還有 >11% 的單日還原價變動視為資料問題，報酬歸零並向前調整"""
    C = D['C']; Cf = C.ffill(); age = C.notna().cumsum()
    r = C / Cf.shift(1) - 1
    bad = (r.abs() > th) & (age > 5) & C.notna()
    print(f'超過漲跌幅的單日跳動：{int(bad.sum().sum())} 筆（已修正）', flush=True)
    adj = (1 + r.where(bad, 0)).iloc[::-1].cumprod().iloc[::-1].shift(-1).fillna(1.0)
    for k in ('O', 'H', 'L', 'C'): D[k] = D[k] * adj
    return D


def prepare(D):
    C, H, L, O = D['C'], D['H'], D['L'], D['O']
    pc = C.shift(1)
    tr = np.maximum(H - L, np.maximum((H - pc).abs(), (L - pc).abs()))
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    e12 = C.ewm(span=12, adjust=False).mean(); e200 = C.ewm(span=200, adjust=False).mean()
    amt20 = (D['RC'] * D['V']).rolling(20, min_periods=15).mean()
    univ = C.notna() & (C.notna().cumsum() >= 250) & amt20.ge(amt20.quantile(0.20, axis=1), axis=0)
    score = ((e12 - e200) / atr).where(univ)
    buy_ok = (O.shift(-1) / C - 1 < 0.095) & O.shift(-1).notna()
    e60 = D['ix'].ewm(span=60, adjust=False).mean(); bull = ((e60 / e60.shift(20) - 1) > 0)
    return dict(dates=C.index, C=C.values, O=O.values, Cf=C.ffill().values, A=atr.values, S=score.values,
                U=univ.values, BUY=buy_ok.values, BULL=bull.values, cols=C.columns)


def run(X, off, kb, kr):
    dates = X['dates']; start = dates.get_indexer([pd.Timestamp(EVAL_START)], method='bfill')[0]; end = len(dates) - 1
    Cv, Ov, Cfv, Av, Sv, Uv, BUY, BULL = (X[k] for k in ('C', 'O', 'Cf', 'A', 'S', 'U', 'BUY', 'BULL'))
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
            if k is not None and c < p[2] - k * Av[t, j]: sell_next.add(j)
        if t in reb:
            sc = np.where(BUY[t] & Uv[t], Sv[t], np.nan); ok = np.where(~np.isnan(sc))[0]
            top = list(ok[np.argsort(-sc[ok])[:N]])
            sell_next |= {j for j in pos if j not in top}
            buy_next = [j for j in top if j not in pos]
    return pd.Series(eq, index=dates).dropna(), pd.DataFrame(log)


def stats(s):
    yrs = (s.index[-1] - s.index[0]).days / 365.25; r = s.pct_change().dropna()
    return dict(年化=(s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1, 最大回撤=(s / s.cummax() - 1).min(),
                Sharpe=r.mean() / r.std() * np.sqrt(252))


if True:  # finlab 線上環境直接執行
    t0 = time.time()
    D = fix_jumps(load()); X = prepare(D)
    print(f'資料 {X["dates"][0].date()} ~ {X["dates"][-1].date()}，{len(X["cols"])} 檔，{time.time() - t0:.0f} 秒', flush=True)
    print(f'空頭天數比例（評估期）：{1 - X["BULL"][X["dates"] >= EVAL_START].mean():.0%}')
    rows = []; keep = None
    for name, kb, kr in (('不停損', None, None), ('全程4ATR', 4, 4), ('多4空3', 4, 3)):
        res = [run(X, off, kb, kr) for off in OFFSETS]
        st = pd.DataFrame([stats(s) for s, _ in res])
        m = st.median()
        loc = LOCAL[name] if not HOLDOUT else (np.nan,) * 3
        rows.append(dict(版本=name, 年化中位=f'{m.年化:.1%}', 年化範圍=f'{st.年化.min():.1%}～{st.年化.max():.1%}',
                         最大回撤=f'{m.最大回撤:.1%}', Sharpe=round(m.Sharpe, 2),
                         自建引擎=f'{loc[0]}% / {loc[1]}% / {loc[2]}'))
        if name == '多4空3': keep = res[0]
        print(rows[-1], flush=True)
    pd.set_option('display.width', 200)
    print(pd.DataFrame(rows).to_string(index=False))
    s, T = keep
    y = s.resample('YE').last(); yr = y.pct_change(); yr.iloc[0] = y.iloc[0] / s.iloc[0] - 1
    print('\n多4空3（起始日 0）逐年報酬：'); print((yr * 100).round(1).set_axis(yr.index.year).to_string())
    print(f'\n交易 {len(T)} 筆，勝率 {(T.ret > 0).mean():.1%}，平均 {T.ret.mean():.2%}，中位 {T.ret.median():.2%}')
    T.to_csv('tld_trades.csv', index=False, encoding='utf-8-sig')

    # 交叉檢查：finlab 內建 sim 跑「不停損」版（每 20 日換股、等權、隔日開盤）
    try:
        from finlab import backtest
        dates = X['dates']; start = dates.get_indexer([pd.Timestamp(EVAL_START)], method='bfill')[0]
        pos = pd.DataFrame(np.nan, index=dates, columns=X['cols'])
        for t in range(start, len(dates) - 1, REB):
            sc = np.where(X['BUY'][t] & X['U'][t], X['S'][t], np.nan); ok = np.where(~np.isnan(sc))[0]
            row = np.zeros(len(X['cols'])); row[ok[np.argsort(-sc[ok])[:N]]] = 1
            pos.iloc[t] = row
        pos = pos.ffill().fillna(0).astype(bool).loc[EVAL_START:]
        r = backtest.sim(pos, resample=None, trade_at_price='open', fee_ratio=FEE, tax_ratio=TAX, upload=False)
        print('\nfinlab sim（不停損、會重新平衡權重）：', r.get_stats().get('cagr'), r.get_stats().get('max_drawdown'))
    except Exception as e:  # noqa: BLE001
        print('finlab sim 交叉檢查失敗：', e)
