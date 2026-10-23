# -*- coding: utf-8 -*-
"""
V3C-PRO 回測（finlab 版）

一、進場（訊號當天收盤全部成立）
  1 大盤：加權指數 EMA60 ÷ 20 日前 EMA60 − 1 > 0
  2 趨勢：EMA200 十日斜率 ≥ 0.1 %/日
  3 突破：收盤 > 前 5 日收盤最高（不含今天）
  4 位置：收盤距 250 日最高價（含今天）≥ −12%，或 ATR% ≤ 2.5（[回]）
  5 排除漲停：當日漲幅 < 9.5%
  6 流動性：20 日均成交值（未還原收盤 × 成交股數）≥ 300 萬
  7 評分 ≥ 2（4 項各 1 分）：ATR% ≤ 3.3、距 250 日高 ≥ −8%、
                               20 日均成交值 < 2,000 萬、未還原股價 < 30
  8 排除產業：生技醫療業、通信網路業
二、選股與資金：分數高優先、同分股價低優先；最多 8 檔；
  每檔 = 當時權益 × 12.5%（現金不足用剩餘現金 × 99%）；整數股；持有中不加碼；不重新平衡
三、出場（收盤檢查、收盤賣）：
  ATR 移動停損：距離 = 進場 ATR% × 4（4%～15%，進場後固定），收盤 < 最高收盤 × (1 − 距離)
  保本：最高收盤曾 ≥ 進場收盤 × 1.06 後，收盤 < 進場收盤 × 1.005
四、成本：手續費 0.0225%/邊、證交稅 0.405%；滑價依進場時均成交值：
  < 300 萬 0.5%、300～1,000 萬 0.3%、其餘 0.1%（每邊，出場沿用進場級距）

所有指標都在「個股自己的 K 棒」上計算（停牌日不算一根），與 Python 引擎一致。
主要結果由 simulate() 逐日記錄股數與現金計算；finlab backtest.sim 只作為平台報告（見最下方）。
"""
import numpy as np
import pandas as pd

# ───────── 參數 ─────────
P = dict(
    data_start='2015-01-01',   # 資料起點（需要約 1 年暖身）
    trade_start='2016-01-01',  # 交易起點
    cap0=1_000_000,
    slots=8, weight=0.125,
    mkt_len=60, mkt_look=20,
    trend_len=200, slope_look=10, slope_min=0.1,
    brk_len=5,
    hh_len=250, pos_max=-0.12, rescue_atr=2.5, use_rescue=True,
    limit_up=0.095,
    adv_len=20, adv_min=3e6,
    atr_len=14,
    min_score=2,
    s_atr=3.3, s_dist=-0.08, s_adv=2e7, s_px=30,
    excl_ind=('生技醫療業', '通信網路業'),
    atr_mult=4, dist_min=0.04, dist_max=0.15,
    be_trig=1.06, be_lvl=1.005,
    fee=0.000225, tax=0.00405,
    slip_tiers=((3e6, 0.005), (1e7, 0.003), (float('inf'), 0.001)),
)


# ───────── 指標（逐檔、只用有交易的 K 棒） ─────────
def stock_features(adj, ah, al, raw, vol, p=P):
    """單一股票（皆為 Series，已去除停牌日）→ 各項指標 DataFrame"""
    C = adj
    n = np.arange(len(C))
    ema = C.ewm(span=p['trend_len'], adjust=False).mean()
    slope = (ema / ema.shift(p['slope_look']) - 1) * 100 / p['slope_look']
    brk = C > C.shift(1).rolling(p['brk_len']).max()
    dist = C / ah.rolling(p['hh_len']).max() - 1
    pc = C.shift(1)
    tr = pd.concat([ah - al, (ah - pc).abs(), (al - pc).abs()], axis=1).max(axis=1)
    atrp = tr.ewm(alpha=1 / p['atr_len'], adjust=False).mean() / C * 100
    chg = C / pc - 1
    adv = (raw * vol).rolling(p['adv_len']).mean()
    score = ((atrp <= p['s_atr']).astype(int) + (dist >= p['s_dist']).astype(int)
             + (adv < p['s_adv']).astype(int) + (raw < p['s_px']).astype(int))
    pos_main = dist >= p['pos_max']
    rescue = (~pos_main) & (atrp <= p['rescue_atr']) & p['use_rescue']
    sig = ((n >= p['hh_len']) & (slope >= p['slope_min']) & brk & (pos_main | rescue)
           & (chg < p['limit_up']) & (adv >= p['adv_min']) & (score >= p['min_score']))
    return pd.DataFrame({'sig': sig, 'score': score, 'atrp': atrp, 'adv': adv, 'rescue': rescue})


def build_signals(raw, adj, hi, lo, vol, mkt_ok, ind_map, p=P):
    """寬表（日期 × 代號）→ 訊號寬表與排序用寬表"""
    f = adj / raw
    ah, al = hi * f, lo * f
    out = {k: {} for k in ('sig', 'score', 'atrp', 'adv', 'rescue')}
    for c in raw.columns:
        m = adj[c].notna() & raw[c].notna() & vol[c].fillna(0).gt(0)
        if m.sum() < p['hh_len'] + 1:
            continue
        d = stock_features(adj[c][m], ah[c][m], al[c][m], raw[c][m], vol[c][m], p)
        for k in out:
            out[k][c] = d[k]
    W = {k: pd.DataFrame(v).reindex(raw.index) for k, v in out.items()}
    sig = W['sig'].fillna(False).astype(bool)
    sig = sig.mul(mkt_ok.reindex(raw.index).ffill().fillna(False).astype(bool), axis=0)
    bad = [c for c in sig.columns if ind_map.get(c) in p['excl_ind']]
    sig[bad] = False
    return sig, W['score'], W['atrp'], W['adv'], W['rescue']


# ───────── 組合模擬（逐日股數與現金，不重新平衡） ─────────
def simulate(sig, score, atrp, adv, rescue, raw, adj, p=P):
    dates = raw.index[raw.index >= p['trade_start']]
    cols = list(sig.columns)
    ci = {c: i for i, c in enumerate(cols)}
    A = adj.loc[dates, cols].to_numpy(); R = raw.loc[dates, cols].to_numpy()
    S = sig.loc[dates, cols].to_numpy(); SC = score.loc[dates, cols].to_numpy()
    AT = atrp.loc[dates, cols].to_numpy(); AD = adv.loc[dates, cols].to_numpy()
    RS = rescue.loc[dates, cols].fillna(False).to_numpy()
    last = adj[cols].ffill().loc[dates].to_numpy()          # 停牌日用最後收盤估值
    cash = float(p['cap0']); pos = {}; trades = []; eq = []; nh = []
    rng = np.random.default_rng(p.get('seed'))
    weights = np.zeros((len(dates), len(cols)))               # 給 finlab 的實際權重
    for t, d in enumerate(dates):
        row = A[t]
        # 1) 出場（進場當天不檢查；停牌日不檢查）
        for j in list(pos):
            q = pos[j]
            Cj = row[j]
            if np.isnan(Cj) or t == q['t']:
                continue
            q['peak'] = max(q['peak'], Cj)
            why = None
            if Cj < q['peak'] * (1 - q['dist']):
                why = 'ATR'
            elif q['peak'] >= q['ec'] * p['be_trig'] and Cj < q['ec'] * p['be_lvl']:
                why = '保本'
            if why:
                g = q['sh'] * Cj * (1 - q['sl'])
                proceeds = g * (1 - p['fee'] - p['tax'])
                cash += proceeds
                trades.append(dict(code=cols[j], entry=dates[q['t']], exit=d, ret=proceeds / q['cost'] - 1,
                                   days=t - q['t'], why=why, score=q['score'], rescue=q['rescue']))
                del pos[j]
        # 2) 進場
        equity = cash + sum(q['sh'] * last[t, j] for j, q in pos.items())
        free = p['slots'] - len(pos)
        if free > 0:
            ids = np.where(S[t])[0]
            ids = [j for j in ids if j not in pos]
            rk = p.get('rank', 'score_price')
            if rk == 'random':                      # 同日候選完全隨機
                rng.shuffle(ids)
            elif rk == 'score_random':              # 分數高優先、同分隨機
                rng.shuffle(ids); ids.sort(key=lambda j: -SC[t, j])
            else:                                   # 分數高優先、同分股價低優先
                ids.sort(key=lambda j: (-SC[t, j], R[t, j]))
            for j in ids:
                if free <= 0:
                    break
                target = equity * p['weight']
                amt = target if cash >= target else cash * 0.99
                sl = p['slip_flat'] if p.get('slip_flat') is not None else next(s for lim, s in p['slip_tiers'] if AD[t, j] < lim)
                rpx = R[t, j] * (1 + sl)
                rsh = np.floor(amt / (rpx * (1 + p['fee'])))      # 整數股（零股）
                if rsh < 1:
                    continue
                gross = rsh * rpx
                cost = gross * (1 + p['fee'])
                cash -= cost
                pos[j] = dict(t=t, ec=row[j], peak=row[j], sh=gross / (row[j] * (1 + sl)), cost=cost, sl=sl,
                              dist=min(max(AT[t, j] * p['atr_mult'] / 100, p['dist_min']), p['dist_max']),
                              score=int(SC[t, j]), rescue=bool(RS[t, j]))
                free -= 1
        mv = {j: q['sh'] * last[t, j] for j, q in pos.items()}
        e = cash + sum(mv.values())
        for j, v in mv.items():
            weights[t, j] = v / e
        eq.append(e); nh.append(len(pos))
    E = pd.Series(eq, index=dates, name='equity')
    T = pd.DataFrame(trades)
    W = pd.DataFrame(weights, index=dates, columns=cols)
    return E, T, W, pd.Series(nh, index=dates)


def metrics(E, T=None, bench=None):
    r = E.pct_change().dropna()
    yrs = (E.index[-1] - E.index[0]).days / 365.25
    dd = E / E.cummax() - 1
    o = {'年化': (E.iloc[-1] / E.iloc[0]) ** (1 / yrs) - 1, '最大回撤': dd.min(),
         'Sharpe': r.mean() / r.std() * np.sqrt(252), '波動': r.std() * np.sqrt(252),
         '總倍數': E.iloc[-1] / E.iloc[0]}
    if T is not None and len(T):
        w, l = T[T.ret > 0], T[T.ret <= 0]
        o.update({'交易數': len(T), '勝率': len(w) / len(T), '平均獲利': w.ret.mean(),
                  '平均虧損': l.ret.mean(), '平均持有天數': T.days.mean()})
    ye = E.resample('YE').last()
    base = pd.concat([pd.Series([E.iloc[0]], index=[E.index[0] - pd.Timedelta(days=1)]), ye])
    o['逐年'] = {int(k.year): round(float(v), 4) for k, v in base.pct_change().dropna().items()}  # 含第一年
    if bench is not None:
        b = bench.reindex(E.index).ffill()
        o['大盤年化'] = (b.iloc[-1] / b.iloc[0]) ** (1 / yrs) - 1
        o['大盤最大回撤'] = (b / b.cummax() - 1).min()
    return o


# ───────── 主程式（finlab） ─────────
if __name__ == '__main__':
    from finlab import data, backtest
    from finlab import FinlabDataFrame

    S0 = P['data_start']
    raw = data.get('price:收盤價').loc[S0:]
    adj = data.get('etl:adj_close').loc[S0:]
    hi = data.get('price:最高價').loc[S0:]
    lo = data.get('price:最低價').loc[S0:]
    vol = data.get('price:成交股數').loc[S0:]
    op = data.get('price:開盤價').loc[S0:]

    # 股票池：上市、上櫃 4 位數普通股；排除 00 開頭（ETF）與存託憑證
    sc = data.get('security_categories')
    sc = sc.set_index('stock_id') if 'stock_id' in sc.columns else sc
    bad_cat = sc[sc['category'].astype(str).str.contains('存託憑證|ETF|ETN|受益', regex=True)].index.astype(str)
    common = [c for c in raw.columns
              if len(str(c)) == 4 and str(c).isdigit() and not str(c).startswith('00') and str(c) not in set(bad_cat)
              and c in adj.columns and c in hi.columns and c in lo.columns and c in vol.columns]
    # 注意：不要用「目前仍上市」名單篩選，否則會排除下市股，產生倖存者偏誤
    raw, adj, hi, lo, vol, op = raw[common], adj[common], hi[common], lo[common], vol[common], op.reindex(columns=common)
    # 資料清理：最高價至少涵蓋開盤與收盤、最低價至多為開盤與收盤（避免資料源的不合理 K 棒）
    hi = np.fmax(np.fmax(hi, op.reindex_like(hi)), raw)
    lo = np.fmin(np.fmin(lo, op.reindex_like(lo)), raw)

    ix = data.get('stock_index_price:收盤指數').loc[S0:]['上市發行量加權股價指數']
    e60 = ix.ewm(span=P['mkt_len'], adjust=False).mean()
    mkt_ok = (e60 / e60.shift(P['mkt_look']) - 1) > 0

    ind_map = data.get('director_compensation:產業類別').ffill().iloc[-1].to_dict()

    sig, score, atrp, adv, rescue = build_signals(raw, adj, hi, lo, vol, mkt_ok, ind_map)
    E, T, W, N = simulate(sig, score, atrp, adv, rescue, raw, adj)

    m = metrics(E, T, ix)
    print({k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in m.items()})
    print('平均持股數', round(float(N.mean()), 2))
    T.to_csv('v3c_pro_trades.csv', index=False, encoding='utf-8-sig')
    E.to_csv('v3c_pro_equity.csv')

    # ── finlab 平台報告（選用）──
    # 把模擬出來的「實際每日權重」交給 finlab，讓它照實際持股走，不會把贏家砍回 12.5%。
    # 成交價：訊號當天還原收盤 ± 滑價（finlab 在下一列成交，所以用前一列價格）。
    data.free_memory()
    # 滑價已含在 simulate 的成交價中；finlab 報告僅供檢視，數字以 simulate 為準
    prev_adj = adj.loc[W.index].shift(1).bfill()
    report = backtest.sim(FinlabDataFrame(W), resample=None, trade_at_price=FinlabDataFrame(prev_adj),
                          fee_ratio=P['fee'], tax_ratio=P['tax'], name='V3C-PRO', upload=True)
    print('finlab 報告年化', round(float(report.metrics.annual_return()), 4),
          '最大回撤', round(float(report.metrics.max_drawdown()), 4))
