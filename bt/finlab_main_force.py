"""主力集中度（券商分點前 15 大買超 − 前 15 大賣超）÷ 成交量：短線事件前後與報酬的關係（finlab，單檔可直接執行）
資料 broker_transactions 從 2018 起；只跑開發期
只跑開發期 2016 起（保留樣本已用過三次，不再用來調整）

以下是 v4 的說明（引擎沿用）：

選股（每天收盤後計算，只用當天以前的資料）
- 股票池：4 位數普通股、上市滿 250 根、20 日均成交額 ≥ 當日全市場 20 分位
- 分數 T_ldf = (EMA12 − EMA200) / max(ATR14, 當日全市場 ATR% 的 5 分位 × 股價)
- 族群過濾：所屬產業 T_ldf 中位數在全市場前 50%（查不到產業視為通過）
- （候選）上漲速度：速度 = log(120 日最高價 ÷ 120 日最低價) ÷ 低點到高點經過的天數
       （高點在低點之前 = 下跌，不算）；強勢股（分數前 10%）中速度最快的 25% 只買半個部位
- 近 5 日出現過一字跌停（最高 = 最低、跌幅 ≥ 跌停 −0.5%）不買
- 候選池 = 通過過濾後分數前 K 名（K=30）

進出場（持股 10 檔、等權）
- 每天收盤有空位就從候選池挑排名最高、未持有的，隔日開盤買
  隔日開盤漲停或一字漲停買不到 → 跳過換下一檔
- 出場（先到者）：收盤 < 持有期最高收盤 − k×ATR14（多頭 k=4、空頭 k=3；多空 = 加權指數 EMA60 的 20 日斜率 > 0）
                  或 收盤 < 60 日密集區下緣（前 60 根最低價的中位數，到昨天為止）
  → 隔日開盤賣；當天一字跌停賣不掉，順延到打開那天
- 每 20 個交易日：排名跌出前 2K（60 名）的持股賣出
- 漲跌停：2015-06-01 前 7%，之後 10%
- 成本：手續費 0.085%/邊、證交稅 0.3%；20 個起始日取中位數

版本：標準版（等權）、標準版＋速度前 25% 半個部位、短線事件版、標準＋短線合併。

短線事件版（獨立策略）
- 事件（強勢股＝通過族群過濾後分數前 10%，且近 5 日無一字跌停）出現以下任一：
  ① 布林收斂後首次突破上軌：近 10 日內帶寬（4σ÷中軌）曾 ≤ 自身前 120 日 20 分位，今天收盤首次站上上軌（20 日、2σ）
  ② 布林收斂後帶寬張開＋收上軌：同①的收斂，帶寬比 3 天前擴大 15% 以上，且收盤在上軌之上
  ③ 整理後突破 20 日高：昨天為止 20 日高低區間 ÷ ATR ≤ 自身前 250 日 30 分位，今天收盤 > 前 20 日最高價
- 隔日開盤買（開盤漲停／一字漲停跳過），同一天多檔依分數排序，10 個部位
- 出場：分批固定天數，1/3 資金各跑持有 15／20／25 天的版本後平均（不設停損；一字跌停順延）
HOLDOUT=False 跑 2016 起開發期，和自建引擎（20 個起始日）對照；True 跑 2010–2015 保留樣本。
"""
import time
import numpy as np, pandas as pd

HOLDOUT = False   # 固定開發期
DATA_START, EVAL_START, EVAL_END = ('2008-06-01', '2010-01-04', '2015-12-31') if HOLDOUT else ('2014-06-01', '2016-02-01', None)
N, K, REB, FEE, TAX = 10, 30, 20, 0.00085, 0.003
OFFSETS = tuple(range(20))   # 20 個起始日；只取 5 個的雜訊太大（模擬測試時速度過濾 25% 和 15% 差了 5 個百分點）
# 自建引擎結果（年化 / 回撤 / Sharpe），對照用
# 自建引擎（20 個起始日中位數）：年化 / 最大回撤 / 最差回撤 / Sharpe
LOCAL = {'標準版': (42.9, -25.9, -33.2, 1.86), '標準版＋速度前25%半個部位': (41.2, -24.9, -28.8, 1.99)}
# 短線與合併為單一路徑（標準版用起始日 0）：年化 / 最大回撤 / Sharpe
LOCAL_S = {'短線事件版（分批 15/20/25 日）': (40.8, -31.5, 1.97), '標準 70%＋短線 30%': (42.2, -24.3, 1.98), '標準 50%＋短線 50%': (42.0, -26.3, 2.05)}


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


def prepare(D, half_cut=0.0):
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
    raw_score = score.copy()
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
    q = speed.where(strong).rank(axis=1, pct=True)
    W = pd.DataFrame(1.0, index=C.index, columns=C.columns)
    if half_cut: W = W.mask(q > 1 - half_cut, 0.5)        # 速度最快的 half_cut 只買半個部位
    nxt_open_up = (ROP.shift(-1) / RC - 1).ge(lim.shift(-1), axis=0)
    buy_ok = ~nxt_open_up & O.shift(-1).notna()
    medL = L.rolling(60, min_periods=60).median().shift(1)
    # 短線事件
    m20 = C.rolling(20).mean(); sd20 = C.rolling(20).std(); up = m20 + 2 * sd20; bw = 4 * sd20 / m20
    squeeze = bw.rolling(10).min().shift(1) <= bw.rolling(120, min_periods=100).quantile(0.2).shift(1)
    ev1 = squeeze & (C > up) & (C.shift(1) <= up.shift(1))
    ev2 = squeeze & (bw > bw.shift(3) * 1.15) & (C > up)
    rng20 = (H.rolling(20).max() - L.rolling(20).min()) / atr
    tight = rng20.shift(1) <= rng20.rolling(250, min_periods=150).quantile(0.3).shift(1)
    ev3 = tight & (C > H.rolling(20).max().shift(1))
    ldn5 = lockdn.rolling(5, min_periods=1).sum() > 0
    EVS = strong & ~ldn5 & (ev1 | ev2 | ev3)
    e60 = D['ix'].ewm(span=60, adjust=False).mean(); bull = ((e60 / e60.shift(20) - 1) > 0)
    return dict(dates=C.index, C=C.values, O=O.values, Cf=C.ffill().values, A=atr.values, ML=medL.values,
                LD=lockdn.values, LU=lockup.values, W=W.values,
                RK=score.where(buy_ok).rank(axis=1, ascending=False).values,
                RKall=score.rank(axis=1, ascending=False).values,
                RKs=raw_score.where(EVS & buy_ok).rank(axis=1, ascending=False, method='first').values,
                BULL=bull.values, cols=C.columns)


def run(X, off, kb=4, kr=3):
    dates = X['dates']; start = dates.get_indexer([pd.Timestamp(EVAL_START)], method='bfill')[0]; end = len(dates) - 1
    Cv, Ov, Cfv, Av, ML, LD, LU, RK, RKall, BULL, Wt = (X[k] for k in ('C', 'O', 'Cf', 'A', 'ML', 'LD', 'LU', 'RK', 'RKall', 'BULL', 'W'))
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
                amt = min(val / N * Wt[t - 1, j], cash)
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


def run_short(X, Hd):
    """短線事件版：持有 Hd 天後隔日開盤賣"""
    dates = X['dates']; start = dates.get_indexer([pd.Timestamp(EVAL_START)], method='bfill')[0]; end = len(dates) - 1
    Ov, Cfv, LD, LU, RKs = (X[k] for k in ('O', 'Cf', 'LD', 'LU', 'RKs'))
    cash = 1.0; pos = {}; sell_next, buy_next = set(), []; eq = np.full(len(dates), np.nan); rets = []
    for t in range(start, end + 1):
        keep = set()
        for j in sell_next:
            if j not in pos: continue
            if LD[t, j]: keep.add(j); continue
            px = Ov[t, j] if not np.isnan(Ov[t, j]) else Cfv[t, j]
            p = pos.pop(j); cash += p[0] * px * (1 - FEE - TAX); rets.append(px * (1 - FEE - TAX) / (p[1] / (1 - FEE)) - 1)
        if buy_next:
            val = cash + sum(p[0] * Cfv[t - 1, j] for j, p in pos.items())
            for j in buy_next:
                if j in pos or len(pos) >= N: continue
                px = Ov[t, j]
                if np.isnan(px) or LU[t, j]: continue
                amt = min(val / N, cash)
                if amt <= 0: break
                pos[j] = [amt * (1 - FEE) / px, px, t]; cash -= amt
        sell_next, buy_next = keep, []
        eq[t] = cash + sum(p[0] * Cfv[t, j] for j, p in pos.items())
        for j, p in pos.items():
            if j not in sell_next and t - p[2] + 1 >= Hd: sell_next.add(j)
        free = N - len(pos) + len(sell_next & set(pos))
        if free > 0:
            cand = np.where(~np.isnan(RKs[t]))[0]
            buy_next = [j for j in cand[np.argsort(RKs[t, cand])] if j not in pos][:free]
    return pd.Series(eq, index=dates).dropna(), np.array(rets)


def rstats(r):
    r = r.dropna(); e = (1 + r).cumprod(); yrs = (e.index[-1] - e.index[0]).days / 365.25
    return dict(年化=e.iloc[-1] ** (1 / yrs) - 1, 最大回撤=(e / e.cummax() - 1).min(), Sharpe=r.mean() / r.std() * np.sqrt(252))


def stats(s):
    yrs = (s.index[-1] - s.index[0]).days / 365.25; r = s.pct_change().dropna()
    return dict(年化=(s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1, 最大回撤=(s / s.cummax() - 1).min(),
                Sharpe=r.mean() / r.std() * np.sqrt(252))





# ───────────────────────── 券商分點：主力買賣超 ─────────────────────────
def load_main_force(idx, cols):
    from finlab import data
    bt = pd.DataFrame(data.get('broker_transactions'))
    print('券商分點欄位：', list(bt.columns), '；前 2 列：', bt.head(2).to_dict('records'), flush=True)
    bt = bt.reset_index(drop=not ('date' not in bt.columns and bt.index.name == 'date'))
    cd = next(c for c in bt.columns if c in ('date', '日期')); cs = next(c for c in bt.columns if c in ('stock_id', 'symbol', '代號'))
    for c in ('buy', 'sell'): bt[c] = pd.to_numeric(bt[c], errors='coerce')
    g = bt.groupby([cd, cs])[['buy', 'sell']].sum(); del bt
    g = g.reset_index(); g[cd] = pd.to_datetime(g[cd]); g[cs] = g[cs].astype(str)
    net = g.pivot(index=cd, columns=cs, values='buy') - g.pivot(index=cd, columns=cs, values='sell')
    net = pd.DataFrame(net).reindex(index=idx, columns=cols)
    print('券商分點期間', net.dropna(how='all').index.min().date(), '～', net.dropna(how='all').index.max().date(), flush=True)
    return net


if True:  # finlab 線上環境直接執行
    t0 = time.time()
    D = fix_jumps(load())
    idx, cols = D['C'].index, D['C'].columns
    net = load_main_force(idx, cols)
    V = D['V']
    ratio = (net.abs().stack().median() / V.where(net.notna()).stack().median())
    unit = 1000.0 if ratio < 0.01 else 1.0                                   # 分點若以「張」計，換成股
    print(f'單位檢查：|主力買賣超| 中位數 ÷ 成交股數中位數 = {ratio:.4f} → 分點數量乘以 {unit:.0f}', flush=True)
    net = net * unit
    have = net.notna()
    cn = net.fillna(0).cumsum(); cv = V.where(have).fillna(0).cumsum()
    def conc(a, b):
        """t+a ～ t+b 日（含）的主力集中度，用 t 的列表示；a、b 可為負（過去）"""
        n_ = cn.shift(-b) - cn.shift(-(a - 1)); v_ = cv.shift(-b) - cv.shift(-(a - 1))
        ok = have.astype(float).fillna(0).cumsum(); k = ok.shift(-b) - ok.shift(-(a - 1))
        return (n_ / v_.replace(0, np.nan)).where(k >= (b - a + 1) * 0.8)
    pre5, pre20 = conc(-4, 0), conc(-19, 0)                                    # 事件當天（含）往前 5／20 日，收盤後已知
    post5 = conc(1, 5)                                                         # 事件後第 1～5 日（第 5 天收盤後才知道）
    X = prepare(D, 0)
    C, O = D['C'], D['O']; Cf = C.ffill()
    sc_all = pd.DataFrame(X['RKall'], index=idx, columns=cols); nstock = (~np.isnan(X['RKall'])).sum(1)
    strong = sc_all.le(np.ceil(nstock * 0.10), axis=0)
    ev = pd.DataFrame(X['RKs'], index=idx, columns=cols).notna()
    d0 = max(pd.Timestamp(EVAL_START), net.dropna(how='all').index.min() + pd.Timedelta(days=40))
    def rel(entry_shift, exit_shift, entry_px='O'):
        ent = (O if entry_px == 'O' else Cf).shift(-entry_shift); R = Cf.shift(-exit_shift) / ent - 1
        return R.sub(R.where(strong).mean(1), axis=0), R
    print(f'\n分析期間 {d0.date()} 起；短線事件 {int(ev.loc[d0:].sum().sum())} 次', flush=True)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    tests = [('A 事件前 5 日集中度 → 隔日開盤進、持有到第 20 日', ev, pre5, (1, 20)),
             ('A 事件前 20 日集中度 → 隔日開盤進、持有到第 20 日', ev, pre20, (1, 20)),
             ('B 事件後第 1～5 日集中度 → 第 6 日開盤進、持有到第 20 日（可交易）', ev, post5, (6, 20)),
             ('B 事件後第 1～5 日集中度 → 第 6 日開盤進、持有到第 25 日（可交易）', ev, post5, (6, 25)),
             ('C 對照（不可交易）：事件後第 1～5 日集中度 vs 同期報酬', ev, post5, (1, 5)),
             ('D 一般強勢股：前 5 日集中度 → 隔日開盤進、持有 20 日', strong, pre5, (1, 20)),
             ('D 一般強勢股：前 20 日集中度 → 隔日開盤進、持有 20 日', strong, pre20, (1, 20))]
    rows = []
    for lab, G, Xf, (a, b) in tests:
        ex, R = rel(a, b)
        g = G.loc[d0:] & Xf.loc[d0:].notna() & R.loc[d0:].notna()
        v = Xf.loc[d0:].where(g).stack().dropna(); q1, q2 = v.quantile([1 / 3, 2 / 3])
        o = dict(測試=lab, 筆數=len(v), 切點=f'{q1:+.1%} / {q2:+.1%}')
        for i, (lo, hi, nm) in enumerate(((-np.inf, q1, '低'), (q1, q2, '中'), (q2, np.inf, '高'))):
            m = g & (Xf.loc[d0:] > lo) & (Xf.loc[d0:] <= hi)
            e = ex.loc[d0:].where(m).stack().dropna(); r = R.loc[d0:].where(m).stack().dropna()
            yr = e.groupby(e.index.get_level_values(0).year).mean()
            o[f'{nm}：超額'] = f'{e.mean():+.2%}'; o[f'{nm}：正年'] = f'{(yr > 0).sum()}/{len(yr)}'; o[f'{nm}：勝率'] = f'{(r > 0).mean():.0%}'
        rows.append(o); print(o, flush=True)
    print('\n══ 主力集中度三等分（超額＝相對同日強勢股；低＝主力賣超、高＝主力買超）')
    print(pd.DataFrame(rows).to_string(index=False))
    for code, dt, lab in (('6223', '2025-09-26', '旺矽'), ('6215', '2025-02-18', '和椿'), ('4743', '2020-04-29', '合一')):
        if code in cols:
            d_ = idx[idx.searchsorted(pd.Timestamp(dt))]
            f = lambda Z: None if pd.isna(Z.at[d_, code]) else f'{Z.at[d_, code]:+.1%}'
            print(lab, d_.date(), '前 5 日', f(pre5), '前 20 日', f(pre20), '進場後 1～5 日', f(post5))
    print(f'\n完成，{time.time() - t0:.0f} 秒。請把「══」表格、單位檢查與最後三行貼給我。')
