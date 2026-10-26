"""集保股權分散（大戶／散戶）因子測試：在 T_ld 強勢股、短線事件、標準版上（finlab，單檔可直接執行）
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




# ───────────────────────── 集保資料 ─────────────────────────
LEVEL_LO = {1: 1, 2: 1000, 3: 5001, 4: 10001, 5: 15001, 6: 20001, 7: 30001, 8: 40001, 9: 50001, 10: 100001,
            11: 200001, 12: 400001, 13: 600001, 14: 800001, 15: 1000001}


def load_inventory():
    from finlab import data
    tried = []
    for key in ('inventory', '集保戶股權分散表', 'tdcc', 'shareholding'):
        try:
            df = data.get(key); print(f'集保資料：data.get("{key}") 成功，型態 {type(df).__name__}', flush=True); return df
        except Exception as e:  # noqa: BLE001
            tried.append(f'{key}: {str(e)[:80]}')
    for kw in ('inventory', '集保', '股權分散', '持股分級'):
        try: print(f'data.search("{kw}") →', list(data.search(kw))[:20], flush=True)
        except Exception as e: print(f'data.search("{kw}") 失敗：{e}')  # noqa: BLE001
    raise SystemExit('找不到集保股權分散資料，請把上面幾行輸出貼給我。嘗試過：' + '；'.join(tried))


def level_lower(v):
    """持股分級 → 該級下限股數；數字 1～15 或文字區間（如 '400,001-600,000'）都可以"""
    if pd.isna(v): return np.nan
    try:
        k = int(float(v))
        if 1 <= k <= 15: return LEVEL_LO[k]
        return np.nan
    except (TypeError, ValueError):
        pass
    import re
    txt = str(v).replace(',', '').replace('，', '')
    nums = re.findall(r'\d+', txt)
    if not nums or '合計' in txt or '差異' in txt or 'total' in txt.lower(): return np.nan
    lo = int(nums[0])
    if '以上' in txt or 'more' in txt.lower() or '>' in txt: lo = max(lo, 1000001)
    return lo


def build_hold(inv, idx, cols):
    """回傳每日（可交易日）面板：大戶≥400張、≥1000張、散戶<50張 的持股比例（%）與股東總人數"""
    inv = pd.DataFrame(inv)                                          # 轉成一般 DataFrame：時間對齊由下面自行處理（週五資料→下一交易日）
    if isinstance(inv.index, pd.RangeIndex) or inv.index.name in inv.columns: df = inv.reset_index(drop=True)
    else: df = inv.reset_index()
    c = {k: next((x for x in df.columns if any(t in str(x) for t in ts)), None) for k, ts in
         dict(date=('date', '日期'), sid=('stock_id', '代號', 'symbol'), lv=('持股分級', 'level', '分級'),
              pct=('占集保庫存數比例', '比例', 'percent', 'proportion'), ppl=('人數', 'people', 'holders')).items()}
    print('集保欄位對應：', c, '；前 3 列：', df.head(3).to_dict('records'), flush=True)
    if None in (c['date'], c['sid'], c['lv'], c['pct']):
        raise SystemExit('集保資料欄位辨識失敗，請把上面兩行輸出貼給我。')
    df = df[[c['date'], c['sid'], c['lv'], c['pct']] + ([c['ppl']] if c['ppl'] else [])].copy()
    df.columns = ['date', 'sid', 'lv', 'pct'] + (['ppl'] if c['ppl'] else [])
    df['date'] = pd.to_datetime(df['date']); df = df[df.date >= pd.Timestamp(DATA_START) - pd.Timedelta(days=120)]
    df['sid'] = df['sid'].astype(str).str.strip(); df = df[df.sid.isin(set(map(str, cols)))]
    lv_map = {v: level_lower(v) for v in df['lv'].unique()}
    df['lo'] = df['lv'].map(lv_map); df['pct'] = pd.to_numeric(df['pct'], errors='coerce')
    tot_ppl = None
    if 'ppl' in df:
        df['ppl'] = pd.to_numeric(df['ppl'], errors='coerce')
        tot_ppl = pd.DataFrame(df[df.lo.notna()].pivot_table(index='date', columns='sid', values='ppl', aggfunc='sum'))
    d = df[df.lo.notna()]
    piv = lambda m: pd.DataFrame(d[m].pivot_table(index='date', columns='sid', values='pct', aggfunc='sum'))
    out = {'大戶400': piv(d.lo >= 400001), '大戶1000': piv(d.lo >= 1000001), '散戶50': piv(d.lo < 50001)}
    if tot_ppl is not None: out['股東人數'] = tot_ppl
    # 週五資料、週末公布 → 下一個交易日起才可用
    res = {}
    for k, w in out.items():
        w = pd.DataFrame(w).sort_index(); avail = idx[idx.searchsorted(w.index + pd.Timedelta(days=1))[:len(w)].clip(max=len(idx) - 1)]
        w.index = avail; w = w[~w.index.duplicated(keep='last')]
        res[k] = w.reindex(idx.union(w.index)).ffill().reindex(idx).reindex(columns=cols)
        res[k + '_週'] = w.reindex(columns=cols)                       # 週資料（用來算 4／12 週變化）
    print('集保資料期間', min(v.index.min() for v in out.values()).date(), '～', max(v.index.max() for v in out.values()).date(),
          '；有資料的股票', int(res['大戶400'].iloc[-1].notna().sum()), flush=True)
    return res


def weekly_change(Hd, key, n, idx, pct_change=False):
    w = Hd[key + '_週']
    ch = (w / w.shift(n) - 1) * 100 if pct_change else w - w.shift(n)
    return ch.reindex(idx.union(ch.index)).ffill().reindex(idx)


if True:  # finlab 線上環境直接執行
    t0 = time.time()
    D = fix_jumps(load())
    idx, cols = D['C'].index, D['C'].columns
    Hd = build_hold(load_inventory(), idx, cols)
    X = prepare(D, 0)
    F = {'大戶≥400張 4 週變化（百分點）': weekly_change(Hd, '大戶400', 4, idx),
         '大戶≥400張 12 週變化': weekly_change(Hd, '大戶400', 12, idx),
         '大戶≥1000張 4 週變化': weekly_change(Hd, '大戶1000', 4, idx),
         '散戶<50張 4 週變化（負＝散戶減少）': weekly_change(Hd, '散戶50', 4, idx),
         '大戶≥400張 持股比例（水準）': Hd['大戶400']}
    if '股東人數' in Hd: F['股東人數 4 週變化 %（負＝籌碼集中）'] = weekly_change(Hd, '股東人數', 4, idx, pct_change=True)
    # 強勢股（族群過濾後分數前 10%）
    C, O = D['C'], D['O']
    sc_all = pd.DataFrame(X['RKall'], index=idx, columns=cols)              # 排名（含一字跌停排除）
    raw = pd.DataFrame(-X['RKs'], index=idx, columns=cols)                    # 只為取得事件遮罩
    ev = raw.notna()
    nstock = (~np.isnan(X['RKall'])).sum(1)
    strong = sc_all.le(np.ceil(nstock * 0.10), axis=0)
    ent = O.shift(-1); Cf = C.ffill(); d0 = pd.Timestamp(EVAL_START)
    print(f'\n資料準備完成 {time.time() - t0:.0f} 秒；強勢股每日約 {int(strong.loc[d0:].sum(1).median())} 檔，短線事件每年約 {int(ev.loc[d0:].sum().sum() / 10.7)} 次', flush=True)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    for grp, G in (('強勢股', strong), ('短線事件（隔日開盤買）', ev)):
        rows = []
        for h in (20, 60):
            R = (Cf.shift(-h) / ent - 1).loc[d0:]
            base = R.where(strong.loc[d0:]).mean(1)
            for k, Xf in F.items():
                g = G.loc[d0:] & R.notna() & Xf.loc[d0:].notna()
                q = Xf.loc[d0:].where(g).rank(axis=1, pct=True) if grp == '強勢股' else None
                if grp != '強勢股':                                          # 事件很少，改用全期三等分門檻
                    v = Xf.loc[d0:].where(g).stack().dropna(); a, b = v.quantile([1 / 3, 2 / 3])
                    lab = pd.DataFrame(np.select([Xf.loc[d0:] <= a, Xf.loc[d0:] <= b], [0, 1], 2), index=R.index, columns=cols).where(g)
                else:
                    lab = (q > 1 / 3).astype(float) + (q > 2 / 3).astype(float); lab = lab.where(g)
                o = dict(持有=h, 因子=k)
                for i, nm in enumerate(('低', '中', '高')):
                    m = lab == i; ex = R.sub(base, axis=0).where(m).stack().dropna(); rr = R.where(m).stack().dropna()
                    yr = ex.groupby(ex.index.get_level_values(0).year).mean()
                    o[f'{nm}：超額'] = f'{ex.mean():+.2%}'; o[f'{nm}：正年'] = f'{(yr > 0).sum()}/{len(yr)}'
                    if h == 60: o[f'{nm}：崩/噴'] = f'{(rr < -.2).mean():.0%}/{(rr > .3).mean():.0%}'
                rows.append(o)
        print(f'\n══ {grp}：依集保因子分三組（相對同日強勢股平均）'); print(pd.DataFrame(rows).to_string(index=False), flush=True)
    for code, dt, lab in (('6223', '2025-09-26', '旺矽'), ('6215', '2025-02-18', '和椿'), ('4743', '2020-04-29', '合一')):
        if code in cols:
            d_ = idx[idx.searchsorted(pd.Timestamp(dt))]
            print(lab, d_.date(), {k: round(float(v.at[d_, code]), 2) if pd.notna(v.at[d_, code]) else None for k, v in F.items()})
    print(f'\n完成，{time.time() - t0:.0f} 秒。請把「══」兩張表和最後三行貼給我。')
