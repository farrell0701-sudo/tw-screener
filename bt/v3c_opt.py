# -*- coding: utf-8 -*-
"""
V3C 系列策略優化（finlab 單檔版，整份貼到 finlab 環境執行）

方法（與 EMA12 優化相同）
  訓練期 2010-01 ~ 2017-12：選因子與參數；測試期 2018-01 起：只驗證
  每個設定用 3 個錯開的起始日平均；參數格點以「平台分數」選
執行
  預設 t 日收盤決策、t+1 收盤成交（成交日漲停鎖死換下一位、跌停鎖死續抱）；另測「訊號當天收盤成交」
  每格 = 權益 ÷ 格數，不再平衡；手續費 0.0225%/邊、證交稅 0.405%；滑價依 20 日均成交值分層
預估時間：特徵約 1 分鐘、因子階段約 5 分鐘、參數格點約 20 分鐘
"""
import time
import itertools
import numpy as np
import pandas as pd

TRAIN = ('2010-01-01', '2017-12-31')
TEST = ('2018-01-01', '2099-12-31')
DATA_START = '2007-01-01'
N_STARTS, STAGGER = 3, 40

# V3C-PRO 規格
PRO = dict(
    slots=8, exec_delay=1,
    slope_min=0.1, use_brk=True, use_lu=True, use_ind=True, mkt='slope60',
    pos_max=-0.12, use_rescue=True, rescue_atr=2.5, min_adv=3e6,
    items=('s_atr', 's_dist', 's_adv', 's_px'), min_score=2,
    rank='score_price',                     # score_price / score_ld / ld / price / random
    mult=4.0, dmin=0.04, dmax=0.15, be_trig=1.06, be_lvl=1.005,
    ts_days=0, ts_ret=0.15,
    fee=0.000225, tax=0.00405, tiers=((3e6, 0.005), (1e7, 0.003), (np.inf, 0.001)), seed=0,
)
# 使用者 finlab 改良版：無突破、無排除漲停、評分只有 ATR 與成交值且不設門檻、時間停損 120 天 / 15%
USER = dict(PRO, use_brk=False, use_lu=False, items=('s_atr', 's_adv'), min_score=0, ts_days=120)


# ───────── 資料 ─────────
def load_finlab():
    from finlab import data
    raw = data.get('price:收盤價'); adj = data.get('etl:adj_close')
    hi = data.get('price:最高價'); lo = data.get('price:最低價'); op = data.get('price:開盤價')
    vol = data.get('price:成交股數')
    sc = data.get('security_categories')
    sc = sc.set_index('stock_id') if 'stock_id' in sc.columns else sc
    bad = set(sc[sc['category'].astype(str).str.contains('存託憑證|ETF|ETN|受益', regex=True)].index.astype(str))
    cols = [c for c in raw.columns if len(str(c)) == 4 and str(c).isdigit() and not str(c).startswith('0')
            and str(c) not in bad and all(c in x.columns for x in (adj, hi, lo, vol))]
    f = lambda x: x.reindex(columns=cols).loc[DATA_START:]
    raw, adj, hi, lo, op, vol = map(f, (raw, adj, hi, lo, op, vol))
    hi = np.fmax(np.fmax(hi, op), raw); lo = np.fmin(np.fmin(lo, op), raw)
    ix = None
    for key, col in (('stock_index_price:收盤指數', '上市發行量加權股價指數'),
                     ('benchmark_return:發行量加權股價報酬指數', None)):
        try:
            s = data.get(key)
            s = (s[col] if col else (s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s)).astype(float).dropna()
            print(f'大盤資料 {key} 起始 {s.index[0].date()}', flush=True)
            if s.index[0] <= pd.Timestamp('2009-06-30'):
                ix = s.loc[DATA_START:]; break
        except Exception as e:  # noqa: BLE001
            print(f'大盤資料 {key} 讀取失敗：{e}', flush=True)
    if ix is None:
        raise SystemExit('找不到 2009 年以前就有資料的大盤序列，請回報。')
    ind_map = data.get('director_compensation:產業類別').ffill().iloc[-1].to_dict()
    return raw, adj, hi, lo, vol, ix, ind_map


def build_panel(raw, adj, hi, lo, vol, ix, ind_map):
    t0 = time.time()
    f = adj / raw; ah, al = hi * f, lo * f
    C = adj
    e200 = C.ewm(span=200, adjust=False).mean()
    slope = (e200 / e200.shift(10) - 1) * 100 / 10
    pc = C.shift()
    tr = pd.DataFrame(np.fmax(np.fmax((ah - al).values, (ah - pc).abs().values), (al - pc).abs().values),
                      index=C.index, columns=C.columns)
    atrp = tr.ewm(alpha=1 / 14, adjust=False).mean() / C * 100
    dist = C / ah.rolling(250, min_periods=200).max() - 1
    adv = (raw * vol).rolling(20, min_periods=15).mean()
    e12 = C.ewm(span=12, adjust=False).mean()
    atr = atrp * C / 100
    raw_ret = raw / raw.shift() - 1
    lim = np.where(C.index < pd.Timestamp('2015-06-01'), 0.065, 0.094)[:, None]
    e60 = ix.ewm(span=60, adjust=False).mean()
    mk = ((e60 / e60.shift(20) - 1) > 0).reindex(C.index).ffill()
    if mk.loc['2009-01-01':].isna().any():
        raise SystemExit('大盤濾網在 2009 年後有缺值，請回報。')
    mk2 = (ix > ix.ewm(span=120, adjust=False).mean()).reindex(C.index).ffill()
    B = lambda x: x.fillna(False).values.astype(bool)
    Fz = lambda x: np.clip(x.replace([np.inf, -np.inf], np.nan).values, -3e38, 3e38).astype(np.float32)
    cols = list(C.columns)
    P = dict(
        idx=C.index, cols=cols, C=Fz(C), RC=Fz(raw), ATRP=Fz(atrp), ADV=Fz(adv),
        slope=Fz(slope), dist=Fz(dist), chg=Fz(C / pc - 1),
        brk=B(C > C.shift(1).rolling(5).max()),
        LD=Fz((e12 - e200) / atr),
        valid=B(C.notna().cumsum() >= 250),
        LU=B((raw_ret > lim) & (raw >= hi)), LDN=B((raw_ret < -lim) & (raw <= lo)),
        MK=mk.fillna(False).values.astype(bool), MK2=mk2.fillna(False).values.astype(bool),
        ind_bad=np.array([ind_map.get(c) in ('生技醫療業', '通信網路業') for c in cols]),
        bench=ix.reindex(C.index).ffill(),
    )
    lastv = C.notna().values[::-1].argmax(axis=0); P['last'] = len(C) - 1 - lastv
    print(f'特徵面板完成 {time.time() - t0:.0f} 秒，{len(P["idx"])} 天 × {len(cols)} 檔', flush=True)
    return P


# ───────── 模擬 ─────────
def simulate(P, cfg, start, end):
    c = dict(PRO); c.update(cfg)
    idx = P['idx']; i0 = idx.searchsorted(pd.Timestamp(start)); i1 = idx.searchsorted(pd.Timestamp(end), side='right')
    C, RC, LU, LDN = P['C'], P['RC'], P['LU'], P['LDN']
    n = C.shape[1]; last = P['last']; d = c['exec_delay']
    rng = np.random.default_rng(c['seed'])
    hold = np.zeros(n, bool); ec = np.zeros(n); peak = np.zeros(n); dst = np.zeros(n); age = np.zeros(n, int)
    sh = np.zeros(n); cost = np.zeros(n); sl = np.zeros(n); lastpx = np.full(n, np.nan)
    q_sell = np.array([], int); q_buy = np.array([], int); q_slip = np.array([])
    cash = 1.0; eq = []; trades = []
    for t in range(i0, i1):
        row = C[t]; ok = np.isfinite(row); lastpx[ok] = row[ok]

        def do_sell(js):
            nonlocal cash
            for j in js:
                g = sh[j] * row[j] * (1 - sl[j]); pro = g * (1 - c['fee'] - c['tax'])
                cash += pro; trades.append(pro / cost[j] - 1); sh[j] = 0

        def do_buy(js, slips):
            nonlocal cash
            equity = cash + np.nansum(sh * np.nan_to_num(lastpx))
            for j, s in zip(js, slips):
                amt = equity / c['slots']; amt = amt if cash >= amt else cash * 0.99
                sh[j] = amt / (row[j] * (1 + s) * (1 + c['fee'])); cash -= amt; cost[j] = amt; sl[j] = s

        if d == 1:
            do_sell(q_sell); do_buy(q_buy, q_slip)
            q_sell = np.array([], int); q_buy = np.array([], int)
        for j in np.where(hold & (last == t))[0]:                 # 下市
            do_sell([j]); hold[j] = False
        if t + d >= i1:
            eq.append(cash + np.nansum(sh * np.nan_to_num(lastpx))); break
        nx = t + d
        nx_ok = np.isfinite(C[nx])
        # 出場
        act = hold & ok
        if act.any():
            age = np.where(act, age + 1, age)
            peak = np.where(act, np.fmax(peak, row), peak)
            hit = row < peak * (1 - dst)
            if c['be_trig']:
                hit |= (peak >= ec * c['be_trig']) & (row < ec * c['be_lvl'])
            if c['ts_days']:
                hit |= (age >= c['ts_days']) & (row < ec * (1 + c['ts_ret']))
            ex = act & hit & nx_ok
            if d == 1:
                ex &= ~LDN[nx]
            js = np.where(ex)[0]; hold[ex] = False
            if d == 1:
                q_sell = js
            else:
                do_sell(js)
        # 進場
        free = c['slots'] - int(hold.sum())
        if free > 0:
            E = P['valid'][t] & (P['slope'][t] >= c['slope_min']) & (P['ADV'][t] >= c['min_adv']) & np.isfinite(P['ATRP'][t])
            pm = P['dist'][t] >= c['pos_max']
            pos = pm | ((P['ATRP'][t] <= c['rescue_atr']) & ~pm) if c['use_rescue'] else pm
            E &= pos
            if c['use_brk']: E &= P['brk'][t]
            if c['use_lu']: E &= P['chg'][t] < 0.095
            if c['use_ind']: E &= ~P['ind_bad']
            if c['mkt'] == 'slope60': E &= P['MK'][t]
            elif c['mkt'] == 'above120': E &= P['MK2'][t]
            sc = np.zeros(n, int)
            for k in c['items']:
                if k == 's_atr': sc += P['ATRP'][t] <= 3.3
                elif k == 's_dist': sc += P['dist'][t] >= -0.08
                elif k == 's_adv': sc += P['ADV'][t] < 2e7
                elif k == 's_px': sc += RC[t] < 30
            E &= sc >= c['min_score']
            cand = np.where(E & ~hold)[0]
            if len(cand):
                rk = c['rank']
                if rk == 'score_price':
                    order = np.lexsort((RC[t, cand], -sc[cand]))
                elif rk == 'score_ld':
                    order = np.lexsort((-np.nan_to_num(P['LD'][t, cand], nan=-9), -sc[cand]))
                elif rk == 'ld':
                    order = np.argsort(-np.nan_to_num(P['LD'][t, cand], nan=-9), kind='stable')
                elif rk == 'price':
                    order = np.argsort(RC[t, cand], kind='stable')
                else:
                    order = rng.permutation(len(cand))
                cand = cand[order]
                blk = ~nx_ok[cand] | (LU[nx, cand] if d == 1 else False)
                ch = cand[~blk][:free]
                a = P['ADV'][t, ch]
                slips = np.array([next(s for lim_, s in c['tiers'] if v < lim_) for v in a])
                hold[ch] = True; ec[ch] = row[ch]; peak[ch] = row[ch]; age[ch] = 0
                dst[ch] = np.clip(P['ATRP'][t, ch] * c['mult'] / 100, c['dmin'], c['dmax'])
                if d == 1:
                    q_buy, q_slip = ch, slips
                else:
                    do_buy(ch, slips)
        eq.append(cash + np.nansum(sh * np.nan_to_num(lastpx)))
    return pd.Series(eq, index=idx[i0:i0 + len(eq)]), np.array(trades)


def stats(E, T=None):
    r = E.pct_change().dropna(); yrs = (E.index[-1] - E.index[0]).days / 365.25
    cagr = (E.iloc[-1] / E.iloc[0]) ** (1 / yrs) - 1; mdd = (E / E.cummax() - 1).min()
    o = dict(CAGR=cagr, MDD=mdd, Sharpe=r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0,
             Calmar=cagr / -mdd if mdd < 0 else 0)
    if T is not None and len(T):
        o.update(N=len(T), Win=(T > 0).mean())
    return o


def multi(P, cfg, a, b):
    idx = P['idx']; i = idx.searchsorted(pd.Timestamp(a)); res = []
    for k in range(N_STARTS):
        E, T = simulate(P, cfg, idx[min(i + k * STAGGER, len(idx) - 1)], b); res.append(stats(E, T))
    return pd.DataFrame(res).mean().to_dict()


def both(P, cfg, label, rows):
    out = dict(label=label)
    for nm, (a, b) in (('訓練', TRAIN), ('測試', TEST)):
        out.update({f'{nm}_{k}': v for k, v in multi(P, cfg, a, b).items()})
    rows.append(out)
    print(f"{label:<34} 訓練 {out['訓練_CAGR']:6.1%} / {out['訓練_MDD']:6.1%} / Sh {out['訓練_Sharpe']:.2f}   "
          f"測試 {out['測試_CAGR']:6.1%} / {out['測試_MDD']:6.1%} / Sh {out['測試_Sharpe']:.2f}", flush=True)
    return out


def run_all(P, n_random=10):
    t0 = time.time()
    for nm, (a, z) in (('訓練', TRAIN), ('測試', TEST)):
        s = stats(P['bench'].loc[a:z].dropna())
        print(f'大盤 {nm}期 年化 {s["CAGR"]:.1%} 回撤 {s["MDD"]:.1%} Sharpe {s["Sharpe"]:.2f}')
    i0, i1 = P['idx'].searchsorted(pd.Timestamp(TRAIN[0])), P['idx'].searchsorted(pd.Timestamp(TRAIN[1]))
    print(f'訓練期大盤濾網偏多天數 {P["MK"][i0:i1].mean():.0%}（應約 50%～75%）', flush=True)

    print('\n① 版本對照'); R = []
    both(P, PRO, 'V3C-PRO（t+1 收盤成交）', R)
    both(P, dict(PRO, exec_delay=0), 'V3C-PRO（訊號當天收盤成交）', R)
    both(P, USER, '你的 finlab 改良版', R)

    print('\n② 你的改良版：每項改動單獨加在 PRO 上')
    both(P, dict(PRO, use_brk=False), 'PRO 拿掉 5 日新高', R)
    both(P, dict(PRO, use_lu=False), 'PRO 拿掉 排除漲停', R)
    both(P, dict(PRO, items=('s_atr', 's_adv'), min_score=0), 'PRO 評分改 2 項、無門檻', R)
    both(P, dict(PRO, ts_days=120), 'PRO 加 時間停損 120 天 / 15%', R)

    print('\n③ PRO 因子逐項開關')
    for k in ('s_atr', 's_dist', 's_adv', 's_px'):
        both(P, dict(PRO, items=tuple(i for i in PRO['items'] if i != k)), f'評分拿掉 {k}', R)
    for ms in (1, 3):
        both(P, dict(PRO, min_score=ms), f'評分門檻 {ms}', R)
    both(P, dict(PRO, use_ind=False), '不排除生技、通訊', R)
    both(P, dict(PRO, use_rescue=False), '取消救回組', R)
    both(P, dict(PRO, mkt=False), '不用大盤濾網', R)
    both(P, dict(PRO, mkt='above120'), '大盤濾網改 站上 EMA120', R)
    for s in (0.05, 0.15):
        both(P, dict(PRO, slope_min=s), f'EMA200 斜率門檻 {s}', R)
    for a in (1e7, 3e7):
        both(P, dict(PRO, min_adv=a), f'均額門檻 {a / 1e4:.0f} 萬', R)

    print('\n④ 排序方式')
    for rk, l in (('score_ld', '分數 → LD'), ('ld', '只用 LD'), ('price', '只用低價')):
        both(P, dict(PRO, rank=rk), l, R)
    rs = [both(P, dict(PRO, rank='random', seed=s), f'隨機 #{s}', []) for s in range(n_random)]
    rr = pd.DataFrame(rs)
    R.append(dict(label=f'隨機 ×{n_random} 中位數', **rr.drop(columns='label').median().to_dict()))
    print(f"隨機中位數：訓練 {rr['訓練_CAGR'].median():.1%}  測試 {rr['測試_CAGR'].median():.1%}")
    pd.DataFrame(R).to_csv('v3c_opt_factors.csv', index=False, encoding='utf-8-sig')
    print(f'因子階段完成 {(time.time() - t0) / 60:.1f} 分鐘')

    print('\n⑤ 出場參數格點（只看訓練期）')
    G = dict(mult=[3.0, 4.0, 5.0], dmax=[0.12, 0.15, 0.20], be_trig=[0, 1.04, 1.06, 1.08], slots=[8, 10])
    keys = list(G); rows = []
    for i, v in enumerate(itertools.product(*G.values()), 1):
        k = dict(zip(keys, v))
        s = multi(P, dict(PRO, **k), *TRAIN)
        rows.append(dict(k, **{f'訓練_{a}': x for a, x in s.items()}))
        if i % 24 == 0:
            print(f'  {i}/72  {(time.time() - t0) / 60:.1f} 分鐘', flush=True)
    D = pd.DataFrame(rows)

    def plateau(r):
        dist = np.zeros(len(D)); m = np.ones(len(D), bool)
        for k in keys:
            pos = {v: i for i, v in enumerate(G[k])}
            dd = (D[k].map(pos) - pos[r[k]]).abs().values; dist += dd; m &= dd <= 1
        return D.loc[m & (dist <= 1), '訓練_Calmar'].mean()
    D['平台Calmar'] = D.apply(plateau, axis=1)
    D = D.sort_values('平台Calmar', ascending=False)
    D.to_csv('v3c_opt_grid.csv', index=False, encoding='utf-8-sig')
    print(D.head(8).round(3).to_string(index=False))

    print('\n⑥ 平台前 3 名在測試期（第一次看測試期）'); R6 = []
    for r in D.head(3).itertuples():
        k = {kk: getattr(r, kk) for kk in keys}
        both(P, dict(PRO, **k), ' '.join(f'{kk}={k[kk]}' for kk in keys), R6)
    both(P, PRO, '（對照）V3C-PRO 原參數', R6)
    pd.DataFrame(R6).to_csv('v3c_opt_final.csv', index=False, encoding='utf-8-sig')
    print(f'全部完成 {(time.time() - t0) / 60:.1f} 分鐘')


try:
    import finlab  # noqa: F401
    _HAS = True
except ImportError:
    _HAS = False
if _HAS:
    P = build_panel(*load_finlab())
    run_all(P)
