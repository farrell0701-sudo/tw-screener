# -*- coding: utf-8 -*-
"""
EMA12 策略優化（finlab 單檔版，整份貼到 finlab 環境執行）

方法
  訓練期 2010-01 ~ 2017-12：用來選因子與參數
  測試期 2018-01 ~ 2026-xx：只用來驗證，選擇時完全不看
  參數只用整齊格點；以「訓練期 + 相鄰格點平均」選，避免挑到單一尖峰
  每個設定用 3 個錯開的起始日平均，降低路徑運氣（單次回測換個起點就可能差 ±10%）
  第二輪（修正大盤資料後）：大盤濾網重測約 3 分鐘，參數格點 240 組約 15～20 分鐘

帳務（所有測試相同）
  t 日收盤決策、t+1 日收盤成交；漲停鎖死買不到（隔日重試一次）、跌停鎖死賣不掉（續抱）
  每格 = 當時權益 ÷ 格數，持股不再平衡；閒置資金留現金
  手續費每邊 0.0225%、賣出證交稅 0.405%（同 V3C-PRO）、滑價每邊 0.1%
  停牌續抱；資料結束（下市）以最後收盤出場

輸出
  ① 原參數基準  ② 因子逐項開關  ③ 排序方式  ④ 參數格點（含平台分數）  ⑤ 最終候選在測試期的表現
  全部存成 ema12_opt_*.csv
"""
import time
import itertools
import numpy as np
import pandas as pd

TRAIN = ('2010-01-01', '2017-12-31')
TEST = ('2018-01-01', '2099-12-31')
DATA_START = '2007-01-01'

BASE = dict(
    slots=10, min_amt=4.646e6, min_score=2, chase=2.0,
    trail=3.707, act=3.16, floor=1.404, max_ld=4.63, cluster=0.2,
    use_cross=True, use_slope=True, use_align=True, use_chase=True,
    items=('s_up', 's_pull', 's_gap', 's_ld'),       # 評分項目
    extra=(),                                         # 額外的硬條件（見 COND）
    mkt=False,                                        # 大盤濾網（加權 EMA60 二十日上升才進場）
    rank='ld',                                        # ld / ld_cluster / low_price / ret60 / random
    fee=0.000225, tax=0.00405, slip=0.001, seed=0,
)
BASE_ORIG = dict(BASE, rank='ld_cluster')             # 原始規則


# ───────── 資料與特徵 ─────────
def load_finlab():
    from finlab import data
    with data.universe(market='TSE_OTC'):
        close = data.get('etl:adj_close'); high = data.get('etl:adj_high'); low = data.get('etl:adj_low')
        amt = data.get('price:成交金額'); raw_c = data.get('price:收盤價')
        raw_h = data.get('price:最高價'); raw_l = data.get('price:最低價'); vol = data.get('price:成交股數')
    cols = [c for c in close.columns if len(str(c)) == 4 and str(c).isdigit() and not str(c).startswith('0')]
    f = lambda x: x.reindex(columns=cols).loc[DATA_START:]
    close, high, low, amt, raw_c, raw_h, raw_l, vol = map(f, (close, high, low, amt, raw_c, raw_h, raw_l, vol))
    # 大盤：優先用加權指數；若起始太晚，改用發行量加權股價報酬指數（含息，只用來判斷趨勢方向）
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
        raise SystemExit('找不到 2009 年以前就有資料的大盤序列，訓練期的大盤濾網會失效，請回報。')
    cat = data.get('security_categories')
    cat_map = dict(zip(cat['stock_id'].astype(str), cat['category']))
    return close, high, low, amt, raw_c, raw_h, raw_l, vol, ix, cat_map


def build_panel(close, high, low, amt, raw_c, raw_h, raw_l, vol, ix, cat_map):
    t0 = time.time()
    ema = lambda n: close.ewm(span=n, adjust=False).mean()
    e12, e144, e169, e200 = ema(12), ema(144), ema(169), ema(200)
    pc = close.shift()
    tr = pd.DataFrame(np.fmax(np.fmax((high - low).values, (high - pc).abs().values), (low - pc).abs().values),
                      index=close.index, columns=close.columns)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    valid = close.notna().cumsum() >= 200
    gap = e12 - e144
    raw_ret = raw_c / raw_c.shift() - 1
    lim = pd.Series(np.where(close.index < pd.Timestamp('2015-06-01'), 0.065, 0.094), index=close.index)
    vr = vol / vol.rolling(20).mean()
    e60 = ix.ewm(span=60, adjust=False).mean()
    mk = ((e60 / e60.shift(20) - 1) > 0).reindex(close.index).ffill()
    if mk.loc['2009-01-01':].isna().any():
        raise SystemExit('大盤濾網在 2009 年後有缺值，請回報。')
    mk = mk.fillna(False)
    mk2 = (ix > ix.ewm(span=120, adjust=False).mean()).reindex(close.index).ffill().fillna(False)
    B = lambda x: x.fillna(False).values.astype(bool)
    Fz = lambda x: np.clip(x.replace([np.inf, -np.inf], np.nan).values, -3e38, 3e38).astype(np.float32)
    P = dict(
        idx=close.index, cols=list(close.columns),
        C=Fz(close), H=Fz(high), A=Fz(atr), RC=Fz(raw_c),
        cross=B((close > e12) & (pc <= e12.shift())),
        slope=B(e200 > e200.shift()),
        align=B((e12 > e144) & (e144 > e169) & (e169 > e200)),
        valid=B(valid),
        Z=Fz((close - e12) / atr),                                   # 距 EMA12 幾倍 ATR（追價）
        LD=Fz((e12 - e200) / atr),                                   # 距長線幾倍 ATR
        amt20=Fz(amt.rolling(20).mean()),
        s_up=B(e12 > e12.shift(3)),
        s_pull=B(low.rolling(9).min() >= e144),
        s_gap=B(gap > gap.shift(5)),
        # 候選新因子（硬條件）
        c_near=B(close / high.rolling(250).max() - 1 >= -0.12),      # 距 250 日高 12% 內
        c_calm=B(vr.shift(1).rolling(5).max() <= 3),                 # 前 5 日不爆量
        c_e60=B(close > close.ewm(span=60, adjust=False).mean()),    # 收盤在 EMA60 上
        R60=Fz(close / close.shift(60) - 1),
        LU=B((raw_ret > lim.values[:, None]) & (raw_c >= raw_h)),
        LDN=B((raw_ret < -lim.values[:, None]) & (raw_c <= raw_l)),
        MK=mk.values.astype(bool), MK2=mk2.values.astype(bool),
    )
    codes = pd.factorize(pd.Series([cat_map.get(c) for c in P['cols']]))[0]
    P['codes'] = codes; P['K'] = codes.max() + 2; P['cc'] = np.where(codes >= 0, codes, P['K'] - 1)
    lastv = close.notna().values[::-1].argmax(axis=0)
    P['last'] = len(close) - 1 - lastv                               # 每檔最後一天有收盤
    P['bench'] = ix.reindex(close.index).ffill()
    print(f'特徵面板完成 {time.time() - t0:.0f} 秒，{len(P["idx"])} 天 × {len(P["cols"])} 檔', flush=True)
    return P


# ───────── 模擬 ─────────
def simulate(P, cfg, start, end):
    """原始狀態機（t 日決策、t+1 收盤成交，成交日漲停換下一位、跌停續抱），帳務改為固定格數。"""
    c = dict(BASE); c.update(cfg)
    idx = P['idx']; i0 = idx.searchsorted(pd.Timestamp(start)); i1 = idx.searchsorted(pd.Timestamp(end), side='right')
    C, H, A, RC, LU, LDN = P['C'], P['H'], P['A'], P['RC'], P['LU'], P['LDN']
    n = C.shape[1]; last = P['last']
    LDs = np.where(np.isfinite(P['LD']), P['LD'], -np.inf)
    rng = np.random.default_rng(c['seed'])
    hold = np.zeros(n, bool); ep = np.zeros(n); ea = np.zeros(n); hh = np.zeros(n)
    retry = np.zeros(n, bool)
    sh = np.zeros(n); cost = np.zeros(n); edate = np.zeros(n, int); lastpx = np.full(n, np.nan)
    q_sell = np.array([], int); q_buy = np.array([], int)
    cash = 1.0; eq = []; trades = []
    items = list(c['items'])
    for t in range(i0, i1):
        row = C[t]; ok = np.isfinite(row); lastpx[ok] = row[ok]
        # ── 執行昨天的決定（今天收盤成交）──
        for j in q_sell:
            g = sh[j] * row[j] * (1 - c['slip']); pro = g * (1 - c['fee'] - c['tax'])
            cash += pro; trades.append((P['cols'][j], edate[j], t, pro / cost[j] - 1)); sh[j] = 0
        if len(q_buy):
            equity = cash + np.nansum(sh * np.nan_to_num(lastpx))
            for j in q_buy:
                amt = equity / c['slots']; amt = amt if cash >= amt else cash * 0.99
                px = row[j] * (1 + c['slip']); sh[j] = amt / (px * (1 + c['fee']))
                cash -= amt; cost[j] = amt; edate[j] = t
        q_sell = np.array([], int); q_buy = np.array([], int)
        # 下市：資料最後一天以收盤出場
        for j in np.where(hold & (last == t))[0]:
            g = sh[j] * row[j] * (1 - c['slip']); pro = g * (1 - c['fee'] - c['tax'])
            cash += pro; trades.append((P['cols'][j], edate[j], t, pro / cost[j] - 1)); sh[j] = 0; hold[j] = False
        eq.append(cash + np.nansum(sh * np.nan_to_num(lastpx)))
        if t + 1 >= i1:
            break
        nx_ok = np.isfinite(C[t + 1])
        # ── 今天收盤決策 ──
        act = hold & ok
        if act.any():
            hh = np.where(act, np.fmax(hh, H[t]), hh)
            stop = hh - c['trail'] * ea
            stop = np.where(hh >= ep + c['act'] * ea, np.maximum(stop, ep + c['floor'] * ea), stop)
            ex = act & (row < stop) & ~LDN[t + 1] & nx_ok        # 明天跌停或停牌 → 續抱
            q_sell = np.where(ex)[0]; hold[ex] = False
        E = P['valid'][t] & (P['amt20'][t] > c['min_amt'])
        if c['use_cross']: E = E & P['cross'][t]
        if c['use_slope']: E = E & P['slope'][t]
        if c['use_align']: E = E & P['align'][t]
        if c['use_chase']: E = E & (P['Z'][t] <= c['chase'])
        for k in c['extra']:
            E = E & P[k][t]
        if c['mkt']:
            E = E & (P['MK2'][t] if c['mkt'] == 'above120' else P['MK'][t])
        if items:
            sc = np.zeros(n, int)
            for k in items:
                sc += (P['LD'][t] < c['max_ld']) if k == 's_ld' else P[k][t]
            E = E & (sc >= min(c['min_score'], len(items)))
        sig = E | retry
        retry[:] = False
        cand = np.where(sig & ~hold & np.isfinite(A[t]))[0]
        free = c['slots'] - int(hold.sum())
        if len(cand) and free > 0:
            if c['rank'] in ('ld', 'ld_cluster'):
                pr = LDs[t, cand].astype(float)
                if c['rank'] == 'ld_cluster' and len(cand) > free:
                    K = P['K']; cc = P['cc']
                    ns_ = np.bincount(cc[sig], minlength=K); nh_ = np.bincount(cc[hold], minlength=K)
                    pr = pr + c['cluster'] * (((ns_[cc[cand]] - 1 + nh_[cc[cand]]) >= 1) & (P['codes'][cand] >= 0))
            elif c['rank'] == 'ld_bonus':                  # LD + 條件加分（條件成立的優先）
                pr = LDs[t, cand].astype(float)
                for key, w in c.get('bonus', ()):
                    pr = pr + w * np.nan_to_num(P[key][t, cand], nan=0.0)
            elif c['rank'] == 'ld_ov':                     # LD 扣掉重疊度懲罰
                pr = LDs[t, cand].astype(float) - c.get('ov_pen', 1.0) * np.nan_to_num(P['OVAB'][t, cand], nan=1.0)
            elif c['rank'] == 'low_price':
                pr = -RC[t, cand].astype(float)
            elif c['rank'] == 'ret60':
                pr = np.nan_to_num(P['R60'][t, cand], nan=-9).astype(float)
            else:
                pr = rng.random(len(cand))
            cand = cand[np.argsort(-pr, kind='stable')]
            blk = LU[t + 1, cand] | ~nx_ok[cand]                # 明天漲停鎖死或沒價 → 買不到
            retry[cand[:free][LU[t + 1, cand[:free]]]] = True   # 漲停買不到的隔日再試一次
            ch = cand[~blk][:free]
            hold[ch] = True; ep[ch] = row[ch]; ea[ch] = A[t, ch]; hh[ch] = row[ch]
            q_buy = ch
    E_ = pd.Series(eq, index=idx[i0:i0 + len(eq)])
    T_ = pd.DataFrame(trades, columns=['code', 'e', 'x', 'ret'])
    return E_, T_


def stats(E, T=None):
    r = E.pct_change().dropna(); yrs = (E.index[-1] - E.index[0]).days / 365.25
    cagr = (E.iloc[-1] / E.iloc[0]) ** (1 / yrs) - 1; mdd = (E / E.cummax() - 1).min()
    sh = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
    o = dict(CAGR=cagr, MDD=mdd, Sharpe=sh, Calmar=cagr / -mdd if mdd < 0 else 0)
    if T is not None and len(T):
        o.update(N=len(T), Win=(T.ret > 0).mean())
    return o


N_STARTS = 3          # 每個設定用幾個錯開的起始日（各差約 2 個月）平均，降低路徑運氣
STAGGER = 40          # 交易日


def multi(P, cfg, a, b):
    """同一設定、N_STARTS 個錯開起始日的平均指標"""
    idx = P['idx']; i = idx.searchsorted(pd.Timestamp(a)); res = []
    for k in range(N_STARTS):
        st = idx[min(i + k * STAGGER, len(idx) - 1)]
        E, T = simulate(P, cfg, st, b); res.append(stats(E, T))
    return pd.DataFrame(res).mean().to_dict()


def both(P, cfg, label, rows, extra=None):
    out = dict(label=label)
    for nm, (a, b) in (('訓練', TRAIN), ('測試', TEST)):
        s = multi(P, cfg, a, b)
        out.update({f'{nm}_{k}': v for k, v in s.items()})
    if extra: out.update(extra)
    rows.append(out)
    print(f"{label:<34} 訓練 {out['訓練_CAGR']:6.1%} / {out['訓練_MDD']:6.1%} / Sh {out['訓練_Sharpe']:.2f}   "
          f"測試 {out['測試_CAGR']:6.1%} / {out['測試_MDD']:6.1%} / Sh {out['測試_Sharpe']:.2f}", flush=True)
    return out


# 第一輪因子測試後的改良基準：只用 LD 排序（族群加分兩段都較差）、拿掉多餘的 EMA200 上升、參數取整
IMPROVED = dict(BASE, rank='ld', use_slope=False, min_amt=5e6, max_ld=4.5, chase=2.0)


def run_all(P, n_random=10):
    t0 = time.time()
    b = P['bench']
    for nm, (a, z) in (('訓練', TRAIN), ('測試', TEST)):
        s = stats(b.loc[a:z].dropna())
        print(f'大盤 {nm}期 年化 {s["CAGR"]:.1%} 回撤 {s["MDD"]:.1%} Sharpe {s["Sharpe"]:.2f}')
    mk_share = P['MK'][P['idx'].searchsorted(pd.Timestamp(TRAIN[0])):P['idx'].searchsorted(pd.Timestamp(TRAIN[1]))].mean()
    print(f'訓練期大盤濾網「偏多」天數比例 {mk_share:.0%}（合理範圍約 50%～70%）', flush=True)

    print('\n① 大盤濾網重測'); R1 = []
    both(P, BASE_ORIG, '原始規則', R1)
    both(P, IMPROVED, '改良基準（無濾網）', R1)
    both(P, dict(IMPROVED, mkt='slope60'), '改良 + 大盤 EMA60 上升', R1)
    both(P, dict(IMPROVED, mkt='above120'), '改良 + 大盤站上 EMA120', R1)
    rs = [both(P, dict(IMPROVED, rank='random', seed=s), f'改良、隨機排序 #{s}', []) for s in range(n_random)]
    rr = pd.DataFrame(rs)
    R1.append(dict(label=f'改良、隨機排序 ×{n_random} 中位數', **rr.drop(columns='label').median().to_dict()))
    print(f"隨機中位數：訓練 {rr['訓練_CAGR'].median():.1%}  測試 {rr['測試_CAGR'].median():.1%}")
    pd.DataFrame(R1).to_csv('ema12_opt2_filter.csv', index=False, encoding='utf-8-sig')

    print('\n② 參數格點（只看訓練期選，最後才看測試期）')
    G = dict(trail=[3.0, 3.5, 4.0, 4.5, 5.0], act=[2.0, 3.0, 4.0], floor=[1.0, 1.5, 2.0],
             slots=[8, 10], mkt=[False, 'slope60', 'above120'])
    keys = list(G); rows = []
    combos = [dict(zip(keys, v)) for v in itertools.product(*G.values())]
    combos = [k for k in combos if k['floor'] < k['act']]
    print(f'共 {len(combos)} 組', flush=True)
    for i, k in enumerate(combos, 1):
        s = multi(P, dict(IMPROVED, **k), *TRAIN)
        rows.append(dict(k, **{f'訓練_{a}': v for a, v in s.items()}))
        if i % 40 == 0:
            print(f'  {i}/{len(combos)}  已用 {(time.time() - t0) / 60:.1f} 分鐘', flush=True)
    D = pd.DataFrame(rows)
    D['mkt'] = D['mkt'].astype(str)
    Gs = dict(G, mkt=[str(v) for v in G['mkt']])

    def plateau(r):
        dist = np.zeros(len(D)); m = np.ones(len(D), bool)
        for k in keys:
            if k == 'mkt':                       # 濾網種類不是連續參數：只和同種類比較
                m &= (D[k] == r[k]).values; continue
            pos = {v: i for i, v in enumerate(Gs[k])}
            d = (D[k].map(pos) - pos[r[k]]).abs().values
            dist += d; m &= d <= 1
        return D.loc[m & (dist <= 1), '訓練_Calmar'].mean()
    D['平台Calmar'] = D.apply(plateau, axis=1)
    D = D.sort_values('平台Calmar', ascending=False)
    D.to_csv('ema12_opt2_grid.csv', index=False, encoding='utf-8-sig')
    print(D.head(10).round(3).to_string(index=False))
    print('\n各濾網種類的最佳平台：')
    print(D.groupby('mkt').head(1).round(3).to_string(index=False))

    print('\n③ 測試期驗證（第一次看測試期）：每種濾網的平台第 1 名 + 整體前 3 名')
    pick = pd.concat([D.groupby('mkt').head(1), D.head(3)]).drop_duplicates(subset=keys)
    R3 = []
    for r in pick.itertuples():
        k = {kk: getattr(r, kk) for kk in keys}
        k['mkt'] = False if k['mkt'] == 'False' else k['mkt']
        both(P, dict(IMPROVED, **k), ' '.join(f'{kk}={k[kk]}' for kk in keys), R3)
    both(P, BASE_ORIG, '（對照）原始規則', R3)
    both(P, IMPROVED, '（對照）改良基準', R3)
    pd.DataFrame(R3).to_csv('ema12_opt2_final.csv', index=False, encoding='utf-8-sig')
    print(f'全部完成，{(time.time() - t0) / 60:.1f} 分鐘')


if __name__ == '__main__' or True:
    try:
        import finlab  # noqa: F401
        _HAS_FINLAB = True
    except ImportError:
        _HAS_FINLAB = False
    if _HAS_FINLAB:
        P = build_panel(*load_finlab())
        run_all(P)
