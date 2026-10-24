"""乾淨版價格面板：FinMind 真正未還原價 ＋ 除權息／減資／面額變更事件 → 自己算還原價
取代 Yahoo AdjClose（有大量錯誤），並排除興櫃期間與資料瑕疵。

還原方式（向後還原：最新價格不動，事件日之前的價格乘上調整係數）
  除權息  TaiwanStockDividendResult                 係數 = 參考價 after_price ÷ 前收 before_price
  減資    TaiwanStockCapitalReductionReferencePrice  係數 = 恢復買賣參考價 ÷ 停止買賣前收盤
  面額變更 TaiwanStockSplitPrice                      係數 = after_price ÷ before_price
瑕疵日（bad）：還原後單日漲跌超過漲跌幅上限＋0.5%（2015-06-01 前 7%，之後 10%），
              或開盤價不在當天高低之間（興櫃資料的特徵）
可交易（eligible）：距離最近一個瑕疵日已超過 250 根 → 等同「掛牌滿一年、且近一年資料乾淨」
輸出 /home/claude/bt/fac/panel_clean.pkl（和 panel.pkl 同樣的鍵，另加 eligible、bad、n_events）"""
import os, glob
import numpy as np, pandas as pd

RAW_DIRS = ['/home/claude/rawx', '/home/claude/dlraw']
EV_DIR = '/home/claude/events'
H = '/home/claude/hdata'
OUT = '/home/claude/bt/fac/panel_clean.pkl'
CUT = '2026-10-16'


def event_factors(code):
    f = os.path.join(EV_DIR, f'ev_{code}.csv')
    if not os.path.exists(f): return None
    e = pd.read_csv(f)
    if not len(e) or 'dataset' not in e.columns: return pd.Series(dtype=float)
    out = []
    for _, r in e.iterrows():
        ds = r.get('dataset', '')
        num = den = np.nan
        if ds in ('TaiwanStockDividendResult', 'TaiwanStockSplitPrice'):
            num, den = r.get('after_price', np.nan), r.get('before_price', np.nan)
        elif ds == 'TaiwanStockCapitalReductionReferencePrice':
            num, den = r.get('PostReductionReferencePrice', np.nan), r.get('ClosingPriceonTheLastTradingDay', np.nan)
        try: num, den = float(num), float(den)
        except Exception: continue
        if num > 0 and den > 0 and abs(np.log(num / den)) > 1e-6:
            out.append((pd.Timestamp(r['date']), num / den))
    if not out: return pd.Series(dtype=float)
    s = pd.Series(dict(out)) if len({d for d, _ in out}) == len(out) else pd.DataFrame(out).groupby(0)[1].prod()
    return s.sort_index()


def load_one(code):
    for d in RAW_DIRS:
        f = os.path.join(d, f'{code}.csv')
        if os.path.exists(f):
            x = pd.read_csv(f, parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
            x = x[(x.index < CUT) & (x.c > 0) & (x.v > 0)]
            return x
    return None


def adjust(x, ev):
    """向後還原；回傳還原後 O/H/L/C 與調整係數"""
    adj = pd.Series(1.0, index=x.index)
    if ev is not None and len(ev):
        for d, f in ev.items():
            adj[x.index < d] *= f
    return adj


def build():
    lst = pd.read_csv(f'{H}/stock_list.csv', dtype={'code': str})
    tx = pd.read_csv(f'{H}/TAIEX.csv', parse_dates=['Date']).set_index('Date').Close
    tx = tx[tx.index < CUT]
    dl = pd.read_csv('/home/claude/dlraw/_delisted.csv', dtype={'stock_id': str})
    codes = sorted(set(lst.code) | set(dl.stock_id))
    codes = [c for c in codes if len(c) == 4 and c.isdigit() and not c.startswith('0')]
    S, nev, missing_ev = {}, {}, []
    for c in codes:
        x = load_one(c)
        if x is None or len(x) < 60: continue
        ev = event_factors(c)
        if ev is None: missing_ev.append(c)
        a = adjust(x, ev)
        d = pd.DataFrame({'O': x.o * a, 'H': x.h * a, 'L': x.l * a, 'C': x.c * a, 'RC': x.c, 'V': x.v.astype(float),
                          'AMT': x.tv.astype(float) if 'tv' in x else x.c * x.v})
        S[c] = d; nev[c] = 0 if ev is None else len(ev)
    cal = tx.index.union(pd.DatetimeIndex(sorted(set().union(*[s.index for s in S.values()]))))
    cal = cal[cal >= '2015-01-01']
    P = {k: pd.DataFrame({c: S[c][k] for c in S}).reindex(cal) for k in ('O', 'H', 'L', 'C', 'RC', 'V', 'AMT')}
    # 只保留多數股票有交易的日子
    n = P['C'].notna().sum(axis=1); ok = n >= 0.3 * n.rolling(20, min_periods=1).median()
    for k in P: P[k] = P[k][ok]
    C, O, Hh, L = P['C'], P['O'], P['H'], P['L']
    r = C / C.ffill().shift(1) - 1
    lim = pd.Series(np.where(C.index < '2015-06-01', 0.07, 0.10), index=C.index)
    viol = r.abs().gt(lim + 0.005, axis=0) & C.notna()
    incons = ((O > Hh * 1.0001) | (O < L * 0.9999) | (Hh < L)) & C.notna()
    bad = viol | incons
    since = bad.astype(float).cumsum()
    # 距離最近瑕疵日的根數（以該股自己的交易日計）
    cnt = C.notna().astype(float).cumsum()
    last_bad_cnt = cnt.where(bad).ffill().fillna(-1e9)
    eligible = C.notna() & ((cnt - last_bad_cnt) > 250) & (cnt >= 250)
    meta = lst.set_index('code')
    cols = C.columns
    P.update(eligible=eligible, bad=bad, viol=viol, incons=incons,
             industry=meta.industry.reindex(cols), market=meta.market.reindex(cols).fillna('DL'),
             taiex=tx.reindex(C.index).ffill(), n_events=pd.Series(nev).reindex(cols), missing_ev=missing_ev)
    return P


if __name__ == '__main__':
    import time; t0 = time.time()
    P = build(); pd.to_pickle(P, OUT)
    C = P['C']
    print(f'{C.shape}，{time.time() - t0:.0f} 秒；缺事件檔 {len(P["missing_ev"])} 檔')
    print('超過漲跌幅（還原後）', int(P['viol'].sum().sum()), '次，', int(P['viol'].any().sum()), '檔；開盤不在高低間', int(P['incons'].sum().sum()), '次')
    print('每日可交易股票數（中位）', int(P['eligible'].sum(1).loc['2016':].median()))
