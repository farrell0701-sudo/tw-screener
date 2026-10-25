"""基本面面板（對齊 panel_clean.pkl 的日期 × 代號），全部只用「當時已公布」的資料
來源：/home/claude/fund（fund-data 分支：per_／is_／bs_／rev_），revenue-data 分支 revenue.csv（現存股月營收）
公布延遲（保守版，涵蓋金融保險業較晚的期限）：
  季報 Q1(3/31)→6/1、Q2(6/30)→9/1、Q3(9/30)→12/1、Q4(12/31)→次年 4/1 起才可用
  月營收：FinMind 的 date 欄＝公布月份的 1 號；保守起見從該月 11 號起可用
  本益比／淨值比／殖利率：證交所每日公布，當天收盤後即可用
輸出 /home/claude/bt/fac/fund.pkl：dict(因子名 → DataFrame)"""
import os, glob
_rc = pd.read_csv if False else None
import numpy as np, pandas as pd
_read = pd.read_csv
def _safe(f, **k):
    try: return _read(f, **k)
    except pd.errors.EmptyDataError: return pd.DataFrame()
    finally: pass

FD = '/home/claude/fund'
OUT = '/home/claude/bt/fac/fund.pkl'


def avail_date(q):
    q = pd.Timestamp(q)
    return {3: pd.Timestamp(q.year, 6, 1), 6: pd.Timestamp(q.year, 9, 1),
            9: pd.Timestamp(q.year, 12, 1), 12: pd.Timestamp(q.year + 1, 4, 1)}[q.month]


def to_daily(s_by_avail, idx):
    """依可用日排序的序列 → 每日向前填補"""
    if s_by_avail is None or not len(s_by_avail): return pd.Series(np.nan, index=idx)
    s = s_by_avail[~s_by_avail.index.duplicated(keep='last')].sort_index()
    return s.reindex(idx.union(s.index)).ffill().reindex(idx)


def build(P):
    idx, cols = P['C'].index, P['C'].columns
    out = {k: {} for k in ('PER', 'PBR', 'DY', 'NI_TTM', 'EQ', 'SH', 'REV3', 'REV3_LY', 'REV12', 'GM_TTM', 'OPM_TTM')}
    rev_cur = None
    rp = '/tmp/claude-0/rev.csv'
    if os.path.exists(rp):
        rev_cur = pd.read_csv(rp, dtype={'code': str})
    for c in cols:
        f = f'{FD}/per_{c}.csv'
        if os.path.exists(f):
            p = _safe(f)
            if len(p) and 'date' in p:
                p['date'] = pd.to_datetime(p['date']); p = p.drop_duplicates('date').set_index('date')
                for k, col in (('PER', 'PER'), ('PBR', 'PBR'), ('DY', 'dividend_yield')):
                    out[k][c] = p[col].replace(0, np.nan).reindex(idx) if k != 'DY' else p[col].reindex(idx)
        f = f'{FD}/is_{c}.csv'
        if os.path.exists(f):
            i = _safe(f)
            if len(i) and 'type' in i:
                piv = i.pivot_table(index='date', columns='type', values='value', aggfunc='last')
                piv.index = pd.to_datetime(piv.index); piv = piv.sort_index()
                ni = piv.get('EquityAttributableToOwnersOfParent')
                if ni is not None:
                    ttm = ni.rolling(4, min_periods=4).sum()
                    ttm.index = [avail_date(d) for d in ttm.index]
                    out['NI_TTM'][c] = to_daily(ttm.dropna(), idx)
                rv = piv.get('Revenue')
                if rv is not None and 'GrossProfit' in piv:
                    gm = piv['GrossProfit'].rolling(4).sum() / rv.rolling(4).sum()
                    gm.index = [avail_date(d) for d in gm.index]; out['GM_TTM'][c] = to_daily(gm.dropna(), idx)
                if rv is not None and 'OperatingIncome' in piv:
                    om = piv['OperatingIncome'].rolling(4).sum() / rv.rolling(4).sum()
                    om.index = [avail_date(d) for d in om.index]; out['OPM_TTM'][c] = to_daily(om.dropna(), idx)
        f = f'{FD}/bs_{c}.csv'
        if os.path.exists(f):
            b = _safe(f)
            if len(b) and 'type' in b:
                piv = b.pivot_table(index='date', columns='type', values='value', aggfunc='last')
                piv.index = pd.to_datetime(piv.index); piv = piv.sort_index()
                for k, col in (('EQ', 'EquityAttributableToOwnersOfParent'), ('SH', 'OrdinaryShare')):
                    if col in piv:
                        s = piv[col].copy(); s.index = [avail_date(d) for d in s.index]
                        out[k][c] = to_daily(s.dropna(), idx)
        # 月營收
        rv = None
        f = f'{FD}/rev_{c}.csv'
        if os.path.exists(f):
            r = _safe(f)
            if len(r) and 'revenue' in r: rv = pd.Series(r.revenue.values, index=pd.to_datetime(r.date) + pd.Timedelta(days=10))
        elif rev_cur is not None:
            r = rev_cur[rev_cur.code == c]
            if len(r):
                d = pd.to_datetime(r.fm_date) if 'fm_date' in r else pd.to_datetime(r.month) + pd.offsets.MonthBegin(1)
                rv = pd.Series(r.rev.values, index=d + pd.Timedelta(days=10))
        if rv is not None and len(rv):
            rv = rv[~rv.index.duplicated(keep='last')].sort_index().astype(float)
            r3 = rv.rolling(3, min_periods=3).sum()
            out['REV3'][c] = to_daily(r3.dropna(), idx)
            out['REV3_LY'][c] = to_daily(r3.shift(12).dropna(), idx)
            out['REV12'][c] = to_daily(rv.rolling(12, min_periods=12).sum().dropna(), idx)
    W = {k: pd.DataFrame(v, index=idx).reindex(columns=cols) for k, v in out.items()}
    return W


if __name__ == '__main__':
    import sys, time; sys.path.insert(0, '/home/claude/bt/fac')
    from panel import load_panel
    t0 = time.time(); P = load_panel(); W = build(P); pd.to_pickle(W, OUT)
    d = P['C'].index[-300]
    print({k: int(v.loc[d].notna().sum()) for k, v in W.items()}, f'{time.time() - t0:.0f}s')
