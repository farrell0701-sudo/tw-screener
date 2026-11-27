# M-257：鐘擺理論，使用者更正公式 I＝指數 − 台積電股價 × 權重(%)。定義見 registry_M.md M-257。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel, twdata_path
from feat import load_feat
from chips import _staged, to_grid
from M256 import welch

HS = (20, 60, 120, 250)


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; meta = P['meta']
    j = int(np.nonzero(meta['stock_id'].astype(str).to_numpy() == '2330')[0][0])
    MV = to_grid(_staged('TaiwanStockMarketValue', ['date', 'stock_id', 'market_value']), 'market_value', P).astype(np.float64)
    sm = pd.read_parquet(twdata_path() / 'derived' / 'security_master.parquet')
    tw = set(sm[(sm.market_current == 'twse') & (sm.security_class == 'common')].security_key.astype(str))
    ist = meta['security_key'].astype(str).isin(tw).to_numpy(); tot = np.nansum(np.where(ist[None, :], MV, 0.0), axis=1)
    w = pd.Series(np.where(tot > 0, MV[:, j] / tot, np.nan), index=dates).ffill(limit=5)
    IX = pd.read_parquet(twdata_path() / 'derived' / 'index.parquet'); IX['date'] = pd.to_datetime(IX['date'])
    px = IX.pivot(index='date', columns='index_id', values='price_index').reindex(dates).ffill()
    tr = IX.pivot(index='date', columns='index_id', values='total_return_index').reindex(dates).ffill()
    D = pd.DataFrame(dict(w=w, ix=px['TAIEX'], tsm=pd.Series(np.asarray(P['C'], np.float64)[:, j], index=dates).ffill(),
                          tsa=pd.Series(np.asarray(F['CA'], np.float64)[:, j], index=dates).ffill(), trx=tr['TAIEX'], tro=tr['TPEx']))
    D = D[(D.index >= '2005-01-01') & D.w.notna()]
    rT = D.trx.pct_change(); rS = D.tsa.pct_change(); wl = D.w.shift(1)
    D['exi'] = (1 + ((rT - wl * rS) / (1 - wl)).fillna(0)).cumprod()
    D['I'] = D.ix - D.tsm * D.w * 100
    D['st'] = np.where(D.I > 1000, 'A', np.where(D.I < -1000, 'B', 'N'))
    fut = {}
    for h in HS:
        fut[('O1', h)] = (D.tsa.shift(-h) / D.tsa - 1) - (D.exi.shift(-h) / D.exi - 1)
        fut[('O2', h)] = (D.trx.shift(-h) / D.trx - 1) - (D.tro.shift(-h) / D.tro - 1)
    L = ['# M-257 結果：鐘擺理論（I＝指數 − 台積電股價 × 權重%）\n']
    ex = [('2020-03-19', 8681, 248), ('2020-11-20', 13716, 488)]
    L.append('書中範例核對：' + '；'.join(f"{d} 指數 {D.ix[d]:.0f}（書 {a}）、台積電 {D.tsm[d]:.0f}（書 {b}）、權重 {D.w[d]:.1%}、I {D.I[d]:,.0f}" for d, a, b in ex) + '\n')
    yr = D.groupby(D.index.year).agg(I中位=('I', 'median'), A天數=('st', lambda s: (s == 'A').mean()), B天數=('st', lambda s: (s == 'B').mean()))
    L += ['## 每年狀態\n', '| 年 | I 中位數 | A（>1000）天數 | B（<−1000）天數 | 當年 台積電 − 其餘 | 當年 上市 − 上櫃 |', '|---|---|---|---|---|---|']
    for y, r in yr.iterrows():
        s = D.loc[str(y)]; o1 = (s.tsa.iloc[-1] / s.tsa.iloc[0]) - (s.exi.iloc[-1] / s.exi.iloc[0]); o2 = (s.trx.iloc[-1] / s.trx.iloc[0]) - (s.tro.iloc[-1] / s.tro.iloc[0])
        L.append(f'| {y} | {r.I中位:,.0f} | {r.A天數:.0%} | {r.B天數:.0%} | {o1:+.1%} | {o2:+.1%} |')
    first = D.groupby(D.index.to_period('M')).head(1).index
    L += ['', '## (a) 狀態下的未來表現（每月第一個交易日取樣）\n', '| 結果 | 期限 | A 平均（月數） | N 平均（月數） | B 平均（月數） | A − B（t） |', '|---|---|---|---|---|---|']
    res = {}
    for o in ('O1', 'O2'):
        for h in HS:
            x = pd.DataFrame(dict(s=D.st, f=fut[(o, h)])).loc[first].dropna()
            g = {k: x.f[x.s == k] for k in 'ANB'}; d, t = welch(g['A'], g['B']); res[(o, h)] = (d, t)
            L.append(f"| {'台積電 − 其餘' if o == 'O1' else '上市 − 上櫃'} | {h} 日 | " + ' | '.join(f'{g[k].mean():+.1%}（{len(g[k])}）' for k in 'ANB') + f' | {d:+.1%}（{t:.1f}） |')
    ev = []; lastd = {'down': -999, 'up': -999}; I = D.I.to_numpy(); idx = D.index
    for i in range(1, len(D)):
        for k, cond in (('down', I[i - 1] >= -1000 and I[i] < -1000), ('up', I[i - 1] <= 1000 and I[i] > 1000)):
            if cond and i - lastd[k] > 60: ev.append((idx[i], k)); lastd[k] = i
    L += ['', '## (b) 穿越事件\n', '| 日期 | 方向 | 預期 | I | ' + ' | '.join(f'{h} 日 台積電−其餘｜上市−上櫃' for h in (60, 120, 250)) + ' | 120 日符合預期 |', '|' + '---|' * 8]
    hit = []
    for d, k in ev:
        o = {h: (fut[('O1', h)][d], fut[('O2', h)][d]) for h in (60, 120, 250)}
        good = None if np.isnan(o[120][0]) else ((o[120][0] < 0) if k == 'down' else (o[120][0] > 0))
        if good is not None: hit.append(good)
        L.append(f"| {d:%Y-%m-%d} | {'跌破 −1000' if k == 'down' else '升破 +1000'} | {'中小型較好' if k == 'down' else '權值股較好'} | {D.I[d]:,.0f} | " +
                 ' | '.join(('—' if np.isnan(a) else f'{a:+.1%}｜{b:+.1%}') for a, b in o.values()) + f" | {'—' if good is None else ('✓' if good else '✗')} |")
    d60, t60 = res[('O1', 60)]; hr = np.mean(hit) if hit else np.nan
    ok = d60 > 0 and t60 >= 2 and res[('O2', 60)][0] > 0 and hit and hr >= 2 / 3
    c = '明確現象' if ok else '無明確現象'
    L += ['', f'判準：(a) A − B 60 日 台積電−其餘 {d60:+.1%}（t {t60:.1f}）、上市−上櫃 {res[("O2", 60)][0]:+.1%}；(b) 穿越事件 120 日符合預期 {sum(hit)}／{len(hit)} → **{c}**']
    h = f'M-257 鐘擺理論（原始公式）：{c}；(a) A−B 60 日 {d60:+.1%}（t {t60:.1f}），(b) 穿越 {sum(hit)}/{len(hit)}'
    return dict(summary='\n'.join(L[:1] + [f'**{h}**\n'] + L[1:]), headline=h, n_variants=0)
