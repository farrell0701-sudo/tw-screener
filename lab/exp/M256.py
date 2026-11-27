# M-256：書中鐘擺理論（台積電 vs 其餘大盤點數差 ±1000）。定義見 registry_M.md M-256。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
from chips import _staged, to_grid, twdata_path

ERAS = (('2005–15', '2005-01-01', '2015-12-31'), ('2016–26', '2016-01-01', '2026-12-31'))
HS = (20, 60, 120)


def welch(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 3 or len(b) < 3: return np.nan, np.nan
    d = a.mean() - b.mean(); se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)); return d, d / se


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']
    meta = P['meta']; j = int(np.nonzero(meta['stock_id'].astype(str).to_numpy() == '2330')[0][0])
    MV = to_grid(_staged('TaiwanStockMarketValue', ['date', 'stock_id', 'market_value']), 'market_value', P).astype(np.float64)
    sm = pd.read_parquet(twdata_path() / 'derived' / 'security_master.parquet')
    tw = set(sm[(sm.market_current == 'twse') & (sm.security_class == 'common')].security_key.astype(str))
    istw = meta['security_key'].astype(str).isin(tw).to_numpy()
    tot = np.nansum(np.where(istw[None, :], MV, 0.0), axis=1)
    w = pd.Series(np.where(tot > 0, MV[:, j] / tot, np.nan), index=dates).ffill(limit=5)
    IX = pd.read_parquet(twdata_path() / 'derived' / 'index.parquet'); IX['date'] = pd.to_datetime(IX['date'])
    px = IX.pivot(index='date', columns='index_id', values='price_index').reindex(dates).ffill()
    tr = IX.pivot(index='date', columns='index_id', values='total_return_index').reindex(dates).ffill()
    tsm_raw = pd.Series(np.asarray(P['C'], np.float64)[:, j], index=dates).ffill()
    tsm_adj = pd.Series(np.asarray(F['CA'], np.float64)[:, j], index=dates).ffill()
    ok = (dates >= '2005-01-01') & w.notna().to_numpy()
    D = pd.DataFrame(dict(w=w, ix=px['TAIEX'], tsm=tsm_raw, tsa=tsm_adj, trx=tr['TAIEX'], tro=tr['TPEx']))[ok]
    D['lit'] = D.ix - D.tsm * D.w; D['T'] = D.ix * D.w; D['X'] = D.ix * (1 - D.w)
    # 其餘大盤報酬指數：由 TAIEX_TR 與台積電還原報酬依前一日權重反推
    rT = D.trx.pct_change(); rS = D.tsa.pct_change(); wl = D.w.shift(1)
    D['exi'] = (1 + ((rT - wl * rS) / (1 - wl)).fillna(0)).cumprod()
    L = ['# M-256 結果：鐘擺理論（台積電 vs 其餘大盤）\n',
         f"權重 w：2005 年平均 {D.w.loc['2005'].mean():.1%}、2015 {D.w.loc['2015'].mean():.1%}、2020 {D.w.loc['2020'].mean():.1%}、2026 {D.w.loc['2026'].mean():.1%}（台積電市值 ÷ 上市普通股總市值）。\n",
         f"字面版（大盤 − 台積電股價 × 權重）：最小 {D.lit.min():,.0f}、中位 {D.lit.median():,.0f}、最大 {D.lit.max():,.0f}；> 1000 的天數比例 {(D.lit > 1000).mean():.0%}、< −1000 {(D.lit < -1000).mean():.0%} → 字面版無法區分狀態。\n"]
    first = D.groupby(D.index.to_period('M')).head(1).index        # 每月第一個交易日
    res = {}
    for W, thr_g in ((60, 0.10), (20, 0.06), (120, 0.15)):
        G = (D.X - D.X.shift(W)) - (D['T'] - D['T'].shift(W))
        g = (D.exi / D.exi.shift(W) - 1) - (D.tsa / D.tsa.shift(W) - 1)
        for nm, sig, th in ((f'S1 點數背離 {W} 日 ±1000', G, 1000), (f'S2 報酬背離 {W} 日 ±{thr_g:.0%}', g, thr_g)):
            L += [f'## {nm}\n', '| 期間 | 結果 | 天數比例 高｜低 | 月樣本 高｜低｜全 | ' + ' | '.join(f'{h} 日 高組｜低組｜全部｜差（t）' for h in HS) + ' |', '|' + '---|' * (4 + len(HS))]
            for era, lo, hi in ERAS:
                for on, o in (('O1 台積電 − 其餘', None), ('O2 上市 − 上櫃', None)):
                    row = []
                    for h in HS:
                        if on.startswith('O1'): fut = (D.tsa.shift(-h) / D.tsa - 1) - (D.exi.shift(-h) / D.exi - 1)
                        else: fut = (D.trx.shift(-h) / D.trx - 1) - (D.tro.shift(-h) / D.tro - 1)
                        x = pd.DataFrame(dict(s=sig, f=fut)).loc[first].loc[lo:hi].dropna()
                        a, b = x.f[x.s > th], x.f[x.s < -th]; d, t = welch(a, b)
                        row.append(f'{a.mean():+.1%}｜{b.mean():+.1%}｜{x.f.mean():+.1%}｜{d:+.1%}（{t:.1f}）')
                        res[(nm, era, on, h)] = (d, t, len(a), len(b))
                    dd = sig.loc[lo:hi].dropna()
                    L.append(f'| {era} | {on} | {(dd > th).mean():.0%}｜{(dd < -th).mean():.0%} | {res[(nm, era, on, 60)][2]}｜{res[(nm, era, on, 60)][3]}｜{len(x)} | ' + ' | '.join(row) + ' |')
            L.append('')
    m = 'S1 點數背離 60 日 ±1000'
    r = [res[(m, e, 'O1 台積電 − 其餘', 60)] for e, _, _ in ERAS]; r2 = [res[(m, e, 'O2 上市 − 上櫃', 60)] for e, _, _ in ERAS]
    if all(d > 0 and t >= 2 for d, t, _, _ in r) and all(d > 0 for d, *_ in r2): c = '明確現象（鐘擺成立）'
    elif all(d < 0 and t <= -2 for d, t, _, _ in r): c = '現象相反（動能，不是鐘擺）'
    else: c = '無明確現象'
    L += [f"判準（S1 60 日 O1 兩組差）：" + '；'.join(f"{e} {d:+.1%}（t {t:.1f}，高 {na}／低 {nb} 月）" for (e, _, _), (d, t, na, nb) in zip(ERAS, r)) +
          '；O2 ' + '、'.join(f'{d:+.1%}' for d, *_ in r2) + f' → **{c}**']
    h = f'M-256 鐘擺理論：{c}'
    return dict(summary='\n'.join(L[:1] + [f'**{h}**\n'] + L[1:]), headline=h, n_variants=0)
