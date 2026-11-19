# M-137：外資、投信（連續型因子＋投信首買／連買事件）。定義見 registry_M.md M-137。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, M120
from M125 import monthly_from_daily
from M127 import tdcc_factors
from M107 import nw_t
from chips import _staged, to_grid

ERAS = (('2012–18', '2012-07-01', '2018-12-31'), ('2019–26', '2019-01-01', '2026-12-31'))


def inst_wide():
    """逐年讀取，回傳 {'Foreign_Investor','Investment_Trust','NumberOfSharesIssued'}：日期 × stock_id 寬表（float32）。"""
    from chips import twdata_path
    base = twdata_path() / 'staged'
    out = {'Foreign_Investor': [], 'Investment_Trust': [], 'NumberOfSharesIssued': []}
    for f in sorted((base / 'TaiwanStockInstitutionalInvestorsBuySell').glob('*.parquet')):
        d = pd.read_parquet(f, columns=['date', 'stock_id', 'name', 'buy', 'sell'])
        d = d[d.name.isin(['Foreign_Investor', 'Investment_Trust'])]
        d['net'] = (d.buy - d.sell).astype('float32'); d['date'] = pd.to_datetime(d.date)
        for nm in ('Foreign_Investor', 'Investment_Trust'):
            out[nm].append(d[d.name == nm].pivot_table(index='date', columns='stock_id', values='net', aggfunc='sum').astype('float32'))
        del d
    for f in sorted((base / 'TaiwanStockShareholding').glob('*.parquet')):
        if int(f.stem) < 2011: continue
        d = pd.read_parquet(f, columns=['date', 'stock_id', 'NumberOfSharesIssued']); d['date'] = pd.to_datetime(d.date)
        out['NumberOfSharesIssued'].append(d.pivot_table(index='date', columns='stock_id', values='NumberOfSharesIssued', aggfunc='last').astype('float32')); del d
    return {k: pd.concat(v).sort_index() for k, v in out.items()}


def inst_factors(dates):
    W0 = inst_wide()
    ds = pd.Series(dates, index=dates); s = ds[ds.dt.day > 10]; samp = s.groupby(s.dt.to_period('M')).min()
    prev = {p: dates[dates.get_loc(t) - 1] for p, t in samp.items() if dates.get_loc(t) > 0}
    W = {k: v.reindex(dates) for k, v in W0.items()}; del W0
    shr = W['NumberOfSharesIssued'].ffill(); shr = shr.where(shr > 0)
    cols = W['Investment_Trust'].columns
    fi, it = W['Foreign_Investor'].reindex(columns=cols).fillna(0), W['Investment_Trust'].fillna(0)
    shr = shr.reindex(columns=cols)
    b = (it > 0).to_numpy(); run = np.zeros(b.shape, np.float32)
    for i in range(b.shape[0]): run[i] = (run[i - 1] + 1) * b[i] if i else b[i]
    I3 = pd.DataFrame(run, index=it.index, columns=cols).where(W['Investment_Trust'].notna())
    res = {'I1 外資 20 日買超': fi.rolling(20).sum() / shr, 'I2 投信 20 日買超': it.rolling(20).sum() / shr, 'I3 投信連買天數': I3}
    return {k: pd.DataFrame({p - 1: v.loc[d] for p, d in prev.items() if d in v.index}).T.sort_index() for k, v in res.items()}


def events(P, F, U, pool, odd):
    dates = P['dates']; T, N = P['C'].shape
    W = inst_wide()['Investment_Trust'].reindex(dates)
    sid = P['meta']['stock_id'].astype(str).to_numpy(); ci = pd.Index(W.columns.astype(str)).get_indexer(sid)
    IT = np.where(ci >= 0, W.to_numpy()[:, np.clip(ci, 0, None)], np.nan); del W
    has = np.isfinite(IT); b = np.nan_to_num(IT) > 0
    Cf = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill().to_numpy(); VALID = np.asarray(F['VALID'])
    HA = np.asarray(F['HA'], np.float64); LA = np.asarray(F['LA'], np.float64)
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CX = np.cumsum(R - ew[:, None], axis=0)
    prior = pd.DataFrame(b.astype(float)).shift(1).rolling(60, min_periods=60).sum().to_numpy()
    cov60 = pd.DataFrame(has.astype(float)).shift(1).rolling(60, min_periods=60).sum().to_numpy()
    first = b & (prior == 0) & (cov60 >= 55)
    ev = []
    for d, j in zip(*np.nonzero(first)):
        if d + 63 >= T or not odd[j]: continue
        cont = b[d + 1, j] and b[d + 2, j]
        ev.append((d + 1, j, '首買')); ev.append((d + 3, j, '首買＋連買' if cont else '首買未連買'))
    rows = []
    cache = {}
    for t, j, typ in ev:
        if t + 60 >= T: continue
        for layer, mask in (('全市場', U), ('強勢池', pool)):
            if not mask[t, j]: continue
            key = (t, layer)
            if key not in cache:
                jj = np.nonzero(mask[t])[0]; y = {h: CX[t + h, jj] - CX[t, jj] for h in (5, 20, 60)}
                with np.errstate(all='ignore'):
                    up = np.nanmax(HA[t + 1:t + 61, jj], axis=0) / Cf[t, jj] - 1; dn = np.nanmin(LA[t + 1:t + 61, jj], axis=0) / Cf[t, jj] - 1
                cache[key] = dict(m={h: np.nanmean(v) for h, v in y.items()}, q=np.nanpercentile(y[60], [10, 90]), mu=np.nanmedian(up), md=np.nanmedian(dn))
            c = cache[key]
            with np.errstate(all='ignore'):
                up = np.nanmax(HA[t + 1:t + 61, j]) / Cf[t, j] - 1; dn = np.nanmin(LA[t + 1:t + 61, j]) / Cf[t, j] - 1
            y60 = CX[t + 60, j] - CX[t, j]
            rows.append(dict(t=dates[t], typ=typ, layer=layer, d5=CX[t + 5, j] - CX[t, j] - c['m'][5], d20=CX[t + 20, j] - CX[t, j] - c['m'][20], d60=y60 - c['m'][60],
                             left=y60 <= c['q'][0], right=y60 >= c['q'][1], up=up - c['mu'], dn=dn - c['md']))
    return pd.DataFrame(rows)


def _events_only():
    P = load_panel(); F = load_feat('AF')
    S = signals.compute('fix', P=P, F=F); U, pool = S['_dbg']['U'], S['_dbg']['pool']; del S
    odd = (P['meta']['stock_id'].astype(str).str[-1].astype(int) % 2 == 1).to_numpy()
    return events(P, F, U, pool, odd), P['dates']


def main():
    import gc
    E, dates = _events_only(); gc.collect()
    ref = {'R4': M120.rev_factors()['R4 營收創高'], 'B4': tdcc_factors(dates)['B4 散戶 1 月變化']}
    L1, summ = M120.evaluate(1, None, fac_fn=inst_factors, ref=ref, eras=ERAS)
    out = ['# M-137 結果：外資、投信（奇數代碼；期間 2012–18／2019–26；F 計數 40）\n', '## 一、連續型因子\n',
           '| 因子 | 層 | 期間 | IC5 | IC20 | IC60 | IC120 | 高−低 60 日（t） |', '|---|---|---|---|---|---|---|---|']
    for fn, la, era, ic, sp, t in summ:
        out.append(f"| {fn} | {la} | {era} | " + ' | '.join(f'{v:+.3f}' for v in ic) + f" | {sp:+.2%}（{t:.1f}） |")
    out += ['\n## 二、投信首買／連買事件（與同日同層平均之差；t 以月份平均計）\n',
            '| 層 | 期間 | 事件 | 件數 | 月數 | 後5日差（t） | 後20日差（t） | 後60日差（t） | 左尾率 | 右尾率 | 60日最高漲幅 | 60日最大跌幅 |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for layer in ('全市場', '強勢池'):
        for era, lo, hi in ERAS:
            for typ in ('首買', '首買＋連買', '首買未連買'):
                x = E[(E.layer == layer) & (E.typ == typ) & (E.t >= lo) & (E.t <= hi)]
                if len(x) < 20: out.append(f'| {layer} | {era} | {typ} | {len(x)} | | （樣本不足） | | | | | | |'); continue
                mm = x.groupby(x.t.dt.to_period('M'))[['d5', 'd20', 'd60']].mean()
                out.append(f"| {layer} | {era} | {typ} | {len(x)} | {len(mm)} | {x.d5.mean():+.2%}（{nw_t(mm.d5, 1):.1f}） | {x.d20.mean():+.2%}（{nw_t(mm.d20, 1):.1f}） | {x.d60.mean():+.2%}（{nw_t(mm.d60, 3):.1f}） | "
                           f"{x.left.mean():.0%} | {x.right.mean():.0%} | {x.up.mean():+.1%} | {x.dn.mean():+.1%} |")
    return dict(summary='\n'.join(out + [''] + L1), headline='外資、投信評估完成', n_variants=0)
