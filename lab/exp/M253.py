# M-253：定版報酬拆解加入 STM、H52、REV（與 M-254 共用因子）。定義見 registry_M.md M-253。
import sys, pickle
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
from chips import _staged, to_grid
from M107 import nw_t
from M251 import ols, monthly, RF
from L123 import daily_from_monthly
import M180

ALL = ['MKT', 'SMB', 'HML', 'UMD', 'LVOL', 'STM', 'H52', 'REV', 'RTL']
CACHE = LAB / 'cache' / 'M253_factors.pkl'


def factors(P=None, F=None):
    if CACHE.exists(): return pickle.loads(CACHE.read_bytes())
    P = P or load_panel(); F = F or load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    sid = P['meta']['stock_id'].astype(str)
    stock = (~sid.str.startswith('0') & (sid.str.len() == 4)).to_numpy()
    EMG = np.asarray(P['EMG']); VALID = np.asarray(F['VALID'])
    C = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill(limit=5).to_numpy()
    with np.errstate(all='ignore'): R = C / np.vstack([np.full((1, N), np.nan), C[:-1]]) - 1
    R[~VALID] = np.nan; R[np.abs(R) > 0.5] = np.nan
    MV = pd.DataFrame(to_grid(_staged('TaiwanStockMarketValue', ['date', 'stock_id', 'market_value']), 'market_value', P).astype(np.float64)).ffill(limit=5).to_numpy()
    PB = to_grid(_staged('TaiwanStockPER', ['date', 'stock_id', 'PBR']), 'PBR', P).astype(np.float64)
    PB[~(PB > 0)] = np.nan; BM = 1 / pd.DataFrame(PB).ffill(limit=20).to_numpy(); del PB
    Cm = pd.DataFrame(C)
    V = dict(HML=BM, UMD=(Cm.shift(21) / Cm.shift(252) - 1).to_numpy(),
             LVOL=-pd.DataFrame(R).rolling(60, min_periods=40).std().to_numpy(),
             STM=(Cm.shift(5) / Cm.shift(63) - 1).to_numpy(),
             H52=(Cm / Cm.rolling(250, min_periods=200).max()).to_numpy())
    keys = P['meta']['security_key'].astype(str).to_numpy()
    V['REV'] = daily_from_monthly(M180.r6_factors(dates)['R6 營收短長比'], dates, keys).astype(np.float64)
    V['RTL'] = -np.asarray(np.load(LAB / 'cache' / 'M244' / 'B4.npy', mmap_mode='r'), np.float64)
    me = pd.Series(np.arange(T), index=dates).groupby(dates.to_period('M')).max().to_numpy()
    out = {k: np.full(T, np.nan) for k in ['SMB'] + list(V)}
    for a, b in zip(me[:-1], me[1:]):
        ok = stock & ~EMG[a] & VALID[a] & (MV[a] > 0)
        if ok.sum() < 100: continue
        big = MV[a] >= np.nanmedian(MV[a][ok]); w = np.where(ok, MV[a], 0.0); rr = R[a + 1:b + 1]
        x0 = np.where(np.isnan(rr), 0.0, rr); nn = ~np.isnan(rr)
        def port(m):
            ww = np.where(m, w, 0.0)
            return (x0 * ww).sum(1) / np.maximum((nn * ww).sum(1), 1e-9)
        smbs = []
        for k, X in V.items():
            x = X[a]; okx = ok & np.isfinite(x)
            if okx.sum() < 60: continue
            q3, q7 = np.nanpercentile(x[okx], [30, 70]); lo = okx & (x <= q3); hi = okx & (x >= q7); md = okx & ~lo & ~hi
            g = {s + kk: port(sm & m) for s, sm in (('S', ~big), ('B', big)) for kk, m in (('L', lo), ('M', md), ('H', hi))}
            out[k][a + 1:b + 1] = (g['SH'] + g['BH']) / 2 - (g['SL'] + g['BL']) / 2
            if k in ('HML', 'UMD', 'LVOL'): smbs.append(sum(g['S' + q] for q in 'LMH') / 3 - sum(g['B' + q] for q in 'LMH') / 3)
        if smbs: out['SMB'][a + 1:b + 1] = np.mean(smbs, axis=0)
    D = pd.DataFrame(out, index=dates)
    D['MKT'] = P['index']['TAIEX_TR'].reindex(dates).ffill().pct_change().to_numpy() - RF / 252
    Fm = monthly(D[ALL]); Fm[D[ALL].groupby(dates.to_period('M')).count() == 0] = np.nan
    CACHE.write_bytes(pickle.dumps(Fm)); return Fm


def fit_table(sm, Fm, models, eras, title):
    L = [f'## {title}\n']
    for mn, cols in models.items():
        L += [f'### {mn}：' + '＋'.join(cols) + '\n', '| 期間 | 月數 | alpha 年化（t） | ' + ' | '.join(f'β {k}（t）' for k in cols) + ' | R² |', '|' + '---|' * (4 + len(cols))]
        for era, lo, hi in eras:
            J = pd.concat([sm.rename('y'), Fm[cols]], axis=1).dropna()
            J = J[(J.index >= pd.Period(lo, 'M')) & (J.index <= pd.Period(hi, 'M'))]
            b, t, r2 = ols(J.y.to_numpy(), J[cols].to_numpy())
            L.append(f'| {era} | {len(J)} | {b[0] * 12:+.1%}（{t[0]:.1f}） | ' + ' | '.join(f'{b[i + 1]:+.2f}（{t[i + 1]:.1f}）' for i in range(len(cols))) + f' | {r2:.2f} |')
            yield_ = (mn, era, b[0] * 12, t[0])
            fit_table.res.append(yield_)
        L.append('')
    return L
fit_table.res = []


def factor_summary(Fm, eras):
    L = ['## 因子本身（年化平均｜NW t）\n', '| 期間 | ' + ' | '.join(ALL) + ' |', '|' + '---|' * (len(ALL) + 1)]
    for era, lo, hi in eras:
        x = Fm[(Fm.index >= pd.Period(lo, 'M')) & (Fm.index <= pd.Period(hi, 'M'))]
        L.append(f'| {era} | ' + ' | '.join((f'{x[k].mean() * 12:+.1%}｜{nw_t(x[k].dropna(), 3):.1f}' if x[k].notna().sum() > 24 else '—') for k in ALL) + ' |')
    c = Fm.dropna().corr()
    L += ['', '因子相關（2010-06 起共同期間；|ρ| ≥ 0.3）：' + '；'.join(f'{a}-{b} {c.loc[a, b]:+.2f}' for i, a in enumerate(ALL) for b in ALL[i + 1:] if abs(c.loc[a, b]) >= 0.3), '']
    return L


ERAS = (('2005–15', '2005-01', '2015-12'), ('2016–26', '2016-01', '2026-12'), ('2016–24.10', '2016-01', '2024-10'))
FF5 = ['MKT', 'SMB', 'HML', 'UMD', 'LVOL']


def main():
    Fm = factors(); fit_table.res.clear()
    z = np.load(LAB / 'results' / 'L113' / 'series.npz', allow_pickle=True)
    sd = pd.to_datetime(z['dates'], unit='D'); names = list(z['names'])
    L = ['# M-253 結果：定版報酬拆解加入 STM、H52、REV\n'] + factor_summary(Fm, ERAS)
    models = {'M0': FF5, 'M1': FF5 + ['STM', 'H52'], 'M2': FF5 + ['STM', 'H52', 'REV']}
    for nm in ('fix_merge', 'fix_X', 'fix_V'):
        s = pd.Series(z['values'][:, names.index(nm)].astype(np.float64), index=sd)
        L += fit_table(monthly(s) - RF / 12, Fm, models, ERAS, nm)
    r = {(m, e): (a, t) for m, e, a, t in fit_table.res[:9]}
    h = '合併版 alpha（M0→M1→M2）：' + '；'.join(f"{e} {r[('M0', e)][0]:+.1%}→{r[('M1', e)][0]:+.1%}→{r[('M2', e)][0]:+.1%}（t {r[('M2', e)][1]:.1f}）" for e, _, _ in ERAS[:2])
    return dict(summary='\n'.join(L[:1] + [f'**{h}**\n'] + L[1:]), headline=h, n_variants=0)
