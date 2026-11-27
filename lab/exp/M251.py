# M-251：定版報酬拆解（Fama-French 式五因子）。定義見 registry_M.md M-251。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
from chips import _staged, to_grid
from M107 import nw_t

RF = 0.01
ERAS = (('2005–15', '2005-01-01', '2015-12-31'), ('2016–26', '2016-01-01', '2026-12-31'), ('2016–24.10（排除近兩年）', '2016-01-01', '2024-10-31'))
FN = ['MKT', 'SMB', 'HML', 'UMD', 'LVOL']


def factors(P=None, F=None):
    P = P or load_panel(); F = F or load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    sid = P['meta']['stock_id'].astype(str)
    stock = (~sid.str.startswith('0') & (sid.str.len() == 4)).to_numpy()
    EMG = np.asarray(P['EMG']); VALID = np.asarray(F['VALID'])
    C = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill(limit=5).to_numpy()
    with np.errstate(all='ignore'): R = C / np.vstack([np.full((1, N), np.nan), C[:-1]]) - 1
    R[~VALID] = np.nan; R[np.abs(R) > 0.5] = np.nan
    MV = to_grid(_staged('TaiwanStockMarketValue', ['date', 'stock_id', 'market_value']), 'market_value', P).astype(np.float64)
    MV = pd.DataFrame(MV).ffill(limit=5).to_numpy()
    PB = to_grid(_staged('TaiwanStockPER', ['date', 'stock_id', 'PBR']), 'PBR', P).astype(np.float64)
    PB[~(PB > 0)] = np.nan; BM = 1 / pd.DataFrame(PB).ffill(limit=20).to_numpy()
    Cm = pd.DataFrame(C)
    MOM = (Cm.shift(21) / Cm.shift(252) - 1).to_numpy()
    VOL = pd.DataFrame(R).rolling(60, min_periods=40).std().to_numpy()
    me = pd.Series(np.arange(T), index=dates).groupby(dates.to_period('M')).max().to_numpy()   # 月底位置
    out = {k: np.full(T, np.nan) for k in FN[1:]}
    for a, b in zip(me[:-1], me[1:]):
        ok = stock & ~EMG[a] & VALID[a] & (MV[a] > 0)
        if ok.sum() < 100: continue
        big = MV[a] >= np.nanmedian(MV[a][ok]); w = np.where(ok, MV[a], 0.0)
        rr = R[a + 1:b + 1]
        def port(m):
            ww = np.where(m, w, 0.0)
            if ww.sum() <= 0: return np.zeros(b - a)
            x = np.where(np.isnan(rr), 0.0, rr); wm = np.where(np.isnan(rr), 0.0, ww)
            return (x * ww).sum(1) / np.maximum(wm.sum(1), 1e-9)
        def two3(x, hi_good=True):
            okx = ok & np.isfinite(x)
            if okx.sum() < 60: return None, None
            q3, q7 = np.nanpercentile(x[okx], [30, 70]); lo = okx & (x <= q3); hi = okx & (x >= q7); md = okx & ~lo & ~hi
            g = {}
            for s, sm in (('S', ~big), ('B', big)):
                for k, m in (('L', lo), ('M', md), ('H', hi)): g[s + k] = port(sm & m)
            f = (g['SH'] + g['BH']) / 2 - (g['SL'] + g['BL']) / 2
            smb = sum(g['S' + k] for k in 'LMH') / 3 - sum(g['B' + k] for k in 'LMH') / 3
            return (f if hi_good else -f), smb
        hml, s1 = two3(BM[a]); umd, s2 = two3(MOM[a]); lvol, s3 = two3(VOL[a], hi_good=False)
        ss = [s for s in (s1, s2, s3) if s is not None]
        if ss: out['SMB'][a + 1:b + 1] = np.mean(ss, axis=0)
        for k, v in (('HML', hml), ('UMD', umd), ('LVOL', lvol)):
            if v is not None: out[k][a + 1:b + 1] = v
    tr = P['index']['TAIEX_TR'].reindex(dates).ffill().pct_change()
    D = pd.DataFrame(out, index=dates); D['MKT'] = tr.to_numpy() - RF / 252
    return D[FN], dict(MV=MV, VOL=VOL, stock=stock)


def monthly(d):  # 日報酬 → 月複利
    return (1 + d.fillna(0)).groupby(d.index.to_period('M')).prod() - 1


def ols(y, X, lag=3):
    X1 = np.column_stack([np.ones(len(X)), X]); b, *_ = np.linalg.lstsq(X1, y, rcond=None); e = y - X1 @ b
    n, k = X1.shape; XtXi = np.linalg.inv(X1.T @ X1); S = np.zeros((k, k))
    for l in range(lag + 1):
        G = (X1[l:] * e[l:, None]).T @ (X1[:n - l] * e[:n - l, None])
        S += (G if l == 0 else (1 - l / (lag + 1)) * (G + G.T))
    V = XtXi @ S @ XtXi; t = b / np.sqrt(np.diag(V)); r2 = 1 - e.var() / y.var()
    return b, t, r2


def main():
    P = load_panel(); F = load_feat('AF')
    Fd, _ = factors(P, F); Fm = monthly(Fd)
    z = np.load(LAB / 'results' / 'L113' / 'series.npz', allow_pickle=True)
    sd = pd.to_datetime(z['dates'], unit='D'); names = list(z['names'])
    L = ['# M-251 結果：定版報酬拆解（五因子，月報酬，Newey-West t 落後 3）\n',
         f'rf 假設年 {RF:.0%}；因子為自建（上市櫃普通股、市值加權 2×3）。alpha 為年化（月 alpha ×12）。\n']
    L += ['## 因子本身（月平均年化｜t）\n', '| 期間 | ' + ' | '.join(FN) + ' |', '|---|' + '---|' * len(FN)]
    for era, lo, hi in ERAS:
        x = Fm[(Fm.index >= pd.Period(lo[:7], 'M')) & (Fm.index <= pd.Period(hi[:7], 'M'))].dropna()
        L.append(f'| {era} | ' + ' | '.join(f'{x[k].mean() * 12:+.1%}｜{nw_t(x[k], 3):.1f}' for k in FN) + ' |')
    cor = Fm.dropna().corr()
    L += ['', '因子相關（全期）：' + '；'.join(f'{a}-{b} {cor.loc[a, b]:+.2f}' for i, a in enumerate(FN) for b in FN[i + 1:]), '']
    head = {}
    for nm in ('fix_merge', 'fix_X', 'fix_V'):
        s = pd.Series(z['values'][:, names.index(nm)].astype(np.float64), index=sd)
        sm = monthly(s) - RF / 12
        L += [f'## {nm}\n', '| 期間 | 月數 | 年化超額 | alpha 年化（t） | ' + ' | '.join(f'β {k}（t）' for k in FN) + ' | R² | 單因子 alpha（t）｜β |', '|' + '---|' * (6 + len(FN))]
        C2 = ['', '各因子貢獻（β×因子年化平均）：']
        for era, lo, hi in ERAS:
            J = pd.concat([sm.rename('y'), Fm], axis=1).dropna()
            J = J[(J.index >= pd.Period(lo[:7], 'M')) & (J.index <= pd.Period(hi[:7], 'M'))]
            b, t, r2 = ols(J.y.to_numpy(), J[FN].to_numpy())
            b1, t1, _ = ols(J.y.to_numpy(), J[['MKT']].to_numpy())
            L.append(f'| {era} | {len(J)} | {J.y.mean() * 12:+.1%} | {b[0] * 12:+.1%}（{t[0]:.1f}） | ' +
                     ' | '.join(f'{b[i + 1]:+.2f}（{t[i + 1]:.1f}）' for i in range(len(FN))) + f' | {r2:.2f} | {b1[0] * 12:+.1%}（{t1[0]:.1f}）｜{b1[1]:.2f} |')
            C2.append(f'- {era}：' + '；'.join(f'{k} {b[i + 1] * J[k].mean() * 12:+.1%}' for i, k in enumerate(FN)) + f'；alpha {b[0] * 12:+.1%}')
            head[(nm, era)] = (b[0] * 12, t[0])
        L += C2 + ['']
    m1, m2 = head[('fix_merge', '2005–15')], head[('fix_merge', '2016–26')]
    h = f"合併版五因子 alpha：2005–15 {m1[0]:+.1%}（t {m1[1]:.1f}）、2016–26 {m2[0]:+.1%}（t {m2[1]:.1f}）"
    return dict(summary='\n'.join(L[:1] + [f'**{h}**\n'] + L[1:]), headline=h, n_variants=0)
