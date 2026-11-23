# M-178：營收池翻上多空點才進場＋近兩月營收排序（奇數代碼探索集）。定義見 registry_M.md M-178。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals
from signals import prk
from M122 import fin_factors
from M107 import nw_t
import M165

PARITY = 1


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    keys = P['meta']['security_key'].astype(str).to_numpy()
    par = (P['meta']['stock_id'].astype(str).str[-1].astype(int) % 2 == PARITY).to_numpy()
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; U, pool, Tr = dbg['U'], dbg['pool'], dbg['Tr']; cols = dbg['cols']; del S
    if Tr.shape[1] != N: o = np.full((T, N), np.nan); o[:, cols] = Tr; Tr = o
    dT = Tr - np.vstack([np.full((10, N), np.nan), Tr[:-10]])
    VALID = np.asarray(F['VALID']); CAv = np.where(VALID, np.asarray(F['CA'], np.float64), np.nan)
    Cf = pd.DataFrame(CAv).ffill().to_numpy()
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CX = np.cumsum(R - ew[:, None], axis=0)
    # 個股多空點（上個月第一個交易日到當日的收盤高低點中點）：逐月累計
    mon = dates.to_period('M'); X = np.full((T, N), np.nan)
    months = pd.Series(np.arange(T), index=dates).groupby(mon)
    mlist = list(months.groups.keys())
    for i, mm in enumerate(mlist):
        idx = months.get_group(mm).to_numpy()
        prev = months.get_group(mlist[i - 1]).to_numpy() if i > 0 else np.array([], int)
        base = CAv[prev] if len(prev) else np.full((0, N), np.nan)
        with np.errstate(all='ignore'):
            ph = np.nanmax(base, 0) if len(prev) else np.full(N, np.nan); pl = np.nanmin(base, 0) if len(prev) else np.full(N, np.nan)
            ch = np.fmax.accumulate(np.nan_to_num(CAv[idx], nan=-np.inf), axis=0); cl = np.fmin.accumulate(np.nan_to_num(CAv[idx], nan=np.inf), axis=0)
            hh = np.fmax(ch, np.nan_to_num(ph, nan=-np.inf)); ll = np.fmin(cl, np.nan_to_num(pl, nan=np.inf))
        hh[~np.isfinite(hh)] = np.nan; ll[~np.isfinite(ll)] = np.nan
        X[idx] = (hh + ll) / 2
    above = Cf > X
    A = M165.layers(U, pool, dates, keys)[0][1]
    Q5 = fin_factors(dates)['Q5 毛利率創高']; qj = pd.Index(Q5.columns.astype(str)).get_indexer(keys)
    R4 = M165.r4_table(); rj = pd.Index(R4.columns.astype(str)).get_indexer(keys)
    pick = lambda W, j, m: np.where(j >= 0, W.loc[m].to_numpy(float)[np.clip(j, 0, None)], np.nan) if m in W.index else np.full(N, np.nan)
    sp = sorted(M165.samp_prev(dates).items(), key=lambda x: x[1][0])
    rows = []; base_rows = []; last = np.full(N, -999)
    for i, (m, (t, _)) in enumerate(sp):
        if not A[t].any(): continue
        mem = np.nonzero(A[t])[0]
        q = pick(Q5, qj, m)[mem]; fin = np.isfinite(q); drop = np.zeros(len(mem), bool)
        if fin.sum() >= 20: drop[fin] = q[fin] <= np.nanpercentile(q[fin], 20)
        cand = mem[~drop]; cand = cand[par[cand]]
        t2 = sp[i + 1][1][0] if i + 1 < len(sp) else T
        if dates[t] < pd.Timestamp('2005-01-01') or t2 + 121 >= T: continue
        era = '2005–15' if dates[t] <= pd.Timestamp('2015-12-31') else '2016–26'
        rs = 1.1 * pick(R4, rj, m) + 1.0 * pick(R4, rj, m - 1)
        e60u = (CX[t + 60] - CX[t])[U[t]]; q10, q90 = np.nanpercentile(e60u, [10, 90])
        for j in cand:
            if U[t, j]: base_rows.append((str(m), era, CX[t + 60, j] - CX[t, j], q10, q90))
        for d in range(max(t, 1), t2):
            for j in cand:
                if not (U[d, j] and above[d, j] and not above[d - 1, j] and np.isfinite(X[d - 1, j])) or d - last[j] < 20: continue
                last[j] = d
                x1 = 0.5 * np.nan_to_num(prk(Tr[d:d + 1], U[d:d + 1] & np.isin(np.arange(N), cand)[None, :])[0, j]) + 0.5 * np.nan_to_num(prk(dT[d:d + 1], U[d:d + 1] & np.isin(np.arange(N), cand)[None, :])[0, j])
                rows.append((str(m), era, j, rs[j], x1, CX[d + 20, j] - CX[d, j], CX[d + 60, j] - CX[d, j], CX[d + 120, j] - CX[d, j], q10, q90))
    E = pd.DataFrame(rows, columns=['m', 'era', 'j', 'rs', 'x1', 'e20', 'e60', 'e120', 'q10', 'q90'])
    B = pd.DataFrame(base_rows, columns=['m', 'era', 'e60', 'q10', 'q90'])
    E['L'] = E.e60 <= E.q10; E['Rt'] = E.e60 >= E.q90; B['L'] = B.e60 <= B.q10; B['Rt'] = B.e60 >= B.q90
    L = ['# M-178 結果：營收池翻上多空點才進場＋近兩月營收排序（奇數代碼）\n']
    tab = ['| 組 | 期間 | 筆數 | 20 日 | 60 日 | 120 日 | 60 日中位數 | 勝率 | 左尾 | 右尾 |', '|---|---|---|---|---|---|---|---|---|---|']
    res = {}
    for era in ('2005–15', '2016–26'):
        e, b = E[E.era == era], B[B.era == era]
        tab.append(f"| E1 翻上事件 | {era} | {len(e)} | {e.e20.mean():+.2%} | {e.e60.mean():+.2%} | {e.e120.mean():+.2%} | {e.e60.median():+.2%} | {(e.e60 > 0).mean():.0%} | {e.L.mean():.0%} | {e.Rt.mean():.0%} |")
        tab.append(f"| E2 取樣日直接進場 | {era} | {len(b)} | — | {b.e60.mean():+.2%} | — | {b.e60.median():+.2%} | {(b.e60 > 0).mean():.0%} | {b.L.mean():.0%} | {b.Rt.mean():.0%} |")
        d1 = (e.groupby('m').e60.mean() - b.groupby('m').e60.mean()).dropna()
        res[('t', era)] = (d1.mean(), nw_t(d1.to_numpy(), 3), e.L.mean(), b.L.mean())
        for key, nm in (('rs', 'E3 營收分數'), ('x1', 'E4 X1 分數（參考）')):
            e2 = e.copy(); e2['top'] = e2.groupby('m')[key].transform(lambda s: s.rank(pct=True) > 0.5)
            g = e2.groupby(['m', 'top']).e60.mean().unstack()
            dd = (g[True] - g[False]).dropna() if (True in g and False in g) else pd.Series(dtype=float)
            res[(key, era)] = (dd.mean(), nw_t(dd.to_numpy(), 3))
            tp_, bt_ = e2[e2.top], e2[~e2.top]
            tab.append(f"| {nm} 前半 | {era} | {len(tp_)} | {tp_.e20.mean():+.2%} | {tp_.e60.mean():+.2%} | {tp_.e120.mean():+.2%} | {tp_.e60.median():+.2%} | {(tp_.e60 > 0).mean():.0%} | {tp_.L.mean():.0%} | {tp_.Rt.mean():.0%} |")
            tab.append(f"| {nm} 後半 | {era} | {len(bt_)} | {bt_.e20.mean():+.2%} | {bt_.e60.mean():+.2%} | {bt_.e120.mean():+.2%} | {bt_.e60.median():+.2%} | {(bt_.e60 > 0).mean():.0%} | {bt_.L.mean():.0%} | {bt_.Rt.mean():.0%} |")
    def ok(key, lt=None):
        a, b = res[(key, '2005–15')], res[(key, '2016–26')]
        c = a[0] > 0 and b[0] > 0 and max(a[1], b[1]) >= 2 and min(a[1], b[1]) >= 1
        if lt: c = c and a[2] <= a[3] + 0.01 and b[2] <= b[3] + 0.01
        return c
    c1, c2, c4 = ok('t', True), ok('rs'), ok('x1')
    L.append(f"**判準 1 進場時機：{'通過' if c1 else '不通過'}**（E1 − 同月 E2，60 日：{res[('t', '2005–15')][0]:+.2%}（t {res[('t', '2005–15')][1]:.1f}）｜{res[('t', '2016–26')][0]:+.2%}（t {res[('t', '2016–26')][1]:.1f}））；"
             f"**判準 2 營收排序：{'通過' if c2 else '不通過'}**（前半 − 後半：{res[('rs', '2005–15')][0]:+.2%}（t {res[('rs', '2005–15')][1]:.1f}）｜{res[('rs', '2016–26')][0]:+.2%}（t {res[('rs', '2016–26')][1]:.1f}））；"
             f"參考 X1 分數排序：{res[('x1', '2005–15')][0]:+.2%}（t {res[('x1', '2005–15')][1]:.1f}）｜{res[('x1', '2016–26')][0]:+.2%}（t {res[('x1', '2016–26')][1]:.1f}）\n")
    L += tab
    return dict(summary='\n'.join(L), headline=f"翻上進場{'過' if c1 else '不過'}、營收排序{'過' if c2 else '不過'}", n_variants=0)
