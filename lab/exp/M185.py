# M-185：第 1 層續｜依市值、ATR%、產業細分的報酬集中（奇數代碼；到 2024-11-13）。定義見 registry_M.md M-185。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals
import M165
from L140 import fixed_industry

END = pd.Timestamp('2024-11-13')


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    meta = P['meta']; keys = meta['security_key'].astype(str).to_numpy()
    ind = fixed_industry(meta)[0].astype(str).to_numpy()
    odd = (meta['stock_id'].astype(str).str[-1].astype(int) % 2 == 1).to_numpy()
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; cols = dbg['cols']
    def full(a, fill):
        if a.shape[1] == N: return a
        o = np.full((T, N), fill, a.dtype); o[:, cols] = a; return o
    U, pool = full(dbg['U'], False), full(dbg['pool'], False); del S
    VALID = np.asarray(F['VALID']); CA = np.where(VALID, np.asarray(F['CA'], np.float64), np.nan)
    ATRP = np.asarray(F['ATR'], np.float64) / np.asarray(F['CA'], np.float64)
    Cf = pd.DataFrame(CA).ffill().to_numpy(); Craw = pd.DataFrame(np.where(np.asarray(P['C']) > 0, P['C'], np.nan)).ffill(limit=5).to_numpy()
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CX = np.cumsum(R - ew[:, None], axis=0)
    A = M165.layers(U, pool, dates, keys)[0][1]
    SH = M165.shares_table(dates); shj = pd.Index(SH.columns.astype(str)).get_indexer(keys)
    rows = []
    for m, (t, tp) in sorted(M165.samp_prev(dates).items(), key=lambda x: x[1][0]):
        if dates[t] < pd.Timestamp('2005-01-01') or dates[t] > END or t + 61 >= T: continue
        era = '2005–15' if dates[t] <= pd.Timestamp('2015-12-31') else '2016–24/11'
        Ut = U[t]
        sh = np.where(shj >= 0, SH.loc[m].to_numpy(float)[np.clip(shj, 0, None)], np.nan) if m in SH.index else np.full(N, np.nan)
        mc = Craw[tp] * sh
        def ter(x):
            q = pd.Series(np.where(Ut, x, np.nan)).rank(pct=True).to_numpy()
            return np.where(np.isnan(q), '', np.where(q <= 1 / 3, '低', np.where(q <= 2 / 3, '中', '高')))
        mct, att = ter(mc), ter(ATRP[tp])
        e60 = CX[t + 60] - CX[t]; q10, q90 = np.nanpercentile(e60[Ut], [10, 90])
        grp = np.where(pool[t], '強勢池', np.where(A[t], '營收池', '其餘'))
        for j in np.nonzero(Ut & odd)[0]:
            rows.append((era, grp[j], mct[j], att[j], ind[j], e60[j], q10, q90))
    D = pd.DataFrame(rows, columns=['era', 'pool', 'mc', 'atr', 'ind', 'e60', 'q10', 'q90'])
    D['L'] = D.e60 <= D.q10; D['Rt'] = D.e60 >= D.q90
    top = D[D.era == '2016–24/11'].ind.value_counts().head(10).index
    D['indg'] = np.where(D.ind.isin(top), D.ind, '前十以外')
    L = ['# M-185 結果：依市值、ATR%、產業細分的報酬集中（奇數代碼；60 日超額；2005–15｜2016–24/11）\n']
    for key, nm, order in (('mc', '市值三分位', ['低', '中', '高']), ('atr', 'ATR% 三分位', ['低', '中', '高']), ('indg', '產業（前 10 大）', list(top) + ['前十以外'])):
        L += [f'## {nm}\n', '| 子群 | 強勢池占比 | 營收池占比 | 強勢池 60 日（中位）左／右尾 | 營收池 60 日（中位）左／右尾 | 其餘 60 日 | 強勢池兩期同號 | 營收池兩期同號 |', '|---|---|---|---|---|---|---|---|']
        for g in order:
            cells = {}
            for era in ('2005–15', '2016–24/11'):
                x = D[(D.era == era) & (D[key] == g)]
                if not len(x): cells[era] = None; continue
                s, r, o = x[x.pool == '強勢池'], x[x.pool == '營收池'], x[x.pool == '其餘']
                f = lambda y: f"{y.e60.mean():+.1%}（{y.e60.median():+.1%}）{y.L.mean():.0%}／{y.Rt.mean():.0%}" if len(y) >= 30 else '—'
                cells[era] = (f"{(x.pool == '強勢池').mean():.0%}", f"{(x.pool == '營收池').mean():.0%}", f(s), f(r), f"{o.e60.mean():+.1%}", s.e60.mean() if len(s) >= 30 else np.nan, r.e60.mean() if len(r) >= 30 else np.nan)
            a, b = cells['2005–15'], cells['2016–24/11']
            if a is None or b is None: continue
            same = lambda i: '—' if (np.isnan(a[i]) or np.isnan(b[i])) else ('✓' if np.sign(a[i]) == np.sign(b[i]) else '✗')
            L.append(f"| {g} | {a[0]}｜{b[0]} | {a[1]}｜{b[1]} | {a[2]}｜{b[2]} | {a[3]}｜{b[3]} | {a[4]}｜{b[4]} | {same(5)} | {same(6)} |")
        L.append('')
    return dict(summary='\n'.join(L), headline='第 1 層子群細分完成', n_variants=0)
