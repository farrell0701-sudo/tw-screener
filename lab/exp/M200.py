# M-200：X1 候選股與 V4.1 訊號內的因子相關與增量（奇數代碼、排除最近兩年）。定義見 registry_M.md M-200。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals
import M120, M180, M190
from L123 import daily_from_monthly
from M195 import HOLDOUT, ERAS, COST
from M107 import nw_t

FAC = ['T_ld', 'ΔT10', 'ΔT60', 'R1', 'R4', 'R5', 'R6', 'VC', 'VX', 'EXT', 'POS250']


def panel(parity, extra=None, fac=None):
    fac = fac or FAC
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    signals.build('fix', mode='fix', P=P, F=F)
    S_ = signals.compute('fix', P=P, F=F); dbg = S_['_dbg']; cols = dbg['cols']
    def full(a, fill):
        a = np.asarray(a)
        if a.shape[1] == N: return a
        o = np.full((T, N), fill, a.dtype); o[:, cols] = a; return o
    U = full(dbg['U'], False); Tr = full(np.asarray(dbg['Tr'], np.float64), np.nan); del S_
    SG = signals.load_sig('fix'); RKX = np.asarray(SG['RKX']); RKV = np.asarray(SG['RKV']); daily = SG['daily'].reindex(dates)
    below = daily['below'].fillna(True).to_numpy(bool); off = daily['off'].fillna(True).to_numpy(bool)
    VALID = np.asarray(F['VALID']); OA = np.asarray(F['OA'], np.float64)
    Cf = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill().to_numpy()
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CX = np.cumsum(R - ew[:, None], axis=0)
    keys = P['meta']['security_key'].astype(str).to_numpy()
    par = (P['meta']['stock_id'].astype(str).str[-1].astype(int) % 2 == parity).to_numpy()
    FE = M190.features(P, F, Tr)
    rv = M120.rev_factors(dates); r6 = M180.r6_factors(dates)['R6 營收短長比']
    FD = {'T_ld': Tr, 'ΔT10': Tr - np.vstack([np.full((10, N), np.nan), Tr[:-10]]), 'ΔT60': Tr - np.vstack([np.full((60, N), np.nan), Tr[:-60]]),
          'R1': daily_from_monthly(rv['R1 單月年增'], dates, keys), 'R4': daily_from_monthly(rv['R4 營收創高'], dates, keys),
          'R5': daily_from_monthly(rv['R5 年增驚喜'], dates, keys), 'R6': daily_from_monthly(r6, dates, keys),
          'VC': FE['VC'], 'VX': FE['VX'], 'EXT': FE['EXT'], 'POS250': FE['POS250']}
    if extra: FD.update(extra(P, F, Tr, U))
    t0 = dates.searchsorted(pd.Timestamp('2005-01-01')); t1 = min(dates.searchsorted(HOLDOUT), T - 62)
    out = {}
    for nm, RK, gate, step in (('X1 候選', RKX, below, 5), ('V4.1 訊號', RKV, off, 1)):
        rows = []
        for t in range(t0, t1, step):
            if gate[t]: continue
            jj = np.nonzero(np.isfinite(RK[t]) & par)[0]
            if len(jj) == 0: continue
            ent = np.where(np.isfinite(OA[t + 1, jj]), OA[t + 1, jj], Cf[t + 1, jj])
            d = dict(t=t, j=jj, S=-RK[t, jj], e20=CX[t + 20, jj] - CX[t, jj], e60=CX[t + 60, jj] - CX[t, jj], n20=Cf[t + 20, jj] / ent - 1 - COST)
            for f in fac: d[f] = np.asarray(FD[f][t, jj], np.float64)
            rows.append(pd.DataFrame(d))
        D = pd.concat(rows, ignore_index=True); D['date'] = dates[D.t.to_numpy()]
        out[nm] = D[np.isfinite(D.e60)]
    return out


def analyze(out, title, fac=None):
    FAC_ = fac or FAC
    L = [f'# {title}\n']; nomin = []
    for nm, D in out.items():
        grp = 't' if nm.startswith('X1') else 'm'
        D = D.copy(); D['m'] = D.date.dt.to_period('M')
        for era, lo, hi in ERAS:
            E = D[(D.date >= lo) & (D.date <= hi)].copy()
            L += [f"## {nm}｜{era}（{len(E):,} 筆、{E[grp].nunique()} 組；20 日淨勝率 {(E.n20 > 0).mean():.0%}）\n"]
            cm = E[['S'] + FAC_].rank().corr()
            L.append('與 S（X1／V4.1 自身排序）的 Spearman：' + '；'.join(f'{f} {cm.loc["S", f]:+.2f}' for f in FAC_))
            hi_pairs = [(a, b, cm.loc[a, b]) for i, a in enumerate(FAC_) for b in FAC_[i + 1:] if abs(cm.loc[a, b]) >= 0.4]
            L.append('因子間 |ρ| ≥ 0.4：' + ('；'.join(f'{a}–{b} {c:+.2f}' for a, b, c in hi_pairs) or '無') + '\n')
            L += ['| 因子 | 覆蓋 | IC20（t） | IC60（t） | 增量 IC20（t） | 增量 IC60（t） | 殘差前20% 20日淨勝率 | 殘差後20% 20日淨勝率 |', '|---|---|---|---|---|---|---|---|']
            for f in FAC_:
                x = E.dropna(subset=[f]).copy()
                if len(x) < 200: L.append(f'| {f} | {len(x) / len(E):.0%} | 樣本不足 | | | | | |'); continue
                def ric(g, y, col):
                    return g[col].rank().corr(g[y].rank()) if len(g) >= 8 else np.nan
                def resid(g):
                    if len(g) < 8: return pd.Series(np.nan, index=g.index)
                    a = g[f].rank(pct=True).to_numpy(); s = g.S.rank(pct=True).to_numpy(); X = np.column_stack([np.ones(len(g)), s])
                    b = np.linalg.lstsq(X, a, rcond=None)[0]; return pd.Series(a - X @ b, index=g.index)
                x['r'] = x.groupby(grp, group_keys=False).apply(resid)
                ics = {}
                for col in (f, 'r'):
                    for y in ('e20', 'e60'):
                        s = x.groupby(grp).apply(lambda g: ric(g, y, col)).dropna()
                        lag = (4 if y == 'e20' else 12) if grp == 't' else (1 if y == 'e20' else 3)
                        ics[(col, y)] = (s.mean(), nw_t(s, lag))
                x['q'] = x.groupby(grp).r.transform(lambda s: pd.qcut(s.rank(method='first'), 5, labels=False) + 1 if s.notna().sum() >= 5 else np.nan)
                wt, wb = (x[x.q == 5].n20 > 0).mean(), (x[x.q == 1].n20 > 0).mean()
                L.append(f"| {f} | {len(x) / len(E):.0%} | {ics[(f, 'e20')][0]:+.3f}（{ics[(f, 'e20')][1]:.1f}） | {ics[(f, 'e60')][0]:+.3f}（{ics[(f, 'e60')][1]:.1f}） | "
                         f"{ics[('r', 'e20')][0]:+.3f}（{ics[('r', 'e20')][1]:.1f}） | {ics[('r', 'e60')][0]:+.3f}（{ics[('r', 'e60')][1]:.1f}） | {wt:.0%} | {wb:.0%} |")
                nomin.append(dict(s=nm, era=era, f=f, i20=ics[('r', 'e20')], i60=ics[('r', 'e60')], wt=wt, wb=wb, wall=(E.n20 > 0).mean()))
            L.append('')
    N_ = pd.DataFrame(nomin); vt = []
    for (s, f), g in N_.groupby(['s', 'f'], sort=False):
        if len(g) < 2: continue
        ok = False; why = []
        for y in ('i20', 'i60'):
            ic = [v[0] for v in g[y]]; t = sorted(abs(v[1]) for v in g[y])
            if np.sign(ic[0]) == np.sign(ic[1]) and t[1] >= 2 and t[0] >= 1:
                w = g.wt if ic[0] > 0 else g.wb
                if all(w.to_numpy() >= g.wall.to_numpy()): ok = True; why.append(y[1:])
        if ok: vt.append(f"- {s}｜{f}：提名（{'、'.join(why)} 日增量；" + '｜'.join(f"IC20 {a[0]:+.3f} t {a[1]:.1f}／IC60 {b[0]:+.3f} t {b[1]:.1f}" for a, b in zip(g.i20, g.i60)) + '）')
    head = ['## 提名（增量 IC 兩期同號、t ≥ 2／≥ 1，好組 20 日淨勝率 ≥ 全體）\n'] + (vt or ['- 無']) + ['']
    return L[:1] + head + L[1:], vt


def main():
    L, vt = analyze(panel(1), 'M-200 結果：X1 候選與 V4.1 訊號內的因子相關與增量（奇數代碼、排除最近兩年）')
    return dict(summary='\n'.join(L), headline=f"提名 {len(vt)} 個", n_variants=0)
