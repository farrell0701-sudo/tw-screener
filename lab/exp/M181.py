# M-181：月初 1–10 日先漲對 R6 頭部股票的影響、突然強勢日進場 vs 11 日進場、進場後回撤（奇數代碼）。定義見 registry_M.md M-181。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'exp')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core'))
import M180                      # 含日期型別相容層與 R6 定義
from panel import load as load_panel
from feat import load_feat
import signals
from M107 import nw_t

ERAS = (('2005–15', '2005-01-01', '2015-12-31'), ('2016–26', '2016-01-01', '2026-12-31'))
HS = (5, 20, 60, 120)


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    S = signals.compute('fix', P=P, F=F); U = S['_dbg']['U']; del S
    Cf = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill().to_numpy(); LA = np.asarray(F['LA'], np.float64)
    ATR = np.asarray(F['ATR'], np.float64); VALID = np.asarray(F['VALID'])
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    XR = R - ew[:, None]; XRv = np.where(VALID, XR, np.nan); CX = np.cumsum(R - ew[:, None], axis=0)
    keys = P['meta']['security_key'].astype(str).to_numpy()
    odd = (P['meta']['stock_id'].astype(str).str[-1].astype(int) % 2 == 1).to_numpy()
    R6 = M180.r6_factors()['R6 營收短長比']; jmap = pd.Index(R6.columns.astype(str)).get_indexer(keys)
    ds = pd.Series(dates, index=dates); samp = ds[ds.dt.day > 10].groupby(ds[ds.dt.day > 10].dt.to_period('M')).min()
    samp = [dates.get_loc(d) for d in samp if d >= pd.Timestamp('2005-01-01') and dates.get_loc(d) + 120 < T]
    rows = []
    for t in samp:
        m = dates[t].to_period('M') - 1
        if m not in R6.index: continue
        s0 = dates.searchsorted(dates[t].to_period('M').start_time); t0 = s0 - 1
        if t - s0 < 2 or t0 < 61: continue
        fv = R6.loc[m].to_numpy(float); x = np.where(jmap >= 0, fv[np.clip(jmap, 0, None)], np.nan)
        sel = U[t] & odd & np.isfinite(x) & np.isfinite(Cf[t])
        if sel.sum() < 50: continue
        jj = np.nonzero(sel)[0]; xv = x[jj]
        top = pd.Series(xv).rank(pct=True).to_numpy() > 0.8
        pre = CX[t - 1, jj] - CX[t0, jj]
        sd = np.nanstd(XRv[t0 - 59:t0 + 1, jj], axis=0)
        z = XRv[s0:t, jj] / np.where(sd > 0, sd, np.nan)
        hit = z >= 2; has = hit.any(0); sidx = np.where(has, s0 + hit.argmax(0), -1)
        ex = {h: CX[t + h, jj] - CX[t, jj] for h in HS}
        with np.errstate(all='ignore'):
            lo20 = np.nanmin(LA[t + 1:t + 21, jj], axis=0) / Cf[t, jj] - 1
            cmin = np.nanmin(Cf[t + 1:t + 21, jj], axis=0)
            pb = cmin < Cf[t, jj] - ATR[t, jj]
            dmin = np.nanargmin(np.where(np.isnan(Cf[t + 1:t + 21, jj]), np.inf, Cf[t + 1:t + 21, jj]), axis=0) + 1
        for i, j in enumerate(jj):
            si = sidx[i]
            r = dict(t=t, top=top[i], pre=pre[i], has=has[i], lo20=lo20[i], pb=pb[i], dmin=dmin[i], **{f'e{h}': ex[h][i] for h in HS})
            if si >= 0:
                r['s_gone'] = CX[t, j] - CX[si, j]
                for h in (20, 60, 120): r[f's{h}'] = CX[t + h, j] - CX[si, j]
                r['s_day'] = int(si - s0 + 1)
            rows.append(r)
    D = pd.DataFrame(rows); D['date'] = dates[D.t.to_numpy()]
    L = ['# M-181 結果：月初先漲 × R6 頭部（奇數代碼；F 計數 57）\n',
         '進場＝取樣日 t（每月第一個 日>10 的交易日）收盤；超額扣可交易池等權；Q1＝月初超額最少、Q5＝最多（R6 頭部內每期五分組）。\n']
    verdict = {}
    for era, lo, hi in ERAS:
        E = D[(D.date >= lo) & (D.date <= hi)].copy()
        A = E[E.top].copy()
        A['q'] = A.groupby('t').pre.transform(lambda s: pd.qcut(s.rank(method='first'), 5, labels=False)) + 1
        L += [f'## {era}\n',
              f"R6 頭部：每期約 {A.groupby('t').size().mean():.0f} 檔；月初超額 平均 {A.pre.mean():+.2%}、中位 {A.pre.median():+.2%}（全體 {E.pre.mean():+.2%}／{E.pre.median():+.2%}）；"
              f"有突然強勢日 {A.has.mean():.0%}（全體 {E.has.mean():.0%}）。\n",
              '| 組 | 月初超額 | 後5日 | 後20日 | 後60日 | 後120日 | 60日中位 | 60日勝率 | 20日最大跌幅 | 20日回撤率（<進場−1ATR） | 到最低天數 |', '|---|---|---|---|---|---|---|---|---|---|---|']
        for nm, g in [(f'Q{k}', A[A.q == k]) for k in range(1, 6)] + [('R6 頭部全部', A), ('全體可交易池', E)]:
            L.append(f"| {nm} | {g.pre.mean():+.2%} | " + ' | '.join(f'{g[f"e{h}"].mean():+.2%}' for h in HS) +
                     f" | {g.e60.median():+.2%} | {(g.e60 > 0).mean():.0%} | {g.lo20.mean():+.2%} | {g.pb.mean():.0%} | {g.dmin.mean():.1f} |")
        sp = {h: (A[A.q == 5].groupby('t')[f'e{h}'].mean() - A[A.q == 1].groupby('t')[f'e{h}'].mean()).dropna() for h in HS}
        L.append('| Q5−Q1（t） | | ' + ' | '.join(f'{sp[h].mean():+.2%}（{nw_t(sp[h], max(1, h // 21)):.1f}）' for h in HS) + ' | | | | | |\n')
        pbd = A[A.q == 5].pb.mean() - A[A.q == 1].pb.mean()
        verdict[era] = (sp[60].mean(), nw_t(sp[60], 3), pbd)
        # 突然強勢日進場 vs t 進場
        B = A[A.has]
        L += ['**突然強勢日 S 進場 vs 11 日（t）進場**（R6 頭部、有 S 者；S 進場含事後挑選，只是上限）\n',
              f"S 平均在月初第 {B.s_day.mean():.1f} 個交易日；S→t 已漲掉超額 平均 {B.s_gone.mean():+.2%}、中位 {B.s_gone.median():+.2%}\n",
              '| 終點 | S 進場超額 | t 進場超額 | 差 | S 勝率 | t 勝率 |', '|---|---|---|---|---|---|']
        for h in (20, 60, 120):
            L.append(f"| t+{h} | {B[f's{h}'].mean():+.2%} | {B[f'e{h}'].mean():+.2%} | {(B[f's{h}'] - B[f'e{h}']).mean():+.2%} | {(B[f's{h}'] > 0).mean():.0%} | {(B[f'e{h}'] > 0).mean():.0%} |")
        L.append('')
        nb = A[~A.has]
        L += [f"對照：R6 頭部「有 S」vs「無 S」自 t 進場：後20日 {B.e20.mean():+.2%} vs {nb.e20.mean():+.2%}；後60日 {B.e60.mean():+.2%} vs {nb.e60.mean():+.2%}；"
              f"後120日 {B.e120.mean():+.2%} vs {nb.e120.mean():+.2%}；20日回撤率 {B.pb.mean():.0%} vs {nb.pb.mean():.0%}\n"]
    a = [verdict[e] for e, _, _ in ERAS]; ts = sorted([v[1] for v in a])
    h1 = all(v[0] < 0 for v in a) and ts[0] <= -2 and ts[1] <= -1
    h2 = all(v[2] >= 0.05 for v in a)
    vt = (f"判準：H1 月初漲多傷 60 日報酬（Q5−Q1 兩期 < 0、t 一期 ≤ −2 另一期 ≤ −1）{'成立' if h1 else '不成立'}"
          f"（{'｜'.join(f'{v[0]:+.2%} t {v[1]:.1f}' for v in a)}）；H2 Q5 回撤率比 Q1 高 ≥ 5pt {'成立' if h2 else '不成立'}（{'｜'.join(f'{v[2] * 100:+.0f}pt' for v in a)}）\n")
    return dict(summary='\n'.join([L[0], vt] + L[1:]), headline=vt.strip(), n_variants=0)
