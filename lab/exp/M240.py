# M-240：強勢池貼高＋量縮突破事件，營收分組。定義見 registry_M.md M-240。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, M190, M180
from L123 import daily_from_monthly
from M195 import HOLDOUT, ERAS, COST
from M107 import nw_t

HS = (5, 10, 20, 60, 120)


def build(parity=1):
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    signals.build('fix', mode='fix', P=P, F=F)
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; cols = dbg['cols']
    def full(a, fill):
        a = np.asarray(a)
        if a.shape[1] == N: return a
        o = np.full((T, N), fill, a.dtype); o[:, cols] = a; return o
    U = full(dbg['U'], False); pool = full(dbg['pool'], False); Tr = full(np.asarray(dbg['Tr'], np.float64), np.nan); del S
    daily = signals.load_sig('fix')['daily'].reindex(dates); below = daily['below'].fillna(True).to_numpy(bool)
    FE = M190.features(P, F, Tr); BRK = np.asarray(FE['BRK20'], np.float64); VC = np.asarray(FE['VC'], np.float64); VX = np.asarray(FE['VX'], np.float64)
    VALID = np.asarray(F['VALID']); OA = np.asarray(F['OA'], np.float64)
    C = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill(); H = pd.DataFrame(np.asarray(F['HA'], np.float64)).ffill(); Cf = C.to_numpy()
    DH = (C / H.rolling(250, min_periods=200).max() - 1).to_numpy()
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CX = np.cumsum(R - ew[:, None], axis=0)
    keys = P['meta']['security_key'].astype(str).to_numpy()
    par = (P['meta']['stock_id'].astype(str).str[-1].astype(int) % 2 == parity).to_numpy()
    r6 = daily_from_monthly(M180.r6_factors(dates)['R6 營收短長比'], dates, keys)
    pr = lambda x: pd.DataFrame(np.where(pool, x, np.nan)).rank(axis=1, pct=True).to_numpy()
    vcp = pr(VC); vxp = pr(VX); r6p = pr(r6)
    t0 = dates.searchsorted(pd.Timestamp('2005-01-01')); t1 = min(dates.searchsorted(HOLDOUT), T - 122)
    last = np.full(N, -10 ** 9); rows = []
    for t in range(t0, t1):
        jj = np.nonzero(pool[t] & par & (BRK[t] > 0) & ~(BRK[t - 1] > 0) & VALID[t + 1])[0]
        for j in jj:
            if t - last[j] < 20: continue
            last[j] = t
            ent = OA[t + 1, j] if np.isfinite(OA[t + 1, j]) and OA[t + 1, j] > 0 else Cf[t + 1, j]
            path = CX[t + 1:t + 61, j] - CX[t, j]
            rows.append(dict(t=t, j=j, dh=DH[t, j], vc=vcp[t - 1, j], vx=vxp[t - 1, j], r6=r6p[t, j], below=below[t],
                             **{f'e{h}': CX[t + h, j] - CX[t, j] for h in HS}, n20=Cf[t + 20, j] / ent - 1 - COST, pk=int(np.argmax(path)) + 1))
    D = pd.DataFrame(rows); D['date'] = dates[D.t.to_numpy()]
    D['hi'] = D.dh >= -0.05; D['qt'] = (D.vc < 0.4) & (D.vx < 0.4)
    D['rv'] = np.where(D.r6.isna(), '缺', np.where(D.r6 >= 2 / 3, '高', np.where(D.r6 < 1 / 3, '低', '中')))
    return D


def msr(x, col):   # 月平均序列
    return x.groupby(x.date.dt.to_period('M'))[col].mean()


def main():
    D = build(1); L = ['# M-240 結果：強勢池貼高＋量縮突破（奇數代碼）\n']; res = {}
    for era, lo, hi in ERAS:
        E = D[(D.date >= lo) & (D.date <= hi)]; ny = (pd.Timestamp(hi) - pd.Timestamp(lo)).days / 365.25
        G = {'A 全部突破': E, 'B 突破＋貼高': E[E.hi], 'C 突破＋量縮': E[E.qt], 'D 突破＋貼高＋量縮': E[E.hi & E.qt]}
        for k in ('低', '中', '高'): G[f'D｜營收{k}'] = E[E.hi & E.qt & (E.rv == k)]
        for k in ('低', '高'): G[f'A｜營收{k}'] = E[E.rv == k]
        G['D｜大盤 200MA 上'] = E[E.hi & E.qt & ~E.below]; G['D｜大盤 200MA 下'] = E[E.hi & E.qt & E.below]
        L += [f'## {era}\n', '| 組 | 每年事件 | 5日 | 10日 | 20日 | 60日 | 120日 | 20日淨勝率 | 左尾率 | 右尾率 | 高峰中位日 |', '|---|---|---|---|---|---|---|---|---|---|---|']
        for nm, g in G.items():
            L.append(f"| {nm} | {len(g) / ny:.0f} | " + ' | '.join(f"{g[f'e{h}'].mean():+.2%}" for h in HS) +
                     f" | {(g.n20 > 0).mean():.0%} | {(g.n20 <= -0.10).mean():.1%} | {(g.n20 >= 0.20).mean():.1%} | {g.pk.median():.0f} |")
        a, d = G['A 全部突破'], G['D 突破＋貼高＋量縮']
        dd = (msr(d, 'e20') - msr(a, 'e20')).dropna()
        r = {'dA': (dd.mean(), nw_t(dd, 1)), 'win': ((d.n20 > 0).mean(), (a.n20 > 0).mean()), 'pkD': d.pk.median()}
        for h, lag in ((20, 1), (60, 3), (120, 6)):
            x = (msr(G['D｜營收高'], f'e{h}') - msr(G['D｜營收低'], f'e{h}')).dropna(); r[f'rv{h}'] = (x.mean(), nw_t(x, lag), len(x))
            y = (msr(G['A｜營收高'], f'e{h}') - msr(G['A｜營收低'], f'e{h}')).dropna(); r[f'ra{h}'] = (y.mean(), nw_t(y, lag))
        res[era] = r
        L += ['', f"D − A 20 日超額（月序列）：{r['dA'][0]:+.2%}（t {r['dA'][1]:.1f}）",
              'D 內 營收高 − 低：' + '；'.join(f"{h} 日 {r[f'rv{h}'][0]:+.2%}（t {r[f'rv{h}'][1]:.1f}，{r[f'rv{h}'][2]} 月）" for h in (20, 60, 120)),
              'A（全部突破）內 營收高 − 低：' + '；'.join(f"{h} 日 {r[f'ra{h}'][0]:+.2%}（t {r[f'ra{h}'][1]:.1f}）" for h in (20, 60, 120)), '']
    v = list(res.values())
    ts = sorted(abs(x['dA'][1]) for x in v)
    nom = all(x['dA'][0] > 0 for x in v) and ts[1] >= 2 and ts[0] >= 1 and all(x['win'][0] >= x['win'][1] for x in v)
    def rvj(h):
        m = [x[f'rv{h}'] for x in v]
        if all(a < 0 and t <= -1 for a, t, _ in m): return '營收好較差'
        if all(a > 0 and t >= 1 for a, t, _ in m): return '營收好較好'
        return '無明顯差異'
    hz = '；'.join(f"{e} D 高峰 {x['pkD']:.0f} 日（{'短期型' if x['pkD'] <= 20 else '延續型' if x['pkD'] > 30 else '中間'}）" for e, x in res.items())
    head = f"組合{'提名' if nom else '不提名'}；營收 20 日：{rvj(20)}，60 日：{rvj(60)}，120 日：{rvj(120)}；{hz}"
    return dict(summary='\n'.join(L[:1] + [f'**{head}**\n'] + L[1:]), headline=head, n_variants=0)
