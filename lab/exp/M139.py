# M-139：已確認因子在多空頭的表現（強勢池、全部代碼）。定義見 registry_M.md M-139。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, M120
from M107 import nw_t
from M108 import episodes
from M127 import tdcc_factors
from M138 import dt_factors


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    S = signals.compute('fix', P=P, F=F); U, pool = S['_dbg']['U'], S['_dbg']['pool']; del S
    Cf = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill().to_numpy(); VALID = np.asarray(F['VALID'])
    LA = np.asarray(F['LA'], np.float64)
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CX = np.cumsum(R - ew[:, None], axis=0)
    keys = P['meta']['security_key'].astype(str).to_numpy()
    base = M120.rev_factors(); R1, R2, R4, R5 = base['R1 單月年增'], base['R2 三月年增'], base['R4 營收創高'], base['R5 年增驚喜']
    sd = R1.rolling(12, min_periods=10).std(); E4 = R2 / sd.where(sd > 0)
    B4 = tdcc_factors(dates)['B4 散戶 1 月變化']; D1 = dt_factors(dates)['D1 當沖比 20 日']
    def getter(W):
        jm = pd.Index(W.columns.astype(str)).get_indexer(keys)
        return lambda m: (np.where(jm >= 0, W.loc[m].to_numpy(float)[np.clip(jm, 0, None)], np.nan) if m in W.index else np.full(N, np.nan))
    G = {'R4': getter(R4), 'E4': getter(E4), 'R5': getter(R5), 'B4': getter(B4), 'D1': getter(D1)}
    ix = P['index']['TAIEX_P'].reindex(dates).ffill(); below = (ix < ix.rolling(200).mean()).to_numpy()
    phase = pd.Series('其他', index=dates)
    for pk, tr, _ in episodes(ix.loc['2005':]):
        a, b = dates.get_loc(pk), dates.get_loc(tr)
        phase.iloc[a + 1:b + 1] = '下跌段'; phase.iloc[b + 1:b + 61] = '反彈段'
    ds = pd.Series(dates, index=dates); s = ds[ds.dt.day > 10]; samp = s.groupby(s.dt.to_period('M')).min()
    pr = lambda a: pd.Series(a).rank(pct=True).to_numpy()
    rows = []
    for p, t0 in samp.items():
        t = dates.get_loc(t0); m = p - 1
        if t0 < pd.Timestamp('2005-01-01') or t + 60 >= T: continue
        jj0 = np.nonzero(pool[t])[0]
        y60a = CX[t + 60, jj0] - CX[t, jj0]; q10, q90 = np.nanpercentile(y60a, [10, 90])
        with np.errstate(all='ignore'): dna = np.nanmin(LA[t + 1:t + 61, jj0], axis=0) / Cf[t, jj0] - 1
        md = np.nanmedian(dna)
        v = {k: g(m)[jj0] for k, g in G.items()}
        scores = {}
        ok = np.isfinite(v['R4']) & np.isfinite(v['E4'])
        if ok.sum() >= 20: scores['RC 營收合成'] = (ok, (pr(v['R4'][ok]) + pr(v['E4'][ok])) / 2)
        for k, nm, sg in (('R5', 'R5 年增驚喜', 1), ('B4', '散戶減', -1), ('D1', 'D1 當沖比（風險）', 1)):
            ok = np.isfinite(v[k])
            if ok.sum() >= 20: scores[nm] = (ok, sg * v[k][ok])
        y20a = CX[t + 20, jj0] - CX[t, jj0]
        for nm, (ok, sc) in scores.items():
            top = pr(sc) > 0.8; y60, y20, dn = y60a[ok], y20a[ok], dna[ok]
            rows.append(dict(t=t0, f=nm, below=bool(below[t - 1]), ph=phase.iloc[t], d20=np.nanmean(y20[top]) - np.nanmean(y20a), d60=np.nanmean(y60[top]) - np.nanmean(y60a),
                             left=np.nanmean(y60[top] <= q10), right=np.nanmean(y60[top] >= q90), dn=np.nanmean(dn[top]) - md))
    D = pd.DataFrame(rows)
    def line(x, lab):
        if len(x) < 6: return f'| {lab} | {len(x)} | （期數不足） | | | | |'
        return f"| {lab} | {len(x)} | {x.d20.mean():+.2%}（{nw_t(x.d20, 1):.1f}） | {x.d60.mean():+.2%}（{nw_t(x.d60, 3):.1f}） | {x.left.mean():.0%} | {x.right.mean():.0%} | {x.dn.mean():+.1%} |"
    hdr = ['| 環境 | 期數 | 後20日差（t） | 後60日差（t） | 左尾率 | 右尾率 | 60日最大跌幅（扣中位） |', '|---|---|---|---|---|---|---|']
    L = ['# M-139 結果：已確認因子在多空頭的表現（強勢池、全部代碼；前 20% 組與同期強勢池平均之差）\n']
    eras = {'RC 營收合成': ('2005-01-01', '2016-01-01', '2005–15', '2016–26'), 'R5 年增驚喜': ('2005-01-01', '2016-01-01', '2005–15', '2016–26'),
            '散戶減': ('2010-01-01', '2019-01-01', '2010–18', '2019–26'), 'D1 當沖比（風險）': ('2016-01-01', '2021-01-01', '2016–20', '2021–26')}
    for f in ['RC 營收合成', 'R5 年增驚喜', '散戶減', 'D1 當沖比（風險）']:
        x = D[D.f == f]; s0, s1, e0, e1 = eras[f]
        L += [f'## {f}\n', '### 大盤 200MA（無前視）\n'] + hdr
        L += [line(x[~x.below], '200MA 上'), line(x[x.below], '200MA 下'),
              line(x[x.below & (x.t < s1)], f'200MA 下｜{e0}'), line(x[x.below & (x.t >= s1)], f'200MA 下｜{e1}')]
        L += ['', '### M-108 時期（前視）\n'] + hdr + [line(x[x.ph == ph], ph) for ph in ('下跌段', '反彈段', '其他')] + ['']
    return dict(summary='\n'.join(L), headline='多空頭描述完成', n_variants=0)
