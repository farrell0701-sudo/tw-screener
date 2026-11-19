# M-128：散戶增加 × 過熱／最陡／脈衝（強勢池內，奇數代碼，2010–18／2019–26）。定義見 registry_M.md M-128。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals
from M107 import nw_t
from M113 import states
from M127 import tdcc_factors, ERAS

SN = ['上收斂', '上平行', '上發散', '下']; BN = ['散戶減', '中', '散戶增']


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    S = signals.compute('fix', P=P, F=F); d = S['_dbg']; U, pool, hot, steep = d['U'], d['pool'], d['hot'], d['steep']; del S
    st, _ = states(F); s4 = np.where(st >= 3, 3, st)
    Cf = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill().to_numpy(); VALID = np.asarray(F['VALID'])
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CX = np.cumsum(R - ew[:, None], axis=0)
    odd = (P['meta']['stock_id'].astype(str).str[-1].astype(int) % 2 == 1).to_numpy()
    keys = P['meta']['security_key'].astype(str).to_numpy()
    B4 = tdcc_factors(dates)['B4 散戶 1 月變化']; jmap = pd.Index(B4.columns.astype(str)).get_indexer(keys)
    ds = pd.Series(dates, index=dates); s = ds[ds.dt.day > 10]; samp = s.groupby(s.dt.to_period('M')).min()
    L = ['# M-128 結果：散戶增加 × 過熱／最陡／脈衝（強勢池內，奇數代碼）\n',
         '散戶三組在當期強勢池內分；「差」＝該格平均 − 同期強勢池平均（期數平均，NW t）；左／右尾＝同期強勢池後 60 日超額最差／最好 10%。\n']
    for era, lo, hi in ERAS:
        rows, per = [], []
        for p, t0 in samp.items():
            if not (pd.Timestamp(lo) <= t0 <= pd.Timestamp(hi)): continue
            t = dates.get_loc(t0)
            if t + 60 >= T or (p - 1) not in B4.index: continue
            b = B4.loc[p - 1].to_numpy(float); bx = np.where(jmap >= 0, b[np.clip(jmap, 0, None)], np.nan)
            Sm = pool[t] & odd & np.isfinite(bx) & (s4[t] >= 0)
            if Sm.sum() < 30: continue
            jj = np.nonzero(Sm)[0]
            e20 = CX[t + 20, jj] - CX[t, jj]; e60 = CX[t + 60, jj] - CX[t, jj]; q10, q90 = np.nanpercentile(e60, [10, 90])
            bt = pd.qcut(pd.Series(bx[jj]).rank(method='first'), 3, labels=False).to_numpy()
            df = pd.DataFrame(dict(h=np.where(hot[t, jj], '過熱', '非過熱'), st=np.where(steep[t, jj], '最陡', '非最陡'), s=s4[t, jj], b=bt, e20=e20, e60=e60, left=e60 <= q10, right=e60 >= q90))
            m20, m60 = e20.mean(), e60.mean()
            for dim in ('h', 'st', 's'):
                for (a, bb), x in df.groupby([dim, 'b']): per.append(dict(dim=dim, a=a, b=bb, d20=x.e20.mean() - m20, d60=x.e60.mean() - m60))
            rows.append(df)
        X = pd.concat(rows, ignore_index=True); Pd = pd.DataFrame(per)
        L += [f'## {era}（期數 {len(rows)}，觀測 {len(X):,}）\n']
        for dim, levels, nm in (('h', ['過熱', '非過熱'], '表 1：過熱'), ('st', ['最陡', '非最陡'], '表 2：最陡'), ('s', [0, 1, 2, 3], '表 3：脈衝')):
            L += [f'### {nm} × 散戶\n', '| 組 | 散戶 | 觀測 | 後20日差（t） | 後60日差（t） | 左尾率 | 右尾率 |', '|---|---|---|---|---|---|---|']
            for a in levels:
                for bb in range(3):
                    x = X[(X[dim] == a) & (X.b == bb)]; pp = Pd[(Pd.dim == dim) & (Pd.a == a) & (Pd.b == bb)]
                    an = SN[a] if dim == 's' else a
                    if len(x) < 30: L.append(f'| {an} | {BN[bb]} | {len(x)} | （樣本 < 30） | | | |'); continue
                    L.append(f"| {an} | {BN[bb]} | {len(x)} | {pp.d20.mean():+.2%}（{nw_t(pp.d20, 1):.1f}） | {pp.d60.mean():+.2%}（{nw_t(pp.d60, 3):.1f}） | {x.left.mean():.0%} | {x.right.mean():.0%} |")
            L.append('')
    return dict(summary='\n'.join(L), headline='散戶 × 剔除／脈衝觀察完成', n_variants=0)
