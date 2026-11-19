# M-107：整個 T_ld 強勢池中，均線狀態（M-105 六組）的後續報酬方向（觀察式，探索集）。定義見 registry_M.md M-107。
import sys, math
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core'))
from panel import load as load_panel
from feat import load_feat
import signals

START, END = pd.Timestamp('2005-01-01'), pd.Timestamp('2015-12-31')
NAMES = ['上收斂', '上平行', '上發散', '下收斂', '下平行', '下發散']


def nw_t(x, lag):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 10: return float('nan')
    xc = x - x.mean(); v = xc @ xc / n
    for l in range(1, lag + 1): v += 2 * (1 - l / (lag + 1)) * (xc[l:] @ xc[:-l]) / n
    return float(x.mean() / math.sqrt(v / n)) if v > 0 else float('nan')


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; U, pool = dbg['U'], dbg['pool']
    VALID = np.asarray(F['VALID']); Cfd = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill(); Cf = Cfd.to_numpy()
    m120 = Cfd.rolling(120, min_periods=108).mean(); m200 = Cfd.rolling(200, min_periods=180).mean()
    g = (m120 / m200 - 1).to_numpy(); del m120, m200
    ag = np.abs(g); dag = ag - np.vstack([np.full((20, N), np.nan), ag[:-20]])
    st = np.full(g.shape, -1, np.int8); f = np.isfinite(dag) & np.isfinite(g); c = 0.01
    for base, side in ((0, g >= 0), (3, g < 0)):
        m = f & side; st[m & (dag < -c)] = base; st[m & (np.abs(dag) <= c)] = base + 1; st[m & (dag > c)] = base + 2
    with np.errstate(all='ignore'):
        R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'):
        ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CX = np.cumsum(R - ew[:, None], axis=0)
    ix = P['index']['TAIEX_P'].reindex(dates).ffill(); mk = (ix.rolling(120).mean() > ix.rolling(200).mean()).to_numpy()
    odd = (P['meta']['stock_id'].astype(str).str[-1].astype(int) % 2 == 1).to_numpy()
    t_lo = int(dates.searchsorted(START)); t_hi = int(dates.searchsorted(END, side='right')) - 1
    rows, per = [], []
    for t in range(t_lo, t_hi - 60 + 1, 21):
        Sm = pool[t] & odd & (st[t] >= 0)
        if Sm.sum() < 15: continue
        jj = np.nonzero(Sm)[0]
        e20 = CX[t + 20, jj] - CX[t, jj]; e60 = CX[t + 60, jj] - CX[t, jj]
        with np.errstate(all='ignore'): a20 = Cf[t + 20, jj] / Cf[t, jj] - 1
        q10, q90 = np.nanpercentile(e20, [10, 90]); s = st[t, jj]
        for k in range(6):
            m = s == k
            if m.sum(): per.append(dict(t=t, k=k, d20=e20[m].mean() - e20.mean(), d60=e60[m].mean() - e60.mean(), da=a20[m].mean() - np.nanmean(a20), n=int(m.sum())))
        rows.append(pd.DataFrame(dict(k=s, e20=e20, e60=e60, a20=a20, left=e20 <= q10, right=e20 >= q90, mk=mk[t])))
    X = pd.concat(rows, ignore_index=True); Pd = pd.DataFrame(per)
    L = ['# M-107 結果：整個強勢池的均線狀態後續報酬（觀察式，探索集：奇數代碼 2005–15）\n',
         f"期數 {Pd.t.nunique()}（每 21 日），觀測 {len(X):,} 筆（股票 × 期）。超額＝扣可交易池等權；「差」＝該組平均減同期強勢池平均，期數平均與 NW t。\n",
         '| 狀態 | 觀測 | 占比 | 後 20 日超額 | 差（t） | 後 60 日超額 | 差（t） | 後 20 日絕對報酬 | 差（t） | 勝率（超額 > 0） | 左尾率 | 右尾率 |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for k, nm in enumerate(NAMES):
        x = X[X.k == k]; p = Pd[Pd.k == k]
        if not len(x): continue
        L.append(f"| {nm} | {len(x):,} | {len(x) / len(X):.0%} | {x.e20.mean():+.2%} | {p.d20.mean():+.2%}（{nw_t(p.d20, 2):.1f}） | {x.e60.mean():+.2%} | {p.d60.mean():+.2%}（{nw_t(p.d60, 3):.1f}） | "
                 f"{x.a20.mean():+.2%} | {p.da.mean():+.2%}（{nw_t(p.da, 2):.1f}） | {(x.e20 > 0).mean():.0%} | {x.left.mean():.0%} | {x.right.mean():.0%} |")
    L.append(f"| 全部 | {len(X):,} | 100% | {X.e20.mean():+.2%} | — | {X.e60.mean():+.2%} | — | {X.a20.mean():+.2%} | — | {(X.e20 > 0).mean():.0%} | 10% | 10% |")
    L += ['\n## 依大盤 120 日線位置（後 20 日絕對報酬 ｜ 超額；觀測數）\n', '| 狀態 | 大盤 120 在上 | 大盤 120 在下 |', '|---|---|---|']
    for k, nm in enumerate(NAMES):
        cells = []
        for mv in (True, False):
            x = X[(X.k == k) & (X.mk == mv)]
            cells.append(f"{x.a20.mean():+.2%} ｜ {x.e20.mean():+.2%}（{len(x):,}）" if len(x) else '—')
        L.append(f'| {nm} | ' + ' | '.join(cells) + ' |')
    for mv, nm in ((True, '全部（大盤在上）'), (False, '全部（大盤在下）')):
        x = X[X.mk == mv]; L.append(f"| {nm} | " + (f"{x.a20.mean():+.2%} ｜ {x.e20.mean():+.2%}（{len(x):,}）" if mv else '') + ' | ' + ('' if mv else f"{x.a20.mean():+.2%} ｜ {x.e20.mean():+.2%}（{len(x):,}）") + ' |')
    return dict(summary='\n'.join(L), headline='觀察式描述完成', n_variants=0)
