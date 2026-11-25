# M-191：強勢池內安靜度（VC、VX、QUIET、BBWp、OBVD）對超額與快速成功（奇數代碼、排除最近兩年）。定義見 registry_M.md M-191。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals
import M190
from M107 import nw_t

HOLDOUT = pd.Timestamp('2024-11-25')
ERAS = (('2005–15', '2005-01-01', '2015-12-31'), ('2016–24', '2016-01-01', '2024-11-24'))
FE5 = ['VC', 'VX', 'QUIET', 'BBWp', 'OBVD']


def extra(P, F):
    C = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill(); H = pd.DataFrame(np.asarray(F['HA'], np.float64)).ffill()
    L = pd.DataFrame(np.asarray(F['LA'], np.float64)).ffill(); A = pd.DataFrame(np.asarray(P['AMT'], np.float64)).fillna(0)
    pc = C.shift(1); trr = pd.concat([H - L, (H - pc).abs(), (L - pc).abs()]).groupby(level=0).max()
    atr20 = trr.rolling(20, min_periods=15).mean()
    bbw = 4 * C.rolling(20, min_periods=15).std() / C.rolling(20, min_periods=15).mean()
    bbwp = bbw.rolling(120, min_periods=60).rank(pct=True)
    sgn = np.sign(C.diff()).fillna(0); obv = (sgn * A).cumsum()
    obvn = (obv - obv.shift(20)) / A.rolling(20, min_periods=15).sum().replace(0, np.nan)
    prn = (C - C.shift(20)) / (20 * atr20)
    return dict(ATR20=atr20.to_numpy(), BBWp=bbwp.to_numpy(), OBVD=(obvn - prn).to_numpy())


def quick(base, atr, H, L):
    """20 日內先碰 base+2ATR（最高價）再碰 base−1ATR（最低價）＝成功；同日都碰＝失敗。"""
    up = H >= (base + 2 * atr)[None, :]; dn = L <= (base - atr)[None, :]
    n = H.shape[0]
    iu = np.where(up.any(0), up.argmax(0), n + 1); idn = np.where(dn.any(0), dn.argmax(0), n + 1)
    with np.errstate(all='ignore'):
        mfe = (np.nanmax(H, 0) - base) / atr; mae = (np.nanmin(L, 0) - base) / atr
    return (iu < idn) & (iu <= n), mfe, mae, np.where(iu <= n, iu + 1, np.nan)


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    signals.build('fix', mode='fix', P=P, F=F)
    S_ = signals.compute('fix', P=P, F=F); dbg = S_['_dbg']; cols = dbg['cols']
    def full(a, fill):
        if a.shape[1] == N: return np.asarray(a)
        o = np.full((T, N), fill, np.asarray(a).dtype); o[:, cols] = a; return o
    U = full(dbg['U'], False); pool = full(dbg['pool'], False); Tr = full(np.asarray(dbg['Tr'], np.float64), np.nan); del S_
    FE = M190.features(P, F, Tr); EX = extra(P, F)
    HA = np.asarray(F['HA'], np.float64); LA = np.asarray(F['LA'], np.float64); VALID = np.asarray(F['VALID'])
    Cf = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill().to_numpy()
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CX = np.cumsum(R - ew[:, None], axis=0)
    odd = (P['meta']['stock_id'].astype(str).str[-1].astype(int) % 2 == 1).to_numpy()
    t0 = dates.searchsorted(pd.Timestamp('2005-01-01')); t1 = dates.searchsorted(HOLDOUT) - 1
    rows = []
    for t in range(t0, min(t1, T - 61), 5):
        jj = np.nonzero(pool[t] & odd & VALID[t])[0]
        if len(jj) < 10: continue
        e60u = CX[t + 60, U[t]] - CX[t, U[t]]; q10, q90 = np.nanpercentile(e60u, [10, 90])
        vc, vx = FE['VC'][t, jj], FE['VX'][t, jj]
        qt = (pd.Series(-vc).rank(pct=True).to_numpy() + pd.Series(-vx).rank(pct=True).to_numpy()) / 2
        atr = EX['ATR20'][t, jj]; base = Cf[t, jj]
        ok, mfe, mae, _ = quick(base, atr, HA[t + 1:t + 21, jj], LA[t + 1:t + 21, jj])
        e20 = CX[t + 20, jj] - CX[t, jj]; e60 = CX[t + 60, jj] - CX[t, jj]
        rows.append(pd.DataFrame(dict(t=t, VC=vc, VX=vx, QUIET=qt, BBWp=EX['BBWp'][t, jj], OBVD=EX['OBVD'][t, jj], Tld=Tr[t, jj],
                                      e20=e20, e60=e60, ok=ok.astype(float), mfe=mfe, mae=mae, left=(e60 <= q10).astype(float), right=(e60 >= q90).astype(float))))
    D = pd.concat(rows, ignore_index=True); D['date'] = dates[D.t.to_numpy()]
    D = D[np.isfinite(D.e60) & np.isfinite(D.mfe)]
    L = ['# M-191 結果：強勢池內安靜度（奇數代碼、排除最近兩年；每 5 日取樣）\n',
         '快速成功＝20 日內先 +2ATR 再 −1ATR；MFE／MAE 以 ATR20 為單位；超額扣可交易池等權；Q1＝特徵最低。\n']
    res = {}
    for era, lo, hi in ERAS:
        E = D[(D.date >= lo) & (D.date <= hi)]
        L += [f"## {era}（{E.t.nunique()} 個取樣日、{len(E):,} 筆；池內快速成功 {E.ok.mean():.1%}、60 日超額 平均 {E.e60.mean():+.2%}、中位 {E.e60.median():+.2%}；左尾 {E.left.mean():.1%}、右尾 {E.right.mean():.1%}）\n",
              '| 特徵 | IC20（t） | IC60（t） | IC 快速成功（t） | Q1 快速成功 | Q5 快速成功 | Q1 60日中位 | Q5 60日中位 | Q1 60日平均 | Q5 60日平均 | Q1 MFE／MAE | Q5 MFE／MAE | Q1 左／右尾 | Q5 左／右尾 |',
              '|' + '---|' * 14]
        for f in FE5 + ['Tld']:
            x = E.dropna(subset=[f]).copy()
            x['q'] = x.groupby('t')[f].transform(lambda s: pd.qcut(s.rank(method='first'), 5, labels=False) + 1 if len(s) >= 5 else np.nan)
            ics = {}
            for y in ('e20', 'e60', 'ok'):
                s = x.groupby('t').apply(lambda g: g[f].rank().corr(g[y].rank()) if len(g) >= 5 else np.nan).dropna()
                ics[y] = (s.mean(), nw_t(s, 4 if y != 'e60' else 12))
            g = x.groupby('q')
            res[(f, era)] = dict(ic=ics['e60'][0], t=ics['e60'][1], ok1=g.ok.mean()[1], ok5=g.ok.mean()[5], m1=g.e60.median()[1], m5=g.e60.median()[5], okall=E.ok.mean(), mall=E.e60.median())
            L.append(f"| {f} | {ics['e20'][0]:+.3f}（{ics['e20'][1]:.1f}） | {ics['e60'][0]:+.3f}（{ics['e60'][1]:.1f}） | {ics['ok'][0]:+.3f}（{ics['ok'][1]:.1f}） | {g.ok.mean()[1]:.1%} | {g.ok.mean()[5]:.1%} | "
                     f"{g.e60.median()[1]:+.2%} | {g.e60.median()[5]:+.2%} | {g.e60.mean()[1]:+.2%} | {g.e60.mean()[5]:+.2%} | {g.mfe.mean()[1]:.2f}／{g.mae.mean()[1]:.2f} | {g.mfe.mean()[5]:.2f}／{g.mae.mean()[5]:.2f} | "
                     f"{g.left.mean()[1]:.0%}／{g.right.mean()[1]:.0%} | {g.left.mean()[5]:.0%}／{g.right.mean()[5]:.0%} |")
        cm = E[FE5 + ['Tld']].rank().corr()
        L += ['', '特徵間 Spearman 相關：' + '；'.join(f'{a}–{b} {cm.loc[a, b]:+.2f}' for i, a in enumerate(FE5 + ['Tld']) for b in (FE5 + ['Tld'])[i + 1:]) + '\n']
    vt = []
    for f in FE5:
        a = [res[(f, e)] for e, _, _ in ERAS]
        same = np.sign(a[0]['ic']) == np.sign(a[1]['ic']); ts = sorted(abs(x['t']) for x in a)
        c1 = same and ts[1] >= 2 and ts[0] >= 1
        hi_good = a[0]['ic'] + a[1]['ic'] > 0
        c2 = all((x['ok5'] if hi_good else x['ok1']) > x['okall'] for x in a)
        c3 = all((x['m5'] if hi_good else x['m1']) > x['mall'] for x in a)
        ics = '｜'.join(f"{x['ic']:+.3f} t {x['t']:.1f}" for x in a)
        vt.append(f"- {f}：①IC60 {'✓' if c1 else '✗'}（{ics}）②好組({'Q5' if hi_good else 'Q1'})快速成功 {'✓' if c2 else '✗'} ③好組 60 日中位 {'✓' if c3 else '✗'} → {'提名' if c1 and c2 and c3 else '不提名'}")
    head = ['## 判準（兩期；IC60 同號 t ≥ 2／≥ 1、好組快速成功率與 60 日中位數高於池內全體）\n'] + vt + ['']
    return dict(summary='\n'.join(L[:2] + head + L[2:]), headline='；'.join(v[2:].split('：')[0] + v.split('→')[-1] for v in vt), n_variants=0)
