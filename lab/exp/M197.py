# M-197：強勢股回檔解剖學（現象＋因子擬合；奇數代碼、排除最近兩年）。定義見 registry_M.md M-197。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals
import M191, M185
from M195 import pct_in, HOLDOUT, ERAS, COST
from M107 import nw_t

FE = ['DEPTH', 'DUR', 'RETR', 'VOLPB', 'ATRC', 'DTLD', 'EMA20D', 'EMA60D', 'CLV', 'MKT']
CACHE = LAB / 'cache' / 'M197'
TAG = 'M-197'


def ffill_at(mask, val):
    return pd.DataFrame(np.where(mask, val, np.nan)).ffill().to_numpy()


def build(parity=1, a0=False):
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    signals.build('fix', mode='fix', P=P, F=F)
    S_ = signals.compute('fix', P=P, F=F); dbg = S_['_dbg']; cols = dbg['cols']
    def full(a, fill):
        a = np.asarray(a)
        if a.shape[1] == N: return a
        o = np.full((T, N), fill, a.dtype); o[:, cols] = a; return o
    U = full(dbg['U'], False); Tr = full(np.asarray(dbg['Tr'], np.float64), np.nan); del S_
    VALID = np.asarray(F['VALID']); HA = np.asarray(F['HA'], np.float64); LA = np.asarray(F['LA'], np.float64); OA = np.asarray(F['OA'], np.float64)
    Cdf = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill(); Cf = Cdf.to_numpy()
    A = pd.DataFrame(np.asarray(P['AMT'], np.float64)).fillna(0)
    ATR = M191.extra(P, F)['ATR20']
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CX = np.cumsum(R - ew[:, None], axis=0)
    odd = (P['meta']['stock_id'].astype(str).str[-1].astype(int) % 2 == parity).to_numpy()
    dT60 = Tr - np.vstack([np.full((60, N), np.nan), Tr[:-60]]); strong = pct_in(dT60, U) >= 0.9
    strong20 = pd.DataFrame(strong.astype(float)).rolling(20, min_periods=1).max().to_numpy() > 0
    H20 = Cdf.rolling(20, min_periods=15).max().to_numpy()
    # M-197B 修正：高點日＝近 20 日（含當日）最高收盤「最近一次出現」的那一天（原版以「當日為自身 20 日高」錨定，會錨到更早的高點）
    from numpy.lib.stride_tricks import sliding_window_view
    Cz = np.where(np.isfinite(Cf), Cf, -np.inf); hidx = np.full((T, N), -1, np.int64)
    for s0 in range(19, T, 400):
        e0 = min(T, s0 + 400); w = sliding_window_view(Cz[s0 - 19:e0], 20, axis=0)   # (e0-s0, N, 20)
        last = 19 - np.argmax(w[:, :, ::-1], axis=2); hidx[s0:e0] = np.arange(s0, e0)[:, None] - 19 + last
    hc = np.clip(hidx, 0, None); cj = np.broadcast_to(np.arange(N)[None, :], (T, N))
    DUR = (np.arange(T)[:, None] - hidx).astype(float); DUR[hidx < 0] = np.nan
    L20b = Cdf.rolling(20, min_periods=15).min().to_numpy(); lowb = L20b[hc, cj]
    A20 = A.rolling(20, min_periods=15).mean().to_numpy(); a20h = A20[hc, cj]
    CS = A.cumsum().to_numpy(); csh = CS[hc, cj]
    atrh = ATR[hc, cj]; tldh = Tr[hc, cj]
    # 回檔低點（自高點日之後到 t 的最低收盤；以 20 日滾動最小值近似於高點後區間）
    PL = np.full((T, N), np.nan)
    for t in range(20, T):
        d = DUR[t]; ok = np.isfinite(d) & (d >= 1)
        win = Cf[t - 19:t + 1]                                      # 20×N
        mask = (np.arange(20)[:, None] >= (19 - d[None, :] + 1)) & ok[None, :]
        PL[t] = np.where(ok, np.nanmin(np.where(mask, win, np.inf), axis=0), Cf[t])
    ema20 = Cdf.ewm(span=20, adjust=False, min_periods=20).mean().to_numpy(); ema60 = Cdf.ewm(span=60, adjust=False, min_periods=60).mean().to_numpy()
    mkt = ~M185.taiex_below(dates)
    t0 = dates.searchsorted(pd.Timestamp('2005-01-01')); t1 = dates.searchsorted(HOLDOUT)
    rows = []; A0r = []
    for t in range(t0, min(t1, T - 61), 3):
        if a0 and (t - t0) % 15 == 0:
            j0 = np.nonzero(U[t] & VALID[t] & odd & strong[t])[0]
            if len(j0): A0r.append(pd.DataFrame(dict(t=t, e20=CX[t + 20, j0] - CX[t, j0])))
        st = U[t] & VALID[t] & odd & strong20[t] & ((H20[t] - Cf[t]) >= ATR[t]) & (DUR[t] >= 2)
        jj = np.nonzero(st)[0]
        if len(jj) == 0: continue
        c = Cf[t, jj]; a = ATR[t, jj]; h = H20[t, jj]
        with np.errstate(all='ignore'):
            f = dict(DEPTH=(h - c) / a, DUR=DUR[t, jj], RETR=(h - c) / (h - lowb[t, jj]),
                     VOLPB=((CS[t, jj] - csh[t, jj]) / np.maximum(DUR[t, jj], 1)) / a20h[t, jj], ATRC=a / atrh[t, jj], DTLD=Tr[t, jj] - tldh[t, jj],
                     EMA20D=(c - ema20[t, jj]) / a, EMA60D=(c - ema60[t, jj]) / a,
                     CLV=(c - LA[t, jj]) / np.where(HA[t, jj] > LA[t, jj], HA[t, jj] - LA[t, jj], np.nan), MKT=np.full(len(jj), float(mkt[t])))
        fut = Cf[t + 1:t + 21, jj]
        up = fut > h[None, :]; dn = fut < (PL[t, jj] - a)[None, :]
        iu = np.where(up.any(0), up.argmax(0), 99); idn = np.where(dn.any(0), dn.argmax(0), 99)
        ent = np.where(np.isfinite(OA[t + 1, jj]), OA[t + 1, jj], Cf[t + 1, jj])
        e60 = CX[t + 60, jj] - CX[t, jj]; q10 = np.nanpercentile(CX[t + 60, U[t]] - CX[t, U[t]], 10)
        rows.append(pd.DataFrame(dict(t=t, j=jj, e20=CX[t + 20, jj] - CX[t, jj], e60=e60, n20=Cf[t + 20, jj] / ent - 1 - COST,
                                      res=(iu < idn).astype(float), fail=(idn < iu).astype(float), left=(e60 <= q10).astype(float), h=hidx[t, jj],
                                      R5=Cf[t, jj] / Cf[t - 5, jj] - 1, R20=Cf[t, jj] / Cf[t - 20, jj] - 1, TLD=Tr[t, jj], **f)))
    D = pd.concat(rows, ignore_index=True); D['date'] = dates[D.t.to_numpy()]
    D = D[np.isfinite(D.e60) & np.isfinite(D.n20)]
    if a0:
        A0 = pd.concat(A0r, ignore_index=True); A0['date'] = dates[A0.t.to_numpy()]; return D, A0
    return D


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    D = build(); D.to_pickle(CACHE / 'D.pkl') if TAG == 'M-197' else None
    L = ['# M-197 結果：強勢股回檔解剖學（奇數代碼、排除最近兩年；每 3 日取樣）\n',
         '恢復先到＝20 日內先收盤創 20 日新高，早於收盤跌破回檔低點 − 1ATR；20 日淨＝t+1 開盤進場扣 0.77%。\n']
    for era, lo, hi in ERAS:
        E = D[(D.date >= lo) & (D.date <= hi)]
        L += [f"## {era}（{len(E):,} 筆、{E.t.nunique()} 個取樣日）：20 日超額 {E.e20.mean():+.2%}、20 日淨勝率 {(E.n20 > 0).mean():.0%}、恢復先到 {E.res.mean():.0%}、失敗先到 {E.fail.mean():.0%}、左尾 {E.left.mean():.1%}\n",
              '| 特徵 | IC20（t） | IC 恢復（t） | Q1 20日超額 | Q5 20日超額 | Q1 勝率 | Q5 勝率 | Q1 恢復／失敗 | Q5 恢復／失敗 | Q1 左尾 | Q5 左尾 |', '|' + '---|' * 11]
        for f in FE:
            x = E.dropna(subset=[f])
            if f == 'MKT':
                g = x.groupby('MKT'); a, b = g.get_group(0.0) if 0.0 in g.groups else x.iloc[:0], g.get_group(1.0) if 1.0 in g.groups else x.iloc[:0]
                L.append(f"| MKT（0＝多空點下｜1＝上） | | | {a.e20.mean():+.2%} | {b.e20.mean():+.2%} | {(a.n20 > 0).mean():.0%} | {(b.n20 > 0).mean():.0%} | {a.res.mean():.0%}／{a.fail.mean():.0%} | {b.res.mean():.0%}／{b.fail.mean():.0%} | {a.left.mean():.1%} | {b.left.mean():.1%} |"); continue
            ic = x.groupby('t').apply(lambda g: g[f].rank().corr(g.e20.rank()) if len(g) >= 8 else np.nan).dropna()
            icr = x.groupby('t').apply(lambda g: g[f].rank().corr(g.res.rank()) if len(g) >= 8 else np.nan).dropna()
            x = x.assign(q=pd.qcut(x[f].rank(method='first'), 5, labels=False) + 1); g = x.groupby('q')
            L.append(f"| {f} | {ic.mean():+.3f}（{nw_t(ic, 7):.1f}） | {icr.mean():+.3f}（{nw_t(icr, 7):.1f}） | {g.e20.mean()[1]:+.2%} | {g.e20.mean()[5]:+.2%} | {(g.n20.apply(lambda s: (s > 0).mean()))[1]:.0%} | {(g.n20.apply(lambda s: (s > 0).mean()))[5]:.0%} | "
                     f"{g.res.mean()[1]:.0%}／{g.fail.mean()[1]:.0%} | {g.res.mean()[5]:.0%}／{g.fail.mean()[5]:.0%} | {g.left.mean()[1]:.1%} | {g.left.mean()[5]:.1%} |")
        L.append('')
    # 因子擬合：2005–15 訓練，2016–24 樣本外
    X = D.copy(); feats = [f for f in FE if f != 'MKT']
    for f in feats: X[f + '_p'] = X.groupby('t')[f].rank(pct=True)
    X = X.dropna(subset=[f + '_p' for f in feats])
    tr = X[X.date <= '2015-12-31']; te = X[X.date >= '2016-01-01']
    Xm = np.column_stack([np.ones(len(tr))] + [tr[f + '_p'] - 0.5 for f in feats] + [tr.MKT])
    beta = np.linalg.lstsq(Xm, tr.e20.to_numpy(), rcond=None)[0]
    def pred(d): return np.column_stack([np.ones(len(d))] + [d[f + '_p'] - 0.5 for f in feats] + [d.MKT]) @ beta
    L += ['## 因子擬合（2005–15 線性回歸 → 2016–24 樣本外）\n', '係數（20 日超額，每 1.0 百分位差）：' + '；'.join(f'{n} {b:+.4f}' for n, b in zip(['常數'] + feats + ['MKT'], beta)) + '\n']
    for nm, d in (('樣本內 2005–15', tr), ('樣本外 2016–24', te)):
        d = d.assign(p=pred(d)); ic = d.groupby('t').apply(lambda g: g.p.rank().corr(g.e20.rank()) if len(g) >= 8 else np.nan).dropna()
        d['q'] = d.groupby('t').p.transform(lambda s: pd.qcut(s.rank(method='first'), 5, labels=False) + 1 if len(s) >= 5 else np.nan); g = d.groupby('q')
        L += [f"**{nm}**：IC {ic.mean():+.3f}（t {nw_t(ic, 7):.1f}）\n", '| 預測組 | 筆數 | 20日超額 | 20日淨 平均／中位 | 勝率 | 恢復／失敗 | 左尾 |', '|---|---|---|---|---|---|---|']
        for qq, y in g:
            L.append(f"| Q{int(qq)} | {len(y):,} | {y.e20.mean():+.2%} | {y.n20.mean():+.2%}／{y.n20.median():+.2%} | {(y.n20 > 0).mean():.0%} | {y.res.mean():.0%}／{y.fail.mean():.0%} | {y.left.mean():.1%} |")
        L.append('')
    return dict(summary='\n'.join(L), headline='回檔解剖學', n_variants=0)
