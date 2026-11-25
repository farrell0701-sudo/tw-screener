# M-195：強勢股回檔到 EMA10／20／60 的事件（第一層）與第二層預檢（奇數代碼、排除最近兩年）。定義見 registry_M.md M-195。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim
import M190, M191
from M107 import nw_t

HOLDOUT = pd.Timestamp('2024-11-25'); COST = 0.0077
ERAS = (('2005–15', '2005-01-01', '2015-12-31'), ('2016–24', '2016-01-01', '2024-11-24'))
KS = (10, 20, 60)


def pct_in(A, M):
    return pd.DataFrame(np.where(M, A, np.nan)).rank(axis=1, pct=True).to_numpy()


def main():
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
    ATR = M191.extra(P, F)['ATR20']
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CX = np.cumsum(R - ew[:, None], axis=0)
    odd = (P['meta']['stock_id'].astype(str).str[-1].astype(int) % 2 == 1).to_numpy()
    dT60 = Tr - np.vstack([np.full((60, N), np.nan), Tr[:-60]]); dT20 = Tr - np.vstack([np.full((20, N), np.nan), Tr[:-20]])
    strong = pct_in(dT60, U) >= 0.9
    e200 = Cdf.ewm(span=200, adjust=False, min_periods=150).mean(); slope = (e200 / e200.shift(60) - 1).to_numpy()
    steep = pct_in(slope, U) >= 0.8; decay = pct_in(dT20, U) <= 1 / 3
    t0 = dates.searchsorted(pd.Timestamp('2005-01-01')); t1 = dates.searchsorted(HOLDOUT)
    base_mask = U & VALID & strong & odd[None, :]
    base_mask[:t0] = False; base_mask[t1:] = False; base_mask[T - 62:] = False
    # X1 持股（新基準 fix、路徑 0）
    ctx = sim.make_ctx('fix', sig='fix'); _, trX, _ = sim.runX(ctx)
    keys = P['meta']['security_key'].astype(str).to_numpy(); kidx = {k: i for i, k in enumerate(keys)}
    held = np.zeros((T, N), bool)
    for (code, d0, d1, *_ ) in trX:
        j = kidx.get(str(code))
        if j is not None: held[dates.get_loc(pd.Timestamp(d0)):dates.get_loc(pd.Timestamp(d1)) + 1, j] = True

    def outcomes(tt, jj):
        e5, e20, e60 = (CX[tt + h, jj] - CX[tt, jj] for h in (5, 20, 60))
        ent = np.where(np.isfinite(OA[tt + 1, jj]), OA[tt + 1, jj], Cf[tt + 1, jj])
        n20 = Cf[tt + 20, jj] / ent - 1 - COST; n60 = Cf[tt + 60, jj] / ent - 1 - COST
        ok = np.zeros(len(tt)); mfe = np.full(len(tt), np.nan); mae = np.full(len(tt), np.nan)
        for i in range(len(tt)):
            H = HA[tt[i] + 1:tt[i] + 21, jj[i]:jj[i] + 1]; L = LA[tt[i] + 1:tt[i] + 21, jj[i]:jj[i] + 1]
            o, _, _, _ = M191.quick(ent[i:i + 1], ATR[tt[i]:tt[i] + 1, jj[i]], H, L); ok[i] = o[0]
            with np.errstate(all='ignore'): mfe[i] = np.nanmax(H) / ent[i] - 1; mae[i] = np.nanmin(L) / ent[i] - 1
        e60u = {}
        q = np.array([np.nanpercentile(CX[t + 60, U[t]] - CX[t, U[t]], [10, 90]) if t not in e60u else e60u[t] for t in tt])
        D = pd.DataFrame(dict(t=tt, j=jj, e5=e5, e20=e20, e60=e60, n20=n20, n60=n60, ok=ok, mfe=mfe, mae=mae,
                              left=(e60 <= q[:, 0]).astype(float), right=(e60 >= q[:, 1]).astype(float),
                              tld=Tr[tt, jj], steep=steep[tt, jj], decay=decay[tt, jj], x1=held[tt + 1, jj]))
        D['date'] = dates[tt]
        return D[np.isfinite(D.e60) & np.isfinite(D.n60)]

    # 對照 A0：每 5 日取樣
    samp = np.zeros(T, bool); samp[t0::5] = True
    tt, jj = np.nonzero(base_mask & samp[:, None]); A0 = outcomes(tt, jj)
    EV = {}
    for k in KS:
        ema = Cdf.ewm(span=k, adjust=False, min_periods=k).mean().to_numpy()
        prior = pd.DataFrame(Cf).shift(1).rolling(10, min_periods=8).max().to_numpy()
        cond = base_mask & (LA <= ema + 0.5 * ATR) & (Cf >= ema - 0.5 * ATR) & (prior >= ema + ATR)
        recent = pd.DataFrame(cond.astype(float)).shift(1).rolling(10, min_periods=1).sum().to_numpy() > 0
        ev = cond & ~recent
        tt, jj = np.nonzero(ev); D = outcomes(tt, jj)
        D['dist'] = (Cf[D.t, D.j] - ema[D.t, D.j]) / ATR[D.t, D.j]
        D['pull'] = (prior[D.t, D.j] - Cf[D.t, D.j]) / ATR[D.t, D.j]
        EV[k] = D
    L = ['# M-195 結果：強勢股（ΔT60 前 10%）回檔到 EMA10／20／60（奇數代碼、排除最近兩年）\n',
         '超額自訊號日收盤、扣可交易池等權；淨報酬自次日開盤、扣 0.77%；快速成功＝20 日內先 +2ATR 再 −1ATR。\n']
    def row(nm, x):
        return (f"| {nm} | {len(x):,} | {x.e5.mean():+.2%} | {x.e20.mean():+.2%} | {x.e60.mean():+.2%} | {x.e60.median():+.2%} | "
                f"{x.n20.mean():+.2%}／{x.n20.median():+.2%}／{(x.n20 > 0).mean():.0%} | {x.n60.mean():+.2%}／{x.n60.median():+.2%}／{(x.n60 > 0).mean():.0%} | "
                f"{x.ok.mean():.0%} | {x.mfe.mean():+.1%}／{x.mae.mean():+.1%} | {x.left.mean():.1%}／{x.right.mean():.1%} | {x.x1.mean():.0%} |")
    hdr = ['| 組 | 筆數 | 5日超額 | 20日超額 | 60日超額 | 60日超額中位 | 20日淨：平均／中位／勝率 | 60日淨：平均／中位／勝率 | 快速成功 | MFE20／MAE20 | 左尾／右尾 | 被 X1 持有 |', '|' + '---|' * 12]
    res = {}
    for era, lo, hi in ERAS:
        sel = lambda d: d[(d.date >= lo) & (d.date <= hi)]
        a0 = sel(A0)
        L += [f'## {era}\n'] + hdr + [row('A0 強勢母體全部（每 5 日）', a0)]
        for k in KS:
            x = sel(EV[k]); L.append(row(f'E{k} 回檔到 EMA{k}', x))
            m = x.groupby(x.date.dt.to_period('M')).e60.mean()
            res[(k, era)] = dict(m=m.mean(), t=nw_t(m, 3), win=(x.n20 > 0).mean(), left=x.left.mean(), a_win=(a0.n20 > 0).mean(), a_left=a0.left.mean())
        L.append('')
        # 第二層預檢
        for k in KS:
            x = sel(EV[k]).copy()
            if len(x) < 50: continue
            x['q'] = x.groupby(x.date.dt.year).tld.transform(lambda s: pd.qcut(s.rank(method='first'), 5, labels=False) + 1 if len(s) >= 5 else np.nan)
            g = x.groupby('q')
            L += [f'**E{k}：事件內依 T_ld 五分組（年內分組）**\n', '| T_ld 組 | 筆數 | 60日超額 平均／中位 | 20日淨勝率 | 60日淨勝率 | 左尾／右尾 | 收盤距 EMA（ATR） | 自 10 日高回落（ATR） |', '|---|---|---|---|---|---|---|---|']
            for qq, y in g:
                L.append(f"| Q{int(qq)} | {len(y):,} | {y.e60.mean():+.2%}／{y.e60.median():+.2%} | {(y.n20 > 0).mean():.0%} | {(y.n60 > 0).mean():.0%} | {y.left.mean():.0%}／{y.right.mean():.0%} | {y.dist.mean():+.2f} | {y.pull.mean():.2f} |")
            ex = x[x.steep | x.decay]; kp = x[~(x.steep | x.decay)]
            L += ['', f"剔除過陡或動能衰退：剔除 {len(ex) / len(x):.0%}（過陡 {x.steep.mean():.0%}、衰退 {x.decay.mean():.0%}）；"
                  f"保留者 60日超額 {kp.e60.mean():+.2%}、20日淨勝率 {(kp.n20 > 0).mean():.0%}、左尾 {kp.left.mean():.1%}、右尾 {kp.right.mean():.1%}；"
                  f"被剔除者 60日超額 {ex.e60.mean():+.2%}、20日淨勝率 {(ex.n20 > 0).mean():.0%}、左尾 {ex.left.mean():.1%}、右尾 {ex.right.mean():.1%}\n"]
    vt = []
    for k in KS:
        a = [res[(k, e)] for e, _, _ in ERAS]; ts = sorted(x['t'] for x in a)
        c1 = all(x['m'] > 0 for x in a) and ts[1] >= 2 and ts[0] >= 1
        c2 = all(x['win'] >= x['a_win'] for x in a); c3 = all(x['left'] <= x['a_left'] for x in a)
        ms = '｜'.join(f"{x['m']:+.2%} t {x['t']:.1f}" for x in a)
        vt.append(f"- E{k}：①60日超額 {'✓' if c1 else '✗'}（{ms}）②20日淨勝率 ≥ A0 {'✓' if c2 else '✗'}（" + '｜'.join(f"{x['win']:.0%} vs {x['a_win']:.0%}" for x in a) +
                  f"）③左尾 ≤ A0 {'✓' if c3 else '✗'}（" + '｜'.join(f"{x['left']:.1%} vs {x['a_left']:.1%}" for x in a) + f"）→ {'成立，可進第二層' if c1 and c2 and c3 else '不成立'}")
    head = ['## 第一層判準（兩期）\n'] + vt + ['']
    return dict(summary='\n'.join(L[:2] + head + L[2:]), headline='；'.join(v[2:].split('：')[0] + v.split('→')[-1] for v in vt), n_variants=0)
