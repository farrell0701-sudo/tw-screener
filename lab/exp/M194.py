# M-194：MACD(200,201,202) 走緩之後的剩餘超額（強勢池；奇數代碼；到 2024-11-13）。定義見 registry_M.md M-194。
import sys, os
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals

END = pd.Timestamp('2024-11-13')
PARITY = int(os.environ.get('M194_PARITY', '1'))
ERAS = ('2005–15', '2016–24/11')
EVS = ('E1 轉降', 'E2 走緩震盪', 'E3 跌破訊號線')


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    meta = P['meta']
    par = (meta['stock_id'].astype(str).str[-1].astype(int) % 2 == PARITY).to_numpy()
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; cols = dbg['cols']
    def full(a, fill):
        a = np.asarray(a)
        if a.shape[1] == N: return a
        o = np.full((T, N), fill, a.dtype); o[:, cols] = a; return o
    U, pool = full(dbg['U'], False), full(dbg['pool'], False); del S
    VALID = np.asarray(F['VALID']); CA = np.where(VALID, np.asarray(F['CA'], np.float64), np.nan)
    Cdf = pd.DataFrame(CA).ffill(); C = Cdf.to_numpy()
    with np.errstate(all='ignore'): R = C / np.vstack([np.full((1, N), np.nan), C[:-1]]) - 1
    Rz = np.where(VALID & ~np.isnan(R), R, 0.0)
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, Rz, np.nan), axis=1))
    CX = np.cumsum(Rz - ew[:, None], axis=0)
    Md = Cdf.ewm(span=200, adjust=False).mean() - Cdf.ewm(span=201, adjust=False).mean()
    SIG = Md.ewm(span=202, adjust=False).mean()
    M = Md.to_numpy(); SG = SIG.to_numpy()
    d5 = (Md - Md.shift(5)).to_numpy()
    hi120 = (Md >= Md.rolling(120, min_periods=120).max() - 1e-15).to_numpy()
    hi60 = (Md >= Md.rolling(60, min_periods=60).max() - 1e-15).to_numpy().astype(float)
    hi120_10 = pd.DataFrame(hi120.astype(float)).rolling(10, min_periods=1).max().to_numpy() > 0
    no20 = pd.DataFrame(hi60).rolling(20, min_periods=20).sum().to_numpy() == 0
    cnt = pd.DataFrame(hi60).shift(20).rolling(40, min_periods=40).sum().to_numpy()
    e1 = (d5 < 0) & (np.vstack([np.full((1, N), np.nan), d5[:-1]]) >= 0) & np.vstack([np.zeros((1, N), bool), hi120_10[:-1]])
    c2 = no20 & (cnt >= 20); e2 = c2 & ~np.vstack([np.zeros((1, N), bool), c2[:-1]])
    e3 = (M < SG) & (np.vstack([np.full((1, N), np.nan), M[:-1] - SG[:-1]]) >= 0) & (M > 0)
    EV = {EVS[0]: e1, EVS[1]: e2, EVS[2]: e3}
    anyev = e1 | e2 | e3; recent = pd.DataFrame(anyev.astype(float)).rolling(20, min_periods=1).max().to_numpy() > 0
    MX250 = Cdf.rolling(250, min_periods=200).max().to_numpy()
    idx = np.arange(T)[:, None]
    athi = np.where(np.isfinite(C) & (C >= MX250 - 1e-9), idx, -1); lastHi = np.maximum.accumulate(athi, axis=0)
    tend = int(dates.searchsorted(END, side='right')) - 1; s0 = int(dates.searchsorted(pd.Timestamp('2005-01-01')))
    rows = []
    for nm, E in EV.items():
        last = np.full(N, -10 ** 9)
        for t in range(s0, tend - 60):
            js = np.nonzero(E[t] & pool[t] & U[t] & par)[0]
            if not len(js): continue
            ctrl = pool[t] & U[t] & ~recent[t]
            c20 = np.nanmean(CX[t + 20, ctrl] - CX[t, ctrl]) if ctrl.sum() >= 5 else np.nan
            c60 = np.nanmean(CX[t + 60, ctrl] - CX[t, ctrl]) if ctrl.sum() >= 5 else np.nan
            e60all = CX[t + 60, U[t]] - CX[t, U[t]]; q10, q90 = np.nanpercentile(e60all, [10, 90])
            for j in js:
                if t - last[j] < 60: continue
                last[j] = t
                x20, x60 = CX[t + 20, j] - CX[t, j], CX[t + 60, j] - CX[t, j]
                rows.append((nm, ERAS[0] if dates[t] <= pd.Timestamp('2015-12-31') else ERAS[1], x20, x60, x20 - c20, x60 - c60, x60 <= q10, x60 >= q90,
                             t - lastHi[t, j] if lastHi[t, j] >= 0 else np.nan, C[t, j] / MX250[t, j] - 1, CX[t, j] - CX[t - 60, j]))
    D = pd.DataFrame(rows, columns=['ev', 'era', 'x20', 'x60', 'd20', 'd60', 'L', 'Rt', 'lag', 'dd', 'prev60'])
    out = [f"# M-194 結果：MACD(200,201,202) 走緩之後的剩餘超額（強勢池；{'奇' if PARITY else '偶'}數代碼；到 2024-11-13）\n",
           '同日差 = 事件股 − 同日強勢池內近 20 日無事件股票的平均。左尾／右尾 = 當日 U 60 日超額的後／前 10%。\n',
           '| 事件 | 期間 | n | 20 日超額（同日差） | 60 日超額（中位）（同日差） | 左尾｜右尾 | 距 250 日高天數（中位） | 距高點跌幅（中位） | 事件前 60 日已賺 | 判準 |', '|---|---|---|---|---|---|---|---|---|---|']
    noms = []
    for nm in EVS:
        oks = []
        for e in ERAS:
            x = D[(D.ev == nm) & (D.era == e)]
            if len(x) < 30: out.append(f'| {nm} | {e} | {len(x)} | — |'); oks.append(False); continue
            ok = (x.d60.mean() <= -0.02) and (x.x60.mean() <= 0.01); oks.append(ok)
            out.append(f"| {nm} | {e} | {len(x)} | {x.x20.mean():+.1%}（{x.d20.mean():+.1%}） | {x.x60.mean():+.1%}（{x.x60.median():+.1%}）（{x.d60.mean():+.1%}） | {x.L.mean():.0%}｜{x.Rt.mean():.0%} | {x.lag.median():.0f} | {x.dd.median():+.1%} | {x.prev60.mean():+.1%} | {'✓' if ok else '✗'} |")
        if all(oks): noms.append(nm)
    out += ['', '## 提名\n'] + ([f'- {n}：出場訊號候選' for n in noms] or ['- 無'])
    return dict(summary='\n'.join(out), headline=('提名：' + '、'.join(noms)) if noms else '無提名', n_variants=0 if PARITY else len(EVS))
