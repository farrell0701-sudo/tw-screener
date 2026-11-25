# M-197：強勢池與營收池的相關性與相互 beta（奇數代碼；到 2024-11-13）。定義見 registry_M.md M-197。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, M165

END = pd.Timestamp('2024-11-13')
SEGS = (('2005–15', '2005-01-01', '2015-12-31'), ('2016–24/11', '2016-01-01', '2024-11-13'), ('2016–20', '2016-01-01', '2020-12-31'), ('2021–24/11', '2021-01-01', '2024-11-13'))


def beta(y, x):
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    return np.cov(y, x)[0, 1] / np.var(x, ddof=1)


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    meta = P['meta']; keys = meta['security_key'].astype(str).to_numpy()
    odd = (meta['stock_id'].astype(str).str[-1].astype(int) % 2 == 1).to_numpy()
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; cols = dbg['cols']
    def full(a, fill):
        a = np.asarray(a)
        if a.shape[1] == N: return a
        o = np.full((T, N), fill, a.dtype); o[:, cols] = a; return o
    U, pool = full(dbg['U'], False), full(dbg['pool'], False); del S
    A = M165.layers(U, pool, dates, keys)[0][1]
    VALID = np.asarray(F['VALID']); CA = np.where(VALID, np.asarray(F['CA'], np.float64), np.nan)
    C = pd.DataFrame(CA).ffill(limit=5).to_numpy()
    with np.errstate(all='ignore'): R = C / np.vstack([np.full((1, N), np.nan), C[:-1]]) - 1
    R = np.where(VALID & np.isfinite(R), np.clip(R, -0.5, 1.0), np.nan)
    samp = sorted(t for _, (t, _) in M165.samp_prev(dates).items())
    mS = np.zeros((T, N), bool); mA = np.zeros((T, N), bool); mU = np.zeros((T, N), bool)
    for a, b in zip(samp, samp[1:] + [T - 1]):
        mS[a + 1:b + 1] = pool[a] & U[a] & odd; mA[a + 1:b + 1] = A[a] & U[a] & odd; mU[a + 1:b + 1] = U[a] & odd
    with np.errstate(all='ignore'):
        rs = {k: np.nanmean(np.where(m, R, np.nan), axis=1) for k, m in (('S', mS), ('A', mA), ('U', mU))}
    df = pd.DataFrame(rs, index=dates).loc['2005-01-01':END].dropna()
    ix = P['index']['TAIEX_P'].reindex(dates).ffill(); bear = (ix < ix.rolling(200).mean()).reindex(df.index)
    df['eS'] = df.S - df.U; df['eA'] = df.A - df.U
    out = ['# M-197 結果：強勢池（S）與營收池（A）的相關性與相互 beta（奇數代碼；每月成員、日等權；到 2024-11-13）\n']
    out += ['## 報酬特性\n', '| 期間 | 池 | 年化報酬 | 年化波動 | 最大回撤 |', '|---|---|---|---|---|']
    for nm, lo, hi in SEGS:
        for k in ('S', 'A', 'U'):
            r = df.loc[lo:hi, k]; eq = (1 + r).cumprod()
            out.append(f"| {nm} | {k} | {eq.iloc[-1] ** (252 / len(r)) - 1:+.1%} | {r.std() * np.sqrt(252):.1%} | {(eq / eq.cummax() - 1).min():.1%} |")
    out += ['', '## 相關與 beta\n', '| 期間 | 相關 日｜週｜月 | beta(S|A) | beta(A|S) | beta(S|U) | beta(A|U) | 超額相關 日｜月 | 超額 beta(eS|eA)｜(eA|eS) | 大跌日相關（n） | 空頭期相關 |', '|---|---|---|---|---|---|---|---|---|---|']
    for nm, lo, hi in SEGS:
        d = df.loc[lo:hi]
        w = (1 + d[['S', 'A', 'U']]).resample('W').prod() - 1; m = (1 + d[['S', 'A', 'U']]).resample('ME').prod() - 1
        me = m.S - m.U, m.A - m.U
        big = d[d.U <= -0.015]; br = d[bear.loc[lo:hi].fillna(False).to_numpy()]
        out.append(f"| {nm} | {d.S.corr(d.A):.2f}｜{w.S.corr(w.A):.2f}｜{m.S.corr(m.A):.2f} | {beta(d.S.values, d.A.values):.2f} | {beta(d.A.values, d.S.values):.2f} | {beta(d.S.values, d.U.values):.2f} | {beta(d.A.values, d.U.values):.2f} | {d.eS.corr(d.eA):+.2f}｜{me[0].corr(me[1]):+.2f} | {beta(d.eS.values, d.eA.values):+.2f}｜{beta(d.eA.values, d.eS.values):+.2f} | {big.S.corr(big.A):.2f}（{len(big)}） | {br.S.corr(br.A):.2f} |")
    roll = pd.DataFrame({'日報酬相關': df.S.rolling(250).corr(df.A), '超額相關': df.eS.rolling(250).corr(df.eA)}).dropna()
    yr = roll.resample('YE').last()
    out += ['', '## 滾動 250 日相關（每年年底）\n', '| 年 | 日報酬相關 | 超額相關 |', '|---|---|---|'] + [f"| {i.year} | {r['日報酬相關']:.2f} | {r['超額相關']:+.2f} |" for i, r in yr.iterrows()]
    roll.to_csv('/tmp/M197_roll.csv'); df.to_csv('/tmp/M197_ret.csv')
    return dict(summary='\n'.join(out), headline='兩池相關與 beta', n_variants=0)
