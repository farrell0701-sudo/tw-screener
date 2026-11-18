# 評估：分段指標、全期回撤、時間區塊 bootstrap P、DSR、高原表（設定見 L-113 補充「評估設定」）
import numpy as np, pandas as pd
from math import erf, sqrt, exp, log

SEGS = {'2005–15': ('2005-01-01', '2015-12-31'), '2016–20': ('2016-01-01', '2020-12-31'),
        '2021–26': ('2021-01-01', '2026-12-31'), '2016–26': ('2016-01-01', '2026-12-31'), '全期': ('2005-01-01', '2026-12-31')}
BLOCK, NBOOT, SEED = 63, 5000, 0
EULER = 0.5772156649015329


def met(r):
    """日報酬 → (年化, 最大回撤, Calmar, Sharpe)。同參考實作：252 日、Sharpe 不扣無風險利率；少於 60 日回傳 NaN。"""
    r = pd.Series(r).dropna()
    if len(r) < 60: return (np.nan,) * 4
    e = (1 + r).cumprod(); y = len(r) / 252; cg = e.iloc[-1] ** (1 / y) - 1; md = (e / e.cummax() - 1).min()
    sd = r.std()
    return cg, md, (cg / -md if md < 0 else np.nan), (r.mean() / sd * np.sqrt(252) if sd > 0 else np.nan)


def seg_table(paths, segs=SEGS):
    """paths：各路徑日報酬 Series 的 list。回傳每段的中位數指標、Calmar P10、最差路徑回撤。"""
    rows = {}
    for nm, (lo, hi) in segs.items():
        M = np.array([met(p.loc[lo:hi]) for p in paths])
        rows[nm] = dict(年化=np.nanmedian(M[:, 0]), 回撤=np.nanmedian(M[:, 1]), Calmar=np.nanmedian(M[:, 2]),
                        Sharpe=np.nanmedian(M[:, 3]), Calmar_P10=np.nanpercentile(M[:, 2], 10), 最差回撤=np.nanmin(M[:, 1]))
    return pd.DataFrame(rows).T


def path_mean(paths):
    return pd.concat(paths, axis=1).fillna(0).mean(axis=1)


def _calmar_arr(R):
    """R：(重抽次數, 天數) 日報酬矩陣 → 每列 Calmar。"""
    e = np.cumprod(1 + R, axis=1); y = R.shape[1] / 252
    cg = e[:, -1] ** (1 / y) - 1; md = (e / np.maximum.accumulate(e, axis=1) - 1).min(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(md < 0, cg / -md, np.nan)


def boot_p(var, base, lo='2016-01-01', hi='2026-12-31', block=BLOCK, n=NBOOT, seed=SEED, chunk=500):
    """配對循環區塊 bootstrap：P＝重抽後 Calmar(var) > Calmar(base) 的比例。var、base 為日報酬 Series（通常是路徑平均）。"""
    a, b = pd.Series(var).loc[lo:hi].align(pd.Series(base).loc[lo:hi], fill_value=0)
    x, z = a.values, b.values; L = len(x); nb = int(np.ceil(L / block)); rng = np.random.default_rng(seed)
    win = 0; tot = 0
    for k in range(0, n, chunk):
        m = min(chunk, n - k)
        st = rng.integers(0, L, (m, nb))
        idx = ((st[:, :, None] + np.arange(block)[None, None, :]) % L).reshape(m, -1)[:, :L]
        ca, cb = _calmar_arr(x[idx]), _calmar_arr(z[idx])
        ok = np.isfinite(ca) & np.isfinite(cb); win += int((ca[ok] > cb[ok]).sum()); tot += int(ok.sum())
    return win / tot if tot else np.nan


def _ncdf(x): return 0.5 * (1 + erf(x / sqrt(2)))


def _nppf(p):
    """標準常態分位數（Acklam 近似，誤差 < 1e-9）。"""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    pl = 0.02425
    if p < pl:
        q = sqrt(-2 * log(p)); return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > 1 - pl:
        q = sqrt(-2 * log(1 - p)); return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5; r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def psr(r, sr0=0.0):
    """機率夏普（Probabilistic Sharpe Ratio）：真實日 Sharpe > sr0 的機率，含偏態、峰態。"""
    r = pd.Series(r).dropna().values; T = len(r); sr = r.mean() / r.std(ddof=1)
    g3 = ((r - r.mean()) ** 3).mean() / r.std(ddof=0) ** 3; g4 = ((r - r.mean()) ** 4).mean() / r.std(ddof=0) ** 4
    return _ncdf((sr - sr0) * sqrt(T - 1) / sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))


def dsr(r, n_trials, var_sr_daily):
    """Deflated Sharpe Ratio（Bailey & López de Prado 2014）。var_sr_daily：試驗間「日」Sharpe 的變異數。"""
    if n_trials < 2 or not np.isfinite(var_sr_daily) or var_sr_daily <= 0: return np.nan
    sr0 = sqrt(var_sr_daily) * ((1 - EULER) * _nppf(1 - 1 / n_trials) + EULER * _nppf(1 - 1 / (n_trials * exp(1))))
    return psr(r, sr0)


def fmt_table(df):
    f1 = lambda x: f'{x:.1%}' if np.isfinite(x) else '—'; f2 = lambda x: f'{x:.2f}' if np.isfinite(x) else '—'
    out = df.copy().astype(object)
    for c in out.columns:
        out[c] = [f1(v) if c in ('年化', '回撤', '最差回撤', '全期回撤') else f2(v) for v in df[c].values]
    return out
