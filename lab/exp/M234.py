# M-234：套牢賣壓／舊強勢的其他辨識方式。定義見 registry_M.md M-234。
import sys
from pathlib import Path
import numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
sys.path.insert(0, str(Path(__file__).resolve().parent))
import M200

FAC = ['T_ld', 'DH250', 'SUP', 'VWG', 'RG', 'SUP|DH', 'VWG|DH', 'RG|DH']


def extra(P, F, Tr, U):
    C = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill(); H = pd.DataFrame(np.asarray(F['HA'], np.float64)).ffill()
    A = pd.DataFrame(np.asarray(P['AMT'], np.float64)).fillna(0.0)
    T, N = C.shape; Cn = C.to_numpy()
    dh = (C / H.rolling(250, min_periods=200).max() - 1).to_numpy()
    # 套牢量比：過去 250 日（含今日）成交值中，收盤高於今日收盤的比例
    Cv = np.where(np.isfinite(Cn), Cn, np.nan).astype(np.float32); Av = A.to_numpy(np.float32)
    padC = np.vstack([np.full((249, N), np.nan, np.float32), Cv]); padA = np.vstack([np.zeros((249, N), np.float32), Av])
    ovh = np.full((T, N), np.nan)
    for a in range(0, T, 100):
        b = min(T, a + 100)
        WC = sliding_window_view(padC[a:b + 249], 250, axis=0); WA = sliding_window_view(padA[a:b + 249], 250, axis=0)
        cur = Cv[a:b][:, :, None]
        num = np.where(WC > cur, WA, 0).sum(2); den = np.where(np.isfinite(WC), WA, 0).sum(2)
        with np.errstate(all='ignore'): ovh[a:b] = num / den
    cnt = C.notna().cumsum().to_numpy(); ovh[cnt < 200] = np.nan
    vol = A / C
    vwap = A.rolling(250, min_periods=200).sum() / vol.rolling(250, min_periods=200).sum()
    vwg = (C / vwap - 1).to_numpy()
    pr = lambda x: pd.DataFrame(np.where(U, x, np.nan)).rank(axis=1, pct=True).to_numpy()
    rg = pr((C / C.shift(60) - 1).to_numpy()) - pr((C / C.shift(250) - 1).to_numpy())
    out = {'DH250': dh, 'SUP': -ovh, 'VWG': vwg, 'RG': rg}
    pdh = pr(dh)
    for k in ('SUP', 'VWG', 'RG'):
        y = pr(out[k]); r = np.full((T, N), np.nan)
        for t in range(T):
            m = np.isfinite(y[t]) & np.isfinite(pdh[t])
            if m.sum() < 30: continue
            X = np.column_stack([np.ones(m.sum()), pdh[t, m]]); bb = np.linalg.lstsq(X, y[t, m], rcond=None)[0]; r[t, m] = y[t, m] - X @ bb
        out[k + '|DH'] = r
    return out


def main():
    L, vt = M200.analyze(M200.panel(1, extra=extra, fac=FAC), 'M-234 結果：套牢賣壓／舊強勢的其他辨識方式（奇數代碼、排除最近兩年）', fac=FAC)
    return dict(summary='\n'.join(L), headline=f"提名 {len(vt)} 個", n_variants=0)
