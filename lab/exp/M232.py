# M-232：「退潮」直接描述（DH250、R20、DSH、FADE）vs POS250，X1 候選／V4.1 內增量。定義見 registry_M.md M-232。
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import M200

FAC = ['T_ld', 'POS250', 'DH250', 'R20', 'DSH', 'FADE']


def extra(P, F, Tr, U):
    C = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill(); H = pd.DataFrame(np.asarray(F['HA'], np.float64)).ffill()
    T, N = C.shape
    dh = (C / H.rolling(250, min_periods=200).max() - 1).to_numpy()
    r20 = (C / C.shift(20) - 1).to_numpy()
    # 距 250 日最高點的交易日數（分塊滑動視窗）
    Hv = H.to_numpy(np.float32); Hv = np.where(np.isfinite(Hv), Hv, -np.inf)
    pad = np.vstack([np.full((249, N), -np.inf, np.float32), Hv]); dsh = np.full((T, N), np.nan)
    from numpy.lib.stride_tricks import sliding_window_view
    for a in range(0, T, 150):
        b = min(T, a + 150); W = sliding_window_view(pad[a:b + 249], 250, axis=0)  # (b-a, N, 250)
        dsh[a:b] = -np.argmax(W[:, :, ::-1], axis=2)  # 距最近一次最高點的日數（同值取最近者）
    cnt = H.notna().cumsum().to_numpy(); dsh[cnt < 200] = np.nan
    pr = lambda x: pd.DataFrame(np.where(U, x, np.nan)).rank(axis=1, pct=True).to_numpy()
    fade = (pr(dh) + pr(r20)) / 2
    return {'DH250': dh, 'R20': r20, 'DSH': dsh, 'FADE': fade}


def main():
    L, vt = M200.analyze(M200.panel(1, extra=extra, fac=FAC), 'M-232 結果：「退潮」直接描述 vs POS250（奇數代碼、排除最近兩年）', fac=FAC)
    return dict(summary='\n'.join(L), headline=f"提名 {len(vt)} 個", n_variants=0)
