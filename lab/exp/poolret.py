# 股池日報酬（M-197 起共用）：強勢池 S、營收池 A、U；奇數代碼；每月成員；日等權。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, M165

END = pd.Timestamp('2024-11-13')


def pool_returns(parity=1):
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    meta = P['meta']; keys = meta['security_key'].astype(str).to_numpy()
    odd = (meta['stock_id'].astype(str).str[-1].astype(int) % 2 == parity).to_numpy()
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
    ix = P['index'][['TAIEX_P', 'TAIEX_TR']].reindex(dates).ffill()
    return df, ix
