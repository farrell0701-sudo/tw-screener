"""上漲路徑的波段結構：zigzag（收盤價、反轉門檻 = k×ATR14）切出上漲波／回檔波
每個訊號日只用「當天已確認」的轉折點（+ 最後一段進行中的波），取近 120 根內的波段
特徵：上漲波速度（log 漲幅/根）、回檔波速度、兩者比值、回檔深度 ÷ 前一段漲幅、回檔天數 ÷ 上漲天數"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
import numpy as np, pandas as pd, time
from panel import load_panel; from evaluate import prepare; from factors import base; from factors3 import tld_floor
P = load_panel(); E = prepare(P); U = E['univ']
C = P['C']; idx, cols = C.index, C.columns
atrp = (base(P)['atr'] / C).values
ldf = tld_floor(P).where(U)
indv = P['industry'].reindex(cols); known = indv.notna().values
g = ldf.loc[:, known].T.groupby(indv[known].values).transform('median').T.reindex(columns=cols)
gp = g.where(U).rank(axis=1, pct=True)
cur = ldf.where((gp >= 0.5) | gp.isna())
strong = (cur.rank(axis=1, pct=True) >= 0.9).values
import os; K = float(os.environ.get('ZK', 3.0)); WIN = 120
lc = np.log(C.values)
names = ['上漲速度', '回檔速度', '速度比 漲÷跌', '回檔深度÷前段漲幅', '回檔天數÷上漲天數', '上漲波數']
OUT = {n: np.full(lc.shape, np.nan) for n in names}
t0 = time.time()
for j in range(lc.shape[1]):
    x = lc[:, j]; a = atrp[:, j]
    if np.isnan(x).all(): continue
    piv = []            # (時間, log價, 類型 +1 高點 / -1 低點, 確認時間)
    trend = 0; ext_i = None
    for t in range(len(x)):
        if np.isnan(x[t]) or np.isnan(a[t]):
            continue
        th = K * a[t]
        if ext_i is None: ext_i = t; continue
        if trend >= 0:
            if x[t] >= x[ext_i]: ext_i = t
            elif x[ext_i] - x[t] >= th:
                piv.append((ext_i, x[ext_i], 1, t)); trend = -1; ext_i = t
        if trend < 0:
            if x[t] <= x[ext_i]: ext_i = t
            elif x[t] - x[ext_i] >= th:
                piv.append((ext_i, x[ext_i], -1, t)); trend = 1; ext_i = t
        if not strong[t, j]: continue
        # 當天可用：已確認的轉折 + 目前極值點 + 今天
        lo0 = max(0, t - WIN); w = x[lo0:t + 1]
        if np.isnan(w).all(): continue
        tmin = lo0 + int(np.nanargmin(w))                         # 只看近 120 日最低點之後的上漲段
        pts = [p for p in piv if p[0] > tmin]
        seq = [(tmin, x[tmin])] + [(p[0], p[1]) for p in pts if p[0] < ext_i] + ([(ext_i, x[ext_i])] if ext_i > tmin else [])
        if seq[-1][0] != t: seq.append((t, x[t]))
        up, dn, depth = [], [], []
        last_up = None
        for (t1, p1), (t2, p2) in zip(seq[:-1], seq[1:]):
            n = t2 - t1
            if n <= 0: continue
            d = p2 - p1
            if d > 0: up.append((d, n)); last_up = d
            elif d < 0:
                dn.append((-d, n))
                if last_up: depth.append(-d / last_up)
        if not up: continue
        us = sum(d for d, _ in up) / sum(n for _, n in up)
        OUT['上漲速度'][t, j] = us; OUT['上漲波數'][t, j] = len(up)
        if dn:
            ds = sum(d for d, _ in dn) / sum(n for _, n in dn)
            OUT['回檔速度'][t, j] = ds; OUT['速度比 漲÷跌'][t, j] = us / ds
            OUT['回檔天數÷上漲天數'][t, j] = sum(n for _, n in dn) / sum(n for _, n in up)
        if depth: OUT['回檔深度÷前段漲幅'][t, j] = np.mean(depth)
    if j % 300 == 0: print(j, f'{time.time() - t0:.0f}s', flush=True)
F = {k: pd.DataFrame(v, index=idx, columns=cols) for k, v in OUT.items()}
pd.to_pickle(F, f'/home/claude/bt/fac/swing_shape_{K}_lo.pkl')
