"""OF 差值（near5 − far5）與其柱狀圖（差值 − 差值的 EMA9）的預測力（乾淨資料）"""
import sys; sys.path.insert(0, '/home/claude/bt/fac')
import numpy as np, pandas as pd
from panel import load_panel; from evaluate import prepare, evaluate; from factors import build; from factors3 import tld_floor
P = load_panel(); E = prepare(P); H, L = P['H'].values, P['L'].values
near = np.zeros_like(H); far = np.zeros_like(H)
for k in range(1, 121):
    o = np.zeros_like(H, dtype=bool)
    o[k:] = (np.minimum(H[k:], H[:-k]) - np.maximum(L[k:], L[:-k])) > 0
    if k <= 20: near += o
    else: far += o
near /= 20; far /= 100; near[:120] = np.nan; far[:120] = np.nan
idx, cols = P['C'].index, P['C'].columns
n5 = pd.DataFrame(near, idx, cols).rolling(5).mean(); f5 = pd.DataFrame(far, idx, cols).rolling(5).mean()
of = n5 - f5
hist = of - of.ewm(span=9, adjust=False).mean()
F1 = build(P); CTRL = [F1['X_atrpct'][0], F1['X_amt'][0], tld_floor(P)]
for name, X in (('near5', n5), ('far5（低好）', -f5), ('OF 差值 near5−far5', of), ('OF 柱狀（差值−EMA9）', hist), ('OF 差值 20 日變化', of - of.shift(20))):
    o, _ = evaluate(name, X, 1, E, P, CTRL)
    print(f"{name:22s} IC20 {o['ICva20']:+.3f} IC60 {o['ICva60']:+.3f} t {o['t60']:+.1f} | 扣趨勢增量 {o['ICn20']:+.3f} {o['ICn60']:+.3f} | 正年 {o['正年60']} | 前20 {o['Top20_60']:+.2f} 前50 {o['Top50_60']:+.2f}", flush=True)
