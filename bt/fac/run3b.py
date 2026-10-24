import sys; sys.path.insert(0, '/home/claude/bt/fac')
import numpy as np, pandas as pd
from panel import load_panel; from evaluate import prepare, evaluate; from factors import build; from factors3 import build3
P = load_panel(); E = prepare(P); F1 = build(P); F = build3(P)
CTRL = [F1['X_atrpct'][0], F1['X_amt'][0], F['T_ldf'][0]]
for k in ('R_downday', 'R_upday'):
    o, _ = evaluate(k, F[k][0], F[k][1], E, P, CTRL)
    print(k, f"ICva60 {o['ICva60']:+.3f} t {o['t60']:+.1f} 增量 {o['ICn60']:+.3f} t {o['tn60']:+.1f} 正年 {o['正年60']} 前20 {o['Top20_60']:+.2f} 前50 {o['Top50_60']:+.2f}")
