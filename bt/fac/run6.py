import sys, time
sys.path.insert(0, '/home/claude/bt/fac')
import numpy as np, pandas as pd
from panel import load_panel; from evaluate import prepare, evaluate; from factors import build; from factors6 import build6
t0 = time.time(); P = load_panel(); E = prepare(P); W = pd.read_pickle('/home/claude/bt/fac/fund.pkl')
F1 = build(P); F = build6(P, W)
CTRL = [F1['X_atrpct'][0], F1['X_amt'][0]]; CTRL_LD = CTRL + [F['T_ldf'][0]]
rows = []
for name, (X, sign, why) in F.items():
    o, _ = evaluate(name, X, sign, E, P, CTRL if name == 'T_ldf' else CTRL_LD); o['邏輯'] = why; rows.append(o)
    print(f"{name:10s} n {o['n']:.0f} | IC20 {o['ICva20']:+.3f} IC60 {o['ICva60']:+.3f} t {o['t60']:+.1f} | 增量 {o['ICn20']:+.3f} {o['ICn60']:+.3f} t {o['tn60']:+.1f} | "
          f"正年 {o['正年60']} | 前/後 {o['前半60']:+.3f} {o['後半60']:+.3f} | Q1 {o['Q1_60']:+.2f} Q5 {o['Q5_60']:+.2f} 前50 {o['Top50_60']:+.2f} 前20 {o['Top20_60']:+.2f}", flush=True)
pd.DataFrame(rows).to_csv('/home/claude/bt/fac/result_batch6.csv', index=False, encoding='utf-8-sig')
