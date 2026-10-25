import sys, time
sys.path.insert(0, '/home/claude/bt/fac')
import numpy as np, pandas as pd
from panel import load_panel
from evaluate import prepare, evaluate
from factors import build
from factors5 import build5

t0 = time.time()
P = load_panel(); E = prepare(P)
F1 = build(P); F = build5(P)
CTRL = [F1['X_atrpct'][0], F1['X_amt'][0]]
CTRL_LD = CTRL + [F['T_ldf'][0]]
rows = []
for name, (X, sign, why) in F.items():
    if name.startswith('G_'):
        o, _ = evaluate(name, X, sign, E, P, None)
    else:
        o, _ = evaluate(name, X, sign, E, P, CTRL if name == 'T_ldf' else CTRL_LD)
    o['邏輯'] = why; rows.append(o)
    print(f"{name:12s} ICva20 {o['ICva20']:+.3f} | ICva60 {o['ICva60']:+.3f} t {o['t60']:+.1f} | 增量 {o.get('ICn20', np.nan):+.3f} {o.get('ICn60', np.nan):+.3f} "
          f"t {o.get('tn60', np.nan):+.1f} | 正年 {o['正年60']} | 前/後 {o['前半60']:+.3f} {o['後半60']:+.3f} | Q5 {o['Q5_60']:+.2f} 前50 {o['Top50_60']:+.2f} 前20 {o['Top20_60']:+.2f}  ({time.time() - t0:.0f}s)", flush=True)
pd.DataFrame(rows).to_csv('/home/claude/bt/fac/result_batch5.csv', index=False, encoding='utf-8-sig')
