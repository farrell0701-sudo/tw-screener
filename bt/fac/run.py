import sys, time
sys.path.insert(0, '/home/claude/bt/fac')
import numpy as np, pandas as pd
from panel import load_panel
from evaluate import prepare, evaluate
from factors import build

t0 = time.time()
P = load_panel(); E = prepare(P)
D = pd.read_pickle('/home/claude/bt/struct/daily.pkl')
D = {k: v.reindex(index=P['C'].index, columns=P['C'].columns) for k, v in D.items()}
F = build(P, D)
only = sys.argv[1:]
CTRL = [F['X_atrpct'][0] * -1 * -1, F['X_amt'][0]]   # 控制：波動、成交額
rows, years = [], []
for name, (X, sign, why) in F.items():
    if only and name not in only: continue
    o, y = evaluate(name, X, sign, E, P, None if name.startswith('X_') else CTRL); o['邏輯'] = why
    rows.append(o); years.append(y)
    print(f"{name:12s} ICva20 {o['ICva20']:+.3f} t {o['t20']:+.1f} | ICva60 {o['ICva60']:+.3f} t {o['t60']:+.1f} | "
          f"中性 {o.get('ICn20',np.nan):+.3f} {o.get('ICn60',np.nan):+.3f} | 正年 {o['正年20']} {o['正年60']} | Q5-Q1 {o['Q5-Q1_20']:+.2f}% {o['Q5-Q1_60']:+.2f}%  ({time.time() - t0:.0f}s)", flush=True)
R = pd.DataFrame(rows); Y = pd.concat(years, axis=1)
R.to_csv('/home/claude/bt/fac/result_batch1.csv', index=False, encoding='utf-8-sig')
Y.to_csv('/home/claude/bt/fac/yearly_batch1.csv', encoding='utf-8-sig')
