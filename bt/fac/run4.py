import sys, time, os
sys.path.insert(0, '/home/claude/bt/fac')
import numpy as np, pandas as pd
from panel import load_panel
from evaluate import prepare, evaluate
from factors import build
t0 = time.time()
P = load_panel(); E = prepare(P)
CACHE = '/home/claude/bt/fac/f4_clean.pkl' if os.environ.get('CLEAN', '1') == '1' else '/home/claude/bt/fac/f4.pkl'
if os.path.exists(CACHE):
    F = pd.read_pickle(CACHE)
else:
    from factors4 import build4
    D = pd.read_pickle('/home/claude/bt/struct/daily.pkl')
    D = {k: v.reindex(index=P['C'].index, columns=P['C'].columns) for k, v in D.items()}
    F = build4(P, None if 'eligible' in P else D); pd.to_pickle(F, CACHE)   # 乾淨版不用舊的重疊度檔
print(f'因子計算 {time.time() - t0:.0f}s', flush=True)
F1 = build(P)
CTRL = [F1['X_atrpct'][0], F1['X_amt'][0]]; CTRL_LD = CTRL + [F['T_ldf'][0]]
rows = []
only = sys.argv[1:]
for name, (X, sign, why) in F.items():
    if only and name not in only: continue
    o, _ = evaluate(name, X, sign, E, P, CTRL if name == 'T_ldf' else CTRL_LD)
    o['邏輯'] = why; rows.append(o)
    print(f"{name:11s} IC20 {o['ICva20']:+.3f} IC60 {o['ICva60']:+.3f} t {o['t60']:+.1f} | 增量 {o['ICn20']:+.3f} {o['ICn60']:+.3f} t {o['tn60']:+.1f} | "
          f"正年 {o['正年60']} | 前/後 {o['前半60']:+.3f} {o['後半60']:+.3f} | Q1 {o['Q1_60']:+.2f} Q5 {o['Q5_60']:+.2f} 前50 {o['Top50_60']:+.2f} 前20 {o['Top20_60']:+.2f} 自相關 {o['自相關20']:.2f}", flush=True)
pd.DataFrame(rows).to_csv('/home/claude/bt/fac/result_batch4_clean.csv', index=False, encoding='utf-8-sig')
