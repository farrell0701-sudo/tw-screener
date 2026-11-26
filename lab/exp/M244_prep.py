# M-244 前置：籌碼因子先算成 T×N float32 快取（避免與面板同時佔記憶體）。
import sys, gc
from pathlib import Path
import numpy as np
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
OUT = LAB / 'cache' / 'M244'; OUT.mkdir(parents=True, exist_ok=True)
P = load_panel(); dates = P['dates']; keys = P['meta']['security_key'].astype(str).to_numpy()
from M127 import tdcc_factors
from L123 import daily_from_monthly
td = tdcc_factors(dates)
for k, nm in {'B1': 'B1 千張大戶 1 月變化', 'B2': 'B2 千張大戶 3 月變化', 'B3': 'B3 400 張以上 1 月變化', 'B4': 'B4 散戶 1 月變化', 'B5': 'B5 股東人數 1 月變化率'}.items():
    np.save(OUT / f'{k}.npy', daily_from_monthly(td[nm], dates, keys).astype(np.float32))
del td; gc.collect()
from chips import crowd
F = load_feat('AF'); TURN, MUSE = crowd(P, F, 20)
np.save(OUT / 'TURN20.npy', np.asarray(TURN, np.float32)); np.save(OUT / 'MUSE.npy', np.asarray(MUSE, np.float32))
print('ok', OUT)
