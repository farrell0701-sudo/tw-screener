# 步驟 5 檢查：多程序 run_paths 與單程序結果相同（不計算任何績效）
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim
if __name__ == '__main__':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass
    for mode in ('compat', 'fix'):
        t0 = time.time(); R = sim.run_paths(mode, npath=6); t1 = time.time() - t0
        c = sim.make_ctx(mode); ok = True
        for i in (0, 5):
            a = sim.runX(c, off=i % 20, seed=1000 + i)[0]; b = sim.runV(c, off=i % 20, seed=1000 + i)[0]
            ok &= np.array_equal(a.values, R['X'][i][0].values) and np.array_equal(b.values, R['V'][i][0].values)
        print(f"{'通過' if ok else '不通過'}  {mode}：多程序 6 路徑 {t1:.0f} 秒，與單程序相同")
