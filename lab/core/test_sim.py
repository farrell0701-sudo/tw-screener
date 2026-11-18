# 步驟 4 檢查之一：相容模式模擬器 vs 參考實作
# 從 lab/finlab/fl_small_report.py 原文抽出漲跌停推算（第 103–106 行）與模擬器（第 158–238 行），
# 在同一份輸入上執行，逐路徑比對日報酬與交易紀錄。判準：日報酬最大差 < 1e-12、交易紀錄完全相同。
# 用法：python lab/core/test_sim.py [路徑數]
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core'))
from panel import load as load_panel
from feat import load_feat
from signals import load_sig
import sim

try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
NP = int(sys.argv[1]) if len(sys.argv) > 1 else 3
P = load_panel(); F = load_feat('AF_C'); S = load_sig('compat'); dates = P['dates']; M = P['meta']
t0 = time.time(); c = sim.make_ctx('compat', P, F, S); print(f'建立 ctx {time.time() - t0:.0f} 秒', flush=True)
names = M['security_key'].to_numpy()
DFa = lambda a: pd.DataFrame(np.asarray(a), index=dates, columns=names)

src = (LAB / 'finlab' / 'fl_small_report.py').read_text(encoding='utf-8').splitlines()
def lines(a, b):
    i = next(k for k, s in enumerate(src) if s.startswith(a)); j = next(k for k in range(i, len(src)) if src[k].startswith(b))
    return src[i:j + 1]
code = lines('lim = pd.Series', 'LD, LU = lockdn.values')
k1 = next(k for k, s in enumerate(src) if s.startswith('Ov, Cf = O.values')); k2 = next(k for k, s in enumerate(src) if s.startswith('def mix'))
body = [s for s in src[k1:k2] if not s.startswith('del ') and not s.startswith('print(')]
RC = np.asarray(P['C'], float); RO = np.asarray(P['O'], float)
ns = dict(np=np, pd=pd, dates=dates, T_=len(dates), N_=len(names), FEE=sim.FEE, TAX=sim.TAX, SLIP=sim.SLIP,
          RC=DFa(RC), RH=DFa(np.fmax(np.fmax(np.asarray(P['H'], float), RO), RC)), RL=DFa(np.fmin(np.fmin(np.asarray(P['L'], float), RO), RC)),
          O=DFa(F['OA']), C=DFa(F['CA']), VV=np.asarray(F['VALID']), RKX=c['RKX'], RKV=c['RKV'], ATRP=c['ATRP'],
          below=c['below'], MD=c['MD'], OFF=c['OFF'])
exec('\n'.join(code), ns)
body = [s.replace('START = dates[300]', 'START = pd.Timestamp("2005-01-01")') for s in body]
exec('\n'.join(body), ns)

res = {}
print(f"漲跌停推算：跌停鎖死不同格數 {int((ns['LD'] != c['LD']).sum())}、漲停 {int((ns['LU'] != c['LU']).sum())}")
res['鎖死推算'] = int((ns['LD'] != c['LD']).sum() + (ns['LU'] != c['LU']).sum())
for i in range(NP):
    for kind in ('X', 'V'):
        t1 = time.time()
        a = (sim.runX if kind == 'X' else sim.runV)(c, off=i % 20, seed=1000 + i); t_me = time.time() - t1
        b = (ns['runX'] if kind == 'X' else ns['runV'])(off=i % 20, seed=1000 + i)
        dr = float(np.max(np.abs(a[0].values - b[0].values))) if len(a[0]) == len(b[0]) else float('inf')
        ta = [x[:8] for x in a[1]]; tb = [tuple(x) for x in b[1]]
        same_tr = len(ta) == len(tb) and all(x[:6] == y[:6] or (x[:3] == y[:3] and abs(x[4] - y[4]) < 1e-12) for x, y in zip(ta, tb)) \
            and all(x[6] == y[6] for x, y in zip(ta, tb))
        res[f'路徑 {i} {kind} 日報酬'] = dr; res[f'路徑 {i} {kind} 交易'] = 0 if same_tr else 1
        # 不印績效：驗收數字在 evalu.py 判準補登記之後才看（L-113 1b）
        print(f'路徑 {i} {kind}：日報酬最大差 {dr:.2e}｜交易 {len(ta)} 筆 {"相同" if same_tr else "不同"}｜{t_me:.1f} 秒', flush=True)
bad = [k for k, v in res.items() if v > (1e-12 if '日報酬' in k else 0)]
print('結論：' + ('全部通過' if not bad else f'不通過 {bad}'))
