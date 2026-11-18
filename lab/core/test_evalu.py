# 步驟 5 檢查：evalu.py 與 run.py（用合成資料，不看任何策略績效）
import sys, subprocess
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(LAB / 'core'))
import evalu as E

try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
RES = []
def check(nm, ok, detail=''):
    RES.append(bool(ok)); print(f"{'通過' if ok else '不通過'}  {nm} {detail}")

# 1. met：已知序列
d = pd.bdate_range('2010-01-01', periods=504)
r = pd.Series(0.0, index=d); r.iloc[0] = 0.10; r.iloc[100] = -0.20; r.iloc[200] = 0.25
cg, md, cal, sh = E.met(r)
tot = 1.10 * 0.80 * 1.25; check('met 年化', abs(cg - (tot ** (252 / 504) - 1)) < 1e-12, f'{cg:.6f}')
check('met 最大回撤', abs(md - (-0.20)) < 1e-12, f'{md:.6f}')
check('met Calmar', abs(cal - cg / 0.20) < 1e-12)
check('met Sharpe', abs(sh - r.mean() / r.std() * np.sqrt(252)) < 1e-12)
check('met 少於 60 日回傳空值', np.isnan(E.met(r.iloc[:59])[0]))

# 2. 跨段回撤：回撤橫跨 2015/2016，分段各只看到一半，全期看到完整深度
d = pd.bdate_range('2014-01-01', '2017-12-31'); r = pd.Series(0.0005, index=d)
r.loc['2015-09-01':'2015-12-31'] = -0.003; r.loc['2016-01-01':'2016-03-31'] = -0.003
segs = {'2014–15': ('2014-01-01', '2015-12-31'), '2016–17': ('2016-01-01', '2017-12-31'), '全期': ('2014-01-01', '2017-12-31')}
t = E.seg_table([r], segs)
check('全期回撤深於兩段各自的回撤', t.loc['全期', '回撤'] < min(t.loc['2014–15', '回撤'], t.loc['2016–17', '回撤']) - 0.05,
      f"{t.loc['2014–15', '回撤']:.1%} / {t.loc['2016–17', '回撤']:.1%} / 全期 {t.loc['全期', '回撤']:.1%}")

# 3. bootstrap P：相同序列 → 0（嚴格大於）；每日多 5bp → ≈ 1；互換 → 互補
rng = np.random.default_rng(1); d = pd.bdate_range('2016-01-01', '2026-06-30')
base = pd.Series(rng.normal(0.0006, 0.012, len(d)), index=d)
check('bootstrap 相同序列 P＝0', E.boot_p(base, base, n=1000) == 0)
p_up = E.boot_p(base + 0.0005, base, n=1000); check('bootstrap 每日 +5bp → P≈1', p_up > 0.99, f'{p_up:.3f}')
noise = pd.Series(rng.normal(0, 0.003, len(d)), index=d)
p1 = E.boot_p(base + noise, base, n=2000); p2 = E.boot_p(base, base + noise, n=2000)
check('bootstrap 互換後互補', abs(p1 + p2 - 1) < 0.01, f'{p1:.3f} + {p2:.3f}')
check('bootstrap 可重現（固定種子）', E.boot_p(base + noise, base, n=1000) == E.boot_p(base + noise, base, n=1000))

# 4. PSR／DSR：常態化簡、試驗越多 DSR 越低
x = pd.Series(rng.normal(0.0008, 0.01, 2520)); psr0 = E.psr(x)
check('PSR 介於 0、1', 0 < psr0 < 1, f'{psr0:.3f}')
check('常態分位數近似', abs(E._nppf(0.975) - 1.959963985) < 1e-8 and abs(E._nppf(0.01) + 2.326347874) < 1e-8)
ds = [E.dsr(x, n, 0.0004) for n in (10, 100, 520, 2000)]
check('DSR 隨試驗次數遞減', all(a > b for a, b in zip(ds, ds[1:])), ' '.join(f'{v:.3f}' for v in ds))
check('DSR 試驗間變異數未知 → 空值', np.isnan(E.dsr(x, 520, np.nan)))

# 5. run.py：未登記編號被拒絕
r = subprocess.run([sys.executable, str(LAB / 'core' / 'run.py'), 'L-999', '--no-push'], capture_output=True, text=True, encoding='utf-8')
check('run.py 拒絕未登記編號', r.returncode != 0 and '拒絕執行' in (r.stderr + r.stdout))
sys.path.insert(0, str(LAB / 'core')); import run
check('run.py 找得到已登記編號 L-113', run.registered('L-113') == 'registry_L.md')
check('run.py 試驗次數起始 520', run.n_trials() >= 520, str(run.n_trials()))
print('結論：' + ('全部通過' if all(RES) else f'{RES.count(False)} 項不通過'))
