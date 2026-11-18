# 步驟 4 檢查之二：假股票、可手算的情境（主要測修正模式的新規則）
# 每個情境寫出預期值的手算式，與模擬器結果比對（誤差 < 1e-12）。
# 用法：python lab/core/test_sim_toy.py
import sys
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core'))
import sim
from sim import FEE, TAX, SLIP

try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
RES = []


def ctx(close, opn=None, valid=None, fix=True, rkx=None, rkv=None, atrp=0.01, ld=None, lu=None, osus=None, last=None, rel=None,
        md=None, off=None, below=None):
    close = np.asarray(close, float); T, N = close.shape
    opn = close.copy() if opn is None else np.asarray(opn, float)
    valid = np.ones((T, N), bool) if valid is None else np.asarray(valid, bool)
    CA = np.where(valid, close, np.nan).astype(np.float32); OA = np.where(valid, opn, np.nan).astype(np.float32)
    z = lambda: np.zeros((T, N), bool)
    return dict(dates=pd.bdate_range('2005-01-03', periods=T), T=T, N=N, codes=np.array([f'S{j}' for j in range(N)]), fix=fix,
                Ov=OA, CA=CA, Cf=pd.DataFrame(CA).ffill().values, VALID=valid,
                RKX=np.full((T, N), np.nan, np.float32) if rkx is None else np.asarray(rkx, np.float32),
                RKV=np.full((T, N), np.nan, np.float32) if rkv is None else np.asarray(rkv, np.float32),
                ATRP=np.full((T, N), atrp, np.float32),
                below=np.zeros(T, bool) if below is None else below, MD=np.zeros(T, bool) if md is None else md,
                OFF=np.zeros(T, bool) if off is None else off,
                STALE=pd.DataFrame(valid).rolling(20, min_periods=20).sum().values == 0, s0=0,
                LD=z() if ld is None else ld, LU=z() if lu is None else lu, OSUS=z() if osus is None else osus,
                LAST=np.full(N, -1) if last is None else np.asarray(last), REL=np.full(N, T) if rel is None else np.asarray(rel))


def check(name, got, exp, tol=1e-12):
    # 假股票價格以 float32 存（同引擎），與 float64 手算差約 1e-8 → 數值以相對誤差 1e-6 判定
    ok = abs(got - exp) <= max(tol, 1e-6 * max(1.0, abs(exp))) if isinstance(exp, float) else got == exp
    RES.append(ok); print(f"{'通過' if ok else '不通過'}  {name}：結果 {got}，預期 {exp}")


def final(r): return float((1 + r[0]).prod())
def rk_at(T, N, t, j):
    a = np.full((T, N), np.nan, np.float32); a[t, j] = 1; return a


# 1. 移動停損：第 1 日收盤選入，第 2 日開盤 100 買；第 3 日收 110（新高）；第 4 日收 105 < 110×0.96 → 第 5 日開盤 104 賣
T = 8; c = np.array([[100, 100, 100, 110, 105, 104, 104, 104]], float).T; o = c.copy(); o[5] = 104
for fix in (False, True):
    r = sim.runX(ctx(c, o, fix=fix, rkx=rk_at(T, 1, 1, 0)), n=1, drop=0)
    px_in = 100 * (1 + SLIP); sh = (1 - FEE) / px_in; exp = sh * 104 * (1 - SLIP) * (1 - FEE - TAX)
    check(f'1 移動停損（{"修正" if fix else "相容"}）期末淨值', final(r), exp)
    check(f'1 移動停損（{"修正" if fix else "相容"}）出場日與原因', (str(r[1][0][2].date()), r[1][0][6]), (str(r[0].index[5].date()), '移動停損'))

# 2. 停牌：買進後第 3–6 日無有效資料；第 7 日恢復收 90（< 100.15×… 觸發停損）→ 第 8 日開盤 89 賣。停牌期間淨值不變。
T = 10; c = np.array([[100, 100, 100, 100, 100, 100, 100, 90, 89, 89]], float).T; v = np.ones((T, 1), bool); v[3:7] = False
o = c.copy(); o[8] = 89
r = sim.runX(ctx(c, o, valid=v, rkx=rk_at(T, 1, 1, 0)), n=1, drop=0)
eqc = (1 + r[0]).cumprod()
check('2 停牌期間淨值不變（第 3–6 日）', bool(np.allclose(eqc.iloc[3:7].values, eqc.iloc[2])), True)
check('2 恢復後才出場（第 8 日）', str(r[1][0][2].date()), str(r[0].index[8].date()))
check('2 停牌：期末淨值', final(r), (1 - FEE) / (100 * (1 + SLIP)) * 89 * (1 - SLIP) * (1 - FEE - TAX))

# 3. 250 日上限在停牌中到期（cap＝4）：修正模式等恢復後第一個有效開盤賣；相容模式在停牌日以最後收盤賣
T = 12; c = np.full((T, 1), 100.0); c[9:] = 120; v = np.ones((T, 1), bool); v[4:9] = False
r = sim.runX(ctx(c, valid=v, rkx=rk_at(T, 1, 1, 0)), n=1, drop=0, cap=4)
check('3 修正：上限到期於停牌中 → 恢復日開盤賣', (str(r[1][0][2].date()), r[1][0][6]), (str(r[0].index[9].date()), '250日上限'))
r = sim.runX(ctx(c, valid=v, rkx=rk_at(T, 1, 1, 0), fix=False), n=1, drop=0, cap=4)
check('3 相容：上限到期於停牌中 → 停牌日以最後收盤賣（舊偏誤 ③）', (str(r[1][0][2].date()), round(float(r[1][0][4]), 6)),
      (str(r[0].index[6].date()), round((1 - FEE) / (100 * (1 + SLIP)) * 100 * (1 - SLIP) * (1 - FEE - TAX) - 1, 6)))

# 4. 下市 A：最後有效日第 5 日收 80 → 當日收盤出清
T = 10; c = np.full((T, 1), 100.0); c[5] = 80; v = np.ones((T, 1), bool); v[6:] = False
r = sim.runX(ctx(c, valid=v, rkx=rk_at(T, 1, 1, 0), last=[5]), n=1, drop=0)
check('4 下市 A：出場日與原因', (str(r[1][0][2].date()), r[1][0][6]), (str(r[0].index[5].date()), '下市出清'))
check('4 下市 A：期末淨值', final(r), (1 - FEE) / (100 * (1 + SLIP)) * 80 * (1 - SLIP) * (1 - FEE - TAX))

# 5. 下市 B（折扣 50%）：最後有效日跌停鎖死 → 凍結、估值 80×0.5、佔名額；第 9 日下市回收。期間另一檔有訊號也買不到。
T = 14; c = np.full((T, 2), 100.0); c[5, 0] = 80; v = np.ones((T, 2), bool); v[6:, 0] = False
ld = np.zeros((T, 2), bool); ld[5, 0] = True
rk = np.full((T, 2), np.nan, np.float32); rk[1, 0] = 1; rk[6, 1] = 1; rk[10, 1] = 1
r = sim.runX(ctx(c, valid=v, rkx=rk, ld=ld, last=[5, -1], rel=[9, 14]), n=1, drop=0, delist=('B', 0.5))
tr = {x[0]: x for x in r[1]}
check('5 下市 B：回收日', (str(tr['S0'][2].date()), tr['S0'][6]), (str(r[0].index[9].date()), '下市回收'))
check('5 下市 B：回收金額（報酬）', float(tr['S0'][4]), ((1 - FEE) / (100 * (1 + SLIP)) * 80 * 0.5 * (1 - FEE - TAX) - 1))
check('5 下市 B：凍結期間佔名額（第 2 檔在回收後才買）', str(tr['S1'][1].date()), str(r[0].index[11].date()))
eqc = (1 + r[0]).cumprod()
check('5 下市 B：第 6 日起以折扣估值', float(eqc.iloc[6]), (1 - FEE) / (100 * (1 + SLIP)) * 40)
r2 = sim.runX(ctx(c, valid=v, rkx=rk, ld=ld, last=[5, -1], rel=[9, 14]), n=1, drop=0, delist='A')
check('5 同情境主版本 A：最後有效日收盤出清', r2[1][0][6], '下市出清')

# 6. 開盤價可疑：當日不買（第 2 日可疑 → 第 2 日買不到；名單隔日重算）；持有中遇可疑日要賣 → 改用收盤
T = 8; c = np.array([[100, 100, 100, 100, 110, 100, 100, 100]], float).T; o = c.copy(); o[6] = 95
os_ = np.zeros((T, 1), bool); os_[2] = True; os_[6] = True
rk = np.full((T, 1), np.nan, np.float32); rk[1:3, 0] = 1
r = sim.runX(ctx(c, o, rkx=rk, osus=os_), n=1, drop=0)
check('6 可疑開盤日不買（改在第 3 日）', str(r[1][0][1].date()), str(r[0].index[3].date()))
check('6 可疑開盤日賣出改用收盤 100', float(r[1][0][4]), ((1 - FEE) / (100 * (1 + SLIP)) * 100 * (1 - SLIP) * (1 - FEE - TAX) - 1))

# 7. 跌停鎖死賣不掉 → 順延；漲停鎖死買不到
T = 9; c = np.array([[100, 100, 100, 100, 90, 85, 84, 84, 84]], float).T
ld = np.zeros((T, 1), bool); ld[5] = True
r = sim.runX(ctx(c, rkx=rk_at(T, 1, 1, 0), ld=ld), n=1, drop=0)
check('7 第 4 日停損訊號、第 5 日跌停鎖死 → 第 6 日賣', str(r[1][0][2].date()), str(r[0].index[6].date()))
lu = np.zeros((T, 1), bool); lu[2] = True
r = sim.runX(ctx(c, rkx=rk_at(T, 1, 1, 0), lu=lu), n=1, drop=0)
check('7 漲停鎖死買不到', len(r[1]), 0)

# 8. V4.1 持有期滿：持有日數＝亂數 18–22 日中的抽樣值（同參考實作）
T = 40; c = np.full((T, 1), 100.0)
r = sim.runV(ctx(c, rkv=rk_at(T, 1, 1, 0)), n=1, drop=0, seed=5)
h = int(np.random.default_rng(5).integers(18, 23))
check('8 V4.1 持有期滿日數', (r[1][0][5], r[1][0][6]), (h, '持有期滿'))

# 9. 合併：每月第一個交易日再平衡 60/40
d = pd.bdate_range('2005-01-03', '2005-03-31'); a = pd.Series(0.01, index=d); b = pd.Series(0.0, index=d)
m = sim.mix(a, b); jan = d[d.month == 1]; feb0 = d[d.month == 2][0]
va = 0.6 * 1.01 ** len(jan); vb = 0.4; tot = va + vb
check('9 合併：2 月第一天報酬＝0.6×1%', float(m.loc[feb0]), 0.006, 1e-12)
k = len(jan); check('9 合併：1 月最後一天（權重已漂移）', float(m.loc[jan[-1]]), 0.6 * 1.01 ** (k - 1) * 0.01 / (0.6 * 1.01 ** (k - 1) + 0.4))

print('結論：' + ('全部通過' if all(RES) else f'{RES.count(False)} 項不通過'))
