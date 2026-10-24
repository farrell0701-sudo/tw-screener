"""C→B→A 結構事件偵測（改進版）與檢驗
基本轉變條件同 struct/scan.py：near5 ≥ 0.35、far5 ≤ 0.10、near5 上升、5 日低 ≥ 舊高×0.98、15～25 天前 far5 ≥ 0.30
改進（使用者提出）：
  1. B 創新高：B 段最高價 ≥ 突破前 250 根最高價（真突破，不是區間變寬）
  2. B 失敗重置：等待進場期間收盤跌回 C 上緣×0.97 以下，或 far5 回到 ≥ 0.30 → 取消
  3. 密集區上緣進場：A 出現後，等收盤突破 A 區（最近 5 根）最高價才進場，最多等 20 天
檢驗：訊號日 t → t+1 收盤進場，20／60 日超額報酬（相對股票池平均），
     並和「同一天、T_ldf 百分位相近（±5%）」的非事件股比較，看結構本身有沒有額外價值。"""
import sys; sys.path.insert(0, '/home/claude/bt/fac'); sys.path.insert(0, '/home/claude/bt/struct')
import numpy as np, pandas as pd
from panel import load_panel
from evaluate import prepare
from factors3 import tld_floor
from scan import overlap_lines

COOL = 20


def detect(H, L, C):
    near, far = overlap_lines(H, L)
    s = pd.DataFrame({'near': near, 'far': far, 'H': H, 'L': L})
    near5 = s.near.rolling(5).mean().values; far5 = s.far.rolling(5).mean().values
    old_top = s.H.shift(21).rolling(100).quantile(0.9).values
    base_before = pd.Series(far5).shift(15).rolling(11).max().values
    lo5 = s.L.rolling(5).min().values
    up = near5 > pd.Series(near5).shift(3).values
    trans = np.nan_to_num((near5 >= 0.35) & (far5 <= 0.10) & up & (lo5 >= old_top * 0.98) & (base_before >= 0.30)).astype(bool)
    hmax250 = s.H.rolling(250, min_periods=200).max().values
    out = []; last = -999; n = len(C)
    for t in np.where(trans)[0]:
        if t - last < COOL or t < 260 or t + 2 >= n: continue
        last = t
        top = old_top[t]; a0 = t - 4
        cand = [i for i in range(max(1, t - 30), a0) if C[i] > top and C[i - 1] <= top]
        bo = cand[0] if cand else a0 - 1
        bH = H[bo:a0].max() if a0 > bo else H[bo]
        newhigh = bH >= hmax250[bo - 1] if bo >= 1 and not np.isnan(hmax250[bo - 1]) else False
        aH = H[a0:t + 1].max()
        # 密集區上緣突破進場 + 失敗重置
        trig, fail = -1, False
        for k in range(t + 1, min(n - 1, t + 21)):
            if C[k] < top * 0.97 or far5[k] >= 0.30: fail = True; break
            if C[k] > aH: trig = k; break
        out.append(dict(t=t, newhigh=bool(newhigh), trig=trig, fail=fail, b_ext=bH / top - 1))
    return out


if __name__ == '__main__':
    import time
    t0 = time.time()
    P = load_panel(); E = prepare(P); U = E['univ']
    Cdf = P['C']; dates = Cdf.index
    ldp = tld_floor(P).where(U).rank(axis=1, pct=True)
    ex = {h: E['fwd'][h]['raw'].sub(E['fwd'][h]['raw'].where(U).mean(1), axis=0) for h in (20, 60)}
    rows = []
    for j, code in enumerate(Cdf.columns):
        c = Cdf[code]; ok = c.notna().values
        if ok.sum() < 300: continue
        idx = np.where(ok)[0]
        H, L, C = (P[k][code].values[idx] for k in ('H', 'L', 'C'))
        for ev in detect(H, L, C):
            for kind, ti in (('A出現即進', ev['t']), ('上緣突破進', ev['trig'])):
                if ti < 0: continue
                d = dates[idx[ti]]
                if not U.at[d, code] or d < pd.Timestamp('2016-01-01'): continue
                rows.append(dict(code=code, date=d, kind=kind, newhigh=ev['newhigh'], fail=ev['fail'] if kind == 'A出現即進' else False,
                                 triggered=ev['trig'] >= 0, b_ext=ev['b_ext'], ldp=ldp.at[d, code],
                                 ex20=ex[20].at[d, code], ex60=ex[60].at[d, code]))
        if j % 400 == 0: print(j, len(rows), f'{time.time() - t0:.0f}s', flush=True)
    R = pd.DataFrame(rows)
    # 對照組：同一天 T_ldf 百分位 ±5% 的股票平均超額
    for h in (20, 60):
        ctl = []
        for d, g in R.groupby('date'):
            lp = ldp.loc[d]; x = ex[h].loc[d]
            for i, r in g.iterrows():
                m = (lp - r.ldp).abs() <= 0.05
                ctl.append((i, x[m].mean()))
        s = pd.Series(dict(ctl)); R[f'ctl{h}'] = s
    R.to_pickle('/home/claude/bt/fac/struct_events.pkl')
    print('事件數', len(R), f'{time.time() - t0:.0f}s')
