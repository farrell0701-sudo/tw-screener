"""逐日事件回測：T_ld 排序 + 大盤濾網 + 依多空切換的 ATR 停損／停利
- 每 20 日換股：t 收盤排序 → t+1 開盤；不在前 N 的賣出，空位以「當下總資產/N」買入新股（不重新平衡舊部位）
- 大盤：加權指數 EMA60 的 20 日斜率 > 0 = 多頭（之前驗證過最穩健的寫法）
- 濾網開啟：空頭時換股日只賣不買
- 停損：收盤 < 持有期最高收盤 − k×ATR14 → 隔日開盤賣出；k 依當天多空 = k_bull / k_bear
- 空頭停利：空頭時收盤 ≥ 進場價 + m×ATR14 → 隔日開盤賣出
- 被停掉的空位留現金，下個換股日再補
成本：手續費 0.085%/邊、證交稅 0.3%（合計約 0.47%）"""
import sys, itertools; sys.path.insert(0, '/home/claude/bt/fac')
import numpy as np, pandas as pd
from panel import load_panel
from evaluate import prepare
from factors import build, base

FEE, TAX = 0.00085, 0.003
REB, N = 20, 20

P = load_panel(); E = prepare(P); C, O = P['C'], P['O']; U = E['univ']
F = build(P)
SC = (F['T_ld'][0]).where(U)
atr = base(P)['atr']
tx = P['taiex']; e60 = tx.ewm(span=60, adjust=False).mean(); BULL = ((e60 / e60.shift(20) - 1) > 0).values
dates = C.index; d0 = dates.get_indexer([pd.Timestamp('2016-02-01')], method='bfill')[0]
Cv, Ov, Cfv, Av = C.values, O.values, C.ffill().values, atr.values
Uv, Sv = U.values, SC.values
BUY = ((O.shift(-1) / C - 1 < 0.095) & O.shift(-1).notna()).values


def run(off=0, filt=False, kb=None, kr=None, tp=None, start=d0, end=None, bull_arr=None):
    BL = BULL if bull_arr is None else bull_arr
    end = end or len(dates) - 1
    reb = set(range(start + off, end - 1, REB))
    cash = 1.0; pos = {}            # j -> [股數, 進場價, 最高收盤]
    sell_next, buy_next = set(), []
    eq = np.full(len(dates), np.nan); ntr = 0; nstop = 0
    for t in range(start, end + 1):
        # 1) 開盤：執行昨天決定的賣出、買進
        for j in sell_next:
            if j in pos:
                px = Ov[t, j] if not np.isnan(Ov[t, j]) else Cfv[t, j]
                cash += pos.pop(j)[0] * px * (1 - FEE - TAX)
        if buy_next:
            val = cash + sum(p[0] * Cfv[t - 1, j] for j, p in pos.items())
            for j in buy_next:
                if j in pos or len(pos) >= N: continue
                px = Ov[t, j]
                if np.isnan(px): continue
                amt = min(val / N, cash)
                if amt <= 0: break
                pos[j] = [amt * (1 - FEE) / px, px, px]; cash -= amt; ntr += 1
        sell_next, buy_next = set(), []
        # 2) 收盤：記錄淨值、更新最高價、檢查停損停利
        eq[t] = cash + sum(p[0] * Cfv[t, j] for j, p in pos.items())
        bull = BL[t]
        for j, p in pos.items():
            c = Cv[t, j]
            if np.isnan(c): continue
            p[2] = max(p[2], c)
            k = kb if bull else kr
            if k is not None and c < p[2] - k * Av[t, j]: sell_next.add(j); nstop += 1; continue
            if tp is not None and not bull and c >= p[1] + tp * Av[t, j]: sell_next.add(j); nstop += 1
        # 3) 換股日
        if t in reb:
            sc = np.where(BUY[t] & Uv[t], Sv[t], np.nan); ok = np.where(~np.isnan(sc))[0]
            top = list(ok[np.argsort(-sc[ok])[:N]])
            sell_next |= {j for j in pos if j not in top}
            if not (filt and not bull):
                buy_next = [j for j in top if j not in pos]
    s = pd.Series(eq, index=dates).dropna()
    return s, ntr, nstop


def stats(s):
    yrs = (s.index[-1] - s.index[0]).days / 365.25; r = s.pct_change().dropna()
    return dict(CAGR=(s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1, MDD=(s / s.cummax() - 1).min(),
                Sharpe=r.mean() / r.std() * np.sqrt(252))


if __name__ == '__main__':
    OFFS = range(0, 20, 4)
    mid = dates.get_indexer([pd.Timestamp('2021-01-04')], method='bfill')[0]
    KB = [None, 5, 4, 3]; KR = [None, 4, 3, 2, 1.5]; TP = [None, 5, 3]
    rows = []
    for filt in (False, True):
        for kb, kr, tp in itertools.product(KB, KR, TP):
            r = []
            for off in OFFS:
                s, ntr, nst = run(off, filt, kb, kr, tp)
                st = stats(s)
                st['前半'] = stats(s[:dates[mid]])['CAGR']; st['後半'] = stats(s[dates[mid]:])['CAGR']
                st['交易'] = ntr; r.append(st)
            m = pd.DataFrame(r).median()
            rows.append(dict(濾網=filt, k多=kb or '無', k空=kr or '無', 空頭停利=tp or '無', **m.to_dict()))
            print(rows[-1], flush=True)
    R = pd.DataFrame(rows); R.to_csv('/home/claude/bt/fac/port2_grid.csv', index=False, encoding='utf-8-sig')
