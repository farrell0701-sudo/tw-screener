# L-154：突破＋營收高長抱子策略（L-109 第一關）。判準見 registry_L.md L-154。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim, evalu as E, M190, M180
from L123 import daily_from_monthly
from L126 import stats, SEGS

FEE, TAX, SLIP = 0.00085, 0.003, 0.0015
LOG = []
def log(s): LOG.append(s); print(s, flush=True)


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    signals.build('fix', mode='fix', P=P, F=F)
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; cols = dbg['cols']
    def full(a, fill):
        a = np.asarray(a)
        if a.shape[1] == N: return a
        o = np.full((T, N), fill, a.dtype); o[:, cols] = a; return o
    pool = full(dbg['pool'], False); Tr = full(np.asarray(dbg['Tr'], np.float64), np.nan); del S
    FE = M190.features(P, F, Tr); BRK = np.asarray(FE['BRK20'], np.float64)
    VALID = np.asarray(F['VALID']); OA = np.asarray(F['OA'], np.float64); ATR = np.asarray(F['ATR'], np.float64)
    Cf = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill().to_numpy()
    keys = P['meta']['security_key'].astype(str).to_numpy()
    r6 = daily_from_monthly(M180.r6_factors(dates)['R6 營收短長比'], dates, keys)
    r6p = pd.DataFrame(np.where(pool, r6, np.nan)).rank(axis=1, pct=True).to_numpy()
    lastv = np.array([np.nonzero(VALID[:, j])[0].max() if VALID[:, j].any() else -1 for j in range(N)])
    t0 = dates.searchsorted(pd.Timestamp('2005-01-01'))
    # 事件（不含 R6 門檻，門檻在版本中套用）
    ev = []; last = np.full(N, -10 ** 9)
    for t in range(t0, T - 1):
        for j in np.nonzero(pool[t] & (BRK[t] > 0) & ~(BRK[t - 1] > 0))[0]:
            if t - last[j] < 20: continue
            last[j] = t
            if np.isfinite(r6p[t, j]): ev.append((t, j, r6p[t, j]))
    EV = pd.DataFrame(ev, columns=['t', 'j', 'r6'])
    trp = P['index']['TAIEX_TR'].reindex(dates).ffill().pct_change().fillna(0)

    def trade(t, j, mode, k):
        """回傳 (進場日 i0, 出場日 i1, 進價, 出價) 或 None；i1 當日以出價結算。"""
        i0 = t + 1
        if i0 >= T or not VALID[i0, j] or not (OA[i0, j] > 0): return None
        px = OA[i0, j] * (1 + SLIP); stop_w = k * ATR[t, j] if np.isfinite(ATR[t, j]) else 0.1 * Cf[t, j]; hi = Cf[i0, j]
        cap = 60 if mode == 'A' else 120
        for i in range(i0, min(T, i0 + cap + 1)):
            if i > lastv[j]: return (i0, lastv[j], px, Cf[lastv[j], j] * (1 - SLIP))
            if not VALID[i, j]: continue
            if i - i0 == cap:
                ox = OA[i, j] if OA[i, j] > 0 else Cf[i, j]; return (i0, i, px, ox * (1 - SLIP), 'open')
            if mode == 'B':
                hi = max(hi, Cf[i, j])
                if Cf[i, j] < hi - stop_w and i + 1 < T:
                    n = i + 1
                    while n < T and n <= lastv[j] and not VALID[n, j]: n += 1
                    if n > lastv[j] or n >= T: return (i0, min(lastv[j], T - 1), px, Cf[min(lastv[j], T - 1), j] * (1 - SLIP))
                    ox = OA[n, j] if OA[n, j] > 0 else Cf[n, j]; return (i0, n, px, ox * (1 - SLIP), 'open')
        i = min(T - 1, lastv[j]); return (i0, i, px, Cf[i, j] * (1 - SLIP))   # 資料末日仍持有：以收盤估值

    def run(mode, k, q, slots):
        ev = EV[EV.r6 >= q].sort_values(['t', 'r6'], ascending=[True, False])
        TR_ = []; ret = np.zeros(T); cnt = np.zeros(T)
        if slots is None:
            num = np.zeros(T); den = np.zeros(T)
            for t, j, _ in ev.itertuples(index=False):
                x = trade(int(t), int(j), mode, k)
                if x is None: continue
                i0, i1, px, ox = x[:4]; buy = px * (1 + FEE); sell = ox * (1 - FEE - TAX)
                TR_.append((dates[i0], sell / buy - 1, i1 - i0))
                v = [buy] + [Cf[i, j] for i in range(i0, i1)] + [sell]   # 價值路徑：進場成本 → 各日收盤 → 出場淨額
                for n, i in enumerate(range(i0, i1 + 1)):
                    num[i] += v[n + 1] / v[n] - 1; den[i] += 1
            ret = np.where(den > 0, num / np.maximum(den, 1), 0.0); cnt = den
        else:
            byt = {t: g for t, g in ev.groupby('t')}; cash = 1.0; pos = {}; pend = []
            eq_prev = 1.0
            for i in range(t0, T):
                # 出場（出價在 i 日）
                for jj in [jj for jj, p in pos.items() if p['i1'] == i]:
                    p = pos.pop(jj); cash += p['sh'] * p['ox'] * (1 - FEE - TAX)
                # 進場（前一日訊號）
                for (t, j, _) in pend:
                    if len(pos) >= slots or j in pos: continue
                    x = trade(t, j, mode, k)
                    if x is None: continue
                    i0, i1, px, ox = x[:4]; amt = min(eq_prev / slots, cash)
                    if amt <= 0: break
                    sh = amt * (1 - FEE) / px; cash -= amt; pos[j] = dict(sh=sh, i1=i1, ox=ox)
                    TR_.append((dates[i0], ox * (1 - FEE - TAX) / (px / (1 - FEE)) - 1, i1 - i0))
                pend = [(int(a), int(b), c) for a, b, c in byt[i].itertuples(index=False)] if i in byt else []
                eq = cash + sum(p['sh'] * Cf[i, jj] for jj, p in pos.items())
                ret[i] = eq / eq_prev - 1; eq_prev = eq; cnt[i] = len(pos)
        r = pd.Series(ret, index=dates).iloc[t0:]
        return r, pd.DataFrame(TR_, columns=['d0', 'ret', 'hold']), pd.Series(cnt, index=dates).iloc[t0:]

    V = {'B6（主）': ('B', 4, 2 / 3, 6), 'B 不限': ('B', 4, 2 / 3, None), 'A6': ('A', 4, 2 / 3, 6), 'A 不限': ('A', 4, 2 / 3, None),
         'B6｜R6≥0.6': ('B', 4, 0.6, 6), 'B6｜R6≥0.75': ('B', 4, 0.75, 6), 'B6｜3ATR': ('B', 3, 2 / 3, 6), 'B6｜5ATR': ('B', 5, 2 / 3, 6)}
    O = {nm: run(*v) for nm, v in V.items()}
    B = sim.run_paths('fix', 100, sig='fix'); bm = E.path_mean([sim.mix(B['X'][i][0], B['V'][i][0]) for i in range(100)])
    log(f'事件數（R6 門檻前）{len(EV):,}；R6 ≥ 2/3：{(EV.r6 >= 2 / 3).sum():,}\n')
    log('## 單獨表現（年化／最大回撤／Calmar／相對 TR 年化超額）\n')
    log('| 版本 | ' + ' | '.join(s for s, _, _ in SEGS) + ' |\n|---|---|---|---|---|---|')
    for nm, (r, _, _) in list(O.items()) + [('加權報酬指數 TR', (trp, None, None)), ('新基準合併版', (bm, None, None))]:
        log(f'| {nm} | ' + ' | '.join('{:+.1%}／{:.1%}／{:.2f}／{:+.1%}'.format(*stats(r, trp, lo, hi)[:4]) for _, lo, hi in SEGS) + ' |')
    yrs = (dates[-1] - pd.Timestamp('2005-01-01')).days / 365.25
    log('\n## 交易\n'); log('| 版本 | 每年交易 | 單筆平均 | 中位 | 勝率 | 平均持有日 | 平均同時持股 | 與新基準日報酬相關 |\n|---|---|---|---|---|---|---|---|')
    for nm, (r, X, c) in O.items():
        log(f"| {nm} | {len(X) / yrs:.0f} | {X.ret.mean():+.2%} | {X.ret.median():+.2%} | {(X.ret > 0).mean():.0%} | {X.hold.mean():.0f} | {c[c > 0].mean():.1f}（空手 {(c == 0).mean():.0%}） | {r.corr(bm.reindex(r.index).fillna(0)):.2f} |")
    log('\n## 逐年報酬\n'); ys = range(2005, dates[-1].year + 1)
    yr = lambda r: r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)
    log('| 年 | ' + ' | '.join(str(y) for y in ys) + ' |\n|' + '---|' * (len(ys) + 1))
    for nm, r in [('B6（主）', O['B6（主）'][0]), ('B 不限', O['B 不限'][0]), ('TR', trp.loc['2005':]), ('新基準', bm.loc['2005':])]:
        yy = yr(r); log(f'| {nm} | ' + ' | '.join(f'{yy.get(y, np.nan):+.0%}' for y in ys) + ' |')
    ex = {nm: (stats(r, trp, '2005-01-01', '2015-12-31')[3], stats(r, trp, '2016-01-01', '2026-12-31')[3]) for nm, (r, _, _) in O.items()}
    main_ok = all(v > 0 for v in ex['B6（主）']); plat = all(all(v > 0 for v in ex[k]) for k in ('B6｜R6≥0.6', 'B6｜R6≥0.75', 'B6｜3ATR', 'B6｜5ATR'))
    ntr = len(O['B6（主）'][1]) >= 100; ok = main_ok and plat and ntr
    log(f"\n## 判準（L-109 第一關）\n\n- 主版本 B6 超額 {ex['B6（主）'][0]:+.1%}｜{ex['B6（主）'][1]:+.1%} {'✓' if main_ok else '✗'}；高原四變體 {'✓' if plat else '✗'}；交易數 {len(O['B6（主）'][1])} {'✓' if ntr else '✗'} → **{'第一關通過' if ok else '第一關不通過'}**")
    head = f"L-154 突破＋營收高子策略：{'第一關通過' if ok else '第一關不通過'}（B6 相對 TR {ex['B6（主）'][0]:+.1%}｜{ex['B6（主）'][1]:+.1%}；B 不限 {ex['B 不限'][0]:+.1%}｜{ex['B 不限'][1]:+.1%}）"
    return dict(summary='# L-154 結果\n\n**' + head + '**\n\n' + '\n'.join(LOG), headline=head, n_variants=len(V))
