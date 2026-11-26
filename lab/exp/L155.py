# L-155：突破＋營收高子策略，100 路徑、固定持有、大盤多空點閘門。判準見 registry_L.md L-155。
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

FEE, TAX, SLIP = 0.00085, 0.003, 0.0015; NP = 100
LOG = []
def log(s): LOG.append(s); print(s, flush=True)


def bull_line(px, dates):
    """使用者大盤多空點：上個月第一個交易日到當日的收盤最高＋最低中點；回傳空頭布林陣列。"""
    s = pd.Series(px, index=dates); m = dates.to_period('M'); out = np.zeros(len(dates), bool)
    first = pd.Series(np.arange(len(dates)), index=dates).groupby(m).min()
    mi = {p: i for i, p in enumerate(first.index)}
    for t in range(len(dates)):
        k = mi[m[t]]; a = first.iloc[k - 1] if k > 0 else first.iloc[0]
        w = s.iloc[a:t + 1]; out[t] = s.iloc[t] < (w.max() + w.min()) / 2
    return out


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    signals.build('fix', mode='fix', P=P, F=F)
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; cols = dbg['cols']
    def full(a, fill):
        a = np.asarray(a)
        if a.shape[1] == N: return a
        o = np.full((T, N), fill, a.dtype); o[:, cols] = a; return o
    pool = full(dbg['pool'], False); Tr = full(np.asarray(dbg['Tr'], np.float64), np.nan); del S
    BRK = np.asarray(M190.features(P, F, Tr)['BRK20'], np.float64)
    VALID = np.asarray(F['VALID']); OA = np.asarray(F['OA'], np.float64); ATR = np.asarray(F['ATR'], np.float64)
    Cf = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill().to_numpy()
    keys = P['meta']['security_key'].astype(str).to_numpy()
    r6 = daily_from_monthly(M180.r6_factors(dates)['R6 營收短長比'], dates, keys)
    r6p = pd.DataFrame(np.where(pool, r6, np.nan)).rank(axis=1, pct=True).to_numpy()
    lastv = np.array([np.nonzero(VALID[:, j])[0].max() if VALID[:, j].any() else -1 for j in range(N)])
    ix = P['index']['TAIEX_P'].reindex(dates).ffill().to_numpy(); bear = bull_line(ix, dates)
    t0 = dates.searchsorted(pd.Timestamp('2005-01-01'))
    ev = []; last = np.full(N, -10 ** 9)
    for t in range(t0 - 20, T - 1):
        for j in np.nonzero(pool[t] & (BRK[t] > 0) & ~(BRK[t - 1] > 0))[0]:
            if t - last[j] < 20: continue
            last[j] = t
            if np.isfinite(r6p[t, j]) and r6p[t, j] >= 2 / 3 and t >= t0: ev.append((t, j, r6p[t, j]))
    byt = {}
    for t, j, r in ev: byt.setdefault(t, []).append((r, j))
    for t in byt: byt[t].sort(reverse=True)
    trp = P['index']['TAIEX_TR'].reindex(dates).ffill().pct_change().fillna(0)
    CACHE = {}

    def trade(t, j, H, k):
        key = (t, j, H, k)
        if key in CACHE: return CACHE[key]
        i0 = t + 1; res = None
        if i0 < T and VALID[i0, j] and OA[i0, j] > 0:
            px = OA[i0, j] * (1 + SLIP); cap = H if k is None else 120; hi = Cf[i0, j]
            sw = (k * ATR[t, j] if np.isfinite(ATR[t, j]) else 0.1 * Cf[t, j]) if k else None
            for i in range(i0, min(T, i0 + cap + 1)):
                if i > lastv[j]: res = (i0, lastv[j], px, Cf[lastv[j], j] * (1 - SLIP)); break
                if not VALID[i, j]: continue
                if i - i0 == cap: res = (i0, i, px, (OA[i, j] if OA[i, j] > 0 else Cf[i, j]) * (1 - SLIP)); break
                if sw is not None:
                    hi = max(hi, Cf[i, j])
                    if Cf[i, j] < hi - sw and i + 1 < T:
                        n = i + 1
                        while n < T and n <= lastv[j] and not VALID[n, j]: n += 1
                        if n > lastv[j] or n >= T: n2 = min(lastv[j], T - 1); res = (i0, n2, px, Cf[n2, j] * (1 - SLIP))
                        else: res = (i0, n, px, (OA[n, j] if OA[n, j] > 0 else Cf[n, j]) * (1 - SLIP))
                        break
            if res is None: i = min(T - 1, lastv[j]); res = (i0, i, px, Cf[i, j] * (1 - SLIP))
        CACHE[key] = res; return res

    def path(H, k, n, gate, p):
        rng = np.random.default_rng(1000 + p); s0 = t0 + p % 20
        cash = 1.0; pos = {}; pend = []; eqp = 1.0; ret = np.zeros(T); tr = []; cnt = np.zeros(T)
        for i in range(s0, T):
            for jj in [jj for jj, q in pos.items() if q['i1'] == i]:
                q = pos.pop(jj); cash += q['sh'] * q['ox'] * (1 - FEE - TAX)
            for j in pend:
                if len(pos) >= n: break
                if j in pos: continue
                x = trade(i - 1, j, H, k)
                if x is None: continue
                i0, i1, px, ox = x; amt = min(eqp / n, cash)
                if amt <= 0: break
                cash -= amt; pos[j] = dict(sh=amt * (1 - FEE) / px, i1=i1, ox=ox)
                tr.append((dates[i0], ox * (1 - FEE - TAX) / (px / (1 - FEE)) - 1, i1 - i0))
            pend = []
            if i in byt and not (gate and bear[i]):
                pend = [j for r, j in byt[i] if rng.random() >= 0.2]
            eq = cash + sum(q['sh'] * Cf[i, jj] for jj, q in pos.items())
            ret[i] = eq / eqp - 1; eqp = eq; cnt[i] = len(pos)
        return pd.Series(ret, index=dates).iloc[t0:], tr, cnt[t0:]

    V = {'主版本 H60／N10／閘門': (60, None, 10, True), 'H40': (40, None, 10, True), 'H90': (90, None, 10, True),
         'N6': (60, None, 6, True), 'N15': (60, None, 15, True), '無閘門': (60, None, 10, False), '移動停損 4ATR（參考）': (None, 4, 10, True)}
    B = sim.run_paths('fix', NP, sig='fix'); bm = E.path_mean([sim.mix(B['X'][i][0], B['V'][i][0]) for i in range(NP)])
    bmm = bm.groupby(bm.index.to_period('M')).apply(lambda x: (1 + x).prod() - 1); worst = bmm[bmm <= bmm.quantile(0.1)].index
    yrs = (dates[-1] - pd.Timestamp('2005-01-01')).days / 365.25
    flips = (np.diff(bear[t0:].astype(int)) != 0).sum() / yrs
    log(f'訊號（R6 ≥ 2/3）{len(ev):,} 個；大盤多空點：空頭天數 {bear[t0:].mean():.0%}、每年翻轉 {flips:.1f} 次\n')
    res = {}
    for nm, (H, k, n, g) in V.items():
        R = [path(H if H else 0, k, n, g, p) for p in range(NP)]
        st = {s: np.median(np.array([stats(r, trp, lo, hi)[:4] for r, _, _ in R]), axis=0) for s, lo, hi in SEGS}
        pm = pd.concat([r for r, _, _ in R], axis=1).mean(axis=1)
        X = pd.DataFrame([x for _, tr, _ in R for x in tr], columns=['d0', 'ret', 'hold'])
        pmm = pm.groupby(pm.index.to_period('M')).apply(lambda x: (1 + x).prod() - 1)
        res[nm] = dict(st=st, X=X, corr=pm.corr(bm.reindex(pm.index).fillna(0)), wm=pmm.reindex(worst).mean(), cnt=np.mean([c.mean() for _, _, c in R]), pm=pm)
    log('## 單獨表現（100 路徑中位數；年化／最大回撤／Calmar／相對 TR 年化超額）\n')
    log('| 版本 | ' + ' | '.join(s for s, _, _ in SEGS) + ' |\n|---|---|---|---|---|---|')
    for nm, r in res.items():
        log(f'| {nm} | ' + ' | '.join('{:+.1%}／{:.1%}／{:.2f}／{:+.1%}'.format(*r['st'][s]) for s, _, _ in SEGS) + ' |')
    log('| 加權報酬指數 TR | ' + ' | '.join('{:+.1%}／{:.1%}／{:.2f}'.format(*stats(trp, trp, lo, hi)[:3]) for _, lo, hi in SEGS) + ' |')
    log('| 新基準合併版 | ' + ' | '.join('{:+.1%}／{:.1%}／{:.2f}'.format(*stats(bm, trp, lo, hi)[:3]) for _, lo, hi in SEGS) + ' |')
    log('\n## 交易與第二關預覽\n'); log('| 版本 | 每年交易（每路徑） | 單筆平均 | 中位 | 勝率 | 持有日 | 平均持股 | 與新基準相關 | 新基準最差 10% 月份的平均月報酬（新基準 ' + f'{bmm.reindex(worst).mean():+.1%}）|\n|---|---|---|---|---|---|---|---|---|')
    for nm, r in res.items():
        X = r['X']; log(f"| {nm} | {len(X) / NP / yrs:.0f} | {X.ret.mean():+.2%} | {X.ret.median():+.2%} | {(X.ret > 0).mean():.0%} | {X.hold.mean():.0f} | {r['cnt']:.1f} | {r['corr']:.2f} | {r['wm']:+.1%} |")
    ys = range(2005, dates[-1].year + 1); yr = lambda r: r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)
    log('\n## 逐年報酬（路徑平均）\n'); log('| 年 | ' + ' | '.join(str(y) for y in ys) + ' |\n|' + '---|' * (len(ys) + 1))
    for nm, r in [('主版本', res['主版本 H60／N10／閘門']['pm']), ('無閘門', res['無閘門']['pm']), ('TR', trp.loc['2005':]), ('新基準', bm.loc['2005':])]:
        yy = yr(r); log(f'| {nm} | ' + ' | '.join(f'{yy.get(y, np.nan):+.0%}' for y in ys) + ' |')
    ex = lambda nm: (res[nm]['st']['2005–15'][3], res[nm]['st']['2016–26'][3])
    mo = all(v > 0 for v in ex('主版本 H60／N10／閘門')); pl = all(all(v > 0 for v in ex(k)) for k in ('H40', 'H90', 'N6', 'N15'))
    nt = len(res['主版本 H60／N10／閘門']['X']) / NP >= 100; ok = mo and pl and nt
    e = ex('主版本 H60／N10／閘門')
    log(f"\n## 判準（L-109 第一關）\n\n- 主版本超額 {e[0]:+.1%}｜{e[1]:+.1%} {'✓' if mo else '✗'}；高原 H40／H90／N6／N15 " + '、'.join(f"{k} {ex(k)[0]:+.1%}｜{ex(k)[1]:+.1%}" for k in ('H40', 'H90', 'N6', 'N15')) + f" {'✓' if pl else '✗'}；交易數 {'✓' if nt else '✗'} → **{'第一關通過' if ok else '第一關不通過'}**")
    head = f"L-155 突破＋營收高子策略（H60／N10／多空點閘門）：{'第一關通過' if ok else '第一關不通過'}（相對 TR {e[0]:+.1%}｜{e[1]:+.1%}）"
    return dict(summary='# L-155 結果\n\n**' + head + '**\n\n' + '\n'.join(LOG), headline=head, n_variants=len(V))
