# L-157：慢資訊策略（營收＋散戶減少，位置剔除），單獨與合併分開判斷。判準見 registry_L.md L-157。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim, evalu as E, M180
from L123 import daily_from_monthly
from L126 import stats

FEE, TAX, SLIP = 0.00085, 0.003, 0.0015; NP = 100
SEGS = (('2010–15', '2010-01-01', '2015-12-31'), ('2016–20', '2016-01-01', '2020-12-31'), ('2021–26', '2021-01-01', '2026-12-31'), ('2016–26', '2016-01-01', '2026-12-31'))
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
    pool = full(dbg['pool'], False); del S
    VALID = np.asarray(F['VALID']); OA = np.asarray(F['OA'], np.float64); ATR = np.asarray(F['ATR'], np.float64)
    C = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill(); H = pd.DataFrame(np.asarray(F['HA'], np.float64)).ffill(); Cf = C.to_numpy()
    DH = (C / H.rolling(250, min_periods=200).max() - 1).to_numpy(); del H, C
    keys = P['meta']['security_key'].astype(str).to_numpy()
    r6 = daily_from_monthly(M180.r6_factors(dates)['R6 營收短長比'], dates, keys)
    B4 = np.load(LAB / 'cache' / 'M244' / 'B4.npy', mmap_mode='r')
    lastv = np.array([np.nonzero(VALID[:, j])[0].max() if VALID[:, j].any() else -1 for j in range(N)])
    t0 = dates.searchsorted(pd.Timestamp('2010-01-01'))
    # 每月排序日與候選名單
    ds = pd.Series(dates, index=dates); s = ds[ds.dt.day > 10]; rdays = [dates.get_loc(d) for d in s.groupby(s.dt.to_period('M')).min() if d >= dates[t0 - 30]]
    lists = {}
    for t in rdays:
        jj = np.nonzero(pool[t])[0]
        d = pd.DataFrame(dict(j=jj, r6=r6[t, jj], b4=np.asarray(B4[t, jj], np.float64), dh=DH[t, jj]))
        d['dhp'] = d.dh.rank(pct=True); d = d[~(d.dhp < 0.2) & d.r6.notna()].copy()
        if len(d) < 5: lists[t] = []; continue
        d['a'] = d.r6.rank(pct=True); d['b'] = 1 - d.b4.rank(pct=True)
        d['C'] = d[['a', 'b']].mean(axis=1)
        d = d.sort_values('C', ascending=False); lists[t] = d.j.to_numpy()[:max(1, int(np.ceil(len(d) * 0.2)))].tolist()
    rd = np.array(rdays); CACHE = {}

    def trade(t, j, k):
        key = (t, j, k)
        if key in CACHE: return CACHE[key]
        i0 = t + 1; res = None
        if i0 < T and VALID[i0, j] and OA[i0, j] > 0:
            px = OA[i0, j] * (1 + SLIP); hi = Cf[i0, j]; sw = k * ATR[t, j] if np.isfinite(ATR[t, j]) else 0.1 * Cf[t, j]
            for i in range(i0, min(T, i0 + 121)):
                if i > lastv[j]: res = (i0, lastv[j], px, Cf[lastv[j], j] * (1 - SLIP)); break
                if not VALID[i, j]: continue
                if i - i0 == 120: res = (i0, i, px, (OA[i, j] if OA[i, j] > 0 else Cf[i, j]) * (1 - SLIP)); break
                hi = max(hi, Cf[i, j])
                if Cf[i, j] < hi - sw and i + 1 < T:
                    n = i + 1
                    while n < T and n <= lastv[j] and not VALID[n, j]: n += 1
                    if n > lastv[j] or n >= T: n2 = min(lastv[j], T - 1); res = (i0, n2, px, Cf[n2, j] * (1 - SLIP))
                    else: res = (i0, n, px, (OA[n, j] if OA[n, j] > 0 else Cf[n, j]) * (1 - SLIP))
                    break
            if res is None: i = min(T - 1, lastv[j]); res = (i0, i, px, Cf[i, j] * (1 - SLIP))
        CACHE[key] = res; return res

    def path(n, k, p):
        rng = np.random.default_rng(1000 + p); s0 = t0 + p % 20
        cash = 1.0; pos = {}; eqp = 1.0; ret = np.zeros(T); tr = []; banned = set(); cur = None
        for i in range(s0, T):
            for jj in [jj for jj, q in pos.items() if q['i1'] == i]:
                q = pos.pop(jj); cash += q['sh'] * q['ox'] * (1 - FEE - TAX); banned.add(jj)
            t = i - 1; k_ = rd[rd <= t]
            if len(k_):
                r = int(k_[-1])
                if r != cur: cur = r; banned = set(jj for jj in banned if jj in pos)
                if len(pos) < n:
                    for j in lists[r]:
                        if len(pos) >= n: break
                        if j in pos or j in banned or rng.random() < 0.2: continue
                        x = trade(t, j, k)
                        if x is None: continue
                        i0, i1, px, ox = x; amt = min(eqp / n, cash)
                        if amt <= 0: break
                        cash -= amt; pos[j] = dict(sh=amt * (1 - FEE) / px, i1=i1, ox=ox)
                        tr.append((dates[i0], ox * (1 - FEE - TAX) / (px / (1 - FEE)) - 1, i1 - i0))
            eq = cash + sum(q['sh'] * Cf[i, jj] for jj, q in pos.items())
            ret[i] = eq / eqp - 1; eqp = eq
        return pd.Series(ret, index=dates).iloc[t0:], tr

    trp = P['index']['TAIEX_TR'].reindex(dates).ffill().pct_change().fillna(0)
    B = sim.run_paths('fix', NP, sig='fix'); BM = [sim.mix(B['X'][i][0], B['V'][i][0]).reindex(dates).fillna(0).iloc[t0:] for i in range(NP)]
    bm = pd.concat(BM, axis=1).mean(axis=1)

    def combine(s, b, mode):
        if isinstance(mode, float): return (1 - mode) * b + mode * s
        vs = s.rolling(250, min_periods=60).std(); vb = b.rolling(250, min_periods=60).std()
        w = ((1 / vs) / (1 / vs + 1 / vb)); w = w.groupby(w.index.to_period('M')).transform('first').shift(1).fillna(0)
        return (1 - w) * b + w * s, w

    V = {'N10（主 A）': (10, 4), 'N10｜3ATR': (10, 3), 'N10｜5ATR': (10, 5), 'N15（主 B）': (15, 4), 'N15｜3ATR': (15, 3), 'N15｜5ATR': (15, 5)}
    yrs = (dates[-1] - dates[t0]).days / 365.25; res = {}
    for nm, (n, k) in V.items():
        R = [path(n, k, p) for p in range(NP)]
        st = {sg: np.median(np.array([stats(r, trp, lo, hi)[:4] for r, _ in R]), axis=0) for sg, lo, hi in SEGS}
        X = pd.DataFrame([x for _, tr in R for x in tr], columns=['d0', 'ret', 'hold'])
        res[nm] = dict(R=R, st=st, X=X, pm=pd.concat([r for r, _ in R], axis=1).mean(axis=1))
    log('## 判斷一：單獨策略（100 路徑中位數；年化／最大回撤／Calmar／相對 TR 年化超額）\n')
    log('| 版本 | ' + ' | '.join(s for s, _, _ in SEGS) + ' | 每年交易 | 單筆平均 | 勝率 | 持有日 |\n|---|---|---|---|---|---|---|---|---|')
    for nm, r in res.items():
        X = r['X']; log(f'| {nm} | ' + ' | '.join('{:+.1%}／{:.1%}／{:.2f}／{:+.1%}'.format(*r['st'][s]) for s, _, _ in SEGS) + f" | {len(X) / NP / yrs:.0f} | {X.ret.mean():+.2%} | {(X.ret > 0).mean():.0%} | {X.hold.mean():.0f} |")
    log('| 加權報酬指數 TR | ' + ' | '.join('{:+.1%}／{:.1%}／{:.2f}'.format(*stats(trp, trp, lo, hi)[:3]) for _, lo, hi in SEGS) + ' | | | | |')
    log('| 新基準合併版 | ' + ' | '.join('{:+.1%}／{:.1%}／{:.2f}／{:+.1%}'.format(*stats(bm, trp, lo, hi)[:4]) for _, lo, hi in SEGS) + ' | | | | |')
    ex = lambda nm: (res[nm]['st']['2010–15'][3], res[nm]['st']['2016–26'][3])
    J1 = {}
    for mn, pl in (('N10（主 A）', ('N10｜3ATR', 'N10｜5ATR')), ('N15（主 B）', ('N15｜3ATR', 'N15｜5ATR'))):
        a = all(v > 0 for v in ex(mn)); b = all(all(v > 0 for v in ex(x)) for x in pl); c = len(res[mn]['X']) / NP >= 100
        J1[mn] = a and b and c
        log(f"\n- {mn}：超額 {ex(mn)[0]:+.1%}｜{ex(mn)[1]:+.1%} {'✓' if a else '✗'}；高原 " + '、'.join(f'{x} {ex(x)[0]:+.1%}｜{ex(x)[1]:+.1%}' for x in pl) + f" {'✓' if b else '✗'}；交易數 {'✓' if c else '✗'} → **{'第一關通過' if J1[mn] else '第一關不通過'}**")
    bmm = bm.groupby(bm.index.to_period('M')).apply(lambda x: (1 + x).prod() - 1); worst = bmm[bmm <= bmm.quantile(0.1)].index
    log('\n## 判斷二：合併策略（新基準 ＋ 本策略；路徑中位數 Calmar（年化）；回撤）\n')
    log('| 組合 | ' + ' | '.join(s for s, _, _ in SEGS) + ' | 2010–15 回撤 | P（16–26） | 平均權重 |\n|---|---|---|---|---|---|---|---|')
    bst = {sg: np.median(np.array([stats(b, trp, lo, hi)[:3] for b in BM]), axis=0) for sg, lo, hi in SEGS}
    log('| 新基準 | ' + ' | '.join(f'{bst[s][2]:.2f}（{bst[s][0]:.1%}）' for s, _, _ in SEGS) + f" | {bst['2010–15'][1]:.1%} | | |")
    J2 = {}
    for mn in ('N10（主 A）', 'N15（主 B）'):
        for mode, lab in (('er', '等風險（主）'), (0.1, '固定 10%'), (0.2, '固定 20%')):
            Cm = []; ws = []
            for (s_, _), b in zip(res[mn]['R'], BM):
                o = combine(s_, b, mode)
                if isinstance(o, tuple): Cm.append(o[0]); ws.append(o[1].loc['2011':].mean())
                else: Cm.append(o); ws.append(mode)
            cs = {sg: np.median(np.array([stats(c, trp, lo, hi)[:3] for c in Cm]), axis=0) for sg, lo, hi in SEGS}
            pv = E.boot_p(pd.concat(Cm, axis=1).mean(axis=1), bm)
            log(f"| {mn[:3]}＋{lab} | " + ' | '.join(f'{cs[s][2]:.2f}（{cs[s][0]:.1%}）' for s, _, _ in SEGS) + f" | {cs['2010–15'][1]:.1%} | {pv:.2f} | {np.mean(ws):.0%} |")
            if mode == 'er':
                ok = all(cs[s][2] > bst[s][2] for s in ('2016–20', '2021–26')) and pv >= 0.8 and cs['2010–15'][1] >= bst['2010–15'][1] - 0.05
                J2[mn] = (ok, cs, pv)
    log('\n## 分散檢查\n'); log(f"| 版本 | 與新基準日報酬相關 | 新基準最差 10% 月份的平均月報酬（新基準 {bmm.reindex(worst).mean():+.1%}） |\n|---|---|---|")
    for mn in ('N10（主 A）', 'N15（主 B）'):
        pm = res[mn]['pm']; pmm = pm.groupby(pm.index.to_period('M')).apply(lambda x: (1 + x).prod() - 1)
        log(f"| {mn} | {pm.corr(bm):.2f} | {pmm.reindex(worst).mean():+.1%} |")
    log('\n## 判準\n')
    for mn in ('N10（主 A）', 'N15（主 B）'):
        ok, cs, pv = J2[mn]
        log(f"- {mn}：判斷一 **{'通過' if J1[mn] else '不通過'}**；判斷二（等風險）合併 Calmar 16–20 {cs['2016–20'][2]:.2f} vs {bst['2016–20'][2]:.2f}、21–26 {cs['2021–26'][2]:.2f} vs {bst['2021–26'][2]:.2f}、P {pv:.2f}、10–15 回撤 {cs['2010–15'][1]:.1%} vs {bst['2010–15'][1]:.1%} → **{'通過' if ok else '不通過'}**")
    head = 'L-157 慢資訊策略：' + '；'.join(f"{mn[:3]} 單獨{'✓' if J1[mn] else '✗'}／合併{'✓' if J2[mn][0] else '✗'}" for mn in ('N10（主 A）', 'N15（主 B）'))
    return dict(summary='# L-157 結果\n\n**' + head + '**\n\n' + '\n'.join(LOG), headline=head, n_variants=len(V))
