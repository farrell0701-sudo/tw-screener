# L-160：多空點加波動緩衝——(A) 慢資訊策略、(B) 定版合併版疊加。判準見 registry_L.md L-160。
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
from L155 import bull_line

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
    ixp = P['index']['TAIEX_P'].reindex(dates).ffill().to_numpy()
    def bear_b(b):
        m_ = dates.to_period('M'); first = pd.Series(np.arange(T), index=dates).groupby(m_).min(); mi = {p: i for i, p in enumerate(first.index)}
        A = pd.Series(np.abs(np.diff(ixp, prepend=np.nan))).rolling(14).mean().to_numpy(); st = np.zeros(T, bool); cur = False
        for t in range(T):
            k = mi[m_[t]]; a = first.iloc[k - 1] if k > 0 else first.iloc[0]; w = ixp[a:t + 1]; w = w[np.isfinite(w)]
            if len(w) == 0 or not np.isfinite(A[t]) or not np.isfinite(ixp[t]): st[t] = cur; continue
            ln = (w.max() + w.min()) / 2
            if not cur and ixp[t] < ln - b * A[t]: cur = True
            elif cur and ixp[t] > ln: cur = False
            st[t] = cur
        return st
    BEARS = {b: bear_b(b) for b in (0.0, 0.5, 1.0, 1.5)}
    bear = BEARS[0.0]

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

    def path(n, k, p, m, bear):
        rng = np.random.default_rng(1000 + p); s0 = t0 + p % 20
        cash = 1.0; pos = {}; eqp = 1.0; ret = np.zeros(T); tr = []; banned = set(); cur = None; state = False
        for i in range(s0, T):
            for jj in [jj for jj, q in pos.items() if q['i1'] == i]:
                q = pos.pop(jj); cash += q['sh'] * q['ox'] * (1 - FEE - TAX); banned.add(jj)
            nb = bool(bear[i - 1]); mult = m if nb else 1.0
            if nb != state:                                    # 曝險調整（開盤）
                for jj, q in pos.items():
                    if not (VALID[i, jj] and OA[i, jj] > 0): continue
                    tgt = q['full'] * mult; d = tgt - q['sh']
                    if d < 0: cash += -d * OA[i, jj] * (1 - SLIP) * (1 - FEE - TAX); q['sh'] = tgt
                    elif d > 0:
                        cost = d * OA[i, jj] * (1 + SLIP) / (1 - FEE)
                        if cost > cash: d = cash * (1 - FEE) / (OA[i, jj] * (1 + SLIP)); cost = cash
                        cash -= cost; q['sh'] += d
                state = nb
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
                        i0, i1, px, ox = x; full_amt = eqp / n; amt = min(full_amt * mult, cash)
                        if amt <= 0 and mult > 0: break
                        full = full_amt * (1 - FEE) / px
                        if mult == 0: pos[j] = dict(sh=0.0, full=full, i1=i1, ox=ox)
                        else: cash -= amt; pos[j] = dict(sh=amt * (1 - FEE) / px, full=full, i1=i1, ox=ox)
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

    V = {'基準 b=0': (10, 4, 0.2, 0.0), 'b=1.0（主）': (10, 4, 0.2, 1.0), 'b=0.5': (10, 4, 0.2, 0.5), 'b=1.5': (10, 4, 0.2, 1.5)}
    yrs = (dates[-1] - dates[t0]).days / 365.25; res = {}
    for nm, (n, k, m, bb) in V.items():
        R = [path(n, k, p, m, BEARS[bb]) for p in range(NP)]
        st = {sg: np.median(np.array([stats(r, trp, lo, hi)[:4] for r, _ in R]), axis=0) for sg, lo, hi in SEGS}
        X = pd.DataFrame([x for _, tr in R for x in tr], columns=['d0', 'ret', 'hold'])
        res[nm] = dict(R=R, st=st, X=X, pm=pd.concat([r for r, _ in R], axis=1).mean(axis=1))
    log(f'大盤多空點：空頭天數 {bear[t0:].mean():.0%}、每年翻轉 {(np.diff(bear[t0:].astype(int)) != 0).sum() / yrs:.1f} 次\n')
    log('## 判斷一：單獨策略（100 路徑中位數；年化／最大回撤／Calmar／相對 TR 年化超額）\n')
    log('| 版本 | ' + ' | '.join(s for s, _, _ in SEGS) + ' | 每年交易 | 單筆平均 | 勝率 | 持有日 |\n|---|---|---|---|---|---|---|---|---|')
    for nm, r in res.items():
        X = r['X']; log(f'| {nm} | ' + ' | '.join('{:+.1%}／{:.1%}／{:.2f}／{:+.1%}'.format(*r['st'][s]) for s, _, _ in SEGS) + f" | {len(X) / NP / yrs:.0f} | {X.ret.mean():+.2%} | {(X.ret > 0).mean():.0%} | {X.hold.mean():.0f} |")
    log('| 加權報酬指數 TR | ' + ' | '.join('{:+.1%}／{:.1%}／{:.2f}'.format(*stats(trp, trp, lo, hi)[:3]) for _, lo, hi in SEGS) + ' | | | | |')
    log('| 新基準合併版 | ' + ' | '.join('{:+.1%}／{:.1%}／{:.2f}／{:+.1%}'.format(*stats(bm, trp, lo, hi)[:4]) for _, lo, hi in SEGS) + ' | | | | |')
    bt = res['基準 b=0']['st']; bpm = res['基準 b=0']['pm']; Pv = {}
    log('\n## (A) 慢資訊策略：相對基準（Calmar 路徑中位數）\n'); log('| 版本 | 2010–15 | 2016–20 | 2021–26 | 2010–15 回撤 | 2016–26 回撤 | P（16–26） | 每年翻轉 |\n|---|---|---|---|---|---|---|---|')
    for nm, v in V.items():
        st = res[nm]['st']; br = BEARS[v[3]][t0:]; fl = (np.diff(br.astype(int)) != 0).sum() / yrs
        p = E.boot_p(res[nm]['pm'], bpm) if nm != '基準 b=0' else float('nan'); Pv[nm] = p
        log(f"| {nm} | {st['2010–15'][2]:.2f} | {st['2016–20'][2]:.2f} | {st['2021–26'][2]:.2f} | {st['2010–15'][1]:.1%} | {st['2016–26'][1]:.1%} | {'' if nm == '基準 b=0' else f'{p:.2f}'} | {fl:.1f} |")
    m = res['b=1.0（主）']['st']
    a1 = all(m[sg][2] > bt[sg][2] for sg in ('2016–20', '2021–26')); a2 = Pv['b=1.0（主）'] >= 0.8; a3 = m['2010–15'][1] >= bt['2010–15'][1] - 0.05
    a4 = any(all(res[x]['st'][sg][2] >= bt[sg][2] for sg in ('2016–20', '2021–26')) for x in ('b=0.5', 'b=1.5'))
    okA = a1 and a2 and a3 and a4
    log(f"\n- (A) 判準：兩段上升 {'✓' if a1 else '✗'}；P {Pv['b=1.0（主）']:.2f} {'✓' if a2 else '✗'}；回撤 {'✓' if a3 else '✗'}；高原 {'✓' if a4 else '✗'} → **{'採用' if okA else '不採用'}**")
    bst = {sg: np.median(np.array([stats(b_, trp, lo, hi)[:3] for b_ in BM]), axis=0) for sg, lo, hi in SEGS}
    log('\n### (A) 與新基準合併（等風險／固定 10%；Calmar 路徑中位數）\n'); log('| 組合 | 2010–15 | 2016–20 | 2021–26 | P |\n|---|---|---|---|---|')
    log('| 新基準 | ' + ' | '.join(f'{bst[s_][2]:.2f}' for s_ in ('2010–15', '2016–20', '2021–26')) + ' | |')
    for nm in ('基準 b=0', 'b=1.0（主）'):
        for mode, lab in (('er', '等風險'), (0.1, '固定 10%')):
            Cm = [combine(s_, b_, mode)[0] if mode == 'er' else combine(s_, b_, mode) for (s_, _), b_ in zip(res[nm]['R'], BM)]
            cs = {sg: np.median(np.array([stats(c, trp, lo, hi)[:3] for c in Cm]), axis=0) for sg, lo, hi in SEGS}
            log(f"| {nm}＋{lab} | " + ' | '.join(f'{cs[s_][2]:.2f}' for s_ in ('2010–15', '2016–20', '2021–26')) + f" | {E.boot_p(pd.concat(Cm, axis=1).mean(axis=1), bm):.2f} |")
    # (B) 定版疊加
    BMfull = [sim.mix(B['X'][i][0], B['V'][i][0]) for i in range(NP)]
    base_t = E.seg_table(BMfull); base_pm = E.path_mean(BMfull)
    def overlay(r, st):
        st = pd.Series(st, index=dates).reindex(r.index).fillna(False).to_numpy()
        ex = np.where(np.r_[False, st[:-1]], 0.2, 1.0); cost = np.r_[0, np.abs(np.diff(ex))] * 0.00385
        return pd.Series(ex * r.to_numpy() - cost, index=r.index)
    log('\n## (B) 定版合併版疊加多空點縮倉（Calmar（年化）；100 路徑中位數）\n')
    SG = ['2005–15', '2016–20', '2021–26']
    log('| 版本 | ' + ' | '.join(SG) + ' | 2005–15 回撤 | 2016–26 回撤 | P |\n|---|---|---|---|---|---|---|')
    log('| 新基準 | ' + ' | '.join(f"{base_t.loc[s_, 'Calmar']:.2f}（{base_t.loc[s_, '年化']:.1%}）" for s_ in SG) + f" | {base_t.loc['2005–15', '回撤']:.1%} | {base_t.loc['2016–26', '回撤']:.1%} | |")
    RB = {}
    for bb in (0.0, 0.5, 1.0, 1.5):
        Ov = [overlay(r, BEARS[bb]) for r in BMfull]; tt = E.seg_table(Ov); pv = E.boot_p(E.path_mean(Ov), base_pm); RB[bb] = (tt, pv)
        log(f"| b={bb}{'（主）' if bb == 1.0 else ''} | " + ' | '.join(f"{tt.loc[s_, 'Calmar']:.2f}（{tt.loc[s_, '年化']:.1%}）" for s_ in SG) + f" | {tt.loc['2005–15', '回撤']:.1%} | {tt.loc['2016–26', '回撤']:.1%} | {pv:.2f} |")
    tm, pm_ = RB[1.0]
    b1 = all(tm.loc[s_, 'Calmar'] > base_t.loc[s_, 'Calmar'] for s_ in ('2016–20', '2021–26')); b2 = pm_ >= 0.8; b3 = tm.loc['2005–15', '回撤'] >= base_t.loc['2005–15', '回撤'] - 0.05
    b4 = any(all(RB[x][0].loc[s_, 'Calmar'] >= base_t.loc[s_, 'Calmar'] for s_ in ('2016–20', '2021–26')) for x in (0.5, 1.5))
    veto = all(RB[x][0].loc['2005–15', 'Calmar'] < base_t.loc['2005–15', 'Calmar'] for x in RB)
    okB = b1 and b2 and b3 and b4 and not veto
    log(f"\n- (B) 判準：兩段上升 {'✓' if b1 else '✗'}；P {pm_:.2f} {'✓' if b2 else '✗'}；05–15 回撤 {'✓' if b3 else '✗'}；高原 {'✓' if b4 else '✗'}；否決 {'觸發' if veto else '未觸發'} → **{'通過' if okB else '不通過'}**")
    head = f"L-160 多空點緩衝 b=1.0：(A) 慢資訊策略 {'採用' if okA else '不採用'}（P {Pv['b=1.0（主）']:.2f}）；(B) 定版疊加 {'通過' if okB else '不通過'}（P {pm_:.2f}）"
    return dict(summary='# L-160 結果\n\n**' + head + '**\n\n' + '\n'.join(LOG), headline=head, n_variants=7)
