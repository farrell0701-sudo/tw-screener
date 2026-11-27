# L-159：L-158 N10 停損寬度改下行 ATR（校準同平均寬度）。判準見 registry_L.md L-159。
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
from M247 import updn

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
    _, DATR, _ = updn(F)
    ent = [(t, j) for t in rdays for j in lists[t]]
    ta_ = np.array([ATR[t, j] for t, j in ent]); td_ = np.array([DATR[t, j] for t, j in ent]); ok_ = np.isfinite(ta_) & np.isfinite(td_) & (td_ > 0)
    KD = 4 * ta_[ok_].mean() / td_[ok_].mean()
    WID = {}
    bear = bull_line(P['index']['TAIEX_P'].reindex(dates).ffill().to_numpy(), dates)

    def trade(t, j, k):
        key = (t, j, k)
        if key in CACHE: return CACHE[key]
        i0 = t + 1; res = None
        if i0 < T and VALID[i0, j] and OA[i0, j] > 0:
            px = OA[i0, j] * (1 + SLIP); hi = Cf[i0, j]; base = ATR[t, j] if k > 0 else DATR[t, j]; sw = abs(k) * base if np.isfinite(base) else 0.1 * Cf[t, j]
            WID.setdefault(k, []).append(sw / Cf[t, j])
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

    def path(n, k, p, m):
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

    V = {'基準 4ATR': (10, 4, 0.2), 'D（主）': (10, -KD, 0.2), 'D−': (10, -KD * 3.5 / 4, 0.2), 'D＋': (10, -KD * 4.5 / 4, 0.2)}
    yrs = (dates[-1] - dates[t0]).days / 365.25; res = {}
    for nm, (n, k, m) in V.items():
        R = [path(n, k, p, m) for p in range(NP)]
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
    log(f"\n校準 kd＝{KD:.2f}（4 × 平均 ATR ÷ 平均下行 ATR）；平均停損寬度（% 訊號日收盤）：" + '；'.join(f"{nm} {np.nanmean(WID[V[nm][1]]):.1%}" for nm in V))
    bt = res['基準 4ATR']['st']; bpm = res['基準 4ATR']['pm']; Pv = {}
    log('\n## 相對基準（Calmar 路徑中位數；P＝2016–26 bootstrap）\n'); log('| 版本 | 2010–15 | 2016–20 | 2021–26 | 2010–15 回撤 | P |\n|---|---|---|---|---|---|')
    for nm in V:
        st = res[nm]['st']; p = E.boot_p(res[nm]['pm'], bpm) if nm != '基準 4ATR' else float('nan'); Pv[nm] = p
        log(f"| {nm} | {st['2010–15'][2]:.2f} | {st['2016–20'][2]:.2f} | {st['2021–26'][2]:.2f} | {st['2010–15'][1]:.1%} | {'' if nm == '基準 4ATR' else f'{p:.2f}'} |")
    m = res['D（主）']['st']
    c1 = all(m[sg][2] > bt[sg][2] for sg in ('2016–20', '2021–26')); c2 = Pv['D（主）'] >= 0.8; c3 = m['2010–15'][1] >= bt['2010–15'][1] - 0.05
    pl = any(all(res[x]['st'][sg][2] >= bt[sg][2] for sg in ('2016–20', '2021–26')) for x in ('D−', 'D＋'))
    ok = c1 and c2 and c3 and pl
    log(f"\n## 判準\n\n- 兩段 Calmar 上升 {'✓' if c1 else '✗'}；P {Pv['D（主）']:.2f} {'✓' if c2 else '✗'}；10–15 回撤 {'✓' if c3 else '✗'}；高原 {'✓' if pl else '✗'} → **{'採用' if ok else '不採用'}**")
    bmm = bm.groupby(bm.index.to_period('M')).apply(lambda x: (1 + x).prod() - 1)
    head = f"L-159 下行 ATR 停損（校準同寬）：{'採用' if ok else '不採用'}（P {Pv['D（主）']:.2f}）"
    return dict(summary='# L-159 結果\n\n**' + head + '**\n\n' + '\n'.join(LOG), headline=head, n_variants=len(V))
