# M-255：慢資訊策略同機制隨機選股對照。策略程式碼取自 L158.py，只換名單。定義見 registry_M.md M-255。
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
    LS = {k: {} for k in ('C', 'A', 'B')}; EL = {}; ELall = {}
    for t in rdays:
        jj = np.nonzero(pool[t])[0]; ELall[t] = jj
        d = pd.DataFrame(dict(j=jj, r6=r6[t, jj], b4=np.asarray(B4[t, jj], np.float64), dh=DH[t, jj]))
        d['dhp'] = d.dh.rank(pct=True); d = d[~(d.dhp < 0.2) & d.r6.notna()].copy()
        EL[t] = d.j.to_numpy()
        if len(d) < 5:
            for k in LS: LS[k][t] = []
            continue
        nn = max(1, int(np.ceil(len(d) * 0.2)))
        d['a'] = d.r6.rank(pct=True); d['b'] = 1 - d.b4.rank(pct=True); d['C'] = d[['a', 'b']].mean(axis=1)
        for k, col in (('C', 'C'), ('A', 'a'), ('B', 'b')):
            LS[k][t] = d.dropna(subset=[col]).sort_values(col, ascending=False).j.to_numpy()[:nn].tolist()
    rd = np.array(rdays); CACHE = {}
    bear = bull_line(P['index']['TAIEX_P'].reindex(dates).ffill().to_numpy(), dates)

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

    def lists_for(kind, p):
        if kind in LS: return LS[kind]
        rr = np.random.default_rng(5000 + p); out = {}
        for t in rdays:
            nn = len(LS['C'][t]); src_ = EL[t] if kind == 'R1' else ELall[t]
            out[t] = rr.permutation(src_)[:nn].tolist() if nn else []
        return out

    def path(n, k, p, m, kind='C'):
        lists = lists_for(kind, p)
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
    from M253 import factors
    from M251 import monthly, ols, RF
    Fm = factors(); cols = ['MKT', 'SMB', 'HML', 'UMD', 'LVOL', 'STM', 'H52', 'REV', 'RTL']
    V = {'C 合成（＝L-158 N10）': 'C', 'R1 隨機（剔 DH）': 'R1', 'R2 全池隨機': 'R2', 'A 只用 R6': 'A', 'B 只用散戶減少': 'B'}
    yrs = (dates[-1] - dates[t0]).days / 365.25; res = {}
    for nm, kind in V.items():
        R = [path(10, 4, p, 0.2, kind) for p in range(NP)]
        st = {sg: np.median(np.array([stats(r, trp, lo, hi)[:4] for r, _ in R]), axis=0) for sg, lo, hi in SEGS}
        ann = {sg: np.array([stats(r, trp, lo, hi)[0] for r, _ in R]) for sg, lo, hi in SEGS}
        X = pd.DataFrame([x for _, tr in R for x in tr], columns=['d0', 'ret', 'hold'])
        pm = pd.concat([r for r, _ in R], axis=1).mean(axis=1); sm = monthly(pm.loc['2016-01':]) - RF / 12
        J = pd.concat([sm.rename('y'), Fm[cols]], axis=1).dropna(); b, t, _ = ols(J.y.to_numpy(), J[cols].to_numpy())
        J0 = pd.concat([sm.rename('y'), Fm[cols[:5]]], axis=1).dropna(); b0, t0_, _ = ols(J0.y.to_numpy(), J0[cols[:5]].to_numpy())
        res[nm] = dict(st=st, ann=ann, X=X, a=(b0[0] * 12, t0_[0], b[0] * 12, t[0]))
        log(f'（{nm} 完成）')
    L = ['## 各版本（100 路徑中位數；年化／最大回撤／Calmar／相對 TR 年化超額）\n',
         '| 版本 | ' + ' | '.join(s for s, _, _ in SEGS) + ' | 每年交易 | 單筆平均 | 勝率 | 2016–26 alpha 五因子｜九因子 |', '|' + '---|' * (len(SEGS) + 5)]
    for nm, r in res.items():
        X = r['X']; a = r['a']
        L.append(f'| {nm} | ' + ' | '.join('{:+.1%}／{:.1%}／{:.2f}／{:+.1%}'.format(*r['st'][s]) for s, _, _ in SEGS) +
                 f" | {len(X) / NP / yrs:.0f} | {X.ret.mean():+.2%} | {(X.ret > 0).mean():.0%} | {a[0]:+.1%}（{a[1]:.1f}）｜{a[2]:+.1%}（{a[3]:.1f}） |")
    L.append('| 加權報酬指數 TR | ' + ' | '.join('{:+.1%}／{:.1%}／{:.2f}'.format(*stats(trp, trp, lo, hi)[:3]) for _, lo, hi in SEGS) + ' | | | | |')
    c, r1 = res['C 合成（＝L-158 N10）'], res['R1 隨機（剔 DH）']
    L += ['', '## 選股貢獻：C − 對照（年化差，路徑中位數；成對路徑勝率）\n', '| 對照 | ' + ' | '.join(s for s, _, _ in SEGS) + ' |', '|' + '---|' * (len(SEGS) + 1)]
    for nm in list(res)[1:]:
        o = res[nm]
        L.append(f'| C − {nm} | ' + ' | '.join(f"{np.median(c['ann'][s] - o['ann'][s]):+.1%}（{(c['ann'][s] > o['ann'][s]).mean():.0%}）" for s, _, _ in SEGS) + ' |')
    d1 = np.median(c['ann']['2010–15'] - r1['ann']['2010–15']); d2 = np.median(c['ann']['2016–26'] - r1['ann']['2016–26'])
    w1 = (c['ann']['2010–15'] > r1['ann']['2010–15']).mean(); w2 = (c['ann']['2016–26'] > r1['ann']['2016–26']).mean()
    ok1 = d1 >= 0.03 and w1 >= 0.7; ok2 = d2 >= 0.03 and w2 >= 0.7
    concl = '選股有實質貢獻' if ok1 and ok2 else '選股貢獻集中在探索期' if ok1 else '選股貢獻不明顯，報酬主要來自機制' if not ok2 else '只在 2016–26 成立（事前未列）'
    L += ['', f'判準：C − R1 2010–15 {d1:+.1%}（勝率 {w1:.0%}）{"✓" if ok1 else "✗"}；2016–26 {d2:+.1%}（勝率 {w2:.0%}）{"✓" if ok2 else "✗"} → **{concl}**']
    head_ = f'M-255：{concl}（C − 隨機 2010–15 {d1:+.1%}、2016–26 {d2:+.1%}）'
    return dict(summary='# M-255 結果\n\n**' + head_ + '**\n\n' + '\n'.join(L), headline=head_, n_variants=0)
