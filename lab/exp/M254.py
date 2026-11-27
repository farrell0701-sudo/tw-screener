# M-254：慢資訊策略（L-158 N10）報酬拆解。策略程式碼逐字取自 L158.py。定義見 registry_M.md M-254。
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


def series():
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

    out = {}
    for nm, m in (('N10 主 A（縮至 20%）', 0.2), ('N10 不縮（m1.0）', 1.0)):
        R = [path(10, 4, p, m) for p in range(NP)]
        out[nm] = pd.concat([r for r, _ in R], axis=1).mean(axis=1)
    return out


def main():
    from M253 import factors, fit_table, factor_summary
    from M251 import monthly, RF
    S = series(); Fm = factors(); fit_table.res.clear()
    eras = (('2010.06–15', '2010-06', '2015-12'), ('2016–26', '2016-01', '2026-12'), ('2016–24.10', '2016-01', '2024-10'))
    FF5 = ['MKT', 'SMB', 'HML', 'UMD', 'LVOL']
    models = {'M0': FF5, 'M2': FF5 + ['STM', 'H52', 'REV'], 'M3': FF5 + ['STM', 'H52', 'REV', 'RTL']}
    L = ['# M-254 結果：慢資訊策略報酬拆解\n'] + factor_summary(Fm, eras)
    for nm, s in S.items():
        sm = monthly(s.loc['2010-06':]) - RF / 12
        L.append(f'策略 {nm}：月超額年化平均 ' + '；'.join(f"{e} {sm[(sm.index >= pd.Period(lo, 'M')) & (sm.index <= pd.Period(hi, 'M'))].mean() * 12:+.1%}" for e, lo, hi in eras) + '\n')
        L += fit_table(sm, Fm, models, eras, nm)
    r = {(m, e): (a, t) for m, e, a, t in fit_table.res[:9]}
    h = '主 A alpha（M0→M3）：' + '；'.join(f"{e} {r[('M0', e)][0]:+.1%}→{r[('M3', e)][0]:+.1%}（t {r[('M3', e)][1]:.1f}）" for e, _, _ in eras[:2])
    return dict(summary='\n'.join(L[:1] + [f'**{h}**\n'] + L[1:]), headline=h, n_variants=0)
