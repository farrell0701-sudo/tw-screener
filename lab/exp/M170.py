# M-170：L-126 與 L-127 B0 差異拆解（機制 J/C × Q5 N/Q）。定義見 registry_M.md M-170。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals
from M120 import rev_factors
from M122 import fin_factors

FEE, TAX, SLIP = 0.00085, 0.003, 0.0015; H = 120; STOP = 0.15
SEGS = (('2005–15', '2005-01-01', '2015-12-31'), ('2016–20', '2016-01-01', '2020-12-31'), ('2021–26', '2021-01-01', '2026-12-31'), ('2016–26', '2016-01-01', '2026-12-31'))


def seg(r, lo, hi):
    r = r.loc[lo:hi]; n = len(r); eq = (1 + r).cumprod(); ann = eq.iloc[-1] ** (245 / n) - 1; dd = (eq / eq.cummax() - 1).min()
    return ann, dd, (ann / -dd if dd < 0 else np.nan)


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    keys = P['meta']['security_key'].astype(str).to_numpy()
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; U, pool = dbg['U'], dbg['pool']; del S
    VALID = np.asarray(F['VALID']); CA = np.asarray(F['CA'], np.float64); OA = np.asarray(F['OA'], np.float64)
    Cf = pd.DataFrame(np.where(VALID, CA, np.nan)).ffill().to_numpy()
    tr = P['index']['TAIEX_TR'].reindex(dates).ffill().pct_change().fillna(0)
    W = rev_factors(dates)['R4 營收創高']; jm = pd.Index(W.columns.astype(str)).get_indexer(keys)
    Q5 = fin_factors(dates)['Q5 毛利率創高']; qj = pd.Index(Q5.columns.astype(str)).get_indexer(keys)
    ds = pd.Series(dates, index=dates); d10 = ds[ds.dt.day > 10]; samp = d10.groupby(d10.dt.to_period('M')).min()
    samp = [dates.get_loc(d) for d in samp if d >= pd.Timestamp('2004-07-01') and dates.get_loc(d) + 2 < T]
    MEM = {'N': {}, 'Q': {}}
    for t in samp:
        m = dates[t].to_period('M') - 1
        if m not in W.index: continue
        v = W.loc[m].to_numpy(float); x = np.where(jm >= 0, v[np.clip(jm, 0, None)], np.nan)
        jj = np.nonzero(U[t] & ~pool[t] & np.isfinite(x))[0]
        if len(jj) < 100: continue
        g = pd.qcut(pd.Series(x[jj]).rank(method='first').to_numpy(), 10, labels=False); top = jj[g == 9]
        top = top[VALID[t + 1, top] & np.isfinite(OA[t + 1, top]) & (OA[t + 1, top] > 0)]
        MEM['N'][t] = top
        q = np.where(qj[top] >= 0, Q5.loc[m].to_numpy(float)[np.clip(qj[top], 0, None)], np.nan) if m in Q5.index else np.full(len(top), np.nan)
        fin = np.isfinite(q); drop = np.zeros(len(top), bool)
        if fin.sum() >= 20: drop[fin] = q[fin] <= np.nanpercentile(q[fin], 20)
        MEM['Q'][t] = top[~drop]

    def jt(members, legacy=False):
        tot = np.zeros(T); cnt = np.zeros(T); live = np.zeros(T)
        for t, jj in members.items():
            e = min(t + 1 + H, T); idx = np.arange(t + 1, e); n = len(idx)
            if n < 2 or not len(jj): continue
            c = Cf[t + 1:e][:, jj]
            px = OA[t + 1, jj] * (1 + SLIP)
            hit = c < px * (1 - STOP)
            k = np.where(hit.any(axis=0), hit.argmax(axis=0), n - 1)
            ar = np.arange(n)[:, None]
            if legacy: val = np.where(ar <= k[None, :], c / px, (c[k, np.arange(len(jj))] / px)[None, :]) * (1 - 0.0077 + SLIP)
            else:
                val = np.where(ar <= k[None, :], c / px * (1 - FEE), (c[k, np.arange(len(jj))] / px * (1 - FEE) * (1 - FEE - TAX - SLIP))[None, :])
            Vc = val.mean(axis=1); rc = np.r_[Vc[0] - 1, Vc[1:] / Vc[:-1] - 1]
            tot[idx] += rc; cnt[idx] += 1; live[idx] += (ar <= k[None, :]).mean(axis=1)
        r = pd.Series(np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0), index=dates)
        cash = pd.Series(np.where(cnt > 0, 1 - live / np.maximum(cnt, 1), 1.0), index=dates)
        return r, cash

    def cr(members):
        SIG = {t + 1: jj for t, jj in members.items()}
        cash = 1.0; pos = {}; eq = np.full(T, np.nan); cs = np.full(T, np.nan); nh = np.zeros(T)
        for t in range(min(SIG), T):
            if t in SIG:
                s = [j for j in SIG[t] if j not in pos]
                if s and cash > 1e-9:
                    a = cash / len(s)
                    for j in s:
                        px = OA[t, j] * (1 + SLIP); pos[j] = dict(sh=a * (1 - FEE) / px, px=px, t0=t)
                    cash = 0.0
            for j in [j for j, p in pos.items() if np.isfinite(Cf[t, j]) and (Cf[t, j] < p['px'] * (1 - STOP) or t - p['t0'] + 1 >= H)]:
                p = pos.pop(j); cash += p['sh'] * Cf[t, j] * (1 - FEE - TAX - SLIP)
            val = cash + sum(p['sh'] * Cf[t, j] for j, p in pos.items() if np.isfinite(Cf[t, j]))
            eq[t] = val; cs[t] = cash / val; nh[t] = len(pos)
        return pd.Series(eq, index=dates).dropna().pct_change().fillna(0), pd.Series(cs, index=dates)

    OUT = {'D1 重疊月份×不剔除（≈L-126）': jt(MEM['N']), 'D1′ 同上、L-126 原成本寫法': jt(MEM['N'], legacy=True),
           'D2 重疊月份×Q5 剔除': jt(MEM['Q']), 'D3 現金再投入×不剔除': cr(MEM['N']), 'D4 現金再投入×Q5 剔除（≈L-127 B0）': cr(MEM['Q'])}
    L = ['# M-170 結果：L-126 與 L-127 B0 差異拆解\n', '## 各格（年化／最大回撤／Calmar）\n', '| 格 | ' + ' | '.join(s[0] for s in SEGS) + ' | 相對 TR 16–26 | 平均現金比例 |', '|---|---|---|---|---|---|---|']
    for nm, (r, cs) in OUT.items():
        L.append(f'| {nm} | ' + ' | '.join('{:+.1%}／{:.1%}／{:.2f}'.format(*seg(r, lo, hi)) for _, lo, hi in SEGS) +
                 f" | {seg(r, '2016-01-01', '2026-12-31')[0] - seg(tr, '2016-01-01', '2026-12-31')[0]:+.1%} | {cs.loc['2005':].mean():.0%} |")
    L.append('| 加權報酬指數 TR | ' + ' | '.join('{:+.1%}／{:.1%}／{:.2f}'.format(*seg(tr, lo, hi)) for _, lo, hi in SEGS) + ' | — | — |')
    a = lambda k, lo, hi: seg(OUT[k][0], lo, hi)[0]
    k1, k2, k3, k4 = list(OUT)[0], list(OUT)[2], list(OUT)[3], list(OUT)[4]
    L += ['', '## 效果拆解（年化差，pt）\n', '| 效果 | ' + ' | '.join(s[0] for s in SEGS) + ' |', '|---|---|---|---|---|']
    for nm, (x, y) in {'Q5 剔除（重疊月份下：D2−D1）': (k2, k1), 'Q5 剔除（現金再投入下：D4−D3）': (k4, k3),
                       '機制（不剔除下：D3−D1）': (k3, k1), '機制（剔除下：D4−D2）': (k4, k2), '合計（D4−D1）': (k4, k1)}.items():
        L.append(f'| {nm} | ' + ' | '.join(f'{(a(x, lo, hi) - a(y, lo, hi)) * 100:+.1f}' for _, lo, hi in SEGS) + ' |')
    # 每年
    yr = lambda r: (1 + r.loc['2005':]).groupby(r.loc['2005':].index.year).prod() - 1
    Y = pd.DataFrame({k.split(' ')[0]: yr(v[0]) for k, v in OUT.items()}); Y['TR'] = yr(tr)
    L += ['', '## 每年報酬\n', '| 年 | ' + ' | '.join(Y.columns) + ' |', '|' + '---|' * (len(Y.columns) + 1)]
    for y, row in Y.iterrows(): L.append(f'| {y} | ' + ' | '.join(f'{x:+.1%}' for x in row) + ' |')
    hl = f"機制 {(a(k3, '2016-01-01', '2026-12-31') - a(k1, '2016-01-01', '2026-12-31')) * 100:+.1f}pt、Q5 {(a(k2, '2016-01-01', '2026-12-31') - a(k1, '2016-01-01', '2026-12-31')) * 100:+.1f}pt（2016–26）"
    return dict(summary='\n'.join(L), headline=hl, n_variants=0)
