# M-196：回檔後創 5／10 日新高再進場（止跌確認）（奇數代碼、排除最近兩年）。定義見 registry_M.md M-196。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals
import M191
from M195 import pct_in, HOLDOUT, ERAS, KS, COST


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    signals.build('fix', mode='fix', P=P, F=F)
    S_ = signals.compute('fix', P=P, F=F); dbg = S_['_dbg']; cols = dbg['cols']
    def full(a, fill):
        a = np.asarray(a)
        if a.shape[1] == N: return a
        o = np.full((T, N), fill, a.dtype); o[:, cols] = a; return o
    U = full(dbg['U'], False); Tr = full(np.asarray(dbg['Tr'], np.float64), np.nan); del S_
    VALID = np.asarray(F['VALID']); HA = np.asarray(F['HA'], np.float64); LA = np.asarray(F['LA'], np.float64); OA = np.asarray(F['OA'], np.float64)
    Cdf = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill(); Cf = Cdf.to_numpy()
    ATR = M191.extra(P, F)['ATR20']
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CX = np.cumsum(R - ew[:, None], axis=0)
    odd = (P['meta']['stock_id'].astype(str).str[-1].astype(int) % 2 == 1).to_numpy()
    dT60 = Tr - np.vstack([np.full((60, N), np.nan), Tr[:-60]]); strong = pct_in(dT60, U) >= 0.9
    t0 = dates.searchsorted(pd.Timestamp('2005-01-01')); t1 = dates.searchsorted(HOLDOUT)
    base_mask = U & VALID & strong & odd[None, :]; base_mask[:t0] = False; base_mask[t1:] = False; base_mask[T - 75:] = False
    q_cache = {}
    def q1090(t):
        if t not in q_cache: q_cache[t] = np.nanpercentile(CX[t + 60, U[t]] - CX[t, U[t]], [10, 90])
        return q_cache[t]
    def out_from(s, j):   # 起點 s（收盤）→ 超額；s+1 開盤 → 淨報酬
        ent = OA[s + 1, j] if np.isfinite(OA[s + 1, j]) else Cf[s + 1, j]
        e60 = CX[s + 60, j] - CX[s, j]; q = q1090(s)
        return dict(n20=Cf[s + 20, j] / ent - 1 - COST, n60=Cf[s + 60, j] / ent - 1 - COST, e60=e60, left=float(e60 <= q[0]), right=float(e60 >= q[1]))
    samp = np.zeros(T, bool); samp[t0::5] = True
    tt, jj = np.nonzero(base_mask & samp[:, None])
    A0 = pd.DataFrame([dict(t=t, **out_from(t, j)) for t, j in zip(tt, jj)]); A0['date'] = dates[A0.t.to_numpy()]
    rows = []
    for k in KS:
        ema = Cdf.ewm(span=k, adjust=False, min_periods=k).mean().to_numpy()
        prior = pd.DataFrame(Cf).shift(1).rolling(10, min_periods=8).max().to_numpy()
        cond = base_mask & (LA <= ema + 0.5 * ATR) & (Cf >= ema - 0.5 * ATR) & (prior >= ema + ATR)
        recent = pd.DataFrame(cond.astype(float)).shift(1).rolling(10, min_periods=1).sum().to_numpy() > 0
        for t, j in zip(*np.nonzero(cond & ~recent)):
            ev = out_from(t, j); r = dict(k=k, t=t, j=j, tld=Tr[t, j], **{f'ev_{a}': b for a, b in ev.items()})
            for w in (5, 10):
                d = None
                for dd in range(t + 1, t + 11):
                    if not VALID[dd, j]: continue
                    hh = HA[dd - w:dd, j]
                    if np.isfinite(hh).sum() >= w - 1 and Cf[dd, j] > np.nanmax(hh): d = dd; break
                if d is None: r[f'c{w}'] = False; continue
                o = out_from(d, j); r[f'c{w}'] = True; r[f'c{w}_days'] = d - t; r[f'c{w}_gone'] = Cf[d, j] / Cf[t, j] - 1
                r.update({f'c{w}_{a}': b for a, b in o.items()})
            rows.append(r)
    D = pd.DataFrame(rows); D['date'] = dates[D.t.to_numpy()]
    L = ['# M-196 結果：回檔後創 5／10 日新高再進場（奇數代碼、排除最近兩年）\n',
         '事件日進場＝t+1 開盤；確認進場＝確認日 d 的次日開盤；淨報酬扣 0.77%；左尾／右尾以各自起點當日可交易池 60 日超額 10%／90% 分位。\n']
    res = {}
    for era, lo, hi in ERAS:
        E = D[(D.date >= lo) & (D.date <= hi)]; a0 = A0[(A0.date >= lo) & (A0.date <= hi)]
        L += [f'## {era}\n', f"A0 強勢母體：20 日淨勝率 {(a0.n20 > 0).mean():.0%}、60 日超額 {a0.e60.mean():+.2%}、左尾 {a0.left.mean():.1%}、右尾 {a0.right.mean():.1%}\n",
              '| 事件 | 確認 | 事件數 | 確認比例 | 確認天數（中位） | t→d 已漲（中位） | 事件日進場：20日淨 平均／中位／勝率 | 確認進場：20日淨 平均／中位／勝率 | 事件日 60日淨 平均／中位／勝率 | 確認 60日淨 平均／中位／勝率 | 60日超額 事件日／確認 | 左尾 事件日／確認／未確認 | 右尾 事件日／確認／未確認 | T_ld 組內勝率差 |',
              '|' + '---|' * 14]
        for k in KS:
            x = E[E.k == k].copy()
            x['q'] = x.groupby(x.date.dt.year).tld.transform(lambda s: pd.qcut(s.rank(method='first'), 5, labels=False) + 1 if len(s) >= 5 else np.nan)
            for w in (5, 10):
                c = x[x[f'c{w}'] == True]; u = x[x[f'c{w}'] != True]
                wd = c.groupby('q').apply(lambda g: (g[f'c{w}_n20'] > 0).mean() - (g.ev_n20 > 0).mean()).mean()
                res[(k, w, era)] = dict(wc=(c[f'c{w}_n20'] > 0).mean(), we=(c.ev_n20 > 0).mean(), lc=c[f'c{w}_left'].mean(), le=c.ev_left.mean(),
                                        ec=c[f'c{w}_e60'].mean(), a0=a0.e60.mean(), wd=wd)
                f3 = lambda s: f"{s.mean():+.2%}／{s.median():+.2%}／{(s > 0).mean():.0%}"
                L.append(f"| E{k} | C{w} | {len(x):,} | {len(c) / len(x):.0%} | {c[f'c{w}_days'].median():.0f} | {c[f'c{w}_gone'].median():+.1%} | {f3(c.ev_n20)} | {f3(c[f'c{w}_n20'])} | "
                         f"{f3(c.ev_n60)} | {f3(c[f'c{w}_n60'])} | {c.ev_e60.mean():+.2%}／{c[f'c{w}_e60'].mean():+.2%} | {c.ev_left.mean():.1%}／{c[f'c{w}_left'].mean():.1%}／{u.ev_left.mean():.1%} | "
                         f"{c.ev_right.mean():.1%}／{c[f'c{w}_right'].mean():.1%}／{u.ev_right.mean():.1%} | {wd * 100:+.1f}pt |")
            u = x[x.c5 != True]
            L.append(f"| E{k} | 未確認（C5） | {len(u):,} | | | | {f3(u.ev_n20)} | | {f3(u.ev_n60)} | | {u.ev_e60.mean():+.2%} | {u.ev_left.mean():.1%} | {u.ev_right.mean():.1%} | |")
        L.append('')
    vt = []
    for k in KS:
        for w in (5, 10):
            a = [res[(k, w, e)] for e, _, _ in ERAS]
            c1 = all(x['wc'] > x['we'] for x in a); c2 = all(x['lc'] < x['le'] for x in a); c3 = all(x['ec'] >= x['a0'] for x in a); c4 = all(x['wd'] > 0 for x in a)
            vt.append(f"- E{k}×C{w}：①勝率 {'✓' if c1 else '✗'}（" + '｜'.join(f"{x['wc']:.0%} vs {x['we']:.0%}" for x in a) + f"）②左尾 {'✓' if c2 else '✗'}（" + '｜'.join(f"{x['lc']:.1%} vs {x['le']:.1%}" for x in a)
                      + f"）③60日超額 ≥ A0 {'✓' if c3 else '✗'}（" + '｜'.join(f"{x['ec']:+.2%} vs {x['a0']:+.2%}" for x in a) + f"）④T_ld 組內 {'✓' if c4 else '✗'}（" + '｜'.join(f"{x['wd'] * 100:+.1f}pt" for x in a)
                      + f"）→ {'提名' if c1 and c2 and c3 and c4 else '不提名'}")
    head = ['## 判準（兩期）\n'] + vt + ['']
    return dict(summary='\n'.join(L[:2] + head + L[2:]), headline='；'.join(v[2:].split('：')[0] + v.split('→')[-1] for v in vt), n_variants=0)
