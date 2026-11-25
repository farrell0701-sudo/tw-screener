# M-182：R6 頭部以月初強勢確認（K1 S、K2 PRE 前 40%）或剔除月初弱勢（X1 剔 Q1）（奇數代碼）。定義見 registry_M.md M-182。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'exp')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core'))
import M180
from panel import load as load_panel
from feat import load_feat
import signals
from M107 import nw_t

ERAS = (('2005–15', '2005-01-01', '2015-12-31'), ('2016–26', '2016-01-01', '2026-12-31'))
COST = 0.0077


def build(parity=1):
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    S = signals.compute('fix', P=P, F=F); U = S['_dbg']['U']; del S
    Cf = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill().to_numpy(); LA = np.asarray(F['LA'], np.float64)
    OA = np.asarray(F['OA'], np.float64); VALID = np.asarray(F['VALID']); R20 = np.asarray(F['R20'], np.float64)
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    XRv = np.where(VALID, R - ew[:, None], np.nan); CX = np.cumsum(R - ew[:, None], axis=0)
    keys = P['meta']['security_key'].astype(str).to_numpy()
    odd = (P['meta']['stock_id'].astype(str).str[-1].astype(int) % 2 == parity).to_numpy()
    R6 = M180.r6_factors()['R6 營收短長比']; jmap = pd.Index(R6.columns.astype(str)).get_indexer(keys)
    ds = pd.Series(dates, index=dates); samp = ds[ds.dt.day > 10].groupby(ds[ds.dt.day > 10].dt.to_period('M')).min()
    samp = [dates.get_loc(d) for d in samp if d >= pd.Timestamp('2005-01-01') and dates.get_loc(d) + 122 < T]
    rows = []
    for t in samp:
        m = dates[t].to_period('M') - 1
        if m not in R6.index: continue
        s0 = dates.searchsorted(dates[t].to_period('M').start_time); t0 = s0 - 1
        if t - s0 < 2 or t0 < 61: continue
        fv = R6.loc[m].to_numpy(float); x = np.where(jmap >= 0, fv[np.clip(jmap, 0, None)], np.nan)
        sel = U[t] & odd & np.isfinite(x) & np.isfinite(Cf[t])
        if sel.sum() < 50: continue
        jj = np.nonzero(sel)[0]
        e60all = CX[t + 60, jj] - CX[t, jj]; q10, q90 = np.nanpercentile(e60all, [10, 90])
        top = pd.Series(x[jj]).rank(pct=True).to_numpy() > 0.8
        jj = jj[top]
        pre = CX[t - 1, jj] - CX[t0, jj]
        sd = np.nanstd(XRv[t0 - 59:t0 + 1, jj], axis=0)
        z = XRv[s0:t, jj] / np.where(sd > 0, sd, np.nan); has = (z >= 2).any(0)
        q = pd.qcut(pd.Series(pre).rank(method='first'), 5, labels=False).to_numpy() + 1
        with np.errstate(all='ignore'):
            lo20 = np.nanmin(LA[t + 1:t + 21, jj], axis=0) / Cf[t, jj] - 1
            ent = np.where(np.isfinite(OA[t + 1, jj]), OA[t + 1, jj], Cf[t + 1, jj])
            C = Cf[t + 1:t + 121, jj]; hit = C < ent * 0.85; any_ = hit.any(0); k = hit.argmax(0)
            ex_open = OA[t + 2 + k, jj]; ex_open = np.where(np.isfinite(ex_open), ex_open, Cf[t + 2 + k, jj])
            exitp = np.where(any_, ex_open, Cf[t + 120, jj])
            net = exitp / ent - 1 - COST
            net_ns = Cf[t + 120, jj] / ent - 1 - COST; net_ns60 = Cf[t + 60, jj] / ent - 1 - COST
        for i, j in enumerate(jj):
            rows.append(dict(t=t, j=int(j), q=q[i], has=has[i], pre=pre[i], r20=R20[t, j],
                             e20=CX[t + 20, j] - CX[t, j], e60=CX[t + 60, j] - CX[t, j], e120=CX[t + 120, j] - CX[t, j],
                             left=(CX[t + 60, j] - CX[t, j]) <= q10, right=(CX[t + 60, j] - CX[t, j]) >= q90,
                             lo20=lo20[i], net=net[i], stop=bool(any_[i]), net_ns=net_ns[i], net_ns60=net_ns60[i]))
    D = pd.DataFrame(rows); D['date'] = dates[D.t.to_numpy()]
    return D


def main():
    D = build()
    V = {'A0 R6頭部全部': lambda d: d, 'K1 確認（S）': lambda d: d[d.has], 'K2 確認（PRE 前40%）': lambda d: d[d.q >= 4],
         'X1 剔除（Q1）': lambda d: d[d.q >= 2], '（被剔除的 Q1）': lambda d: d[d.q == 1]}
    L = ['# M-182 結果：R6 頭部 × 月初強勢確認／月初弱勢剔除（奇數代碼；F 計數 60）\n']
    res = {}
    for era, lo, hi in ERAS:
        E = D[(D.date >= lo) & (D.date <= hi)].dropna(subset=['e60'])
        base = E.groupby('t').e60.mean()
        L += [f'## {era}\n', f"PRE 與 R20 的截面等級相關（R6 頭部內）平均 {E.groupby('t').apply(lambda g: g.pre.rank().corr(g.r20.rank())).mean():+.2f}\n",
              '| 版本 | 每期檔數 | 後20日 | 後60日 | 後120日 | 60日中位 | 60日勝率 | 左尾率 | 右尾率 | 20日最大跌幅 | 交易淨報酬 平均 | 中位 | 勝率 | 停損率 | 最差5% | 60日 vs A0（t） |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for nm, f in V.items():
            g = f(E); d = (g.groupby('t').e60.mean() - base).dropna()
            w5 = g.net.quantile(0.05)
            res[(nm, era)] = dict(diff=d.mean(), t=nw_t(d, 3) if nm[:2] != 'A0' else np.nan, right=g.right.mean(), med=g.net.median())
            L.append(f"| {nm} | {g.groupby('t').size().mean():.0f} | {g.e20.mean():+.2%} | {g.e60.mean():+.2%} | {g.e120.mean():+.2%} | {g.e60.median():+.2%} | {(g.e60 > 0).mean():.0%} | "
                     f"{g.left.mean():.1%} | {g.right.mean():.1%} | {g.lo20.mean():+.2%} | {g.net.mean():+.2%} | {g.net.median():+.2%} | {(g.net > 0).mean():.0%} | {g.stop.mean():.0%} | "
                     f"{g.net[g.net <= w5].mean():+.1%} | {'' if nm[:2] == 'A0' else f'{d.mean():+.2%}（{nw_t(d, 3):.1f}）'} |")
        L.append('')
    vt = []
    for nm in ('K1 確認（S）', 'K2 確認（PRE 前40%）', 'X1 剔除（Q1）'):
        a = [res[(nm, e)] for e, _, _ in ERAS]; b = [res[('A0 R6頭部全部', e)] for e, _, _ in ERAS]
        ts = sorted(x['t'] for x in a)
        c1 = all(x['diff'] > 0 for x in a) and ts[1] >= 2 and ts[0] >= 1
        c2 = all(x['right'] >= y['right'] - 0.01 for x, y in zip(a, b))
        c3 = all(x['med'] > y['med'] for x, y in zip(a, b))
        vt.append(f"- {nm}：①60日超額 {'✓' if c1 else '✗'}（{'｜'.join(f'{x['diff']:+.2%} t {x['t']:.1f}' for x in a)}）②右尾 {'✓' if c2 else '✗'} ③交易中位數 {'✓' if c3 else '✗'} → {'提名' if c1 and c2 and c3 else '不提名'}")
    head = ['## 判準\n'] + vt + ['']
    return dict(summary='\n'.join(L[:1] + head + L[1:]), headline='；'.join(v.split('：')[0][2:] + v.split('→')[-1] for v in vt), n_variants=0)
