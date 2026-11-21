# M-161：R4 高分股內，籌碼（散戶變化 B4、當沖比 D1）能否減左尾而不傷右尾（奇數代碼）。定義與判準見 registry_M.md M-161。
import sys
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals
from M107 import nw_t
from M120 import rev_factors
from M127 import tdcc_factors
from M138 import dt_factors
from M159 import tiered

TABS = (('表一 散戶變化 B4', 'B4', ('散戶減', '中', '散戶增'), (('2010–18', '2010-01-01', '2018-12-31'), ('2019–26', '2019-01-01', '2026-12-31')), '散戶增'),
        ('表二 當沖比 D1', 'D1', ('低', '中', '高'), (('2016–20', '2016-01-01', '2020-12-31'), ('2021–26', '2021-01-01', '2026-12-31')), '高'))


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; del S
    U, pool, Tr = dbg['U'], dbg['pool'], dbg['Tr']; assert Tr.shape[1] == N
    dT = (pd.DataFrame(Tr) - pd.DataFrame(Tr).shift(10)).to_numpy()
    Cf = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill().to_numpy(); LA = np.asarray(F['LA'], np.float64); VALID = np.asarray(F['VALID'])
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    Ush = np.vstack([np.zeros((1, N), bool), U[:-1]])
    with np.errstate(all='ignore'): ew = np.nan_to_num(np.nanmean(np.where(Ush, R, np.nan), axis=1))
    CX = np.cumsum(R - ew[:, None], axis=0)
    keys = P['meta']['security_key'].astype(str).to_numpy()
    odd = (P['meta']['stock_id'].astype(str).str[-1].astype(int) % 2 == 1).to_numpy()
    def getter(W):
        jm = pd.Index(W.columns.astype(str)).get_indexer(keys)
        return lambda m: (np.where(jm >= 0, W.loc[m].to_numpy(float)[np.clip(jm, 0, None)], np.nan) if m in W.index else np.full(N, np.nan))
    G = {'R4': getter(rev_factors(dates)['R4 營收創高']), 'B4': getter(tdcc_factors(dates)['B4 散戶 1 月變化']), 'D1': getter(dt_factors(dates)['D1 當沖比 20 日'])}
    ds = pd.Series(dates, index=dates); d10 = ds[ds.dt.day > 10]; samp = d10.groupby(d10.dt.to_period('M')).min()
    samp = [dates.get_loc(d) for d in samp if d >= pd.Timestamp('2010-01-01') and dates.get_loc(d) + 60 < T]
    rows = []   # 每個 (t, 集合, 變數, 個股)
    for t in samp:
        m = dates[t].to_period('M') - 1
        r4 = G['R4'](m); jp = np.nonzero(pool[t] & odd)[0]
        if len(jp) < 30: continue
        e60 = CX[t + 60] - CX[t]
        with np.errstate(all='ignore'): dn = np.nanmin(LA[t + 1:t + 61], axis=0) / Cf[t] - 1
        q10, q90 = np.nanpercentile(e60[jp], [10, 90]); mdn = np.nanmedian(dn[jp])
        jr = jp[np.isfinite(r4[jp])]
        if len(jr) < 25: continue
        g5 = pd.qcut(pd.Series(r4[jr]).rank(method='first').to_numpy(), 5, labels=False); A = jr[g5 == 4]
        sc = tiered(r4[jp], dT[t, jp]); B = jp[np.argsort(-sc, kind='stable')[:12]]
        for var in ('B4', 'D1'):
            v = G[var](m)
            for sn, sel in (('A 池內 R4 前 20%', A), ('B R4 先排前 12', B)):
                s2 = sel[np.isfinite(v[sel])]
                if len(s2) < 6: continue
                g3 = pd.qcut(pd.Series(v[s2]).rank(method='first').to_numpy(), 3, labels=False)
                for j, g in zip(s2, g3):
                    rows.append((t, sn, var, int(g), e60[j], e60[j] <= q10, e60[j] >= q90, dn[j] - mdn))
    D = pd.DataFrame(rows, columns=['t', 'set', 'var', 'g', 'e60', 'left', 'right', 'dn']); D['date'] = dates[D.t]
    D['e60'] = D.e60.astype(float)
    def stats(x):
        v = x.e60.dropna(); top = np.sort(v.values)[-max(1, len(v) // 10):]
        share = top.sum() / v.sum() if v.sum() > 0 else np.nan
        return dict(n=len(x), e60=v.mean(), left=x.left.mean(), right=x.right.mean(), dn=x.dn.mean(), share=share)
    L = ['# M-161 結果：R4 高分股內，籌碼能否減左尾而不傷右尾（奇數代碼）\n',
         '左尾／右尾＝落在同期整個強勢池後 60 日超額最差／最好 10%（整池＝10%）；最大跌幅扣同期強勢池中位數；「前 10% 占比」＝該格前 10% 個股占該格總超額。\n']
    verdict = {}
    for title, var, gnames, eras, drop in TABS:
        L.append(f'# {title}\n')
        for sn in ('A 池內 R4 前 20%', 'B R4 先排前 12'):
            L += [f'## {sn}\n', '| 期間 | 組 | 觀測數 | 後 60 日超額 | 與集合平均之差（t） | 左尾 | 右尾 | 最大跌幅 | 前 10% 占比 |', '|---|---|---|---|---|---|---|---|---|']
            okall = []
            for en, lo, hi in eras:
                E = D[(D['var'] == var) & (D['set'] == sn) & (D.date >= lo) & (D.date <= hi)]
                if not len(E): continue
                setm = E.groupby('t').e60.mean()
                for k, gn in enumerate(gnames):
                    x = E[E.g == k]; s = stats(x)
                    d = (x.groupby('t').e60.mean() - setm).dropna()
                    sh = f"{s['share']:.0%}" if np.isfinite(s['share']) else '（≤ 0）'
                    L.append(f"| {en} | {gn} | {s['n']} | {s['e60']:+.2%} | {d.mean():+.2%}（{nw_t(d, 3):.1f}） | {s['left']:.0%} | {s['right']:.0%} | {s['dn']:+.1%} | {sh} |")
                s0 = stats(E); s1 = stats(E[E.g != gnames.index(drop)])
                sh0 = f"{s0['share']:.0%}" if np.isfinite(s0['share']) else '—'
                L.append(f"| {en} | **原集合** | {s0['n']} | {s0['e60']:+.2%} | | {s0['left']:.0%} | {s0['right']:.0%} | {s0['dn']:+.1%} | {sh0} |")
                L.append(f"| {en} | **剔除{drop}後** | {s1['n']} | {s1['e60']:+.2%} | | {s1['left']:.0%} | {s1['right']:.0%} | {s1['dn']:+.1%} | |")
                dl, dr, de = (s1['left'] - s0['left']) * 100, (s1['right'] - s0['right']) * 100, (s1['e60'] - s0['e60']) * 100
                c = [dl <= -3, dr >= -1, de >= 0]; okall.append(c)
                L.append(f"| {en} | 剔除後變化 | | {de:+.2f}pt | | {dl:+.1f}pt {'✓' if c[0] else '✗'} | {dr:+.1f}pt {'✓' if c[1] else '✗'} | | 平均 {'✓' if c[2] else '✗'} |")
            full = len(okall) == 2 and all(all(c) for c in okall); only1 = len(okall) == 2 and all(c[0] for c in okall)
            v = '成立（減左尾、不傷右尾）' if full else ('減左尾但付出右尾或平均' if only1 else '不成立')
            verdict[(var, sn[0])] = v
            L.append(f'\n**判讀：{v}**\n')
    head = '；'.join(f"{k[0]}-{k[1]} {v[:3]}" for k, v in verdict.items())
    return dict(summary='\n'.join(L), headline=head, n_variants=0)
