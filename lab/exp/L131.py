# L-131：營收池 × X1 式子策略（L-109 第一、二關；正式引擎 100 路徑）。判準見 registry_L.md L-131。
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim, evalu as E
from signals import prk, rank_desc
from M122 import fin_factors
import M165

NPATH = 100; N_DSR = 604
SEGS = (('2005–15', '2005-01-01', '2015-12-31'), ('2016–20', '2016-01-01', '2020-12-31'), ('2021–26', '2021-01-01', '2026-12-31'), ('2016–26', '2016-01-01', '2026-12-31'), ('全期', '2005-01-01', '2026-12-31'))


def met_seg(r, lo, hi): return E.met(r.loc[lo:hi])


def med(paths, lo, hi):
    M = np.array([met_seg(p, lo, hi) for p in paths]); return np.nanmedian(M, axis=0)


def mix_w(b, s, wfun):
    """b、s 日報酬；每月第一個交易日把 s 權重調回 wfun(該日)；其餘時間各自漂移。"""
    b, s = b.align(s, fill_value=0); m = b.index.to_period('M'); out = np.empty(len(b)); vb, vs = 1.0, 0.0; bv, sv = b.values, s.values
    for i in range(len(b)):
        if i == 0 or m[i] != m[i - 1]:
            w = wfun(i); tot = vb + vs; vs, vb = tot * w, tot * (1 - w)
        prev = vb + vs; vb *= 1 + bv[i]; vs *= 1 + sv[i]; out[i] = (vb + vs) / prev - 1
    return pd.Series(out, index=b.index)


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    keys = P['meta']['security_key'].astype(str).to_numpy()
    signals.build('fix', mode='fix', P=P, F=F)
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; U, pool, Tr = dbg['U'], dbg['pool'], dbg['Tr']; cols = dbg['cols']
    assert S['RKX'].shape == (T, N)
    if U.shape[1] != N:
        def full(a, fill): o = np.full((T, N), fill, a.dtype); o[:, cols] = a; return o
        U, pool, Tr = full(U, False), full(pool, False), full(Tr, np.nan)
    dT = Tr - np.vstack([np.full((10, N), np.nan), Tr[:-10]])
    A = M165.layers(U, pool, dates, keys)[0][1]
    Q5 = fin_factors(dates)['Q5 毛利率創高']; qj = pd.Index(Q5.columns.astype(str)).get_indexer(keys)
    cand = np.zeros((T, N), bool); sp = sorted(M165.samp_prev(dates).items(), key=lambda x: x[1][0])
    for i, (m, (t, _)) in enumerate(sp):
        if not A[t].any(): continue
        mem = np.nonzero(A[t])[0]
        q = np.where(qj[mem] >= 0, Q5.loc[m].to_numpy(float)[np.clip(qj[mem], 0, None)], np.nan) if m in Q5.index else np.full(len(mem), np.nan)
        fin = np.isfinite(q); drop = np.zeros(len(mem), bool)
        if fin.sum() >= 20: drop[fin] = q[fin] <= np.nanpercentile(q[fin], 20)
        t2 = sp[i + 1][1][0] if i + 1 < len(sp) else T
        cand[t:t2, mem[~drop]] = True
    cand &= U
    sc = 0.5 * np.nan_to_num(prk(Tr, cand)) + 0.5 * np.nan_to_num(prk(dT, cand))
    RK = rank_desc(sc, cand)
    d = signals.OUT / 'L131_rev'; d.mkdir(parents=True, exist_ok=True)
    np.save(d / 'RKX.npy', RK.astype(np.float32)); np.save(d / 'RKV.npy', np.asarray(S['RKV'])); np.save(d / 'ATRP.npy', np.asarray(S['ATRP']))
    S['daily'].reset_index(names='date').to_parquet(d / 'daily.parquet', index=False)
    (d / 'info.json').write_text(json.dumps(dict(name='L131_rev', note='L-131 營收池候選'), ensure_ascii=False), encoding='utf-8')
    pool_overlap = float((cand & pool).sum()) / max(1, cand.sum())
    del S
    tr = P['index']['TAIEX_TR'].reindex(dates).ffill().pct_change().fillna(0)
    RB = sim.run_paths('fix', NPATH)
    B = [sim.mix(RB['X'][i][0], RB['V'][i][0]) for i in range(NPATH)]
    OUT = {}
    for n in (6, 4, 10):
        R = sim.run_paths('fix', NPATH, sig='L131_rev', x_kw={'n': n}, v_kw={})
        OUT[n] = ([R['X'][i][0] for i in range(NPATH)], [R['X'][i][1] for i in range(NPATH)])
    L = ['# L-131 結果：營收池 × X1 式子策略（100 路徑中位數）\n', f'候選與強勢池重疊比例：{pool_overlap:.1%}（應為 0）\n',
         '## 第一關：單獨表現（年化／回撤／Calmar；相對 TR 年化超額）\n', '| 版本 | ' + ' | '.join(s[0] for s in SEGS) + ' | 每年交易 |', '|---|---|---|---|---|---|---|']
    g1 = {}
    for n, (paths, trs) in OUT.items():
        cells = []
        for sn, lo, hi in SEGS:
            a, dd, cm, _ = med(paths, lo, hi); ta = E.met(tr.loc[lo:hi])[0]
            cells.append(f'{a:+.1%}／{dd:.1%}／{cm:.2f}／{a - ta:+.1%}'); g1[(n, sn)] = a - ta
        ntr = np.mean([len([x for x in t if x[1] >= pd.Timestamp('2005-01-01')]) for t in trs]) / ((dates[-1] - pd.Timestamp('2005-01-01')).days / 365.25)
        L.append(f'| N＝{n}{"（主）" if n == 6 else ""} | ' + ' | '.join(cells) + f' | {ntr:.0f} |')
    L.append('| 新基準合併版 B | ' + ' | '.join('{:+.1%}／{:.1%}／{:.2f}／—'.format(*med(B, lo, hi)[:3]) for _, lo, hi in SEGS) + ' | — |')
    L.append('| 加權報酬指數 TR | ' + ' | '.join('{:+.1%}／{:.1%}／{:.2f}／—'.format(*E.met(tr.loc[lo:hi])[:3]) for _, lo, hi in SEGS) + ' | — |')
    pass1 = all(g1[(n, s)] > 0 for n in OUT for s in ('2005–15', '2016–26'))
    L += ['', f"**第一關：{'通過' if pass1 else '不通過'}**（三版本 2005–15 與 2016–26 相對 TR 超額皆 > 0）\n"]
    # 第二關
    Sp = OUT[6][0]
    Bm, Sm = E.path_mean(B), E.path_mean(Sp)
    vb = Bm.rolling(250, min_periods=120).std().shift(1); vs = Sm.rolling(250, min_periods=120).std().shift(1)
    wr = (1 / vs) / (1 / vs + 1 / vb); wr = wr.fillna(0.2).clip(0, 1)
    WF = {'等風險（主）': lambda i: float(wr.iloc[i]), '固定 10%': lambda i: 0.10, '固定 20%': lambda i: 0.20}
    L += ['## 第二關' + ('' if pass1 else '（預覽，第一關未過不判定）') + '：B ＋ S（年化／回撤／Calmar，100 路徑中位數）\n', '| 組合 | ' + ' | '.join(s[0] for s in SEGS) + ' |', '|---|---|---|---|---|---|']
    L.append('| B | ' + ' | '.join('{:+.1%}／{:.1%}／{:.2f}'.format(*med(B, lo, hi)[:3]) for _, lo, hi in SEGS) + ' |')
    g2 = {}; CMB = {}
    for nm, wf in WF.items():
        C = [mix_w(B[i], Sp[i], wf) for i in range(NPATH)]; CMB[nm] = C
        L.append(f'| B＋S {nm} | ' + ' | '.join('{:+.1%}／{:.1%}／{:.2f}'.format(*med(C, lo, hi)[:3]) for _, lo, hi in SEGS) + ' |')
        c1 = med(C, '2016-01-01', '2020-12-31')[2] - med(B, '2016-01-01', '2020-12-31')[2]
        c2 = med(C, '2021-01-01', '2026-12-31')[2] - med(B, '2021-01-01', '2026-12-31')[2]
        pb = E.boot_p(E.path_mean(C), Bm); dd = med(C, '2005-01-01', '2015-12-31')[1] - med(B, '2005-01-01', '2015-12-31')[1]
        c0 = med(C, '2005-01-01', '2015-12-31')[2] - med(B, '2005-01-01', '2015-12-31')[2]
        g2[nm] = (c1, c2, pb, dd, c0)
    L += ['', '| 組合 | Calmar Δ 16–20 | Δ 21–26 | P（16–26） | 05–15 回撤變化 | 05–15 Calmar Δ | 結論 |', '|---|---|---|---|---|---|---|']
    for nm, (c1, c2, pb, dd, c0) in g2.items():
        ok = c1 > 0 and c2 > 0 and pb >= 0.8 and dd >= -0.05
        L.append(f"| {nm} | {c1:+.2f} | {c2:+.2f} | {pb:.2f} | {dd * 100:+.1f}pt | {c0:+.2f} | {('**通過**' if ok else '不通過') if (nm.startswith('等風險') and pass1) else ('只報告' if pass1 else '預覽')} |")
    # 分散檢查
    mb = (1 + Bm).groupby(Bm.index.to_period('M')).prod() - 1; ms = (1 + Sm).groupby(Sm.index.to_period('M')).prod() - 1
    mb = mb.loc['2005':]; ms = ms.reindex(mb.index)
    worst = mb[mb <= mb.quantile(0.1)].index
    eqb = (1 + Bm).cumprod(); ddb = eqb / eqb.cummax() - 1; inDD = ddb < -0.05
    L += ['', '## 分散檢查\n', f'- 全期日報酬相關（路徑平均）：{Bm.loc["2005":].corr(Sm.loc["2005":]):.2f}；B 回撤 > 5% 期間相關：{Bm[inDD].corr(Sm[inDD]):.2f}',
          f'- B 最差 10% 月份（{len(worst)} 個月）：B 平均 {mb[worst].mean():+.2%}、S 平均 {ms[worst].mean():+.2%}（L-109：S 須高於 B）',
          f'- 等風險權重：平均 {wr.loc["2005":].mean():.0%}（範圍 {wr.loc["2005":].min():.0%}～{wr.loc["2005":].max():.0%}）']
    yr = lambda r: (1 + r.loc['2005':]).groupby(r.loc['2005':].index.year).prod() - 1
    Y = pd.DataFrame({'S（N6，路徑平均）': yr(Sm), 'B': yr(Bm), 'B＋S 等風險': yr(E.path_mean(CMB['等風險（主）'])), 'TR': yr(tr)})
    L += ['', '## 每年報酬（路徑平均）\n', '| 年 | ' + ' | '.join(Y.columns) + ' |', '|---|---|---|---|---|']
    for y, row in Y.iterrows(): L.append(f'| {y} | ' + ' | '.join(f'{x:+.1%}' for x in row) + ' |')
    hl = f"第一關{'通過' if pass1 else '不通過'}；第二關等風險 Calmar Δ {g2['等風險（主）'][0]:+.2f}／{g2['等風險（主）'][1]:+.2f}、P {g2['等風險（主）'][2]:.2f}"
    return dict(summary='\n'.join(L), headline=hl, n_variants=3, series={'S_N6': Sm, 'B': Bm})
