# M-173：營收池 × X1 式子策略的大回撤分析（描述性）。定義見 registry_M.md M-173。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim, evalu as E
from M144 import weekly_gate

NPATH = 100


def episodes(r, k=8):
    eq = (1 + r).cumprod(); out = []; used = np.zeros(len(eq), bool); idx = eq.index
    dd = eq / eq.cummax() - 1
    for _ in range(k):
        x = dd.where(~used)
        if x.isna().all() or x.min() > -0.03: break
        tr = x.idxmin(); ti = idx.get_loc(tr)
        pk = eq.iloc[:ti + 1].idxmax(); pi = idx.get_loc(pk)
        after = eq.iloc[ti:]; rec = after[after >= eq.loc[pk]]
        ri = idx.get_loc(rec.index[0]) if len(rec) else len(eq) - 1
        used[pi:ri + 1] = True; out.append((pk, tr, idx[ri], float(dd.loc[tr]), bool(len(rec))))
    return out


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    keys = P['meta']['security_key'].astype(str).to_numpy()
    ix = P['index']['TAIEX_P'].reindex(dates).ffill(); g21 = weekly_gate(ix, dates, 21)
    tr = P['index']['TAIEX_TR'].reindex(dates).ffill().pct_change().fillna(0)
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; U, pool = dbg['U'], dbg['pool']; del S
    VALID = np.asarray(F['VALID']); CA = np.asarray(F['CA'], np.float64)
    Cf = pd.DataFrame(CA).ffill().to_numpy()
    with np.errstate(all='ignore'): R = Cf / np.vstack([np.full((1, N), np.nan), Cf[:-1]]) - 1
    R[~VALID] = 0.0; R[np.isnan(R)] = 0.0
    sh = lambda m: np.vstack([np.zeros((1, N), bool), m[:-1]])
    with np.errstate(all='ignore'):
        ewo = pd.Series(np.nan_to_num(np.nanmean(np.where(sh(U & ~pool), R, np.nan), axis=1)), index=dates)
        ewu = pd.Series(np.nan_to_num(np.nanmean(np.where(sh(U), R, np.nan), axis=1)), index=dates)
    RB = sim.run_paths('fix', NPATH); B = E.path_mean([sim.mix(RB['X'][i][0], RB['V'][i][0]) for i in range(NPATH)])
    RUNS = {}
    for nm, kw in (('E21', {'n': 6, 'gate': g21}), ('200MA', {'n': 6})):
        Rr = sim.run_paths('fix', NPATH, sig='L131_rev', x_kw=kw, v_kw={})
        paths = [Rr['X'][i][0] for i in range(NPATH)]
        trades = pd.DataFrame([x for i in range(NPATH) for x in Rr['X'][i][1]], columns=['code', 'd0', 'd1', 'wt', 'ret', 'hold', 'why', 'st', 'lk'])
        expo = pd.concat([Rr['X'][i][2] for i in range(NPATH)], axis=1).mean(axis=1)
        RUNS[nm] = (paths, E.path_mean(paths), trades, expo)
    L = ['# M-173 結果：營收池 × X1 式子策略的大回撤分析\n',
         '分類：(a) 大盤急跌＝TR 同期 ≤ −10%；(b) 小型股落後＝等權池外 − TR ≤ −10pt；(c) 其他。報酬欄皆為「高點 → 低點」期間。\n']
    for nm, (paths, Sm, trades, expo) in RUNS.items():
        Sm = Sm.loc['2005':]
        L += [f'## {nm} 子策略：最深回撤段（路徑平均淨值）\n',
              '| # | 高點 | 低點 | 收復 | S 跌幅 | 中位數路徑／最差路徑 | 天數（到低點） | B | TR | 等權池外 | 等權全體 | 閘門為負 | S 曝險 | 出場筆數（每路徑）／每筆平均 | 主要出場原因 | 分類 |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for k, (pk, trd, rc, dd, ok) in enumerate(episodes(Sm), 1):
            w = slice(pk, trd); ret = lambda s: float((1 + s.loc[w]).prod() - 1)
            pd_ = [float((1 + p.loc[w]).prod() - 1) for p in paths]
            tx = trades[(trades.d1 > pk) & (trades.d1 <= trd)]
            why = tx.why.value_counts(normalize=True).head(2)
            rt, ro = ret(tr), ret(ewo)
            cls = '(a) 大盤急跌' if rt <= -0.10 else ('(b) 小型股落後' if ro - rt <= -0.10 else '(c) 其他')
            if rt <= -0.10 and ro - rt <= -0.10: cls = '(a)＋(b)'
            gi = g21[dates.get_loc(pk):dates.get_loc(trd) + 1].mean()
            L.append(f"| {k} | {pk:%Y-%m-%d} | {trd:%Y-%m-%d} | {rc:%Y-%m-%d}{'' if ok else '（未收復）'} | {dd:.1%} | {np.median(pd_):.1%}／{min(pd_):.1%} | {dates.get_loc(trd) - dates.get_loc(pk)} | "
                     f"{ret(B):+.1%} | {rt:+.1%} | {ro:+.1%} | {ret(ewu):+.1%} | {gi:.0%} | {expo.loc[w].mean():.0%} | {len(tx) / NPATH:.1f}／{tx.ret.mean():+.1%} | "
                     + '、'.join(f'{a} {b:.0%}' for a, b in why.items()) + f' | {cls} |')
        L.append('')
    # 最差月份（E21）
    Sm = RUNS['E21'][1]
    mo = lambda s: (1 + s.loc['2005':]).groupby(s.loc['2005':].index.to_period('M')).prod() - 1
    Mtab = pd.DataFrame({'S E21': mo(Sm), 'S 200MA': mo(RUNS['200MA'][1]), 'B': mo(B), 'TR': mo(tr), '等權池外': mo(ewo), '等權全體': mo(ewu)})
    w12 = Mtab.nsmallest(12, 'S E21')
    L += ['## S（E21）最差 12 個月\n', '| 月份 | ' + ' | '.join(Mtab.columns) + ' |', '|' + '---|' * (len(Mtab.columns) + 1)]
    for m, row in w12.iterrows(): L.append(f'| {m} | ' + ' | '.join(f'{x:+.1%}' for x in row) + ' |')
    # 年度：小型股相對大型股
    yr = lambda s: (1 + s.loc['2005':]).groupby(s.loc['2005':].index.year).prod() - 1
    Ytab = pd.DataFrame({'S E21': yr(Sm), 'B': yr(B), 'TR': yr(tr), '等權池外': yr(ewo)}); Ytab['等權池外−TR'] = Ytab['等權池外'] - Ytab['TR']; Ytab['S−B'] = Ytab['S E21'] - Ytab['B']
    L += ['', '## 年度：S、B 與大小型股\n', '| 年 | ' + ' | '.join(Ytab.columns) + ' |', '|' + '---|' * (len(Ytab.columns) + 1)]
    for y, row in Ytab.iterrows(): L.append(f'| {y} | ' + ' | '.join(f'{x:+.1%}' for x in row) + ' |')
    c = Ytab['S−B'].corr(Ytab['等權池外−TR'])
    L += ['', f'年度「S − B」與「等權池外 − TR」的相關：{c:.2f}']
    return dict(summary='\n'.join(L), headline='營收子策略回撤分析完成', n_variants=0)
