# M-190：X1 交易速度（每持有日報酬）與進場特徵（奇數代碼、排除最近兩年）。定義見 registry_M.md M-190。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim

HOLDOUT = pd.Timestamp('2024-11-25')
ERAS = (('2005–15', '2005-01-01', '2015-12-31'), ('2016–24', '2016-01-01', '2024-11-24'))
FEATS = ['POS250', 'D60H', 'BRK20', 'VC', 'EXT', 'VX', 'D2', 'OFs']; REFS = ['T_ld', 'ΔT_ld10']


def features(P, F, Tr):
    dates = P['dates']
    C = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill(); H = pd.DataFrame(np.asarray(F['HA'], np.float64)).ffill()
    L = pd.DataFrame(np.asarray(F['LA'], np.float64)).ffill(); A = pd.DataFrame(np.asarray(P['AMT'], np.float64))
    pc = C.shift(1); trr = pd.concat([H - L, (H - pc).abs(), (L - pc).abs()]).groupby(level=0).max()
    atr20 = trr.rolling(20, min_periods=15).mean(); atr60 = trr.rolling(60, min_periods=45).mean()
    out = {}
    lo250, hi250 = L.rolling(250, min_periods=200).min(), H.rolling(250, min_periods=200).max()
    out['POS250'] = (C - lo250) / (hi250 - lo250)
    out['D60H'] = C / H.rolling(60, min_periods=45).max() - 1
    out['BRK20'] = (C - H.shift(1).rolling(20, min_periods=15).max()) / atr20
    out['VC'] = atr20 / atr60
    out['EXT'] = (C - C.ewm(span=20, adjust=False).mean()) / atr20
    out['VX'] = A.rolling(20, min_periods=15).mean() / A.rolling(60, min_periods=45).mean()
    # D2：上月第一個交易日到當日
    mon = pd.PeriodIndex(dates, freq='M'); start = pd.Series(dates.searchsorted((mon - 1).start_time), index=range(len(dates)))
    Cn = C.to_numpy(); D2 = np.full(Cn.shape, np.nan)
    for t in range(len(dates)):
        w = Cn[start[t]:t + 1]
        if len(w) < 15: continue
        with np.errstate(all='ignore'): h, l = np.nanmax(w, 0), np.nanmin(w, 0)
        D2[t] = np.where(h > l, (Cn[t] - (h + l) / 2) / (h - l), np.nan)
    out['D2'] = pd.DataFrame(D2)
    lr, hr = L.shift(1).rolling(20, min_periods=20).min(), H.shift(1).rolling(20, min_periods=20).max()
    inter = (np.minimum(H, hr) - np.maximum(L, lr)).clip(lower=0); uni = np.maximum(H, hr) - np.minimum(L, lr)
    out['OFs'] = (inter / uni.where(uni > 0)).rolling(5, min_periods=5).mean()
    out['T_ld'] = pd.DataFrame(Tr); out['ΔT_ld10'] = pd.DataFrame(Tr) - pd.DataFrame(Tr).shift(10)
    return {k: v.to_numpy(np.float64) for k, v in out.items()}


def yic(df, f):
    s = df.groupby(df.date.dt.year).apply(lambda g: g[f].rank().corr(g.spd.rank()) if g[f].notna().sum() > 20 else np.nan).dropna()
    return s.mean(), s.mean() / s.std() * np.sqrt(len(s)) if len(s) > 2 else np.nan


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    signals.build('fix', mode='fix', P=P, F=F)
    S_ = signals.compute('fix', P=P, F=F); dbg = S_['_dbg']; cols = dbg['cols']
    Tr = np.asarray(dbg['Tr'], np.float64)
    if Tr.shape[1] != N: o = np.full((T, N), np.nan); o[:, cols] = Tr; Tr = o
    del S_
    FE = features(P, F, Tr)
    keys = P['meta']['security_key'].astype(str).to_numpy(); kidx = {k: i for i, k in enumerate(keys)}
    odd = (P['meta']['stock_id'].astype(str).str[-1].astype(int) % 2 == 1).to_numpy()
    B = sim.run_paths('fix', 100, sig='fix')
    rows = {}
    for (r, tr, ex) in B['X']:
        for (code, d0, d1, w, ret, hold, why, st, lk) in tr:
            j = kidx.get(str(code)); d0 = pd.Timestamp(d0)
            if j is None or not odd[j] or d0 >= HOLDOUT or d0 < pd.Timestamp('2005-01-01'): continue
            rows.setdefault((j, d0), (ret, max(int(hold), 1), why))
    D = pd.DataFrame([(j, d0, *v) for (j, d0), v in rows.items()], columns=['j', 'date', 'ret', 'hold', 'why'])
    ti = dates.searchsorted(D.date.to_numpy()) - 1
    for f in FEATS + REFS: D[f] = FE[f][ti, D.j.to_numpy()]
    D['spd'] = D.ret / D.hold
    L = ['# M-190 結果：X1 交易速度與進場特徵（奇數代碼、排除 2024-11-25 之後；100 路徑交易去重）\n',
         '每持有日報酬＝淨報酬 ÷ 持有日數；特徵於進場前一交易日計算；五分組 Q1＝特徵最低。\n']
    res = {}
    for era, lo, hi in ERAS:
        E = D[(D.date >= lo) & (D.date <= hi)].copy()
        E['sp3'] = pd.qcut(E.spd.rank(method='first'), 3, labels=['慢', '中', '快'])
        L += [f'## {era}（{len(E):,} 筆；每持有日 {E.spd.mean():+.3%}；平均 {E.ret.mean():+.2%}、中位 {E.ret.median():+.2%}、勝率 {(E.ret > 0).mean():.0%}、持有 {E.hold.mean():.0f} 日）\n']
        L += ['| 速度組 | 筆數 | 平均淨報酬 | 中位 | 勝率 | 平均持有日 | 出場原因（前三） |', '|---|---|---|---|---|---|---|']
        for g, x in E.groupby('sp3', observed=True):
            top = x.why.value_counts(normalize=True).head(3)
            L.append(f"| {g} | {len(x):,} | {x.ret.mean():+.2%} | {x.ret.median():+.2%} | {(x.ret > 0).mean():.0%} | {x.hold.mean():.0f} | " + '、'.join(f'{k} {v:.0%}' for k, v in top.items()) + ' |')
        L += ['', '| 特徵 | 逐年 IC（t） | Q1 每日 | Q3 每日 | Q5 每日 | Q1 中位 | Q5 中位 | Q1 勝率 | Q5 勝率 | Q1 持有日 | Q5 持有日 | Q1 快組比例 | Q5 快組比例 |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for f in FEATS + REFS:
            x = E.dropna(subset=[f]).copy(); x['q'] = pd.qcut(x[f].rank(method='first'), 5, labels=False) + 1
            ic, t = yic(x, f); g = x.groupby('q')
            res[(f, era)] = dict(ic=ic, t=t, q1=g.spd.mean()[1], q5=g.spd.mean()[5], m1=g.ret.median()[1], m5=g.ret.median()[5], all=E.spd.mean(), mall=E.ret.median())
            L.append(f"| {f} | {ic:+.3f}（{t:.1f}） | {g.spd.mean()[1]:+.3%} | {g.spd.mean()[3]:+.3%} | {g.spd.mean()[5]:+.3%} | {g.ret.median()[1]:+.2%} | {g.ret.median()[5]:+.2%} | "
                     f"{(g.ret.apply(lambda s: (s > 0).mean()))[1]:.0%} | {(g.ret.apply(lambda s: (s > 0).mean()))[5]:.0%} | {g.hold.mean()[1]:.0f} | {g.hold.mean()[5]:.0f} | "
                     f"{(x[x.q == 1].sp3 == '快').mean():.0%} | {(x[x.q == 5].sp3 == '快').mean():.0%} |")
        cm = E[FEATS + REFS].rank().corr()
        L += ['', '特徵間 Spearman 相關（|ρ| ≥ 0.5）：' + '；'.join(f'{a}–{b} {cm.loc[a, b]:+.2f}' for i, a in enumerate(FEATS + REFS) for b in (FEATS + REFS)[i + 1:] if abs(cm.loc[a, b]) >= 0.5) + '\n']
    vt = []
    for f in FEATS:
        a = [res[(f, e)] for e, _, _ in ERAS]
        same = np.sign(a[0]['ic']) == np.sign(a[1]['ic']); ts = sorted(abs(x['t']) for x in a)
        c1 = same and ts[1] >= 2 and ts[0] >= 1
        good = 'q5' if a[0]['ic'] > 0 else 'q1'; mk = 'm5' if good == 'q5' else 'm1'
        c2 = all(x[good] > x['all'] for x in a); c3 = all(x[mk] >= x['mall'] for x in a)
        ics = '｜'.join(f"{x['ic']:+.3f} t {x['t']:.1f}" for x in a); gl = 'Q5' if good == 'q5' else 'Q1'
        vt.append(f"- {f}：①IC {'✓' if c1 else '✗'}（{ics}）②好組({gl})速度 {'✓' if c2 else '✗'} ③好組中位數 {'✓' if c3 else '✗'} → {'提名' if c1 and c2 and c3 else '不提名'}")
    head = ['## 判準（特徵 1–8；兩期）\n'] + vt + ['']
    return dict(summary='\n'.join(L[:2] + head + L[2:]), headline='；'.join(v[2:].split('：')[0] + v.split('→')[-1] for v in vt), n_variants=0)
