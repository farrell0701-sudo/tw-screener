# M-192：重做 M-190，改用快速成功、MFE／MAE、達標天數（奇數代碼、排除最近兩年）。定義見 registry_M.md M-192。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim
import M190, M191

FEATS = M190.FEATS + ['BBWp', 'OBVD']; REFS = M190.REFS; ERAS = M190.ERAS


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    signals.build('fix', mode='fix', P=P, F=F)
    S_ = signals.compute('fix', P=P, F=F); dbg = S_['_dbg']; cols = dbg['cols']
    Tr = np.asarray(dbg['Tr'], np.float64)
    if Tr.shape[1] != N: o = np.full((T, N), np.nan); o[:, cols] = Tr; Tr = o
    del S_
    FE = M190.features(P, F, Tr); EX = M191.extra(P, F); FE['BBWp'] = EX['BBWp']; FE['OBVD'] = EX['OBVD']
    HA = np.asarray(F['HA'], np.float64); LA = np.asarray(F['LA'], np.float64); OA = np.asarray(F['OA'], np.float64)
    Cf = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill().to_numpy()
    keys = P['meta']['security_key'].astype(str).to_numpy(); kidx = {k: i for i, k in enumerate(keys)}
    odd = (P['meta']['stock_id'].astype(str).str[-1].astype(int) % 2 == 1).to_numpy()
    B = sim.run_paths('fix', 100, sig='fix')
    rows = {}
    for (r, tr, ex) in B['X']:
        for (code, d0, d1, w, ret, hold, why, st, lk) in tr:
            j = kidx.get(str(code)); d0 = pd.Timestamp(d0)
            if j is None or not odd[j] or d0 >= M190.HOLDOUT or d0 < pd.Timestamp('2005-01-01'): continue
            rows.setdefault((j, d0), (ret, max(int(hold), 1)))
    D = pd.DataFrame([(j, d0, *v) for (j, d0), v in rows.items()], columns=['j', 'date', 'ret', 'hold'])
    te = dates.searchsorted(D.date.to_numpy()); ts = te - 1; jj = D.j.to_numpy()
    for f in FEATS + REFS: D[f] = FE[f][ts, jj]
    base = np.where(np.isfinite(OA[te, jj]), OA[te, jj], Cf[te, jj]); atr = EX['ATR20'][ts, jj]
    ok = np.zeros(len(D)); mfe = np.full(len(D), np.nan); mae = np.full(len(D), np.nan); days = np.full(len(D), np.nan)
    for i in range(len(D)):
        if te[i] + 60 >= T or not np.isfinite(atr[i]) or atr[i] <= 0: continue
        o, a, b, _ = M191.quick(base[i:i + 1], atr[i:i + 1], HA[te[i]:te[i] + 20, jj[i]:jj[i] + 1], LA[te[i]:te[i] + 20, jj[i]:jj[i] + 1])
        ok[i], mfe[i], mae[i] = o[0], a[0], b[0]
        h60 = HA[te[i]:te[i] + 60, jj[i]] >= base[i] + 2 * atr[i]
        days[i] = h60.argmax() + 1 if h60.any() else np.nan
    D['ok'], D['mfe'], D['mae'], D['days'] = ok, mfe, mae, days
    D = D[np.isfinite(D.mfe)]
    L = ['# M-192 結果：X1 交易的速度（快速成功／MFE／MAE／達標天數）與進場特徵（奇數代碼、排除最近兩年）\n']
    res = {}
    for era, lo, hi in ERAS:
        E = D[(D.date >= lo) & (D.date <= hi)]
        L += [f"## {era}（{len(E):,} 筆；快速成功 {E.ok.mean():.1%}；60 日內達 +2ATR {E.days.notna().mean():.0%}、中位 {E.days.median():.0f} 日；MFE {E.mfe.mean():.2f}、MAE {E.mae.mean():.2f}）\n",
              f"快速成功者淨報酬 中位 {E[E.ok == 1].ret.median():+.2%}、勝率 {(E[E.ok == 1].ret > 0).mean():.0%}；未成功者 中位 {E[E.ok == 0].ret.median():+.2%}、勝率 {(E[E.ok == 0].ret > 0).mean():.0%}\n",
              '| 特徵 | 逐年 IC 快速成功（t） | Q1 快速成功 | Q5 快速成功 | Q1 中位淨報酬 | Q5 中位淨報酬 | Q1 MFE／MAE | Q5 MFE／MAE | Q1 達標中位天數 | Q5 達標中位天數 |', '|' + '---|' * 10]
        for f in FEATS + REFS:
            x = E.dropna(subset=[f]).copy(); x['q'] = pd.qcut(x[f].rank(method='first'), 5, labels=False) + 1
            s = x.groupby(x.date.dt.year).apply(lambda g: g[f].rank().corr(g.ok.rank()) if len(g) > 20 else np.nan).dropna()
            ic, t = s.mean(), s.mean() / s.std() * np.sqrt(len(s)) if len(s) > 2 else np.nan
            g = x.groupby('q')
            res[(f, era)] = dict(ic=ic, t=t, ok1=g.ok.mean()[1], ok5=g.ok.mean()[5], m1=g.ret.median()[1], m5=g.ret.median()[5], okall=E.ok.mean(), mall=E.ret.median())
            L.append(f"| {f} | {ic:+.3f}（{t:.1f}） | {g.ok.mean()[1]:.1%} | {g.ok.mean()[5]:.1%} | {g.ret.median()[1]:+.2%} | {g.ret.median()[5]:+.2%} | "
                     f"{g.mfe.mean()[1]:.2f}／{g.mae.mean()[1]:.2f} | {g.mfe.mean()[5]:.2f}／{g.mae.mean()[5]:.2f} | {g.days.median()[1]:.0f} | {g.days.median()[5]:.0f} |")
        L.append('')
    vt = []
    for f in FEATS:
        a = [res[(f, e)] for e, _, _ in ERAS]
        same = np.sign(a[0]['ic']) == np.sign(a[1]['ic']); tt = sorted(abs(x['t']) for x in a)
        c1 = same and tt[1] >= 2 and tt[0] >= 1; hg = a[0]['ic'] + a[1]['ic'] > 0
        c2 = all((x['ok5'] if hg else x['ok1']) > x['okall'] for x in a); c3 = all((x['m5'] if hg else x['m1']) >= x['mall'] for x in a)
        ics = '｜'.join(f"{x['ic']:+.3f} t {x['t']:.1f}" for x in a)
        vt.append(f"- {f}：①IC {'✓' if c1 else '✗'}（{ics}）②好組({'Q5' if hg else 'Q1'})快速成功 {'✓' if c2 else '✗'} ③中位數 {'✓' if c3 else '✗'} → {'提名' if c1 and c2 and c3 else '不提名'}")
    head = ['## 判準（兩期）\n'] + vt + ['']
    return dict(summary='\n'.join(L[:1] + head + L[1:]), headline='；'.join(v[2:].split('：')[0] + v.split('→')[-1] for v in vt), n_variants=0)
