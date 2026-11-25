# L-144：X1 停損寬度的 ATR 倍數 k。定義見 registry_L.md L-144。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim, evalu as E

NPATH = 100; END = '2024-11-13'
SEGS = (('2005–15', '2005-01-01', '2015-12-31'), ('2016–20', '2016-01-01', '2020-12-31'), ('2021–24/11', '2021-01-01', END), ('16–24/11', '2016-01-01', END))
KS = (2, 3, 4, 5, 6)


def med(paths, lo, hi): return np.nanmedian(np.array([E.met(p.loc[lo:hi]) for p in paths]), axis=0)


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    S = signals.compute('fix', P=P, F=F); RKX = np.asarray(S['RKX']); below = S['daily']['below'].to_numpy(bool); cols = S['_dbg']['cols']
    if RKX.shape[1] != N:
        o = np.full((T, N), np.nan); o[:, cols] = RKX; RKX = o
    del S
    CA = np.where(np.asarray(F['VALID']), np.asarray(F['CA'], np.float64), np.nan); ATRP = np.asarray(F['ATR'], np.float64) / pd.DataFrame(CA).ffill().to_numpy()
    end = int(dates.searchsorted(pd.Timestamp(END), side='right'))
    m = (RKX <= 6) & ~below[:, None]; m[:int(dates.searchsorted(pd.Timestamp('2005-01-01')))] = False; m[end:] = False
    a = ATRP[m]; a = a[np.isfinite(a)]
    clip = {k: ((k * a < 0.04).mean(), (k * a > 0.15).mean()) for k in KS}
    VERS = {f'K{k}': {'k': k} for k in KS}; VERS['K2+2R'] = {'k': 2, 'tgt': (2, None)}; VERS['K3+2R'] = {'k': 3, 'tgt': (2, None)}
    OUT, TR = {}, {}
    for nm, xo in VERS.items():
        R = sim.run_paths('fix', NPATH, sig='fix', x_kw=dict(end=end, xopt=xo), v_kw=dict(end=end))
        X = [R['X'][i][0].loc[:END] for i in range(NPATH)]; V = [R['V'][i][0].loc[:END] for i in range(NPATH)]
        OUT[nm] = ([sim.mix(X[i], V[i]) for i in range(NPATH)], X)
        tr = [x for i in range(NPATH) for x in R['X'][i][1] if x[2] <= pd.Timestamp(END)]
        TR[nm] = (np.mean([x[5] for x in tr]), np.mean([x[4] for x in tr]), np.mean([x[4] > 0 for x in tr]), np.mean([x[4] >= 0.30 for x in tr]))
    L = ['# L-144 結果：X1 停損寬度的 ATR 倍數 k（100 路徑中位數；到 2024-11-13）\n',
         'X1 進場股的 ATR% 中位數 {:.2%}；各 k 卡在下限 4%／上限 15% 的比例：'.format(np.median(a)) + '；'.join(f'k={k} {lo:.0%}／{hi:.0%}' for k, (lo, hi) in clip.items()) + '\n',
         '格式：CAGR／最大回撤／Calmar\n', '| 版本 | ' + ' | '.join(s[0] for s in SEGS) + ' |', '|---|---|---|---|---|']
    for nm, (B, X) in OUT.items():
        L.append(f'| {nm} 合併 | ' + ' | '.join('{:+.1%}／{:.1%}／{:.2f}'.format(*med(B, lo, hi)[:3]) for _, lo, hi in SEGS) + ' |')
        L.append(f'| {nm} X1 單獨 | ' + ' | '.join('{:+.1%}／{:.1%}／{:.2f}'.format(*med(X, lo, hi)[:3]) for _, lo, hi in SEGS) + ' |')
    L += ['', '## X1 交易\n', '| 版本 | 平均持有天數 | 平均報酬 | 勝率 | ≥ +30% |', '|---|---|---|---|---|'] + [f'| {k} | {v[0]:.0f} | {v[1]:+.2%} | {v[2]:.0%} | {v[3]:.1%} |' for k, v in TR.items()]
    b0 = OUT['K4'][0]; res = {}
    L += ['', '## 判準（合併 Calmar 變化：05–15｜16–20｜21–24/11｜16–24/11；P）\n']
    for nm in VERS:
        if nm == 'K4': continue
        b = OUT[nm][0]; dC = [med(b, lo, hi)[2] - med(b0, lo, hi)[2] for _, lo, hi in SEGS]
        pb = E.boot_p(E.path_mean(b), E.path_mean(b0), lo='2016-01-01', hi=END); res[nm] = (dC, pb)
        L.append(f"- {nm}：{'｜'.join(f'{x:+.2f}' for x in dC)}；P {pb:.2f}")
    nb = {2: (3,), 3: (2, 4), 5: (4, 6), 6: (5,)}; verdict = []
    for k in (2, 3, 5, 6):
        dC, pb = res[f'K{k}']
        c4 = all(np.sign(res[f'K{q}'][0][3]) == np.sign(dC[3]) for q in nb[k] if q != 4)
        c = [dC[1] > 0 and dC[2] > 0, dC[0] >= -0.05, pb >= 0.8, c4]
        ok = all(c); verdict.append(f"K{k} {'採用' if ok else '不採用'}")
        L.append(f"\nK{k}：(1) {'✓' if c[0] else '✗'} (2) {'✓' if c[1] else '✗'} (3) {'✓' if c[2] else '✗'} (4) {'✓' if c[3] else '✗'} → **{'採用' if ok else '不採用'}**")
    return dict(summary='\n'.join(L), headline='；'.join(verdict), n_variants=4)
