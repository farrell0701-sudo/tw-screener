# L-143：X1 出場的 R 倍數停利（全標的／只限高波動）、固定持有，以及三帳戶。定義見 registry_L.md L-143。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim, evalu as E

NPATH = 100; END = '2024-11-13'; THR = 0.485
SEGS = (('2005–15', '2005-01-01', '2015-12-31'), ('2016–20', '2016-01-01', '2020-12-31'), ('2021–24/11', '2021-01-01', END), ('16–24/11', '2016-01-01', END))
GROUPS = {'A 全標的': ('R2', 'R1', 'R3'), 'B 只限高波動': ('HR2', 'HR1', 'HR3'), 'C 固定持有': ('F20', 'F10', 'F40')}


def med(paths, lo, hi): return np.nanmedian(np.array([E.met(p.loc[lo:hi]) for p in paths]), axis=0)


def mix3(a, b, v, wa, wb):
    a, b = a.align(b, fill_value=0); a, v = a.align(v, fill_value=0); b = b.reindex(a.index, fill_value=0)
    w = np.array([wa, wb, 1 - wa - wb]); val = w.copy(); out = np.empty(len(a)); m = a.index.to_period('M'); R = np.c_[a.values, b.values, v.values]
    for i in range(len(a)):
        if i > 0 and m[i] != m[i - 1]: val = val.sum() * w
        prev = val.sum(); val = val * (1 + R[i]); out[i] = val.sum() / prev - 1
    return pd.Series(out, index=a.index)


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; cols = dbg['cols']
    U = dbg['U'] if np.asarray(dbg['U']).shape[1] == N else None
    if U is None: U = np.zeros((T, N), bool); U[:, cols] = dbg['U']
    del S
    VALID = np.asarray(F['VALID']); CA = np.where(VALID, np.asarray(F['CA'], np.float64), np.nan)
    C = pd.DataFrame(CA).ffill().to_numpy(); ATRP = np.asarray(F['ATR'], np.float64) / C
    with np.errstate(all='ignore'): ar = pd.DataFrame(np.where(U, ATRP, np.nan)).rank(axis=1, pct=True).to_numpy()
    (LAB / 'cache').mkdir(exist_ok=True); hv = str(LAB / 'cache' / 'L143_hv.npy'); np.save(hv, (np.nan_to_num(ar) >= THR))
    end = int(dates.searchsorted(pd.Timestamp(END), side='right'))
    VERS = {'B0 基準': None, 'R2': {'tgt': (2, None)}, 'R1': {'tgt': (1, None)}, 'R3': {'tgt': (3, None)},
            'HR2': {'tgt': (2, hv)}, 'HR1': {'tgt': (1, hv)}, 'HR3': {'tgt': (3, hv)}, 'F20': {'xfix': 20}, 'F10': {'xfix': 10}, 'F40': {'xfix': 40}}
    OUT, TR, V0 = {}, {}, None
    for nm, xo in VERS.items():
        x_kw = dict(end=end)
        if xo: x_kw['xopt'] = xo
        R = sim.run_paths('fix', NPATH, sig='fix', x_kw=x_kw, v_kw=dict(end=end))
        X = [R['X'][i][0].loc[:END] for i in range(NPATH)]; V = [R['V'][i][0].loc[:END] for i in range(NPATH)]
        if V0 is None: V0 = V
        OUT[nm] = ([sim.mix(X[i], V[i]) for i in range(NPATH)], X)
        tr = [x for i in range(NPATH) for x in R['X'][i][1] if x[2] <= pd.Timestamp(END)]
        TR[nm] = (pd.Series([x[6] for x in tr]).value_counts(normalize=True), np.mean([x[5] for x in tr]), np.mean([x[4] for x in tr]))
    L = ['# L-143 結果：X1 出場的 R 倍數停利、固定持有、三帳戶（100 路徑中位數；到 2024-11-13）\n', '格式：CAGR／最大回撤／Calmar\n',
         '| 版本 | ' + ' | '.join(s[0] for s in SEGS) + ' |', '|---|---|---|---|---|']
    for nm, (B, X) in OUT.items():
        L.append(f'| {nm} 合併 | ' + ' | '.join('{:+.1%}／{:.1%}／{:.2f}'.format(*med(B, lo, hi)[:3]) for _, lo, hi in SEGS) + ' |')
        L.append(f'| {nm} X1 單獨 | ' + ' | '.join('{:+.1%}／{:.1%}／{:.2f}'.format(*med(X, lo, hi)[:3]) for _, lo, hi in SEGS) + ' |')
    L += ['', '## 出場原因與交易（X1；全部路徑）\n', '| 版本 | 平均持有天數 | 平均報酬 | 出場原因 |', '|---|---|---|---|']
    for nm, (vc, hd, mr) in TR.items():
        L.append(f"| {nm} | {hd:.0f} | {mr:+.2%} | " + '；'.join(f'{k} {v:.0%}' for k, v in vc.head(5).items()) + ' |')
    b0 = OUT['B0 基準'][0]; res = {}
    L += ['', '## 判準（合併 Calmar 變化：05–15｜16–20｜21–24/11｜16–24/11；P = bootstrap 16–24/11）\n']
    for nm in VERS:
        if nm == 'B0 基準': continue
        b = OUT[nm][0]; dC = [med(b, lo, hi)[2] - med(b0, lo, hi)[2] for _, lo, hi in SEGS]
        pb = E.boot_p(E.path_mean(b), E.path_mean(b0), lo='2016-01-01', hi=END); res[nm] = (dC, pb)
        L.append(f"- {nm}：{'｜'.join(f'{x:+.2f}' for x in dC)}；P {pb:.2f}")
    verdict = []
    for g, (m, p1, p2) in GROUPS.items():
        dC, pb = res[m]
        c = [dC[1] > 0 and dC[2] > 0, dC[0] >= -0.05, pb >= 0.8, all(np.sign(res[k][0][3]) == np.sign(dC[3]) for k in (p1, p2))]
        ok = all(c); verdict.append(f"{m} {'採用' if ok else '不採用'}")
        L.append(f"\n{g}｜{m}：(1) {'✓' if c[0] else '✗'} (2) {'✓' if c[1] else '✗'} (3) {'✓' if c[2] else '✗'} (4) {'✓' if c[3] else '✗'} → **{'採用' if ok else '不採用'}**")
    xb, xr, xh = E.path_mean(OUT['B0 基準'][1]), E.path_mean(OUT['R2'][1]), E.path_mean(OUT['HR2'][1])
    L += ['', '## 三帳戶（描述）\n', f"X1 與 X-R2 路徑平均日報酬相關：{xb.corr(xr):.2f}；X1 與 X-HR2：{xb.corr(xh):.2f}；X-R2 與 V4.1：{xr.corr(E.path_mean(V0)):.2f}；X1 與 V4.1：{xb.corr(E.path_mean(V0)):.2f}\n",
          '| 組合 | ' + ' | '.join(s[0] for s in SEGS) + ' |', '|---|---|---|---|---|']
    XB, XR = OUT['B0 基準'][1], OUT['R2'][1]
    for nm, wa, wb in (('X1 60／V 40（基準）', 0.6, 0.0), ('X1 40／X-R2 20／V 40', 0.4, 0.2), ('X1 30／X-R2 30／V 40', 0.3, 0.3)):
        B = [mix3(XB[i], XR[i], V0[i], wa, wb) for i in range(NPATH)]
        L.append(f'| {nm} | ' + ' | '.join('{:+.1%}／{:.1%}／{:.2f}'.format(*med(B, lo, hi)[:3]) for _, lo, hi in SEGS) + ' |')
    return dict(summary='\n'.join(L), headline='；'.join(verdict), n_variants=9)
