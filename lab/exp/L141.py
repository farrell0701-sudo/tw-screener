# L-141：X1「太抖」降權（ATR% 在 U 內排名 ≥ 0.485 → 部位 × w）。定義見 registry_L.md L-141。
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


def med(paths, lo, hi): return np.nanmedian(np.array([E.met(p.loc[lo:hi]) for p in paths]), axis=0)


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; cols = dbg['cols']
    U = np.zeros((T, N), bool); U[:, cols] = dbg['U'] if dbg['U'].shape[1] != N else dbg['U'][:, cols]
    if dbg['U'].shape[1] == N: U = dbg['U']
    del S
    VALID = np.asarray(F['VALID']); CA = np.where(VALID, np.asarray(F['CA'], np.float64), np.nan)
    C = pd.DataFrame(CA).ffill().to_numpy(); ATRP = np.asarray(F['ATR'], np.float64) / C
    with np.errstate(all='ignore'): ar = pd.DataFrame(np.where(U, ATRP, np.nan)).rank(axis=1, pct=True).to_numpy()
    flag = np.nan_to_num(ar) >= THR
    end = int(dates.searchsorted(pd.Timestamp(END), side='right'))
    (LAB / 'cache').mkdir(exist_ok=True)
    OUT, fr = {}, flag[:end].sum() / max(np.isfinite(ar[:end]).sum(), 1)
    for nm, w in (('B0 基準', None), ('W50', 0.5), ('W33', 1 / 3), ('W67', 2 / 3), ('W0 剔除（描述）', 0.0)):
        x_kw = dict(end=end)
        if w is not None:
            pth = str(LAB / 'cache' / f'L141_w{int(round(w * 100))}.npy')
            np.save(pth, np.where(flag, w, 1.0).astype(np.float32)); x_kw['xopt'] = {'xsize': pth}
        R = sim.run_paths('fix', NPATH, sig='fix', x_kw=x_kw, v_kw=dict(end=end))
        X = [R['X'][i][0].loc[:END] for i in range(NPATH)]; V = [R['V'][i][0].loc[:END] for i in range(NPATH)]
        OUT[nm] = ([sim.mix(X[i], V[i]) for i in range(NPATH)], X)
    L = ['# L-141 結果：X1「太抖」降權（100 路徑中位數；到 2024-11-13）\n', f'U 內被標記比例 {fr:.0%}（全部股票，非只 X1 候選）\n',
         '格式：CAGR／最大回撤／Calmar\n', '| 版本 | ' + ' | '.join(s[0] for s in SEGS) + ' |', '|---|---|---|---|---|']
    for nm, (B, X) in OUT.items():
        L.append(f'| {nm} 合併 | ' + ' | '.join('{:+.1%}／{:.1%}／{:.2f}'.format(*med(B, lo, hi)[:3]) for _, lo, hi in SEGS) + ' |')
        L.append(f'| {nm} X1 單獨 | ' + ' | '.join('{:+.1%}／{:.1%}／{:.2f}'.format(*med(X, lo, hi)[:3]) for _, lo, hi in SEGS) + ' |')
    b0 = OUT['B0 基準'][0]
    L += ['', '## 判準（合併 Calmar 變化：05–15｜16–20｜21–24/11｜16–24/11；P = bootstrap 16–24/11）\n']
    res = {}
    for nm in ('W50', 'W33', 'W67', 'W0 剔除（描述）'):
        b = OUT[nm][0]; dC = [med(b, lo, hi)[2] - med(b0, lo, hi)[2] for _, lo, hi in SEGS]
        pb = E.boot_p(E.path_mean(b), E.path_mean(b0), lo='2016-01-01', hi=END); res[nm] = (dC, pb)
        L.append(f"- {nm}：{'｜'.join(f'{x:+.2f}' for x in dC)}；P {pb:.2f}")
    dC, pb = res['W50']
    c = [dC[1] > 0 and dC[2] > 0, dC[0] >= -0.05, pb >= 0.8, all(np.sign(res[k][0][3]) == np.sign(dC[3]) for k in ('W33', 'W67'))]
    ok = all(c)
    L += ['', f"W50：(1) {'✓' if c[0] else '✗'} (2) {'✓' if c[1] else '✗'} (3) {'✓' if c[2] else '✗'} (4) {'✓' if c[3] else '✗'} → **{'採用' if ok else '不採用'}**"]
    return dict(summary='\n'.join(L), headline=f"W50 {'採用' if ok else '不採用'}：Δ {'｜'.join(f'{x:+.2f}' for x in dC[:3])}，P {pb:.2f}", n_variants=4,
                series={k: E.path_mean(v[0]) for k, v in OUT.items()})
