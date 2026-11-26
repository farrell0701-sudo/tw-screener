# L-156：X1 依安靜度減碼。判準見 registry_L.md L-156。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim, evalu as E, M190, L131, L153

N_DSR = 713; CACHE = LAB / 'cache' / 'L156'
VARS = {'S50（主）': 0.5, 'S25': 0.25, 'S75': 0.75}


def main():
    P = load_panel(); F = load_feat('AF'); T, N = P['C'].shape
    signals.build('fix', mode='fix', P=P, F=F)
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; cols = dbg['cols']
    def full(a, fill):
        a = np.asarray(a)
        if a.shape[1] == N: return a
        o = np.full((T, N), fill, a.dtype); o[:, cols] = a; return o
    pool = full(dbg['pool'], False); Tr = full(np.asarray(dbg['Tr'], np.float64), np.nan); del S
    FE = M190.features(P, F, Tr)
    pr = lambda x: pd.DataFrame(np.where(pool, np.asarray(x, np.float64), np.nan)).rank(axis=1, pct=True).to_numpy()
    qt = -(pr(FE['VC']) + pr(FE['VX'])) / 2; qp = pr(qt); loud = np.isfinite(qp) & (qp < 0.2)
    CACHE.mkdir(parents=True, exist_ok=True); paths = {}
    for nm, s in VARS.items():
        p = CACHE / f'{nm[:3]}.npy'; np.save(p, np.where(loud, s, 1.0).astype(np.float32)); paths[nm] = str(p)
    B = L153.go('新基準'); O = {nm: L153.go(nm, size=paths[nm]) for nm in VARS}
    SEGS = L131.SEGS; bt, bpm = B['t'], E.path_mean(B['M']); Lg = []
    Lg += ['## 合併版（Calmar（年化）；100 路徑中位數）\n', '| 版本 | ' + ' | '.join(SEGS) + ' | 05-15 回撤 | 16-26 回撤 | X1 平均持股比例 | P（16-26） |', '|---|---|---|---|---|---|---|---|---|---|']
    Pv = {}
    for nm, r in [('新基準', B)] + list(O.items()):
        t = r['t']; p = '' if nm == '新基準' else f"{E.boot_p(E.path_mean(r['M']), bpm):.2f}"
        if p: Pv[nm] = float(p)
        Lg.append(f"| {nm} | " + ' | '.join(f"{t.loc[s, 'Calmar']:.2f}（{t.loc[s, '年化']:.1%}）" for s in SEGS) + f" | {t.loc['2005–15', '回撤']:.1%} | {t.loc['2016–26', '回撤']:.1%} | {r['ex']:.0%} | {p} |")
    Lg += ['\n## X1 單獨（Calmar／年化／回撤）\n', '| 版本 | ' + ' | '.join(SEGS[:4]) + ' |', '|---|---|---|---|---|']
    for nm, r in [('新基準', B)] + list(O.items()):
        Lg.append(f"| {nm} | " + ' | '.join(f"{r['tx'].loc[s, 'Calmar']:.2f}／{r['tx'].loc[s, '年化']:.1%}／{r['tx'].loc[s, '回撤']:.1%}" for s in SEGS[:4]) + ' |')
    yb = bpm.groupby(bpm.index.year).apply(lambda r: (1 + r).prod() - 1); ys = [y for y in yb.index if y >= 2005]
    Lg += ['\n## 合併版逐年報酬差（相對新基準）\n', '| 版本 | ' + ' | '.join(str(y) for y in ys) + ' |', '|' + '---|' * (len(ys) + 1)]
    for nm, r in O.items():
        m = E.path_mean(r['M']); ym = m.groupby(m.index.year).apply(lambda x: (1 + x).prod() - 1)
        Lg.append(f'| {nm[:3]} | ' + ' | '.join(f'{ym[y] - yb[y]:+.1%}' for y in ys) + ' |')
    sr = [E.path_mean(O[k]['M']).loc['2005':].pipe(lambda r: r.mean() / r.std()) for k in VARS]
    mt = O['S50（主）']['t']
    c1 = all(mt.loc[s, 'Calmar'] > bt.loc[s, 'Calmar'] for s in ('2016–20', '2021–26')); c2 = Pv['S50（主）'] >= 0.8
    c3 = mt.loc['2005–15', '回撤'] >= bt.loc['2005–15', '回撤'] - 0.05
    plat = any(all(O[k]['t'].loc[s, 'Calmar'] >= bt.loc[s, 'Calmar'] for s in ('2016–20', '2021–26')) for k in ('S25', 'S75'))
    veto = all(O[k]['t'].loc['2005–15', 'Calmar'] < bt.loc['2005–15', 'Calmar'] for k in VARS)
    ok = c1 and c2 and c3 and plat and not veto
    ds = E.dsr(E.path_mean(O['S50（主）']['M']).loc['2005':], N_DSR, float(np.var(sr, ddof=1)))
    Lg.append(f"\n## 判準\n\n- Calmar 兩段上升 {'✓' if c1 else '✗'}；P {Pv['S50（主）']:.2f} {'✓' if c2 else '✗'}；05-15 回撤 {'✓' if c3 else '✗'}；高原 {'✓' if plat else '✗'}；否決 {'觸發' if veto else '未觸發'}；DSR {ds:.2f} → **{'通過' if ok else '不採用'}**")
    head = f"L-156 X1 依安靜度減碼：{'通過' if ok else '不採用'}（S50 P {Pv['S50（主）']:.2f}）"
    return dict(summary='# L-156 結果\n\n**' + head + '**\n\n' + '\n'.join(Lg), headline=head, n_variants=len(VARS))
