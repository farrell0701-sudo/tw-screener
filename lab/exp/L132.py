# L-132：L-131 穩健性——位階回看期間 120／500 日。判準見 registry_L.md L-132。
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim, evalu as E

NPATH = 100
SEGS = ('2005–15', '2016–20', '2021–26', '2016–26', '全期')
LOG = []
def log(s): LOG.append(s); print(s, flush=True)


def go(name, sig):
    t0 = time.time(); R = sim.run_paths('fix', NPATH, sig=sig)
    M = [sim.mix(R['X'][i][0], R['V'][i][0]) for i in range(NPATH)]
    log(f'  （{name}：{time.time() - t0:.0f} 秒）'); return dict(M=M, t=E.seg_table(M))


def pos(F, W):
    C = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill(); H = pd.DataFrame(np.asarray(F['HA'], np.float64)).ffill(); L = pd.DataFrame(np.asarray(F['LA'], np.float64)).ffill()
    lo, hi = L.rolling(W, min_periods=int(W * 0.8)).min(), H.rolling(W, min_periods=int(W * 0.8)).max()
    return ((C - lo) / (hi - lo)).to_numpy(np.float32)


def main():
    P = load_panel(); F = load_feat('AF')
    signals.build('fix', mode='fix', P=P, F=F)
    for W in (120, 500):
        signals.EXTRA[f'POS{W}'] = [pos(F, W)]
        signals.build(f'L132_{W}', mode='fix', P=P, F=F, x_rev=('filter', 0.2, f'POS{W}'))
    B = go('新基準', 'fix'); O = {f'POS{W} 剔後20%': go(f'POS{W}', f'L132_{W}') for W in (120, 500)}
    bt, bpm = B['t'], E.path_mean(B['M'])
    log('## 合併版（Calmar（年化））\n')
    log('| 版本 | ' + ' | '.join(SEGS) + ' | P（16-26） |\n|---|---|---|---|---|---|---|')
    Pv = {}
    for nm, r in [('新基準', B)] + list(O.items()):
        t = r['t']; p = '' if nm == '新基準' else f"{E.boot_p(E.path_mean(r['M']), bpm):.2f}"
        if p: Pv[nm] = float(p)
        log(f"| {nm} | " + ' | '.join(f"{t.loc[s, 'Calmar']:.2f}（{t.loc[s, '年化']:.1%}）" for s in SEGS) + f" | {p} |")
    z = np.load(LAB / 'results' / 'L131' / 'series.npz', allow_pickle=True)
    idx = pd.to_datetime(z['dates'], unit='D'); names = list(z['names'])
    b = pd.Series(z['values'][:, names.index('base')], index=idx); p20 = pd.Series(z['values'][:, names.index('P20（主）')], index=idx)
    yb = (1 + b).groupby(idx.year).prod() - 1; yp = (1 + p20).groupby(idx.year).prod() - 1; dy = (yp - yb).loc[2005:]
    log('\n## L-131 P20 逐年合併報酬差（P20 − 新基準）\n')
    log('| ' + ' | '.join(str(y) for y in dy.index) + ' |\n|' + '---|' * len(dy) + '\n| ' + ' | '.join(f'{v:+.1%}' for v in dy) + ' |')
    log(f"\n正差年數 {int((dy > 0).sum())}/{len(dy)}；最大三年貢獻占總差 {dy.sort_values(ascending=False).head(3).sum() / dy.sum():.0%}（總差 {dy.sum():+.1%}）")
    ok = any(all(O[k]['t'].loc[s, 'Calmar'] >= bt.loc[s, 'Calmar'] for s in ('2016–20', '2021–26')) and Pv[k] >= 0.5 for k in O)
    v = f"判準：{'✓ 高原（至少一個期間版本兩段不低於新基準且 P ≥ 0.5）' if ok else '✗ 脆弱（L-131 降為待觀察）'}"
    log('\n' + v)
    return dict(summary='# L-132 結果\n\n**' + v + '**\n\n' + '\n'.join(LOG), headline=v, n_variants=2)
