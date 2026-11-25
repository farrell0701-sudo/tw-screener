# L-133：X1 候選剔除距 250 日高點最遠的後段（機制驗證）。判準見 registry_L.md L-133。
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim, evalu as E

NPATH = 100; SEGS = ('2005–15', '2016–20', '2021–26', '2016–26', '全期')
LOG = []
def log(s): LOG.append(s); print(s, flush=True)


def go(sig):
    R = sim.run_paths('fix', NPATH, sig=sig); M = [sim.mix(R['X'][i][0], R['V'][i][0]) for i in range(NPATH)]
    return dict(M=M, t=E.seg_table(M))


def main():
    P = load_panel(); F = load_feat('AF')
    signals.build('fix', mode='fix', P=P, F=F)
    C = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill(); H = pd.DataFrame(np.asarray(F['HA'], np.float64)).ffill()
    signals.EXTRA['DH'] = [(C / H.rolling(250, min_periods=200).max() - 1).to_numpy(np.float32)]
    for q in (0.2, 0.3): signals.build(f'L133_{int(q * 100)}', mode='fix', P=P, F=F, x_rev=('filter', q, 'DH'))
    B = go('fix'); O = {'DH20（主）': go('L133_20'), 'DH30': go('L133_30')}
    bt, bpm = B['t'], E.path_mean(B['M'])
    log('| 版本 | ' + ' | '.join(SEGS) + ' | 05-15 回撤 | P（16-26） |\n|---|---|---|---|---|---|---|---|')
    Pv = {}
    for nm, r in [('新基準', B)] + list(O.items()):
        t = r['t']; p = '' if nm == '新基準' else f"{E.boot_p(E.path_mean(r['M']), bpm):.2f}"
        if p: Pv[nm] = float(p)
        log(f"| {nm} | " + ' | '.join(f"{t.loc[s, 'Calmar']:.2f}（{t.loc[s, '年化']:.1%}）" for s in SEGS) + f" | {t.loc['2005–15', '回撤']:.1%} | {p} |")
    mt = O['DH20（主）']['t']
    c1 = all(mt.loc[s, 'Calmar'] > bt.loc[s, 'Calmar'] for s in ('2016–20', '2021–26')); c2 = Pv['DH20（主）'] >= 0.8
    c3 = mt.loc['2005–15', '回撤'] >= bt.loc['2005–15', '回撤'] - 0.05
    plat = all(O['DH30']['t'].loc[s, 'Calmar'] >= bt.loc[s, 'Calmar'] for s in ('2016–20', '2021–26'))
    veto = all(O[k]['t'].loc['2005–15', 'Calmar'] < bt.loc['2005–15', 'Calmar'] for k in O)
    ok = c1 and c2 and c3 and plat and not veto
    v = (f"判準：Calmar 兩段上升 {'✓' if c1 else '✗'}；P {Pv['DH20（主）']:.2f} {'✓' if c2 else '✗'}；05-15 回撤 {'✓' if c3 else '✗'}；高原 {'✓' if plat else '✗'}；否決 {'觸發' if veto else '未觸發'} → {'通過' if ok else '不採用'}")
    log('\n' + v)
    return dict(summary='# L-133 結果\n\n**' + v + '**\n\n' + '\n'.join(LOG), headline=v, n_variants=2,
                series={'base': bpm, **{k: E.path_mean(x['M']) for k, x in O.items()}})
