# L-134：L-131 P20 在奇數／偶數代碼兩半各自是否成立。判準見 registry_L.md L-134。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim, evalu as E
import M190

NPATH = 100; SEGS = ('2005–15', '2016–20', '2021–26', '2016–26')
LOG = []
def log(s): LOG.append(s); print(s, flush=True)


def main():
    P = load_panel(); F = load_feat('AF'); T, N = P['C'].shape
    signals.build('fix', mode='fix', P=P, F=F)
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; cols = dbg['cols']; Tr0 = np.asarray(dbg['Tr'], np.float64); del S
    if Tr0.shape[1] == N: Tr = Tr0
    else: Tr = np.full((T, N), np.nan); Tr[:, cols] = Tr0
    signals.EXTRA['POS'] = [M190.features(P, F, Tr)['POS250'].astype(np.float32)]
    signals.build('L134_P20', mode='fix', P=P, F=F, x_rev=('filter', 0.2, 'POS'))
    par = P['meta']['stock_id'].astype(str).str[-1].astype(int) % 2
    res = {}
    for half, keep in (('奇數代碼', 1), ('偶數代碼', 0)):
        ban = np.nonzero((par != keep).to_numpy())[0]
        for nm, sig in (('新基準', 'fix'), ('P20', 'L134_P20')):
            R = sim.run_paths('fix', NPATH, sig=sig, x_kw={'ban': ban})
            tx = E.seg_table([x[0] for x in R['X']]); tm = E.seg_table([sim.mix(R['X'][i][0], R['V'][i][0]) for i in range(NPATH)])
            res[(half, nm)] = (tx, tm)
    log('| 範圍 | 版本 | X1 單獨 Calmar（年化）：' + '｜'.join(SEGS) + ' | 合併 Calmar：' + '｜'.join(SEGS) + ' |\n|---|---|---|---|')
    verdict = []
    for half in ('奇數代碼', '偶數代碼'):
        for nm in ('新基準', 'P20'):
            tx, tm = res[(half, nm)]
            log(f"| {half} | {nm} | " + '｜'.join(f"{tx.loc[s, 'Calmar']:.2f}（{tx.loc[s, '年化']:.1%}）" for s in SEGS) + ' | ' + '｜'.join(f"{tm.loc[s, 'Calmar']:.2f}" for s in SEGS) + ' |')
        b, p = res[(half, '新基準')][0], res[(half, 'P20')][0]
        d = [p.loc[s, 'Calmar'] - b.loc[s, 'Calmar'] for s in ('2016–20', '2021–26')]
        ok = max(d) > 0 and min(d) > -0.1; verdict.append(ok)
        log(f"| {half} | 差 | 2016–20 {d[0]:+.2f}、2021–26 {d[1]:+.2f} → {'成立' if ok else '不成立'} | |")
    v = f"判準：{'✓ 兩半皆成立（穩健）' if all(verdict) else ('✗ 只有一半成立（L-131 降為待觀察）' if any(verdict) else '✗ 兩半皆不成立')}"
    log('\n' + v)
    return dict(summary='# L-134 結果\n\n**' + v + '**\n\n' + '\n'.join(LOG), headline=v, n_variants=0)
