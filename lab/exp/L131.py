# L-131：X1 候選剔除 POS250 或 R6 後段（filter）。判準見 registry_L.md L-131。
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim, evalu as E
import M180, M190
from L123 import daily_from_monthly

NPATH = 100; N_DSR = 607
SEGS = ('2005–15', '2016–20', '2021–26', '2016–26', '全期')
VARS = {'P20（主）': ('filter', 0.2, 'POS'), 'P10': ('filter', 0.1, 'POS'), 'P30': ('filter', 0.3, 'POS'),
        'R20（主）': ('filter', 0.2, 'R6'), 'R10': ('filter', 0.1, 'R6'), 'R30': ('filter', 0.3, 'R6')}
LOG = []
def log(s): LOG.append(s); print(s, flush=True)


def go(name, sig):
    t0 = time.time()
    R = sim.run_paths('fix', NPATH, sig=sig)
    M = [sim.mix(R['X'][i][0], R['V'][i][0]) for i in range(NPATH)]
    log(f'  （{name}：{time.time() - t0:.0f} 秒）')
    return dict(R=R, M=M, t=E.seg_table(M), tx=E.seg_table([x[0] for x in R['X']]))


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; keys = P['meta']['security_key'].astype(str).to_numpy()
    signals.build('fix', mode='fix', P=P, F=F)
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; cols = dbg['cols']; T, N = P['C'].shape
    Tr0 = np.asarray(dbg['Tr'], np.float64)
    if Tr0.shape[1] == N: Tr = Tr0
    else: Tr = np.full((T, N), np.nan); Tr[:, cols] = Tr0
    del S
    FE = M190.features(P, F, Tr)
    signals.EXTRA['POS'] = [FE['POS250'].astype(np.float32)]
    signals.EXTRA['R6'] = [daily_from_monthly(M180.r6_factors(dates)['R6 營收短長比'], dates, keys)]
    for nm, xr in VARS.items():
        signals.build(f'L131_{nm[:3]}', mode='fix', P=P, F=F, x_rev=xr)
    B = go('新基準', 'fix'); O = {nm: go(nm, f'L131_{nm[:3]}') for nm in VARS}
    bt, bpm = B['t'], E.path_mean(B['M'])
    log('## 合併版（Calmar（年化）；100 路徑中位數）\n')
    log('| 版本 | ' + ' | '.join(SEGS) + ' | 05-15 回撤 | 16-26 回撤 | P（16-26） |\n|---|---|---|---|---|---|---|---|---|')
    Pv = {}
    for nm, r in [('新基準', B)] + list(O.items()):
        t = r['t']; p = '' if nm == '新基準' else f"{E.boot_p(E.path_mean(r['M']), bpm):.2f}"
        if p: Pv[nm] = float(p)
        log(f"| {nm} | " + ' | '.join(f"{t.loc[s, 'Calmar']:.2f}（{t.loc[s, '年化']:.1%}）" for s in SEGS) + f" | {t.loc['2005–15', '回撤']:.1%} | {t.loc['2016–26', '回撤']:.1%} | {p} |")
    log('\n## X1 單獨（Calmar／年化／回撤）\n')
    log('| 版本 | ' + ' | '.join(SEGS[:4]) + ' |\n|---|---|---|---|---|')
    for nm, r in [('新基準', B)] + list(O.items()):
        log(f"| {nm} | " + ' | '.join(f"{r['tx'].loc[s, 'Calmar']:.2f}／{r['tx'].loc[s, '年化']:.1%}／{r['tx'].loc[s, '回撤']:.1%}" for s in SEGS[:4]) + ' |')
    yrs = (dates[-1] - pd.Timestamp('2005-01-01')).days / 365.25
    log('\n## X1 交易（2005 起）與 2020 年報酬差\n')
    log('| 版本 | 每年進場 | 單筆平均 | 中位 | 勝率 | 2020 合併報酬差 |\n|---|---|---|---|---|---|')
    y20b = (1 + bpm.loc['2020']).prod() - 1
    for nm, r in [('新基準', B)] + list(O.items()):
        X = pd.DataFrame([x for i in range(NPATH) for x in r['R']['X'][i][1] if x[1] >= pd.Timestamp('2005-01-01')], columns=['c', 'd0', 'd1', 'w', 'ret', 'hold', 'why', 'st', 'lk'])
        m = E.path_mean(r['M']); y20 = (1 + m.loc['2020']).prod() - 1
        log(f"| {nm} | {len(X) / NPATH / yrs:.1f} | {X.ret.mean():+.2%} | {X.ret.median():+.2%} | {(X.ret > 0).mean():.0%} | {y20 - y20b:+.1%} |")
    sr = [E.path_mean(O[k]['M']).loc['2005':].pipe(lambda r: r.mean() / r.std()) for k in VARS]
    log('\n## 判準\n')
    heads = []
    for fam, main_, others in (('P', 'P20（主）', ('P10', 'P30')), ('R', 'R20（主）', ('R10', 'R30'))):
        mt = O[main_]['t']
        c1 = all(mt.loc[s, 'Calmar'] > bt.loc[s, 'Calmar'] for s in ('2016–20', '2021–26')); c2 = Pv[main_] >= 0.8
        c3 = mt.loc['2005–15', '回撤'] >= bt.loc['2005–15', '回撤'] - 0.05
        plat = any(all(O[k]['t'].loc[s, 'Calmar'] >= bt.loc[s, 'Calmar'] for s in ('2016–20', '2021–26')) for k in others)
        veto = all(O[k]['t'].loc['2005–15', 'Calmar'] < bt.loc['2005–15', 'Calmar'] for k in (main_,) + others)
        ok = c1 and c2 and c3 and plat and not veto
        ds = E.dsr(E.path_mean(O[main_]['M']).loc['2005':], N_DSR, float(np.var(sr, ddof=1)))
        log(f"- {fam} 系列：Calmar 兩段上升 {'✓' if c1 else '✗'}（{mt.loc['2016–20', 'Calmar']:.2f}／{mt.loc['2021–26', 'Calmar']:.2f} vs {bt.loc['2016–20', 'Calmar']:.2f}／{bt.loc['2021–26', 'Calmar']:.2f}）；"
            f"P {Pv[main_]:.2f} {'✓' if c2 else '✗'}；05-15 回撤 {'✓' if c3 else '✗'}；高原 {'✓' if plat else '✗'}；否決 {'觸發' if veto else '未觸發'}；DSR {ds:.2f} → **{'通過' if ok else '不採用'}**")
        heads.append(f"{fam}{'通過' if ok else '不採用'}（P {Pv[main_]:.2f}）")
    head = 'L-131 X1 剔除 POS250／R6 後段：' + '；'.join(heads)
    return dict(summary='# L-131 結果\n\n**' + head + '**\n\n' + '\n'.join(LOG), headline=head, n_variants=len(VARS),
                series={'base': bpm, **{k: E.path_mean(v['M']) for k, v in O.items()}})
