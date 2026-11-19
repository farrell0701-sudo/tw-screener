# L-120：X1 閘門換成週 EMA 下彎（組合回測）。判準見 registry_L.md L-120。
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim, evalu as E
from M144 import weekly_gate

NPATH = 100; VARS = {'EMA21': 21, 'EMA16': 16, 'EMA26': 26}
SEGS = ('2005–15', '2016–20', '2021–26', '2016–26')
LOG = []
def log(s): LOG.append(s); print(s, flush=True)


def go(name, gate=None):
    t0 = time.time()
    R = sim.run_paths('fix', NPATH, sig='fix', x_kw={'gate': gate} if gate is not None else None)
    M = [sim.mix(R['X'][i][0], R['V'][i][0]) for i in range(NPATH)]
    expo = float(np.mean([R['X'][i][2].loc['2005':].mean() for i in range(NPATH)]))
    log(f'  （{name}：{time.time() - t0:.0f} 秒）')
    return dict(M=M, t=E.seg_table(M), tx=E.seg_table([x[0] for x in R['X']]), expo=expo)


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']
    signals.build('fix', mode='fix', P=P, F=F)
    ix = P['index']['TAIEX_P'].reindex(dates).ffill(); below = (ix < ix.rolling(200).mean()).to_numpy()
    B = go('新基準'); O = {}; sw = {}
    for k, n in VARS.items():
        g = weekly_gate(ix, dates, n); O[k] = go(k, g)
        s = pd.Series(g, index=dates).loc['2005':]; sw[k] = (s != s.shift()).sum() / (len(s) / 245)
    s0 = pd.Series(below, index=dates).loc['2005':]; sw['新基準'] = (s0 != s0.shift()).sum() / (len(s0) / 245)
    bt, bpm = B['t'], E.path_mean(B['M'])
    log('## 合併版 Calmar（年化）｜100 路徑中位數\n')
    log('| 版本 | ' + ' | '.join(SEGS) + ' | 05-15 回撤 | 16-26 回撤 | P（16-26） |\n|---|---|---|---|---|---|---|---|')
    Pv = {}
    for nm, r in [('新基準', B)] + list(O.items()):
        t = r['t']; p = '' if nm == '新基準' else f"{E.boot_p(E.path_mean(r['M']), bpm):.2f}"
        if p: Pv[nm] = float(p)
        log(f"| {nm} | " + ' | '.join(f"{t.loc[s, 'Calmar']:.2f}（{t.loc[s, '年化']:.1%}）" for s in SEGS) + f" | {t.loc['2005–15', '回撤']:.1%} | {t.loc['2016–26', '回撤']:.1%} | {p} |")
    log('\n## X1 單獨 Calmar、曝險、閘門切換\n')
    log('| 版本 | ' + ' | '.join(SEGS) + ' | X1 平均曝險 | 閘門切換（次／年） |\n|---|---|---|---|---|---|---|')
    for nm, r in [('新基準', B)] + list(O.items()):
        log(f"| {nm} | " + ' | '.join(f"{r['tx'].loc[s, 'Calmar']:.2f}" for s in SEGS) + f" | {r['expo']:.2f} | {sw[nm]:.1f} |")
    up = lambda k: all(O[k]['t'].loc[s, 'Calmar'] > bt.loc[s, 'Calmar'] for s in ('2016–20', '2021–26'))
    dn = lambda k: all(O[k]['t'].loc[s, 'Calmar'] < bt.loc[s, 'Calmar'] for s in ('2016–20', '2021–26'))
    m = O['EMA21']['t']
    c1 = up('EMA21'); c2 = Pv['EMA21'] >= 0.8; c3 = m.loc['2005–15', '回撤'] >= bt.loc['2005–15', '回撤'] - 0.05
    pl = dn('EMA16') and dn('EMA26'); veto = all(O[k]['t'].loc['2005–15', 'Calmar'] < bt.loc['2005–15', 'Calmar'] for k in VARS)
    ok = c1 and c2 and c3 and not pl and not veto
    log('\n## 判準（事前固定）\n')
    log(f"- EMA21 兩段 Calmar 都上升：{'成立' if c1 else '**不成立**'}（16-20 {m.loc['2016–20', 'Calmar']:.2f} vs {bt.loc['2016–20', 'Calmar']:.2f}；21-26 {m.loc['2021–26', 'Calmar']:.2f} vs {bt.loc['2021–26', 'Calmar']:.2f}）")
    log(f"- P {Pv['EMA21']:.2f} ≥ 0.8：{'成立' if c2 else '**不成立**'}")
    log(f"- 05-15 回撤不加深 > 5pt：{'成立' if c3 else '**不成立**'}")
    log(f"- 高原（16、26 不得兩者都兩段下降）：{'**失敗**' if pl else '成立'}；L-104 否決：{'**觸發**' if veto else '未觸發'}")
    log(f"\n**結論：{'通過（待使用者決定）' if ok else '不採用'}**\n")
    head = f"L-120 X1 閘門改週 EMA21：{'通過' if ok else '不採用'}（16-20 {m.loc['2016–20', 'Calmar']:.2f}、21-26 {m.loc['2021–26', 'Calmar']:.2f}、P {Pv['EMA21']:.2f}）"
    return dict(summary='# L-120 結果\n\n**' + head + '**\n\n' + '\n'.join(LOG), headline=head, n_variants=len(VARS))
