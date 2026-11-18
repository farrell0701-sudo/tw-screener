# L-116：拿掉 X1 的 250 日持有上限（新引擎重測 L-106 X8）。判準見 registry_L.md L-116（L-106 不對稱拿掉判準）。
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data'))
from panel import load as load_panel
from feat import load_feat
import signals, sim, evalu as E, run as RUN

NPATH = 100
VARS = {'無上限': 10 ** 7, '500日': 500}
MAIN = '無上限'
SEGS = ('2005–15', '2016–20', '2021–26', '2016–26')
LOG = []
def log(s): LOG.append(s); print(s, flush=True)

P = load_panel(); F = load_feat('AF')


def go(name, cap=250):
    t0 = time.time()
    R = sim.run_paths('fix', NPATH, sig='fix', x_kw={'cap': cap})
    M = [sim.mix(R['X'][i][0], R['V'][i][0]) for i in range(NPATH)]
    tr = [t for i in range(NPATH) for t in R['X'][i][1]]
    hold = np.array([t[5] for t in tr]); ret = np.array([t[4] for t in tr])
    expo = float(np.mean([R['X'][i][2].loc['2005':].mean() for i in range(NPATH)]))
    log(f'  （{name}：{time.time() - t0:.0f} 秒）')
    return dict(M=M, t=E.seg_table(M), tx=E.seg_table([x[0] for x in R['X']]), hold=hold, ret=ret, expo=expo)


def main():
    signals.build('fix', mode='fix', P=P, F=F)
    B = go('新基準'); O = {k: go(k, v) for k, v in VARS.items()}
    bt, bpm = B['t'], E.path_mean(B['M'])
    log('## 合併版 Calmar（年化）｜100 路徑中位數\n')
    log('| 版本 | ' + ' | '.join(SEGS) + ' | 05-15 回撤 | P（16-26） |\n|---|---|---|---|---|---|---|')
    P_ = {}
    for nm, r in [('新基準', B)] + list(O.items()):
        t = r['t']; p = '' if nm == '新基準' else f"{E.boot_p(E.path_mean(r['M']), bpm):.2f}"
        if nm != '新基準': P_[nm] = float(p)
        log(f"| {nm} | " + ' | '.join(f"{t.loc[s, 'Calmar']:.2f}（{t.loc[s, '年化']:.1%}）" for s in SEGS) + f" | {t.loc['2005–15', '回撤']:.1%} | {p} |")
    log('\n## X1 單獨 Calmar 與交易\n')
    log('| 版本 | ' + ' | '.join(SEGS) + ' | X1 平均曝險 | 持有 > 250 日（筆／路徑） | 其平均報酬 | 最長持有（日） |\n|---|---|---|---|---|---|---|---|---|')
    for nm, r in [('新基準', B)] + list(O.items()):
        m = r['hold'] > 250
        log(f"| {nm} | " + ' | '.join(f"{r['tx'].loc[s, 'Calmar']:.2f}" for s in SEGS) +
            f" | {r['expo']:.2f} | {m.sum() / NPATH:.1f} | {(r['ret'][m].mean() if m.any() else float('nan')):+.1%} | {int(r['hold'].max())} |")
    # 判準（L-106 不對稱）
    m = O[MAIN]['t']
    c1 = all(m.loc[s, 'Calmar'] >= bt.loc[s, 'Calmar'] - 0.05 for s in ('2005–15', '2016–20', '2021–26'))
    c2 = m.loc['2005–15', '回撤'] >= bt.loc['2005–15', '回撤'] - 0.02
    c3 = P_[MAIN] >= 0.5
    ok = c1 and c2 and c3
    log('\n## 判準（事前固定，L-106 不對稱）\n')
    log(f"- 三段 Calmar 皆 ≥ 基準 − 0.05：{'成立' if c1 else '**不成立**'}")
    log(f"- 05-15 回撤 {m.loc['2005–15', '回撤']:.1%}（基準 {bt.loc['2005–15', '回撤']:.1%}）不深於 2pt：{'成立' if c2 else '**不成立**'}")
    log(f"- P {P_[MAIN]:.2f} ≥ 0.5：{'成立' if c3 else '**不成立**'}")
    log(f"\n**結論：{'可拿掉（待使用者決定）' if ok else '保留上限'}**（500 日版只作參考，不列入判準）\n")
    head = f"L-116 拿掉 X1 250 日上限：{'可拿掉' if ok else '保留'}（P {P_[MAIN]:.2f}）"
    return dict(summary='# L-116 結果\n\n**' + head + '**\n\n' + '\n'.join(LOG), headline=head, n_variants=len(VARS),
                series={'base': bpm, **{k: E.path_mean(v['M']) for k, v in O.items()}})
