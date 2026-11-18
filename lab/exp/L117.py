# L-117：V4.1 訊號稀少時暫停 X1 新進場，新基準＋參數高原。判準見 registry_L.md L-117。
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data'))
from panel import load as load_panel
from feat import load_feat
import signals, sim, evalu as E, run as RUN

NPATH = 100
VARS = {'W20k6': (20, 6), 'W20k4': (20, 4), 'W20k8': (20, 8), 'W10k3': (10, 3), 'W40k12': (40, 12)}
MAIN = 'W20k6'
SEGS = ('2005–15', '2016–20', '2021–26', '2016–26')
BEAR = (2008, 2011, 2015, 2018, 2022)
LOG = []
def log(s): LOG.append(s); print(s, flush=True)

P = load_panel(); F = load_feat('AF')


def go(name, **sw):
    t0 = time.time()
    if name != 'fix': signals.build(name, mode='fix', P=P, F=F, **sw)
    R = sim.run_paths('fix', NPATH, sig=name)
    M = [sim.mix(R['X'][i][0], R['V'][i][0]) for i in range(NPATH)]
    d = signals.load_sig(name)['daily']
    log(f'  （{name}：{time.time() - t0:.0f} 秒）')
    expo = float(np.mean([R['X'][i][2].loc['2005':].mean() for i in range(NPATH)]))
    return dict(M=M, X=[x[0] for x in R['X']], t=E.seg_table(M), tx=E.seg_table([x[0] for x in R['X']]), d=d, expo=expo)


def main():
    signals.build('fix', mode='fix', P=P, F=F)
    B = go('fix'); O = {k: go(f'fix_sp_{k}', x_sparse=v) for k, v in VARS.items()}
    bt, bpm = B['t'], E.path_mean(B['M'])
    log('## 合併版 Calmar（年化）｜100 路徑中位數\n')
    log('| 版本 | ' + ' | '.join(SEGS) + ' | 05-15 回撤 | P（16-26） |\n|---|---|---|---|---|---|---|')
    P_ = {}
    for nm, r in [('新基準', B)] + list(O.items()):
        t = r['t']
        p = '' if nm == '新基準' else f"{E.boot_p(E.path_mean(r['M']), bpm):.2f}"
        if nm != '新基準': P_[nm] = float(p)
        log(f"| {nm} | " + ' | '.join(f"{t.loc[s, 'Calmar']:.2f}（{t.loc[s, '年化']:.1%}）" for s in SEGS) + f" | {t.loc['2005–15', '回撤']:.1%} | {p} |")
    log('\n## X1 單獨 Calmar\n')
    log('| 版本 | ' + ' | '.join(SEGS) + ' |\n|---|---|---|---|---|')
    for nm, r in [('新基準', B)] + list(O.items()):
        log(f"| {nm} | " + ' | '.join(f"{r['tx'].loc[s, 'Calmar']:.2f}" for s in SEGS) + ' |')
    log('\n## 閘門與曝險\n')
    log('| 版本 | 閘門日比例 05-15 ｜ 16-26 | 其中 200MA 閘門已關的比例 | X1 平均曝險（基準 ' + f"{B['expo']:.2f}" + '） |\n|---|---|---|---|')
    for nm, r in O.items():
        d = r['d']; a, b = slice('2005', '2015'), slice('2016', '2026')
        g = d['xgate'].astype(bool); bl = d['below'].astype(bool)
        log(f"| {nm} | {g.loc[a].mean():.0%} ｜ {g.loc[b].mean():.0%} | {(g & bl).sum() / max(g.sum(), 1):.0%} | {r['expo']:.2f} |")
    # 判準
    m = O[MAIN]['t']
    c1a = m.loc['2016–20', 'Calmar'] > bt.loc['2016–20', 'Calmar'] and m.loc['2021–26', 'Calmar'] > bt.loc['2021–26', 'Calmar']
    c1b = P_[MAIN] >= 0.8; c1c = m.loc['2005–15', '回撤'] >= bt.loc['2005–15', '回撤'] - 0.05
    nb = [k for k in VARS if k != MAIN]
    ok_nb = [k for k in nb if O[k]['t'].loc['2016–20', 'Calmar'] >= bt.loc['2016–20', 'Calmar'] and O[k]['t'].loc['2021–26', 'Calmar'] >= bt.loc['2021–26', 'Calmar']]
    low1626 = [k for k in nb if O[k]['t'].loc['2016–26', 'Calmar'] < bt.loc['2016–26', 'Calmar']]
    c2 = len(ok_nb) >= 3 and not low1626
    c3 = all(O[k]['t'].loc['2005–15', 'Calmar'] < bt.loc['2005–15', 'Calmar'] for k in VARS)
    log('\n## 判準（事前固定）\n')
    log(f"1. 主判準（{MAIN}）：兩段 Calmar 都上升 {'成立' if c1a else '**不成立**'}；P {P_[MAIN]:.2f} {'≥' if c1b else '**<**'} 0.8；05-15 回撤 {m.loc['2005–15', '回撤']:.1%}（基準 {bt.loc['2005–15', '回撤']:.1%}）{'未加深 5pt' if c1c else '**加深超過 5pt**'} → {'通過' if c1a and c1b and c1c else '**不通過**'}")
    log(f"2. 高原：兩段都不低於基準的鄰近版本 {len(ok_nb)}/4（{', '.join(ok_nb) or '無'}）；16-26 低於基準 {', '.join(low1626) or '無'} → {'成立' if c2 else '**不成立**'}")
    log(f"3. 否決（五個版本 05-15 Calmar 全低於基準）：{'**觸發**' if c3 else '未觸發'}")
    adopt = c1a and c1b and c1c and c2 and not c3
    log(f"\n**結論：{'建議採用（待使用者決定）' if adopt else '不採用'}**\n")
    # 每年報酬差
    log('## 每年合併報酬差（主版本 − 基準，路徑平均；* 為大盤空頭年）\n')
    yr = lambda r: (1 + r).groupby(r.index.year).prod() - 1
    dy = (yr(E.path_mean(O[MAIN]['M'])) - yr(bpm)).loc[2005:]
    log('| 年 | ' + ' | '.join(f"{y}{'*' if y in BEAR else ''}" for y in dy.index) + ' |\n|' + '---|' * (len(dy) + 1))
    log('| 差 | ' + ' | '.join(f'{v * 100:+.1f}' for v in dy.values) + ' |')
    e = dy.loc[:2015]; log(f"\n05-15：空頭年差合計 {e[e.index.isin(BEAR)].sum() * 100:+.1f}pt、其他年 {e[~e.index.isin(BEAR)].sum() * 100:+.1f}pt（描述性）。\n")
    # DSR
    sr = [E.met(E.path_mean(r['M']).loc['2005':])[3] / np.sqrt(252) for r in O.values()]
    n = RUN.n_trials() + len(VARS)
    ds = E.dsr(E.path_mean(O[MAIN]['M']).loc['2005':], n, float(np.var(sr, ddof=1)))
    log(f"DSR（主版本，全期日報酬，N＝{n}，試驗間變異數取本次五版本，暫定）：{ds:.2f}\n")
    head = f"L-117 訊號稀少暫停 X1 {'建議採用' if adopt else '不採用'}：主判準{'通過' if c1a and c1b and c1c else '不通過'}、高原{'成立' if c2 else '不成立'}、否決{'觸發' if c3 else '未觸發'}"
    return dict(summary='# L-117 結果\n\n**' + head + '**\n\n' + '\n'.join(LOG), headline=head, n_variants=len(VARS),
                series={'base': bpm, **{k: E.path_mean(v['M']) for k, v in O.items()}})
