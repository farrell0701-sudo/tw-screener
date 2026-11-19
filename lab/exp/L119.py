# L-119：V4.1 空頭停損寬度依 ATR 調整（出場機制）。判準見 registry_L.md L-119。
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data'))
from panel import load as load_panel
from feat import load_feat
import signals, sim, evalu as E

NPATH = 100
VARS = {'S1': (2.0, 0.05, 0.15), 'S1a': (1.5, 0.05, 0.15), 'S1b': (2.5, 0.05, 0.15), 'S2': (2.0, 0.03, 0.15), 'S0': 'none'}
SEGS = ('2005–15', '2016–20', '2021–26', '2016–26')
LOG = []
def log(s): LOG.append(s); print(s, flush=True)


def go(name, P, vstop=None):
    t0 = time.time()
    R = sim.run_paths('fix', NPATH, sig='fix', v_kw={'vstop': vstop} if vstop is not None else None)
    M = [sim.mix(R['X'][i][0], R['V'][i][0]) for i in range(NPATH)]
    trv = [t for i in range(NPATH) for t in R['V'][i][1]]
    stops = [t for t in trv if t[6] == '空頭5%停損']
    Cf = P['_Cf']; dates = P['dates']; kix = P['_kix']
    after = []
    for (code, d0, d1, *_r) in stops:
        j = kix[code]; t1 = dates.get_loc(d1)
        if t1 + 20 < len(dates) and np.isfinite(Cf[t1, j]): after.append(Cf[t1 + 20, j] / Cf[t1, j] - 1)
    log(f'  （{name}：{time.time() - t0:.0f} 秒）')
    return dict(M=M, t=E.seg_table(M), tv=E.seg_table([x[0] for x in R['V']]), nstop=len(stops) / NPATH,
                sret=float(np.mean([t[4] for t in stops])) if stops else np.nan, after=float(np.nanmean(after)) if after else np.nan)


def main():
    P = load_panel(); F = load_feat('AF')
    signals.build('fix', mode='fix', P=P, F=F)
    P['_Cf'] = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill().to_numpy()
    P['_kix'] = {k: j for j, k in enumerate(P['meta']['security_key'])}
    B = go('新基準', P); O = {k: go(k, P, v) for k, v in VARS.items()}
    bt, bpm = B['t'], E.path_mean(B['M'])
    log('## 合併版 Calmar（年化）｜100 路徑中位數\n')
    log('| 版本 | ' + ' | '.join(SEGS) + ' | 05-15 回撤 | P（16-26） |\n|---|---|---|---|---|---|---|')
    Pv = {}
    for nm, r in [('新基準', B)] + list(O.items()):
        t = r['t']; p = '' if nm == '新基準' else f"{E.boot_p(E.path_mean(r['M']), bpm):.2f}"
        if p: Pv[nm] = float(p)
        log(f"| {nm} | " + ' | '.join(f"{t.loc[s, 'Calmar']:.2f}（{t.loc[s, '年化']:.1%}）" for s in SEGS) + f" | {t.loc['2005–15', '回撤']:.1%} | {p} |")
    log('\n## V4.1 單獨 Calmar 與空頭停損\n')
    log('| 版本 | ' + ' | '.join(SEGS) + ' | 空頭停損（筆／路徑） | 停損筆平均報酬 | 停損後 20 日股價 |\n|---|---|---|---|---|---|---|---|')
    for nm, r in [('新基準', B)] + list(O.items()):
        log(f"| {nm} | " + ' | '.join(f"{r['tv'].loc[s, 'Calmar']:.2f}" for s in SEGS) + f" | {r['nstop']:.1f} | {r['sret']:+.1%} | {r['after']:+.1%} |")
    up = lambda nm: all(O[nm]['t'].loc[s, 'Calmar'] > bt.loc[s, 'Calmar'] for s in ('2016–20', '2021–26'))
    m = O['S1']['t']
    c1 = up('S1'); c2 = Pv['S1'] >= 0.8; c3 = m.loc['2005–15', '回撤'] >= bt.loc['2005–15', '回撤'] - 0.05
    plateau_fail = all(all(O[v]['t'].loc[s, 'Calmar'] < bt.loc[s, 'Calmar'] for s in ('2016–20', '2021–26')) for v in ('S1a', 'S1b'))
    veto = all(O[v]['t'].loc['2005–15', 'Calmar'] < bt.loc['2005–15', 'Calmar'] for v in VARS)
    ok = c1 and c2 and c3 and not plateau_fail and not veto
    log('\n## 判準（事前固定）\n')
    log(f"- S1 兩段 Calmar 都上升：{'成立' if c1 else '**不成立**'}")
    log(f"- S1 P {Pv['S1']:.2f} ≥ 0.8：{'成立' if c2 else '**不成立**'}")
    log(f"- S1 05-15 回撤不加深 > 5pt：{'成立' if c3 else '**不成立**'}")
    log(f"- 高原（S1a、S1b 不得兩者都兩段下降）：{'**失敗**' if plateau_fail else '成立'}；L-104 否決：{'**觸發**' if veto else '未觸發'}")
    log(f"\n**結論：{'通過（待使用者決定）' if ok else '不採用'}**\n")
    head = f"L-119 V4.1 空頭停損依 ATR：{'通過' if ok else '不採用'}（S1 16-20 {m.loc['2016–20', 'Calmar']:.2f}、21-26 {m.loc['2021–26', 'Calmar']:.2f}，P {Pv['S1']:.2f}）"
    return dict(summary='# L-119 結果\n\n**' + head + '**\n\n' + '\n'.join(LOG), headline=head, n_variants=len(VARS))
