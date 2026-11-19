# M-149：X1 持股數 6／10／15 下，週 EMA21 閘門 vs 200 日線（敏感度）。定義見 registry_M.md M-149。
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim, evalu as E
from M144 import weekly_gate

NP = 100; SEGS = ('2005–15', '2016–20', '2021–26', '2016–26')


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']
    signals.build('fix', mode='fix', P=P, F=F)
    ix = P['index']['TAIEX_P'].reindex(dates).ffill(); gw = weekly_gate(ix, dates, 21)
    L = ['# M-149 結果：X1 持股數分散後的週 EMA21 閘門 vs 200 日線（合併版與 X1 單獨 Calmar，100 路徑中位數）\n',
         '| X1 檔數 | 閘門 | ' + ' | '.join(f'合併 {s}' for s in SEGS) + ' | P（16-26） | ' + ' | '.join(f'X1 {s}' for s in SEGS) + ' |', '|---|---|' + '---|' * (2 * len(SEGS) + 1)]
    gaps = {}
    for n in (6, 10, 15):
        res = {}
        for nm, kw in (('200日線', {'n': n}), ('週EMA21', {'n': n, 'gate': gw})):
            t0 = time.time(); R = sim.run_paths('fix', NP, sig='fix', x_kw=kw)
            M = [sim.mix(R['X'][i][0], R['V'][i][0]) for i in range(NP)]
            res[nm] = dict(M=M, t=E.seg_table(M), tx=E.seg_table([x[0] for x in R['X']])); print(n, nm, round(time.time() - t0), flush=True)
        p = E.boot_p(E.path_mean(res['週EMA21']['M']), E.path_mean(res['200日線']['M']))
        for nm in ('200日線', '週EMA21'):
            r = res[nm]
            L.append(f"| {n} | {nm} | " + ' | '.join(f"{r['t'].loc[s, 'Calmar']:.2f}" for s in SEGS) + f" | {'' if nm == '200日線' else f'{p:.2f}'} | " + ' | '.join(f"{r['tx'].loc[s, 'Calmar']:.2f}" for s in SEGS) + ' |')
        gaps[n] = {s: res['週EMA21']['t'].loc[s, 'Calmar'] - res['200日線']['t'].loc[s, 'Calmar'] for s in SEGS}
    L += ['\n## 週 EMA21 − 200 日線（合併 Calmar 差）\n', '| X1 檔數 | ' + ' | '.join(SEGS) + ' |', '|---|' + '---|' * len(SEGS)]
    for n, g in gaps.items(): L.append(f"| {n} | " + ' | '.join(f"{g[s]:+.2f}" for s in SEGS) + ' |')
    g = [gaps[n]['2021–26'] for n in (6, 10, 15)]
    mono = g[0] < g[1] < g[2]
    verdict = '支持「路徑運氣」' if mono and g[2] >= 0 else ('支持「規則本身」' if g[1] <= g[0] and g[2] <= g[1] else '無法判斷')
    L.append(f"\n**解讀（事前規則）：2021–26 差 {g[0]:+.2f} → {g[1]:+.2f} → {g[2]:+.2f}：{verdict}**")
    return dict(summary='\n'.join(L), headline=f'持股分散敏感度：{verdict}', n_variants=0)
