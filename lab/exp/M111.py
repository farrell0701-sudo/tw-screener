# M-111：M-110 B 版本的參數高原（描述性）。定義見 registry_M.md M-111。
import sys
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim
from M110 import wrsi, triggers, stats

NP = 100
VARS = {'B（武裝 30、回到 40、60 日）': (30, 40, 60), '回到 35': (30, 35, 60), '回到 45': (30, 45, 60),
        '武裝 25': (25, 40, 60), '武裝 35': (35, 40, 60), '窗口 40 日': (30, 40, 40), '窗口 120 日': (30, 40, 120)}


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T = len(dates)
    signals.build('fix', mode='fix', P=P, F=F)
    ix = P['index']['TAIEX_P'].reindex(dates).ffill(); below = (ix < ix.rolling(200).mean()).to_numpy()
    wk = pd.Series(dates, index=dates).groupby(dates.to_period('W-FRI')).max(); r = wrsi(ix.loc[wk.values], 14)
    pm = lambda ps: pd.concat(ps, axis=1).mean(axis=1)
    B = sim.run_paths('fix', NP, sig='fix'); rV = [B['V'][i][0] for i in range(NP)]
    rows = [('基準', pm([B['X'][i][0] for i in range(NP)]), pm([sim.mix(B['X'][i][0], rV[i]) for i in range(NP)]), [])]
    for nm, (arm, fire, win) in VARS.items():
        trig = [d for d in triggers(r, arm, fire) if below[dates.get_loc(d)]]
        un = np.zeros(T, bool)
        for d in trig: s = dates.get_loc(d); un[s:s + win] = True
        U = sim.run_paths('fix', NP, sig='fix', x_kw={'unlock': un})
        rows.append((nm, pm([U['X'][i][0] for i in range(NP)]), pm([sim.mix(U['X'][i][0], rV[i]) for i in range(NP)]), trig))
    f = lambda t: f"{t[0]:+.1%} / {t[1]:.1%} / {t[2]:.2f}"
    b = stats(rows[0][1], '2005', '2015')
    L = ['# M-111 結果：B 版本參數高原（描述性、不能提名）\n', '| 版本 | X1 2005–15 | X1 2016–26 | 合併 2005–15 | 合併 2016–26 | 在高原內 |', '|---|---|---|---|---|---|']
    inn = []
    for nm, rx, rc, trig in rows:
        x = stats(rx, '2005', '2015'); ok = '' if nm == '基準' else ('✓' if x[2] >= b[2] and x[1] >= b[1] - 0.02 else '✗')
        if ok: inn.append(ok == '✓')
        L.append(f"| {nm} | {f(x)} | {f(stats(rx, '2016', '2026'))} | {f(stats(rc, '2005', '2015'))} | {f(stats(rc, '2016', '2026'))} | {ok} |")
    L += ['\n## 觸發日\n'] + [f"- {nm}：" + '、'.join(str(d.date()) for d in trig) for nm, _, _, trig in rows[1:]]
    head = f"高原：{sum(inn)}/{len(inn)} 個變體在高原內"
    return dict(summary='\n'.join(L), headline=head, n_variants=0)
