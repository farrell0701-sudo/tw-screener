# M-146：L-120 週 EMA21 閘門的年度拆解（描述性）。定義見 registry_M.md M-146。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim
from M144 import weekly_gate
from M106 import worst_dd

NP = 100


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']
    signals.build('fix', mode='fix', P=P, F=F)
    ix = P['index']['TAIEX_P'].reindex(dates).ffill(); g0 = (ix < ix.rolling(200).mean()).to_numpy(); gw = weekly_gate(ix, dates, 21)
    pm = lambda ps: pd.concat(ps, axis=1).mean(axis=1)
    B = sim.run_paths('fix', NP, sig='fix'); G = sim.run_paths('fix', NP, sig='fix', x_kw={'gate': gw})
    bX, gX = pm([B['X'][i][0] for i in range(NP)]), pm([G['X'][i][0] for i in range(NP)])
    bC, gC = pm([sim.mix(B['X'][i][0], B['V'][i][0]) for i in range(NP)]), pm([sim.mix(G['X'][i][0], G['V'][i][0]) for i in range(NP)])
    bE, gE = pm([B['X'][i][2] for i in range(NP)]), pm([G['X'][i][2] for i in range(NP)])
    yr = lambda r: r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)
    ydd = lambda r: r.groupby(r.index.year).apply(lambda x: ((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min())
    T = pd.DataFrame({'合併 基準': yr(bC), '合併 EMA21': yr(gC), '合併回撤 基準': ydd(bC), '合併回撤 EMA21': ydd(gC),
                      'X1 基準': yr(bX), 'X1 EMA21': yr(gX), 'X1 曝險 基準': bE.groupby(bE.index.year).mean(), 'X1 曝險 EMA21': gE.groupby(gE.index.year).mean()}).loc[2005:]
    gs = pd.DataFrame({'A': g0 & ~gw, 'B': ~g0 & gw}, index=dates)
    d = (gX - bX).reindex(dates).fillna(0)
    state = pd.Series(np.where(gs.A, 'A', np.where(gs.B, 'B', '同')), index=dates)
    # 閘門差異的影響延續：以「最近 20 日內曾有差異」歸類
    recA = gs.A.rolling(20, min_periods=1).max().astype(bool); recB = gs.B.rolling(20, min_periods=1).max().astype(bool)
    cls = pd.Series(np.where(recA, 'A 後 20 日內', np.where(recB, 'B 後 20 日內', '其他')), index=dates)
    L = ['# M-146 結果：L-120 週 EMA21 閘門的年度拆解（100 路徑平均日報酬）\n', '## 每年\n',
         '| 年 | 合併 基準 | 合併 EMA21 | 差 | 合併回撤 基準 | 合併回撤 EMA21 | X1 基準 | X1 EMA21 | X1 曝險 基準 | X1 曝險 EMA21 | A 天數 | B 天數 |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for y, r in T.iterrows():
        a, b = int(gs.A[gs.index.year == y].sum()), int(gs.B[gs.index.year == y].sum())
        L.append(f"| {y} | {r['合併 基準']:+.1%} | {r['合併 EMA21']:+.1%} | {r['合併 EMA21'] - r['合併 基準']:+.1%} | {r['合併回撤 基準']:.1%} | {r['合併回撤 EMA21']:.1%} | {r['X1 基準']:+.1%} | {r['X1 EMA21']:+.1%} | {r['X1 曝險 基準']:.2f} | {r['X1 曝險 EMA21']:.2f} | {a} | {b} |")
    L += ['\nA＝200 日線擋、週 EMA 放行；B＝200 日線放行、週 EMA 擋。\n', '## 2021–26 合併版最大回撤\n']
    for nm, r in (('基準', bC), ('EMA21', gC)):
        L.append(f"- {nm}：" + '；'.join(f'{a.date()}→{b.date()} {x:.1%}' for a, b, x in worst_dd(r.loc['2021':], 3)))
    L += ['\n## X1 日報酬差（EMA21 − 基準）依閘門差異歸類（對數加總）\n', '| 期間 | A 後 20 日內 | B 後 20 日內 | 其他 |', '|---|---|---|---|']
    for era, lo, hi in (('2005–15', '2005', '2015'), ('2016–20', '2016', '2020'), ('2021–26', '2021', '2026')):
        m = (d.index >= lo) & (d.index <= hi + '-12-31')
        L.append(f"| {era} | " + ' | '.join(f"{np.log1p(d[m & (cls == k)]).sum():+.1%}" for k in ('A 後 20 日內', 'B 後 20 日內', '其他')) + ' |')
    L += ['\n## 2021–26 閘門切換時點（週 EMA21 下彎區間 vs 200 日線下區間）\n']
    for nm, arr in (('週 EMA21 下彎', gw), ('200 日線下', g0)):
        s = pd.Series(arr, index=dates).loc['2021':]; ch = s.ne(s.shift()); runs = []
        starts = s.index[ch & s]
        for st in starts:
            after = s.loc[st:]; end = after[~after]
            runs.append(f"{st.date()}–{(end.index[0].date() if len(end) else '迄今')}")
        L.append(f"- {nm}：" + '、'.join(runs))
    return dict(summary='\n'.join(L), headline='週 EMA 閘門年度拆解完成', n_variants=0)
