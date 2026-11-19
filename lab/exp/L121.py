# L-121：X1 閘門＝週 EMA 下彎 且 大盤 < 200 日線（D5）。判準見 registry_L.md L-121。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim, evalu as E
import L120
from M144 import weekly_gate

VARS = {'D5-EMA21': 21, 'D5-EMA16': 16, 'D5-EMA26': 26}
SEGS = L120.SEGS


def main():
    log = L120.log; LOG = L120.LOG
    P = load_panel(); F = load_feat('AF'); dates = P['dates']
    signals.build('fix', mode='fix', P=P, F=F)
    ix = P['index']['TAIEX_P'].reindex(dates).ffill(); below = (ix < ix.rolling(200).mean()).to_numpy()
    B = L120.go('新基準'); O = {}; sw = {}
    for k, n in VARS.items():
        g = weekly_gate(ix, dates, n) & below; O[k] = L120.go(k, g)
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
    log('\n## X1 單獨 Calmar、曝險、閘門切換；2023／2025 合併報酬（路徑平均）\n')
    log('| 版本 | ' + ' | '.join(SEGS) + ' | X1 曝險 | 切換（次／年） | 2023 | 2025 |\n|---|---|---|---|---|---|---|---|---|')
    for nm, r in [('新基準', B)] + list(O.items()):
        pmr = E.path_mean(r['M']); y = lambda yy: (1 + pmr.loc[str(yy)]).prod() - 1
        log(f"| {nm} | " + ' | '.join(f"{r['tx'].loc[s, 'Calmar']:.2f}" for s in SEGS) + f" | {r['expo']:.2f} | {sw[nm]:.1f} | {y(2023):+.1%} | {y(2025):+.1%} |")
    up = lambda k: all(O[k]['t'].loc[s, 'Calmar'] > bt.loc[s, 'Calmar'] for s in ('2016–20', '2021–26'))
    dn = lambda k: all(O[k]['t'].loc[s, 'Calmar'] < bt.loc[s, 'Calmar'] for s in ('2016–20', '2021–26'))
    m = O['D5-EMA21']['t']
    c1 = up('D5-EMA21'); c2 = Pv['D5-EMA21'] >= 0.8; c3 = m.loc['2005–15', '回撤'] >= bt.loc['2005–15', '回撤'] - 0.05
    pl = dn('D5-EMA16') and dn('D5-EMA26'); veto = all(O[k]['t'].loc['2005–15', 'Calmar'] < bt.loc['2005–15', 'Calmar'] for k in VARS)
    ok = c1 and c2 and c3 and not pl and not veto
    log('\n## 判準（事前固定）\n')
    log(f"- D5-EMA21 兩段 Calmar 都上升：{'成立' if c1 else '**不成立**'}（16-20 {m.loc['2016–20', 'Calmar']:.2f} vs {bt.loc['2016–20', 'Calmar']:.2f}；21-26 {m.loc['2021–26', 'Calmar']:.2f} vs {bt.loc['2021–26', 'Calmar']:.2f}）")
    log(f"- P {Pv['D5-EMA21']:.2f} ≥ 0.8：{'成立' if c2 else '**不成立**'}")
    log(f"- 05-15 回撤不加深 > 5pt：{'成立' if c3 else '**不成立**'}")
    log(f"- 高原：{'**失敗**' if pl else '成立'}；L-104 否決：{'**觸發**' if veto else '未觸發'}")
    log(f"\n**結論：{'通過（待使用者決定）' if ok else '不採用'}**\n")
    head = f"L-121 X1 閘門 D5（週EMA21且200日線）：{'通過' if ok else '不採用'}（16-20 {m.loc['2016–20', 'Calmar']:.2f}、21-26 {m.loc['2021–26', 'Calmar']:.2f}、P {Pv['D5-EMA21']:.2f}）"
    return dict(summary='# L-121 結果\n\n**' + head + '**\n\n' + '\n'.join(LOG), headline=head, n_variants=len(VARS))
