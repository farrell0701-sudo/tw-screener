# L-130：V4.1 分批停利＋2ATR 硬停損（觸價單，滑價 0.2%）。判準見 registry_L.md L-130。
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim, evalu as E

NPATH = 100; N_DSR = 601
SEGS = ('2005–15', '2016–20', '2021–26', '2016–26', '全期')
def lad(m=1.0, b3=2.0, keep=True):
    return dict(stop=2.0 * m, steps=[(2 * m, 0.5, 1 * m), (3 * m, 0.5, b3 * m), (4 * m, 1.0, None)], slip=0.002, keep_time=keep)
VARS = {'A（主）': lad(), 'B（3ATR後保本+3）': lad(b3=3.0), 'A 無期滿': lad(keep=False), 'A×0.75': lad(0.75), 'A×1.25': lad(1.25)}
LOG = []
def log(s): LOG.append(s); print(s, flush=True)


def go(name, ladder=None):
    t0 = time.time()
    R = sim.run_paths('fix', NPATH, sig='fix', v_kw={'ladder': ladder} if ladder else None)
    M = [sim.mix(R['X'][i][0], R['V'][i][0]) for i in range(NPATH)]
    log(f'  （{name}：{time.time() - t0:.0f} 秒）')
    return dict(R=R, M=M, t=E.seg_table(M), tv=E.seg_table([x[0] for x in R['V']]))


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']
    signals.build('fix', mode='fix', P=P, F=F)
    B = go('新基準'); O = {k: go(k, v) for k, v in VARS.items()}
    bt, bpm = B['t'], E.path_mean(B['M']); bvm = E.path_mean([x[0] for x in B['R']['V']])
    log('## V4.1 單獨（Calmar（年化）／回撤；100 路徑中位數）\n')
    log('| 版本 | ' + ' | '.join(SEGS[:4]) + ' | P（V 單獨 16-26） |\n|---|---|---|---|---|---|')
    for nm, r in [('新基準', B)] + list(O.items()):
        p = '' if nm == '新基準' else f"{E.boot_p(E.path_mean([x[0] for x in r['R']['V']]), bvm):.2f}"
        log(f"| {nm} | " + ' | '.join(f"{r['tv'].loc[s, 'Calmar']:.2f}（{r['tv'].loc[s, '年化']:.1%}）／{r['tv'].loc[s, '回撤']:.1%}" for s in SEGS[:4]) + f" | {p} |")
    log('\n## 合併版（Calmar（年化））\n')
    log('| 版本 | ' + ' | '.join(SEGS) + ' | 05-15 回撤 | 16-26 回撤 | P（16-26） |\n|---|---|---|---|---|---|---|---|---|')
    Pv = {}
    for nm, r in [('新基準', B)] + list(O.items()):
        t = r['t']; p = '' if nm == '新基準' else f"{E.boot_p(E.path_mean(r['M']), bpm):.2f}"
        if p: Pv[nm] = float(p)
        log(f"| {nm} | " + ' | '.join(f"{t.loc[s, 'Calmar']:.2f}（{t.loc[s, '年化']:.1%}）" for s in SEGS) + f" | {t.loc['2005–15', '回撤']:.1%} | {t.loc['2016–26', '回撤']:.1%} | {p} |")
    yrs = (dates[-1] - pd.Timestamp('2005-01-01')).days / 365.25
    log('\n## V4.1 交易（2005 起；100 路徑合計）\n')
    log('| 版本 | 每年進場（每路徑） | 平均持有日 | 單筆平均 | 中位 | 勝率 | P5 | P95 | 出場原因（前五） |\n|---|---|---|---|---|---|---|---|---|')
    for nm, r in [('新基準', B)] + list(O.items()):
        X = pd.DataFrame([x for i in range(NPATH) for x in r['R']['V'][i][1] if x[1] >= pd.Timestamp('2005-01-01')],
                         columns=['code', 'd0', 'd1', 'w', 'ret', 'hold', 'why', 'st', 'lk'])
        wc = X.why.value_counts(normalize=True).head(5)
        log(f"| {nm} | {len(X) / NPATH / yrs:.1f} | {X.hold.mean():.0f} | {X.ret.mean():+.2%} | {X.ret.median():+.2%} | {(X.ret > 0).mean():.0%} | {X.ret.quantile(.05):+.1%} | {X.ret.quantile(.95):+.1%} | "
            + '、'.join(f'{k} {v:.0%}' for k, v in wc.items()) + ' |')
    log('\n（出場原因 @n＝出場時已達到的停利級數：0＝未達 +2ATR、1＝達 +2ATR、2＝達 +3ATR、3＝達 +4ATR）')
    sr = [E.path_mean(O[k]['M']).loc['2005':].pipe(lambda r: r.mean() / r.std()) for k in VARS]
    ds = E.dsr(E.path_mean(O['A（主）']['M']).loc['2005':], N_DSR, float(np.var(sr, ddof=1)))
    log(f"\nDSR（主版本合併，N＝{N_DSR}）：{ds:.2f}\n")
    mt = O['A（主）']['t']
    c1 = all(mt.loc[s, 'Calmar'] > bt.loc[s, 'Calmar'] for s in ('2016–20', '2021–26')); c2 = Pv['A（主）'] >= 0.8
    c3 = mt.loc['2005–15', '回撤'] >= bt.loc['2005–15', '回撤'] - 0.05
    plat = any(all(O[k]['t'].loc[s, 'Calmar'] >= bt.loc[s, 'Calmar'] for s in ('2016–20', '2021–26')) for k in ('A×0.75', 'A×1.25'))
    veto = all(O[k]['t'].loc['2005–15', 'Calmar'] < bt.loc['2005–15', 'Calmar'] for k in VARS)
    ok = c1 and c2 and c3 and plat and not veto
    log('## 判準（合併版）\n')
    log(f"- 2016–20、2021–26 Calmar 都上升：{'✓' if c1 else '✗'}（{mt.loc['2016–20', 'Calmar']:.2f} vs {bt.loc['2016–20', 'Calmar']:.2f}；{mt.loc['2021–26', 'Calmar']:.2f} vs {bt.loc['2021–26', 'Calmar']:.2f}）")
    log(f"- P ≥ 0.8：{'✓' if c2 else '✗'}（{Pv['A（主）']:.2f}）")
    log(f"- 2005–15 回撤不加深 > 5pt：{'✓' if c3 else '✗'}（{mt.loc['2005–15', '回撤']:.1%} vs {bt.loc['2005–15', '回撤']:.1%}）")
    log(f"- 高原（×0.75 或 ×1.25 兩段不低於新基準）：{'✓' if plat else '✗'}")
    log(f"- 否決（所有變體 2005–15 變差）：{'觸發' if veto else '未觸發'}")
    log(f"\n**結論：{'採用候選（通過）' if ok else '不採用'}**")
    head = f"L-130 V4.1 分批停利＋2ATR 停損：{'通過' if ok else '不通過'}（合併 Calmar {mt.loc['2016–20', 'Calmar']:.2f}／{mt.loc['2021–26', 'Calmar']:.2f} vs {bt.loc['2016–20', 'Calmar']:.2f}／{bt.loc['2021–26', 'Calmar']:.2f}；P {Pv['A（主）']:.2f}）"
    return dict(summary='# L-130 結果\n\n**' + head + '**\n\n' + '\n'.join(LOG), headline=head, n_variants=len(VARS),
                series={'base': bpm, **{k: E.path_mean(v['M']) for k, v in O.items()}})
