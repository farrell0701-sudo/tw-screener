# L-129：X1 路徑停損（N 日內收盤未達進場 × (1 + k × ATR%) → 出場）。判準見 registry_L.md L-129。
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim, evalu as E

NPATH = 100; N_DSR = 596
SEGS = ('2005–15', '2016–20', '2021–26', '2016–26', '全期')
VARS = {'N20／k2（主）': (20, 2.0), 'N15／k2': (15, 2.0), 'N30／k2': (30, 2.0), 'N20／k1.5': (20, 1.5), 'N20／k2.5': (20, 2.5)}
LOG = []
def log(s): LOG.append(s); print(s, flush=True)


def go(name, pstop=None):
    t0 = time.time()
    R = sim.run_paths('fix', NPATH, sig='fix', x_kw={'pstop': pstop} if pstop else None)
    M = [sim.mix(R['X'][i][0], R['V'][i][0]) for i in range(NPATH)]
    log(f'  （{name}：{time.time() - t0:.0f} 秒）')
    return dict(R=R, M=M, t=E.seg_table(M), tx=E.seg_table([x[0] for x in R['X']]))


def trades(R, dates):
    rows = []
    for i in range(NPATH):
        for (code, d0, d1, wt, ret, hold, why, st, lk) in R['X'][i][1]:
            if d0 >= pd.Timestamp('2005-01-01'): rows.append((i, code, d0, ret, hold, why))
    return pd.DataFrame(rows, columns=['i', 'code', 'd0', 'ret', 'hold', 'why'])


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']
    signals.build('fix', mode='fix', P=P, F=F)
    B = go('新基準'); O = {k: go(k, v) for k, v in VARS.items()}
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
    log('\n## X1 交易（2005 起；每路徑平均）\n')
    log('| 版本 | 每年進場 | 平均持有日 | 單筆平均 | 中位 | 勝率 | 出場原因 | 路徑停損出場：平均／中位 |\n|---|---|---|---|---|---|---|---|')
    for nm, r in [('新基準', B)] + list(O.items()):
        X = trades(r['R'], dates); wc = X.why.value_counts(normalize=True); ps = X[X.why == '路徑停損']
        log(f"| {nm} | {len(X) / NPATH / yrs:.1f} | {X.hold.mean():.0f} | {X.ret.mean():+.2%} | {X.ret.median():+.2%} | {(X.ret > 0).mean():.0%} | "
            + '、'.join(f'{k} {v:.0%}' for k, v in wc.head(4).items()) + f" | {'' if ps.empty else f'{ps.ret.mean():+.2%}／{ps.ret.median():+.2%}'} |")
    sr = [E.path_mean(O[k]['M']).loc['2005':].pipe(lambda r: r.mean() / r.std()) for k in VARS]
    m1 = E.path_mean(O['N20／k2（主）']['M'])
    ds = E.dsr(m1.loc['2005':], N_DSR, float(np.var(sr, ddof=1)))
    log(f"\nDSR（主版本，全期日報酬，N＝{N_DSR}）：{ds:.2f}\n")
    mt = O['N20／k2（主）']['t']
    c1 = all(mt.loc[s, 'Calmar'] > bt.loc[s, 'Calmar'] for s in ('2016–20', '2021–26'))
    c2 = Pv['N20／k2（主）'] >= 0.8
    c3 = mt.loc['2005–15', '回撤'] >= bt.loc['2005–15', '回撤'] - 0.05
    plat = sum(all(O[k]['t'].loc[s, 'Calmar'] >= bt.loc[s, 'Calmar'] for s in ('2016–20', '2021–26')) for k in list(VARS)[1:])
    veto = all(O[k]['t'].loc['2005–15', 'Calmar'] < bt.loc['2005–15', 'Calmar'] for k in VARS)
    ok = c1 and c2 and c3 and plat >= 3 and not veto
    log('## 判準\n')
    log(f"- 2016–20、2021–26 Calmar 都上升：{'✓' if c1 else '✗'}（{mt.loc['2016–20', 'Calmar']:.2f} vs {bt.loc['2016–20', 'Calmar']:.2f}；{mt.loc['2021–26', 'Calmar']:.2f} vs {bt.loc['2021–26', 'Calmar']:.2f}）")
    log(f"- P ≥ 0.8：{'✓' if c2 else '✗'}（{Pv['N20／k2（主）']:.2f}）")
    log(f"- 2005–15 回撤不加深 > 5pt：{'✓' if c3 else '✗'}（{mt.loc['2005–15', '回撤']:.1%} vs {bt.loc['2005–15', '回撤']:.1%}）")
    log(f"- 高原（4 個變體中 ≥ 3 個兩段 Calmar 不低於新基準）：{'✓' if plat >= 3 else '✗'}（{plat}/4）")
    log(f"- 否決（所有變體 2005–15 Calmar 變差）：{'觸發' if veto else '未觸發'}")
    log(f"\n**結論：{'採用候選（通過）' if ok else '不採用'}**")
    head = f"L-129 X1 路徑停損：{'通過' if ok else '不通過'}（主版本 Calmar {mt.loc['2016–20', 'Calmar']:.2f}／{mt.loc['2021–26', 'Calmar']:.2f} vs {bt.loc['2016–20', 'Calmar']:.2f}／{bt.loc['2021–26', 'Calmar']:.2f}；P {Pv['N20／k2（主）']:.2f}）"
    return dict(summary='# L-129 結果\n\n**' + head + '**\n\n' + '\n'.join(LOG), headline=head, n_variants=len(VARS),
                series={'base': bpm, **{k: E.path_mean(v['M']) for k, v in O.items()}})
