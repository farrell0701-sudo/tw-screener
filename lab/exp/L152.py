# L-152：X1 排序加入 DH250（加權）。判準見 registry_L.md L-152。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, evalu as E
import M190, L131

N_DSR = 692
VARS = {'W20（主）': ('w', 0.2, 'DH'), 'W10': ('w', 0.1, 'DH'), 'W30': ('w', 0.3, 'DH')}


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']; T, N = P['C'].shape
    signals.build('fix', mode='fix', P=P, F=F)
    S = signals.compute('fix', P=P, F=F); dbg = S['_dbg']; cols = dbg['cols']
    Tr0 = np.asarray(dbg['Tr'], np.float64)
    Tr = Tr0 if Tr0.shape[1] == N else np.full((T, N), np.nan)
    if Tr0.shape[1] != N: Tr[:, cols] = Tr0
    del S
    C = pd.DataFrame(np.asarray(F['CA'], np.float64)).ffill(); H = pd.DataFrame(np.asarray(F['HA'], np.float64)).ffill()
    signals.EXTRA['DH'] = [(C / H.rolling(250, min_periods=200).max() - 1).to_numpy(np.float32)]
    signals.EXTRA['POS'] = [M190.features(P, F, Tr)['POS250'].astype(np.float32)]
    for nm, xr in VARS.items(): signals.build(f'L152_{nm[:3]}', mode='fix', P=P, F=F, x_rev=xr)
    signals.build('L152_P20', mode='fix', P=P, F=F, x_rev=('filter', 0.2, 'POS'))
    log = L131.log
    B = L131.go('新基準', 'fix'); O = {nm: L131.go(nm, f'L152_{nm[:3]}') for nm in VARS}; Q = L131.go('L-131 P20（對照）', 'L152_P20')
    SEGS = L131.SEGS; bt, bpm = B['t'], E.path_mean(B['M'])
    log('## 合併版（Calmar（年化）；100 路徑中位數）\n')
    log('| 版本 | ' + ' | '.join(SEGS) + ' | 05-15 回撤 | 16-26 回撤 | P（16-26） |\n|---|---|---|---|---|---|---|---|---|')
    Pv = {}
    for nm, r in [('新基準', B)] + list(O.items()) + [('L-131 P20（對照）', Q)]:
        t = r['t']; p = '' if nm == '新基準' else f"{E.boot_p(E.path_mean(r['M']), bpm):.2f}"
        if p: Pv[nm] = float(p)
        log(f"| {nm} | " + ' | '.join(f"{t.loc[s, 'Calmar']:.2f}（{t.loc[s, '年化']:.1%}）" for s in SEGS) + f" | {t.loc['2005–15', '回撤']:.1%} | {t.loc['2016–26', '回撤']:.1%} | {p} |")
    log('\n## X1 單獨（Calmar／年化／回撤）\n')
    log('| 版本 | ' + ' | '.join(SEGS[:4]) + ' |\n|---|---|---|---|---|')
    for nm, r in [('新基準', B)] + list(O.items()) + [('L-131 P20（對照）', Q)]:
        log(f"| {nm} | " + ' | '.join(f"{r['tx'].loc[s, 'Calmar']:.2f}／{r['tx'].loc[s, '年化']:.1%}／{r['tx'].loc[s, '回撤']:.1%}" for s in SEGS[:4]) + ' |')
    yrs = (dates[-1] - pd.Timestamp('2005-01-01')).days / 365.25
    log('\n## X1 交易（2005 起）\n')
    log('| 版本 | 每年進場 | 單筆平均 | 中位 | 勝率 | 與新基準相同交易比例 |\n|---|---|---|---|---|---|')
    def tx(r): return pd.DataFrame([x for i in range(L131.NPATH) for x in r['R']['X'][i][1] if x[1] >= pd.Timestamp('2005-01-01')], columns=['c', 'd0', 'd1', 'w', 'ret', 'hold', 'why', 'st', 'lk'])
    XB = tx(B); kb = set(zip(XB.c, XB.d0))
    for nm, r in [('新基準', B)] + list(O.items()) + [('L-131 P20（對照）', Q)]:
        X = tx(r); same = np.mean([(c, d) in kb for c, d in zip(X.c, X.d0)])
        log(f"| {nm} | {len(X) / L131.NPATH / yrs:.1f} | {X.ret.mean():+.2%} | {X.ret.median():+.2%} | {(X.ret > 0).mean():.0%} | {same:.0%} |")
    log('\n## 合併版逐年報酬差（路徑平均；W20 − 新基準｜P20 − 新基準）\n')
    yb = bpm.groupby(bpm.index.year).apply(lambda r: (1 + r).prod() - 1)
    ym = {k: E.path_mean(v['M']).pipe(lambda m: m.groupby(m.index.year).apply(lambda r: (1 + r).prod() - 1)) for k, v in (('W20', O['W20（主）']), ('P20', Q))}
    ys = [y for y in yb.index if y >= 2005]
    log('| 年 | ' + ' | '.join(str(y) for y in ys) + ' |\n|' + '---|' * (len(ys) + 1))
    for k in ('W20', 'P20'): log(f'| {k} | ' + ' | '.join(f'{ym[k][y] - yb[y]:+.1%}' for y in ys) + ' |')
    sr = [E.path_mean(O[k]['M']).loc['2005':].pipe(lambda r: r.mean() / r.std()) for k in VARS]
    mt = O['W20（主）']['t']
    c1 = all(mt.loc[s, 'Calmar'] > bt.loc[s, 'Calmar'] for s in ('2016–20', '2021–26')); c2 = Pv['W20（主）'] >= 0.8
    c3 = mt.loc['2005–15', '回撤'] >= bt.loc['2005–15', '回撤'] - 0.05
    plat = any(all(O[k]['t'].loc[s, 'Calmar'] >= bt.loc[s, 'Calmar'] for s in ('2016–20', '2021–26')) for k in ('W10', 'W30'))
    veto = all(O[k]['t'].loc['2005–15', 'Calmar'] < bt.loc['2005–15', 'Calmar'] for k in VARS)
    ok = c1 and c2 and c3 and plat and not veto
    ds = E.dsr(E.path_mean(mt.pipe(lambda _: O['W20（主）']['M'])).loc['2005':], N_DSR, float(np.var(sr, ddof=1)))
    log(f"\n## 判準\n\n- Calmar 兩段上升 {'✓' if c1 else '✗'}；P {Pv['W20（主）']:.2f} {'✓' if c2 else '✗'}；05-15 回撤 {'✓' if c3 else '✗'}；高原 {'✓' if plat else '✗'}；否決 {'觸發' if veto else '未觸發'}；DSR {ds:.2f} → **{'通過' if ok else '不採用'}**")
    head = f"L-152 X1 排序加 DH250：{'通過' if ok else '不採用'}（W20 P {Pv['W20（主）']:.2f}）"
    return dict(summary='# L-152 結果\n\n**' + head + '**\n\n' + '\n'.join(L131.LOG), headline=head, n_variants=len(VARS))
