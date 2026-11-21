# L-125：營收高的 X1 持股放寬停損為 15%（硬停損／固定移動停損）。判準見 registry_L.md L-125。
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim, evalu as E
from M120 import rev_factors
from L123 import daily_from_monthly

NPATH = 100; N_DSR = 579
SEGS = ('2005–15', '2016–20', '2021–26', '2016–26', '全期')
CACHE = LAB / 'cache' / 'L125'
LOG = []
def log(s): LOG.append(s); print(s, flush=True)


def go(name, sig, xstop=None):
    t0 = time.time()
    R = sim.run_paths('fix', NPATH, sig=sig, x_kw={'xstop': xstop} if xstop else None)
    M = [sim.mix(R['X'][i][0], R['V'][i][0]) for i in range(NPATH)]
    expo = float(np.mean([R['X'][i][2].loc['2005':].mean() for i in range(NPATH)]))
    log(f'  （{name}：{time.time() - t0:.0f} 秒）')
    return dict(R=R, M=M, t=E.seg_table(M), tx=E.seg_table([x[0] for x in R['X']]), expo=expo)


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']
    keys = P['meta']['security_key'].astype(str).to_numpy(); kidx = {k: j for j, k in enumerate(keys)}
    assert (LAB / 'cache' / 'sig' / 'L123_T1' / 'RKX.npy').exists()
    signals.build('fix', mode='fix', P=P, F=F)
    S = signals.compute('fix', P=P, F=F); pool = S['_dbg']['pool']; del S
    r4p = signals.prk(daily_from_monthly(rev_factors(dates)['R4 營收創高'], dates, keys), pool)
    CACHE.mkdir(parents=True, exist_ok=True); mpath = str(CACHE / 'r4hi.npy'); np.save(mpath, (r4p >= 0.8).astype(np.float32))
    VARS = {'W1': ('fix', ('hard', 0.15, mpath)), 'W2': ('fix', ('trail', 0.15, mpath)), 'W3': ('L123_T1', ('hard', 0.15, None)),
            'W4': ('L123_T1', ('trail', 0.15, None)), 'W5': ('fix', ('hard', 0.15, None)),
            'W1_12%': ('fix', ('hard', 0.12, mpath)), 'W1_18%': ('fix', ('hard', 0.18, mpath))}
    B = go('新基準', 'fix'); O = {k: go(k, sg, xs) for k, (sg, xs) in VARS.items()}
    bt, bpm = B['t'], E.path_mean(B['M'])
    log('## 合併版（Calmar（年化）；100 路徑中位數）\n')
    log('| 版本 | ' + ' | '.join(SEGS) + ' | 05-15 回撤 | 16-26 回撤 | P（16-26） |\n|---|---|---|---|---|---|---|---|---|')
    Pv = {}
    for nm, r in [('新基準', B)] + list(O.items()):
        t = r['t']; p = '' if nm == '新基準' else f"{E.boot_p(E.path_mean(r['M']), bpm):.2f}"
        if p: Pv[nm] = float(p)
        log(f"| {nm} | " + ' | '.join(f"{t.loc[s, 'Calmar']:.2f}（{t.loc[s, '年化']:.1%}）" for s in SEGS) + f" | {t.loc['2005–15', '回撤']:.1%} | {t.loc['2016–26', '回撤']:.1%} | {p} |")
    log('\n## X1 單獨（Calmar／年化／回撤）與平均曝險\n')
    log('| 版本 | ' + ' | '.join(SEGS[:4]) + ' | X1 曝險 |\n|---|---|---|---|---|---|')
    for nm, r in [('新基準', B)] + list(O.items()):
        log(f"| {nm} | " + ' | '.join(f"{r['tx'].loc[s, 'Calmar']:.2f}／{r['tx'].loc[s, '年化']:.1%}／{r['tx'].loc[s, '回撤']:.1%}" for s in SEGS[:4]) + f" | {r['expo']:.2f} |")
    log('\n## X1 交易（100 路徑去重，2005 起）：營收高持股（進場前一日 R4 池內百分位 ≥ 0.8）與全部\n')
    log('| 版本 | 每年進場（每路徑） | 營收高筆數 | 營收高平均 | 勝率 | 持有日 | 最大單筆虧損 | 營收高出場原因 | 全部平均 |\n|---|---|---|---|---|---|---|---|---|')
    yrs = (dates[-1] - pd.Timestamp('2005-01-01')).days / 365.25
    for nm, r in [('新基準', B)] + list(O.items()):
        seen = {}
        for i in range(NPATH):
            for (code, d0, d1, wt, ret, hold, why, st, lk) in r['R']['X'][i][1]:
                if d0 >= pd.Timestamp('2005-01-01'): seen.setdefault((code, d0), (code, d0, ret, hold, why))
        X = pd.DataFrame(list(seen.values()), columns=['code', 'd0', 'ret', 'hold', 'why'])
        t0 = dates.get_indexer(pd.DatetimeIndex(X.d0)) - 1; X['hi'] = r4p[t0, X.code.map(kidx).to_numpy()] >= 0.8
        h = X[X.hi]; wc = h.why.value_counts(normalize=True)
        npy = np.mean([len([x for x in r['R']['X'][i][1] if x[1] >= pd.Timestamp('2005-01-01')]) for i in range(NPATH)]) / yrs
        log(f"| {nm} | {npy:.1f} | {len(h)} | {h.ret.mean():+.2%} | {(h.ret > 0).mean():.0%} | {h.hold.mean():.0f} | {h.ret.min():+.1%} | " + '、'.join(f'{k} {v:.0%}' for k, v in wc.head(3).items()) + f" | {X.ret.mean():+.2%} |")
    ix = P['index']['TAIEX_P'].reindex(dates).ffill(); yr_ix = ix.groupby(dates.year).last().pct_change()
    m1 = E.path_mean(O['W1']['M'])
    yb = (1 + bpm).groupby(bpm.index.year).prod() - 1; y1 = (1 + m1).groupby(m1.index.year).prod() - 1
    log('\n## 每年合併報酬（W1 − 新基準；路徑平均）\n')
    log('| 年 | 新基準 | W1 | 差 | 加權指數 |\n|---|---|---|---|---|')
    for y in yb.index:
        if y < 2005: continue
        log(f"| {y}{'（空頭年）' if yr_ix.get(y, 0) < 0 else ''} | {yb[y]:+.1%} | {y1[y]:+.1%} | {y1[y] - yb[y]:+.1%} | {yr_ix.get(y, np.nan):+.1%} |")
    sr = [E.path_mean(O[k]['M']).loc['2005':].pipe(lambda r: r.mean() / r.std()) for k in VARS]
    ds = E.dsr(m1.loc['2005':], N_DSR, float(np.var(sr, ddof=1)))
    log(f"\nDSR（W1，全期日報酬，N＝{N_DSR}，試驗間變異數取本次七版本）：{ds:.2f}\n")
    mt = O['W1']['t']
    up = lambda k: all(O[k]['t'].loc[s, 'Calmar'] > bt.loc[s, 'Calmar'] for s in ('2016–20', '2021–26'))
    dn = lambda k: all(O[k]['t'].loc[s, 'Calmar'] < bt.loc[s, 'Calmar'] for s in ('2016–20', '2021–26'))
    c1 = up('W1'); c2 = Pv['W1'] >= 0.8; c3 = mt.loc['2005–15', '回撤'] >= bt.loc['2005–15', '回撤'] - 0.05
    pl = dn('W1_12%') and dn('W1_18%'); veto = all(O[k]['t'].loc['2005–15', 'Calmar'] < bt.loc['2005–15', 'Calmar'] for k in VARS)
    ok = c1 and c2 and c3 and not pl and not veto
    log('## 判準（事前固定）\n')
    log(f"- W1 兩段 Calmar 都上升：{'成立' if c1 else '**不成立**'}（16-20 {mt.loc['2016–20', 'Calmar']:.2f} vs {bt.loc['2016–20', 'Calmar']:.2f}；21-26 {mt.loc['2021–26', 'Calmar']:.2f} vs {bt.loc['2021–26', 'Calmar']:.2f}）")
    log(f"- P {Pv['W1']:.2f} ≥ 0.8：{'成立' if c2 else '**不成立**'}")
    log(f"- 05-15 回撤不加深 > 5pt：{'成立' if c3 else '**不成立**'}（{mt.loc['2005–15', '回撤']:.1%} vs {bt.loc['2005–15', '回撤']:.1%}）")
    log(f"- 高原（12%、18% 不得都兩段下降）：{'**失敗**' if pl else '成立'}；L-104 否決：{'**觸發**' if veto else '未觸發'}")
    others = [k for k in ('W2', 'W3', 'W4', 'W5') if up(k) and Pv[k] >= 0.8]
    log(f"- 其他版本單獨達主判準：{'、'.join(others) if others else '無'}（只報告，不採用）")
    log(f"\n**結論：{'通過（建議採用，待使用者決定）' if ok else '不採用'}**\n")
    head = f"L-125 營收高持股放寬停損：{'通過' if ok else '不採用'}（W1 16-20 {mt.loc['2016–20', 'Calmar']:.2f}、21-26 {mt.loc['2021–26', 'Calmar']:.2f}、P {Pv['W1']:.2f}）"
    return dict(summary='# L-125 結果\n\n**' + head + '**\n\n' + '\n'.join(LOG), headline=head, n_variants=len(VARS),
                series={'base': bpm, **{k: E.path_mean(v['M']) for k, v in O.items()}})
