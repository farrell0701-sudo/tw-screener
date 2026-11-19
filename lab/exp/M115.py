# M-115：轉弱廣度（上收斂＋120 在下）累積後的跌幅（描述性）。定義見 registry_M.md M-115。
import sys
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim
from M113 import states

NP = 100; ERAS = (('2005–15', '2005-01-01', '2015-12-31'), ('2016–26', '2016-01-01', '2026-12-31'))


def fwd_ret(r, h): return np.expm1(np.log1p(r)[::-1].rolling(h).sum()[::-1].shift(-1))


def fwd_mdd(eq, h):
    v = eq.values; out = np.full(len(v), np.nan)
    for t in range(len(v) - h): out[t] = (v[t + 1:t + 1 + h] / v[t]).min() - 1
    return pd.Series(np.minimum(out, 0), index=eq.index)


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']
    signals.build('fix', mode='fix', P=P, F=F)
    S = signals.compute('fix', P=P, F=F); U, pool = S['_dbg']['U'], S['_dbg']['pool']; del S
    st, above = states(F); ok = st >= 0; weak = ok & ((st == 0) | (st >= 3))
    W = {}
    for nm, den in (('全市場', U & ok), ('強勢池', pool & ok)):
        n = den.sum(1).astype(float); n[n == 0] = np.nan; W[nm] = pd.Series((weak & den).sum(1) / n, index=dates)
    na = (U & ok).sum(1).astype(float); na[na == 0] = np.nan; up200 = pd.Series((above & U & ok).sum(1) / na, index=dates)
    ix = P['index']['TAIEX_P'].reindex(dates).ffill(); below = (ix < ix.rolling(200).mean())
    R = sim.run_paths('fix', NP, sig='fix'); rC = pd.concat([sim.mix(R['X'][i][0], R['V'][i][0]) for i in range(NP)], axis=1).mean(axis=1).reindex(dates).fillna(0)
    ixr = ix.pct_change().fillna(0)
    Y = pd.DataFrame({'指數20': fwd_ret(ixr, 20), '指數60': fwd_ret(ixr, 60), '指數60跌': fwd_mdd(ix, 60),
                      '合併20': fwd_ret(rC, 20), '合併60': fwd_ret(rC, 60), '合併60跌': fwd_mdd((1 + rC).cumprod(), 60)})
    L = ['# M-115 結果：轉弱廣度（上收斂＋120 在下）累積後的跌幅（描述性、不能提名）\n',
         'W＝水準、ΔW＝20 日變化；「60 跌」＝後 60 日內自 t 收盤起的最深回落（平均）；「大跌率」＝後 60 日指數最深回落 ≤ −10% 的比例。\n']
    for era, lo, hi in ERAS:
        sel = (dates >= lo) & (dates <= hi); y = Y[sel]
        L += [f'## {era}\n', '### 相關\n', '| 廣度 | 平均 | 與大盤<200MA | 與站上200比例 |', '|---|---|---|---|']
        for nm, w in W.items():
            for tag, s in (('W', w), ('ΔW', w - w.shift(20))):
                s = s[sel]; L.append(f"| {nm} {tag} | {s.mean():+.1%} | {s.corr(below[sel].astype(float)):+.2f} | {s.corr(up200[sel]):+.2f} |")
        for cond, cn in ((np.ones(sel.sum(), bool), '全部日子'), ((~below[sel]).values, '只看大盤 ≥ 200MA')):
            L += [f'\n### 分五組（{cn}）\n', '| 廣度 | 組 | 天數 | 後20日指數 | 後60日指數 | 指數60跌 | 大跌率 | 後20日合併 | 後60日合併 | 合併60跌 |', '|---|---|---|---|---|---|---|---|---|---|']
            for nm, w in W.items():
                for tag, s in (('W', w), ('ΔW', w - w.shift(20))):
                    s = s[sel][cond]; yy = y[cond]; m = s.notna().values
                    q = pd.qcut(s[m].rank(method='first'), 5, labels=False).values
                    for k in (0, 2, 4):
                        x = yy[m][q == k]
                        L.append(f"| {nm} {tag} | Q{k + 1} | {len(x)} | {x.指數20.mean():+.2%} | {x.指數60.mean():+.2%} | {x.指數60跌.mean():.1%} | {(x.指數60跌 <= -0.10).mean():.0%} | "
                                 f"{x.合併20.mean():+.2%} | {x.合併60.mean():+.2%} | {x.合併60跌.mean():.1%} |")
        L.append('')
    return dict(summary='\n'.join(L), headline='轉弱廣度描述完成', n_variants=0)
