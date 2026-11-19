# M-114：X1 持股「上收斂 且 自高點回落 X×ATR」提前出場的反事實（描述性）。定義見 registry_M.md M-114。
import sys
from pathlib import Path
import numpy as np, pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from feat import load_feat
import signals, sim
from M113 import states

NP = 100; XS = (1.5, 2.0, 2.5, 3.0)


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']
    signals.build('fix', mode='fix', P=P, F=F)
    S = signals.load_sig('fix'); ATRP = np.asarray(S['ATRP'], np.float64)
    st, _ = states(F)
    CA = np.asarray(F['CA'], np.float64); OA = np.asarray(F['OA'], np.float64)
    kix = {k: j for j, k in enumerate(P['meta']['security_key'])}
    R = sim.run_paths('fix', NP, sig='fix')
    seen = {}
    for i in range(NP):
        for (code, d0, d1, w, ret, hold, why, stl, lk) in R['X'][i][1]:
            if why in ('期末未平倉',): continue
            seen.setdefault((code, d0), dict(j=kix[code], t0=dates.get_loc(d0), t1=dates.get_loc(d1), ret=ret, d0=d0))
    T = pd.DataFrame(seen.values())
    rows = []
    for x in T.itertuples():
        j, t0, t1 = x.j, x.t0, x.t1
        px = OA[t0, j] * (1 + sim.SLIP)
        if not np.isfinite(px): continue
        c = CA[t0:t1, j]; hi = np.maximum.accumulate(np.r_[px, np.nan_to_num(c, nan=-np.inf)])[1:]
        atr = ATRP[t0:t1, j] * c; s0 = st[t0:t1, j] == 0
        rec = dict(d0=x.d0, ret=x.ret)
        for X in XS:
            dd = c < hi - X * atr
            for tag, cond in (('c', dd & s0), ('u', dd)):
                k = np.nonzero(np.nan_to_num(cond, nan=0).astype(bool))[0]
                r = x.ret; hit = False
                if len(k):
                    t = t0 + k[0] + 1
                    if t < t1:
                        o = OA[t, j] if np.isfinite(OA[t, j]) else CA[t - 1, j]
                        if np.isfinite(o): r = (1 - sim.FEE) * o * (1 - sim.SLIP) * (1 - sim.FEE - sim.TAX) / px - 1; hit = True
                rec[f'{tag}{X}'] = r; rec[f'h{tag}{X}'] = hit
        rows.append(rec)
    D = pd.DataFrame(rows)
    L = ['# M-114 結果：X1「上收斂 且 自高點回落 X×ATR」提前出場（反事實，忽略騰出名額再投入）\n',
         'c＝有上收斂條件；u＝無條件（對照）。報酬為每筆平均（含成本）。\n']
    for era, lo, hi in (('2005–15', '2005', '2015-12-31'), ('2016–26', '2016', '2026-12-31')):
        d = D[(D.d0 >= lo) & (D.d0 <= hi)]; big = d.ret >= 0.30
        L += [f'## {era}（{len(d)} 筆，原平均 {d.ret.mean():+.2%}；原報酬 ≥ +30% 者 {big.sum()} 筆）\n',
              '| X | 版本 | 受影響比例 | 受影響者原平均 | 反事實平均 | 差（全部交易） | ≥ +30% 被砍比例 | 被砍大贏家平均少賺 |', '|---|---|---|---|---|---|---|---|']
        for X in XS:
            for tag, nm in (('c', '有條件'), ('u', '無條件')):
                h = d[f'h{tag}{X}']; r = d[f'{tag}{X}']
                cut = big & h
                L.append(f"| {X} | {nm} | {h.mean():.0%} | {d.ret[h].mean():+.2%} | {r[h].mean():+.2%} | {(r - d.ret).mean():+.2%} | {cut.sum() / max(big.sum(), 1):.0%} | {(r - d.ret)[cut].mean():+.1%} |")
            L.append(f"| {X} | 有條件 − 無條件 | | | | {(d[f'c{X}'] - d[f'u{X}']).mean():+.2%} | | |")
    return dict(summary='\n'.join(L), headline='上收斂出場反事實完成', n_variants=0)
