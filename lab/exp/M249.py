# M-249：多空點加波動緩衝（指數層模擬）。定義見 registry_M.md M-249。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'exp'))
from panel import load as load_panel
from M195 import HOLDOUT

COST = 0.00385


def main():
    P = load_panel(); dates = P['dates']
    s = P['index']['TAIEX_P'].reindex(dates).ffill().dropna(); d_ = s.index; px = s.to_numpy(); T = len(px)
    m = d_.to_period('M'); first = pd.Series(np.arange(T), index=d_).groupby(m).min(); mi = {p: i for i, p in enumerate(first.index)}
    line = np.full(T, np.nan)
    for t in range(T):
        k = mi[m[t]]; a = first.iloc[k - 1] if k > 0 else first.iloc[0]; w = px[a:t + 1]; line[t] = (w.max() + w.min()) / 2
    A = pd.Series(np.abs(np.diff(px, prepend=np.nan))).rolling(14).mean().to_numpy()
    r = np.r_[0, px[1:] / px[:-1] - 1]
    def states(b, hyst):
        st = np.zeros(T, bool); cur = False
        for t in range(T):
            if not np.isfinite(A[t]): st[t] = cur; continue
            if not cur and px[t] < line[t] - b * A[t]: cur = True
            elif cur and px[t] > line[t] + (b * A[t] if hyst else 0): cur = False
            st[t] = cur
        return st
    ERA = (('探索 2004–15', '2004-03-01', '2015-12-31'), ('確認 2016–24', '2016-01-01', str(HOLDOUT.date())))
    def evalu(st, lo, hi):
        i0, i1 = d_.searchsorted(pd.Timestamp(lo)), d_.searchsorted(pd.Timestamp(hi))
        expo = np.where(np.r_[False, st[:-1]], 0.2, 1.0)        # 昨日狀態決定今日曝險
        chg = np.r_[0, np.abs(np.diff(expo))] * COST
        ret = expo * r - chg; R = pd.Series(ret[i0:i1], index=d_[i0:i1]); eq = (1 + R).cumprod()
        ny = len(R) / 245; cagr = eq.iloc[-1] ** (1 / ny) - 1; mdd = (eq / eq.cummax() - 1).min()
        fl = (np.diff(st[i0:i1].astype(int)) != 0).sum() / ny
        # 段落
        segs = []; t = i0
        while t < i1:
            if st[t] and not st[t - 1]:
                e = t
                while e + 1 < i1 and st[e + 1]: e += 1
                end = min(e + 1, T - 1); Rr = px[end] / px[t] - 1; segs.append(-0.8 * Rr - 0.8 * 2 * COST); t = end + 1
            else: t += 1
        sg = np.array(segs)
        return dict(cagr=cagr, mdd=mdd, cal=cagr / abs(mdd), fl=fl, bear=st[i0:i1].mean(), nr=(sg > 0).sum(), nw=(sg <= 0).sum(), br=sg[sg > 0].sum(), bw=sg[sg <= 0].sum())
    L = ['# M-249 結果：多空點加波動緩衝（指數層，空頭時曝險 20%）\n']
    res = {}
    for era, lo, hi in ERA:
        L += [f'## {era}\n', '| 版本 | 年化 | 最大回撤 | Calmar | 每年翻轉 | 空頭天數 | 真跌段 | 誤判段 | 真跌效益 | 誤判效益 |', '|---|---|---|---|---|---|---|---|---|---|']
        bh = pd.Series(r).iloc[d_.searchsorted(pd.Timestamp(lo)):d_.searchsorted(pd.Timestamp(hi))]; eq = (1 + bh).cumprod(); ny = len(bh) / 245
        cg = eq.iloc[-1] ** (1 / ny) - 1; md = (eq / eq.cummax() - 1).min()
        L.append(f'| 指數持有 | {cg:+.1%} | {md:.1%} | {cg / abs(md):.2f} | — | — | — | — | — | — |')
        for hyst in (False, True):
            for b in (0, 0.5, 1.0, 1.5):
                if hyst and b == 0: continue
                nm = f"{'H 雙向' if hyst else 'E 只翻空'} b={b}" if b else '原規則 b=0'
                x = evalu(states(b, hyst), lo, hi); res[(era, nm)] = x
                L.append(f"| {nm} | {x['cagr']:+.1%} | {x['mdd']:.1%} | {x['cal']:.2f} | {x['fl']:.1f} | {x['bear']:.0%} | {x['nr']} | {x['nw']} | {x['br']:+.1%} | {x['bw']:+.1%} |")
        L.append('')
    noms = []
    for nm in sorted(set(k[1] for k in res)):
        if nm == '原規則 b=0': continue
        e1, e2 = ERA[0][0], ERA[1][0]
        ok = all(res[(e, nm)]['cal'] > res[(e, '原規則 b=0')]['cal'] for e in (e1, e2)) and res[(e2, nm)]['br'] >= 0.7 * res[(e2, '原規則 b=0')]['br']
        if ok: noms.append(nm)
    head = f"提名：{'、'.join(noms) or '無'}"
    return dict(summary='\n'.join(L[:1] + [f'**{head}**\n'] + L[1:]), headline=head, n_variants=0)
