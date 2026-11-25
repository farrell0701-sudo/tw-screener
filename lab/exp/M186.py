# M-186：營收股遇大盤跌破大盤多空點：M0 不動／M1 收緊停損（a=1.5/2/3）／M2 減碼到 20%／M3 全出（奇數代碼）。定義見 registry_M.md M-186。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'exp')); sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core'))
import M182, M185
from panel import load as load_panel
from feat import load_feat

COST = 0.0077; HOLD = 120; AS = (1.5, 2, 3)
ERAS3 = (('2005–15', '2005-01-01', '2015-12-31'), ('2016–20', '2016-01-01', '2020-12-31'), ('2021–26', '2021-01-01', '2026-12-31'))


def sim(S, t, below, ATR_j):
    v = S.v; e = np.searchsorted(v, t + 1)
    if e >= len(v) or v[e] > t + 5: return None
    end = np.searchsorted(v, t + HOLD, side='right') - 1
    if end <= e + 1: return None
    ent = S.o[e]; c = S.c
    def nxt(i): return (S.o[i + 1] if i + 1 <= end else c[i])
    def s15_from(i0, level_fn=None):
        for i in range(i0, end + 1):
            lv = ent * 0.85 if level_fn is None else level_fn(i)
            if c[i] < lv: return nxt(i), i
        return c[end], end
    p0, x0 = s15_from(e)
    out = {'M0': p0 / ent - 1 - COST}
    b_entry = bool(below[v[e - 1]]) if e >= 1 else False
    # 跌破事件：持有期間（含進場日）由上轉下；需前一日在上
    brk = None
    for i in range(e, end + 1):
        if below[v[i]] and not below[v[i - 1]] and i >= 1: brk = i; break
    out['_brk'] = brk is not None and brk < x0
    out['_entry_below'] = b_entry
    if brk is None or brk >= x0:   # 無跌破，或跌破前已停損 → 所有版本同 M0
        for a in AS: out[f'M1 a={a}'] = out['M0']
        out['M2'] = out['M0']; out['M3'] = out['M0']; out['_hold_if'] = np.nan
        return out
    px = nxt(brk)
    out['M3'] = px / ent - 1 - COST
    out['M2'] = 0.8 * (px / ent - 1) + 0.2 * (p0 / ent - 1) - COST
    for a in AS:
        lv_brk = max(ent * 0.85, c[brk] - a * ATR_j[v[brk]]) if np.isfinite(ATR_j[v[brk]]) else ent * 0.85
        p, _ = s15_from(e, lambda i, lv=lv_brk: ent * 0.85 if i <= brk else lv)
        out[f'M1 a={a}'] = p / ent - 1 - COST
    out['_hold_if'] = out['M0'] - out['M3']   # 續抱（M0）相對全出的差：> 0 代表全出賣早了
    return out


def main():
    P = load_panel(); F = load_feat('AF'); dates = P['dates']
    CA = np.asarray(F['CA'], np.float64); OA = np.asarray(F['OA'], np.float64)
    HA = np.asarray(F['HA'], np.float64); LA = np.asarray(F['LA'], np.float64); VALID = np.asarray(F['VALID'])
    AMT = np.asarray(P['AMT'], np.float64); ATR = np.asarray(F['ATR'], np.float64)
    D = M182.build(parity=1); D = D[D.q >= 2].reset_index(drop=True)
    below = M185.taiex_below(dates)
    cache = {}; rows = []
    for r in D.itertuples():
        S = cache.get(r.j) or cache.setdefault(r.j, M185.Stock(r.j, CA, OA, HA, LA, AMT, VALID))
        o = sim(S, r.t, below, ATR[:, r.j])
        if o is None: continue
        o['date'] = r.date; rows.append(o)
    R = pd.DataFrame(rows)
    VS = ['M0'] + [f'M1 a={a}' for a in AS] + ['M2', 'M3']
    L = ['# M-186 結果：營收股遇大盤跌破大盤多空點的處理（R6 頭部剔除 Q1；奇數代碼；F 計數 65）\n']
    res = {}
    for era, lo, hi in ERAS3:
        E = R[(R.date >= lo) & (R.date <= hi)]
        B = E[E._brk]
        L += [f'## {era}（{len(E):,} 筆；持有期間遇到跌破 {E._brk.mean():.0%}；進場時已在多空點之下 {E._entry_below.mean():.0%}）\n',
              '| 版本 | 平均 | 中位數 | 勝率 | P5 | P95 | 遇跌破者：平均 | 遇跌破者：中位數 |', '|---|---|---|---|---|---|---|---|']
        for nm in VS:
            x = E[nm]; y = B[nm]
            res[(nm, era)] = dict(med=x.median(), win=(x > 0).mean(), p5=x.quantile(.05))
            L.append(f"| {nm} | {x.mean():+.2%} | {x.median():+.2%} | {(x > 0).mean():.1%} | {x.quantile(.05):+.1%} | {x.quantile(.95):+.1%} | {y.mean():+.2%} | {y.median():+.2%} |")
        h = B._hold_if
        L += ['', f"假訊號代價（遇跌破者，續抱 M0 − 全出 M3）：平均 {h.mean():+.2%}、中位 {h.median():+.2%}；續抱較好的比例 {(h > 0).mean():.0%}；續抱比全出多 > +10pt 的比例 {(h > 0.10).mean():.0%}。",
              f"進場時已在多空點之下的交易（M0）：中位數 {E[E._entry_below].M0.median():+.2%}、勝率 {(E[E._entry_below].M0 > 0).mean():.0%}；在之上：{E[~E._entry_below].M0.median():+.2%}、{(E[~E._entry_below].M0 > 0).mean():.0%}。\n"]
    def ok(nm):
        return all(res[(nm, e)]['med'] > res[('M0', e)]['med'] and res[(nm, e)]['win'] > res[('M0', e)]['win'] and res[(nm, e)]['p5'] >= res[('M0', e)]['p5'] - 0.005 for e, _, _ in ERAS3)
    o1 = {a: ok(f'M1 a={a}') for a in AS}
    vt = ['- M1 收緊停損：' + '、'.join(f"a={a} {'✓' if o1[a] else '✗'}" for a in AS) + f" → {'提名（主版本 a=2）' if o1[2] and sum(o1.values()) >= 2 else '不提名'}",
          f"- M2 減碼到 20%：{'提名' if ok('M2') else '不提名'}", f"- M3 全出：{'提名' if ok('M3') else '不提名'}"]
    for nm in ('M2', 'M3'):
        vt[-2 if nm == 'M2' else -1] += '（' + '｜'.join(f"中位 {res[(nm, e)]['med'] - res[('M0', e)]['med']:+.1%}、勝率 {(res[(nm, e)]['win'] - res[('M0', e)]['win']) * 100:+.0f}pt、P5 {(res[(nm, e)]['p5'] - res[('M0', e)]['p5']) * 100:+.1f}pt" for e, _, _ in ERAS3) + '）'
    head = ['## 判準（對 M0；2005–15、2016–20、2021–26 三段都要：中位數↑、勝率↑、P5 不深於 −0.5pt）\n'] + vt + ['']
    return dict(summary='\n'.join(L[:1] + head + L[1:]), headline='；'.join(x[2:].split('（')[0] for x in vt), n_variants=0)
